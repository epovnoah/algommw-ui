"""绘图用的数据切片(照 ccm `diagnostics/slices.py`):视图与小图要画的数据全部从这里取。

纯 numpy:输入 ProcessedFrame(及参数),输出数组 / dict,不缓存、不碰 Qt。只读
`ProcessedFrame.products` 里 C 导出的中间量并换算显示单位,不在 Python 里重写处理算法。
帧的中间量已出缓存(products 为 None)或该板没有这个量时返回 None,视图据此显示
「该帧中间量已出缓存」/「不适用」。

dB 口径:RD 图是**绝对**电平(rd_log2 × CFAR_DB_PER_LOG2,幅度 dB);功率量(距离像、
子带功率)是 10·log10(功率);测角谱相对峰值。多普勒轴一律 fftshift 零频居中,列号与
`views.common.rd_column` 一致;轴坐标 TDM 为 m/s,DDM 为子带内 bin(折叠基带)。
"""

import numpy as np

from .metrics import CFAR_DB, doa_index

#: 功率 → dB 时的下限(避免 log(0))
_FLOOR = 1e-30
#: 测角探针后端编号 → (行轴名, 列轴名);同 binding.dtypes.DIAG_PROBE_*
_PROBE_AXES = {
    0: ("Wz(归一化空间频率)", "Wx(归一化空间频率)"),
    1: ("—", "Wx(归一化空间频率)"),
    2: ("俯仰 °", "方位 °"),
    3: ("—", "u = sin(方位)·cos(俯仰)"),
}
#: 指标名 → 所属级(detection_scatter 着色用)
METRIC_STAGES = {
    "fold_margin_db": "doppler",
    "empty_suppression_db": "doppler",
    "cfar_margin_db": "cfar",
    "cfar_range_margin_db": "cfar",
    "cfar_doppler_margin_db": "cfar",
    "coherence": "doa",
    "track_nis": "track",
}
TRACK_STATES = {1: "待确认", 2: "已确认"}


def _db10(power) -> np.ndarray:
    """功率 → 10·log10,C 连续 float32。"""
    p = np.maximum(np.asarray(power, np.float64), _FLOOR)
    return np.ascontiguousarray(10.0 * np.log10(p), dtype=np.float32)


def _get(frame, key: str):
    products = frame.products
    return None if products is None else products.get(key)


def _cell(frame, k: int) -> tuple[int, int]:
    det = frame.detections
    return int(det.range_bin[k]), int(det.doppler_bin[k])


def doppler_axis(radar) -> tuple[np.ndarray, np.ndarray, str]:
    """fftshift 后的列序(列 j 对应原 bin order[j])、轴坐标与单位。"""
    n = radar.n_doppler
    order = np.fft.fftshift(np.arange(n))
    signed = np.arange(n) - n // 2
    if radar.modulation == "tdm":
        return order, signed * radar.doppler_step_mps, "m/s"
    return order, signed.astype(np.float64), "bin"


def doppler_coord(radar, doppler_bin: int) -> float:
    """无符号多普勒 bin → 零频居中轴上的坐标(与 doppler_axis 同口径)。"""
    n = radar.n_doppler
    column = (doppler_bin + n // 2) % n
    _, axis, _ = doppler_axis(radar)
    return float(axis[column])


# ---------------------------------------------------------------- 色阶
def display_levels_db(
    image_db: np.ndarray,
    floor_margin_db: float = 3.0,
    top_quantile: float = 99.95,
    min_span_db: float = 20.0,
    ceiling_db: float | None = 0.0,
) -> tuple[float, float]:
    """功率热图的自动色阶(照 ccm `slices.display_levels_db`):下限取噪声底(中位数)再低
    floor_margin_db,上限取 top_quantile 分位,让最强的少数单元饱和、目标落在色阶中段。
    取整到 1 dB,跨度至少 min_span_db。图是相对本帧峰值的 dB 时上限不超过 ceiling_db
    (默认 0 dB);绝对电平的图(如 rd_db)传 ceiling_db=None。"""
    finite = image_db[np.isfinite(image_db)]
    if finite.size == 0:
        return -40.0, 0.0
    low = float(np.floor(np.median(finite) - floor_margin_db))
    high = float(np.ceil(np.percentile(finite, top_quantile)))
    if ceiling_db is not None:
        high = min(high, ceiling_db)
    if high - low < min_span_db:
        low = high - min_span_db
    return low, high


def adc_complex(raw):
    """IMRE 顺序的数组转复数,只使用 NumPy。"""
    raw = np.asarray(raw)
    if raw.ndim == 0 or raw.shape[-1] != 2:
        raise ValueError("ADC 数组末轴必须是 IMRE 一对")
    return raw[..., 1].astype(np.float32) + 1j * raw[..., 0].astype(np.float32)


# ---------------------------------------------------------------- ADC
def adc_waveforms(frame, rx: int | None, chirp: int) -> dict | None:
    """选中 chirp 的时域波形(实部 / 虚部,经 binding.native.adc_complex 解释 IMRE 定序)。

    rx 为 None 时给各 RX:real / imag 形状 (n_rx, n_samples);否则 (n_samples,)。
    real_sampling = 该 chirp 虚部全零(AFM761 是实采样)。"""
    raw = _get(frame, "adc_raw")
    if raw is None:
        return None

    x = adc_complex(raw[chirp])  # (n_rx, n_samples)
    real_sampling = not bool(np.any(x.imag))
    if rx is not None:
        x = x[rx]
    return {
        "sample": np.arange(x.shape[-1]),
        "real": np.ascontiguousarray(x.real, np.float32),
        "imag": np.ascontiguousarray(x.imag, np.float32),
        "real_sampling": real_sampling,
    }


def adc_image_db(frame, rx: int) -> np.ndarray | None:
    """选中 RX 的 chirp × 采样幅度图,(n_chirps, n_samples) float32,dBFS(相对 32768;
    幅度小于 1 LSB 的按 1 LSB 计)。"""
    raw = _get(frame, "adc_raw")
    if raw is None:
        return None

    amplitude = np.maximum(np.abs(adc_complex(raw[:, rx])), 1.0)
    return np.ascontiguousarray(20.0 * np.log10(amplitude / 32768.0), np.float32)


# ---------------------------------------------------------------- 距离 FFT
def range_axis_m(radar) -> np.ndarray:
    """有效距离门的距离轴(实采样只有前半门有效,见 RadarInfo.gates)。"""
    return np.arange(radar.gates) * float(radar.range_step_m)


def dcsig_power(dcsig: dict, n_range: int) -> np.ndarray:
    """DC 签名按 dcsig 的布局放回距离 bin 的功率 (n_rx, n_range):各分组 |签名|² 的平均
    (每分组 chirp 数相同,即全帧平均)。槽位 → bin:先正侧 0..,再负侧回绕到高 bin 端
    (core/src/dpu/range/dcsig.c 的 prvBinOfSlot)。签名未就绪或功能关闭时全零(没有补偿)。"""
    n_rx = int(dcsig["n_rx"])
    out = np.zeros((n_rx, n_range), np.float64)
    if not (dcsig["enabled"] and dcsig["ready"]):
        return out
    n_groups, n_pos, n_neg = (
        int(dcsig["n_groups"]),
        int(dcsig["n_pos_bins"]),
        int(dcsig["n_neg_bins"]),
    )
    slots = np.arange(n_pos + n_neg)
    bins = np.where(slots < n_pos, slots, n_range - n_neg + (slots - n_pos))
    sig = dcsig["sig"][: n_groups * n_rx, : n_pos + n_neg].reshape(n_groups, n_rx, -1)
    out[:, bins] = (np.abs(sig.astype(np.complex128)) ** 2).mean(axis=0)
    return out


def range_profiles_db(frame, chirp: int) -> dict | None:
    """距离像:选中 chirp(Σ_rx |X|²)对照全帧平均,各 RX;DC 签名去除前 / 后。

    products 只留了距离级之后的功率(range_power = Σ_rx、range_profile = 各 RX 的 chirp 平均),
    所以「选中 chirp」只有各 RX 之和。去除前 = 去除后 + 签名,在功率域相加(略去交叉项;
    签名就是校准时的复均值,残差的均值近零时交叉项近零)。"""
    p = frame.products
    if p is None:
        return None
    after = p["range_profile"].astype(np.float64)  # (n_rx, n_range)
    sig = dcsig_power(
        p["dcsig"], after.shape[1]
    )  # 负侧槽位按完整 n_range 回绕,最后再裁
    g = frame.radar.gates
    after, sig = after[:, :g], sig[:, :g]
    return {
        "range_m": range_axis_m(frame.radar),
        "chirp_db": _db10(p["range_power"][chirp][:g]),  # Σ_rx
        "mean_sum_db": _db10(after.sum(axis=0)),  # Σ_rx 的全帧平均(与 chirp_db 同口径)
        "mean_db": _db10(after),  # (n_rx, gates),DC 签名去除后
        "mean_before_db": _db10(after + sig),  # 去除前
        "dcsig_active": bool(sig.any()),
    }


def rti_db(frame) -> np.ndarray | None:
    """距离-chirp 瀑布图 (n_chirps, gates),10·log10(Σ_rx |X|²)。"""
    power = _get(frame, "range_power")
    return None if power is None else _db10(power[:, : frame.radar.gates])


# ---------------------------------------------------------------- RD / CFAR
def rd_db(frame) -> dict | None:
    """RD 图绝对电平:image (gates, n_doppler) float32 = rd_log2 × CFAR_DB_PER_LOG2,
    多普勒轴 fftshift 零频居中;range_m / doppler 轴与单位。"""
    rd = _get(frame, "rd_log2")
    if rd is None:
        return None
    order, axis, unit = doppler_axis(frame.radar)
    image = rd[: frame.radar.gates][:, order].astype(np.float32) * np.float32(CFAR_DB)
    return {
        "image": np.ascontiguousarray(image),
        "range_m": range_axis_m(frame.radar),
        "doppler": axis,
        "doppler_unit": unit,
    }


def cfar_profiles(frame, k: int, window: int | None = None) -> dict | None:
    """过第 k 个检出点的距离向 / 多普勒向剖面:功率 / 噪声 / 门限(= 噪声 + 门限)三条线(dB)。

    距离向取 [r − window, r + window](None = 全程),轴为米;多普勒向全程,fftshift 后升序,
    轴同 rd_db。该向关闭时 noise_db / threshold_db 为 NaN、on = False。检出点处
    「power_db − threshold_db」即 metrics 的该向余量。"""
    p = frame.products
    if p is None:
        return None
    cfar, rd = p["cfar"], p["rd_log2"]
    r, d = _cell(frame, k)
    n_range = min(rd.shape[0], frame.radar.gates)
    lo, hi = (
        (0, n_range)
        if window is None
        else (max(0, r - window), min(n_range, r + window + 1))
    )
    rows = np.arange(lo, hi)
    order, axis, unit = doppler_axis(frame.radar)

    def lines(power, noise, thr, on):
        power = power.astype(np.float64)
        if not on:
            empty = np.full(power.shape, np.nan)
            return power * CFAR_DB, empty, empty.copy()
        noise = noise.astype(np.float64)
        return power * CFAR_DB, noise * CFAR_DB, (noise + thr) * CFAR_DB

    rp, rn, rt = lines(
        rd[rows, d],
        cfar["noise_range"][rows, d],
        (
            cfar["thr_range_by_range"][rows]
            if "thr_range_by_range" in cfar
            else cfar["thr_range"]
        ),
        cfar["range_on"],
    )
    dp, dn, dt = lines(
        rd[r, order],
        cfar["noise_doppler"][r, order],
        (
            cfar["thr_doppler_by_range"][r]
            if "thr_doppler_by_range" in cfar
            else cfar["thr_doppler"]
        ),
        cfar["doppler_on"],
    )
    radar = frame.radar
    return {
        "range": {
            "axis": rows * float(radar.range_step_m),
            "power_db": rp,
            "noise_db": rn,
            "threshold_db": rt,
            "on": bool(cfar["range_on"]),
            "cursor": r * float(radar.range_step_m),
            "index": r - lo,  # 检出点在本剖面里的下标
        },
        "doppler": {
            "axis": axis,
            "unit": unit,
            "power_db": dp,
            "noise_db": dn,
            "threshold_db": dt,
            "on": bool(cfar["doppler_on"]),
            "cursor": doppler_coord(radar, d),
            "index": (d + radar.n_doppler // 2) % radar.n_doppler,
        },
        "flags": int(cfar["flags"][r, d]),  # binding.dtypes.DIAG_CFAR_* 位
    }


# ---------------------------------------------------------------- DDM
def ddm_raw_matrix_db(frame) -> dict | None:
    """DDM 原始二维矩阵:距离 × 全局慢时间 bin(= band_pow 按 b·n_perSub + d 重排),
    image[r, b·n_perSub + d] = 10·log10(band_pow[r, d, b]);TDM 为 None。"""
    band = _get(frame, "band_pow")
    if band is None:
        return None
    band = band[: frame.radar.gates]
    n_range, n_per, n_sub = band.shape
    image = band.transpose(0, 2, 1).reshape(n_range, n_sub * n_per)
    return {
        "image": _db10(image),
        "range_m": range_axis_m(frame.radar),
        "bin": np.arange(n_sub * n_per),
        "n_per_sub": n_per,
        "n_sub": n_sub,
    }


def ddm_slow_time_db(frame, range_bin: int) -> dict | None:
    """选中距离门的完整慢时间谱(全局 bin,10·log10 Σ_rx |X|²)与每个 bin 所在子带。"""
    band = _get(frame, "band_pow")
    if band is None:
        return None
    _, n_per, n_sub = band.shape
    bins = np.arange(n_sub * n_per)
    return {
        "bin": bins,
        "power_db": _db10(band[range_bin].T.reshape(-1)),
        "subband": bins // n_per,
    }


def ddm_candidates_db(frame, k: int) -> dict | None:
    """第 k 个检出点格的各候选 fold 度量,相对选中 fold 的 dB;lead_db = 选中领先其余最大者多少。"""
    metric = _get(frame, "fold_metric")
    if metric is None:
        return None
    r, d = _cell(frame, k)
    fold = int(frame.products["fold"][r, d])
    values = metric[r, d].astype(np.float64)
    relative = 10.0 * np.log10(np.maximum(values, _FLOOR) / max(values[fold], _FLOOR))
    others = np.delete(relative, fold)
    return {
        "fold": fold,
        "relative_db": relative,
        "lead_db": float(-others.max()) if others.size else float("nan"),
    }


def ddm_tx_bins(frame, k: int) -> dict | None:
    """选中 fold 下各 TX 副本所在子带与全局 bin((f + t) mod n_sub,t < n_tx),及空子带。"""
    band = _get(frame, "band_pow")
    if band is None:
        return None
    _, n_per, n_sub = band.shape
    r, d = _cell(frame, k)
    fold = int(frame.products["fold"][r, d])
    tx = (fold + np.arange(frame.radar.n_tx)) % n_sub
    empty = np.setdiff1d(np.arange(n_sub), tx)
    return {
        "fold": fold,
        "tx_bands": tx,
        "tx_bins": tx * n_per + d,
        "empty_bands": empty,
        "empty_bins": empty * n_per + d,
    }


def fold_map(frame) -> np.ndarray | None:
    """每个 RD 单元选中的 fold,(gates, n_doppler) uint8,多普勒轴与 rd_db 同样 fftshift。"""
    fold = _get(frame, "fold")
    if fold is None:
        return None
    order, _, _ = doppler_axis(frame.radar)
    return np.ascontiguousarray(fold[: frame.radar.gates][:, order])


# ---------------------------------------------------------------- 测角
def doa_spectrum(frame, k: int, probe) -> dict | None:
    """用调用方的 binding.native.DoaProbe 对第 k 个检出点的快照取整张谱。

    power_db 相对峰值;azimuth_cut / elevation_cut 是过峰的列向(方位)/ 行向(俯仰)切片;
    通道残差:contrib_k = w_k·s_k 相对 Σcontrib 的相位(°)与相对最强通道的幅度(dB)。
    probe 为 None / 不可用、无中间量或该点没有测角记录时返回 None。"""
    if probe is None or not getattr(probe, "available", False):
        return None
    p = frame.products
    if p is None:
        return None
    index = int(doa_index(frame)[k])
    snap = p["doa"]["snap"]
    if index < 0 or index >= len(snap):
        return None
    out = probe.probe(snap[index])
    power = out["power"].astype(np.float64)
    row, col = out["peak_row"], out["peak_col"]
    spectrum = 10.0 * np.log10(np.maximum(power, _FLOOR) / max(power[row, col], _FLOOR))
    contrib = out["contrib"].astype(np.complex128)
    total = contrib.sum()
    magnitude = np.abs(contrib)
    with np.errstate(divide="ignore"):
        amplitude = 20.0 * np.log10(magnitude / max(magnitude.max(), _FLOOR))
    row_label, col_label = _PROBE_AXES.get(out["backend"], ("行", "列"))
    n = len(contrib)
    return {
        "power_db": np.ascontiguousarray(spectrum, np.float32),
        "axis_row": out["axis_row"],
        "axis_col": out["axis_col"],
        "row_label": row_label,
        "col_label": col_label,
        "backend": out["backend"],
        "peak_row": row,
        "peak_col": col,
        "azimuth_deg": float(out["azimuth_deg"]),
        "elevation_deg": float(out["elevation_deg"]),
        "azimuth_cut": spectrum[row, :],
        "elevation_cut": spectrum[:, col],
        "phase_deg": np.degrees(np.angle(contrib * np.conj(total))),
        "amplitude_db": amplitude,
        "coherence": float(out["peak_pow"]) / (n * float(out["snap_pow"])),
        "reason": out["reason"],
    }


# ---------------------------------------------------------------- 跟踪
def track_table(frame) -> dict | None:
    """本帧每条非 Free 航迹(products["track"]["tracks"],传感器系)的列表。"""
    track = _get(frame, "track")
    if track is None:
        return None
    t = track["tracks"]
    s = np.asarray(t["s"], np.float64).reshape(-1, 9)
    return {
        "id": np.asarray(t["id"], np.int64),
        "state": np.asarray(t["state"], np.int64),
        "state_name": [TRACK_STATES.get(int(v), str(int(v))) for v in t["state"]],
        "age": np.asarray(t["age"], np.int64),
        "position": s[:, 0:3],
        "velocity": s[:, 3:6],
        "acceleration": s[:, 6:9],
        "p_diag": np.asarray(t["p_diag"], np.float64),
        "n_assoc": np.asarray(t["n_assoc"], np.int64),
        "nis": np.asarray(t["nis"], np.float64),
        "ended_ids": np.asarray(track["ended_ids"], np.int64),
    }


def track_history(frames, track_id: int, n: int) -> dict:
    """从调用方给的若干帧里,取该航迹出现过的最近 n 帧(按帧序号)的世界系位置与 NIS。

    位置取 frame.tracks(对外输出的已确认航迹,待确认时为 NaN);NIS 取该帧中间量,
    中间量已出缓存时为 NaN。"""
    rows = []
    for frame in sorted(frames, key=lambda f: f.index):
        hit = np.flatnonzero(frame.tracks.track_id == track_id)
        table = track_table(frame)
        row = [] if table is None else np.flatnonzero(table["id"] == track_id)
        if len(hit) or len(row):
            xyz = frame.tracks.xyz[hit[0]] if len(hit) else np.full(3, np.nan)
            nis = table["nis"][row[0]] if len(row) else np.nan
            rows.append((frame.index, xyz, nis))
    rows = rows[-n:] if n > 0 else []
    return {
        "index": np.array([r[0] for r in rows], np.int64),
        "xyz": np.array([r[1] for r in rows], np.float64).reshape(-1, 3),
        "nis": np.array([r[2] for r in rows], np.float64),
    }


def gates(frame) -> dict | None:
    """各航迹本帧的预测量测与矩形门(半宽),量测空间:距离 m、方位 / 俯仰 °、径向速度 m/s。
    本帧新建的航迹没有预测,为 NaN。"""
    track = _get(frame, "track")
    if track is None:
        return None
    t = track["tracks"]
    hs = np.asarray(t["hs"], np.float64).reshape(-1, 4)
    limits = np.asarray(t["limits"], np.float64).reshape(-1, 4)
    to_deg = np.array([1.0, np.degrees(1.0), np.degrees(1.0), 1.0])
    return {
        "id": np.asarray(t["id"], np.int64),
        "center": hs * to_deg,  # [range, az, el, doppler]
        "half_width": limits * to_deg,
    }


# ---------------------------------------------------------------- 散点
def detection_scatter(frame, metric: str) -> dict | None:
    """全帧「功率 – 指标」散点:power_db(检出格 RD 绝对电平)、value(DetectionMetrics 的
    该指标)与所属级的等级。帧还没有诊断时为 None。"""
    diag = frame.diagnostics
    if diag is None or diag.metrics is None:
        return None
    m = diag.metrics
    return {
        "power_db": m.power_db,
        "value": np.asarray(getattr(m, metric), np.float64),
        "level": diag.verdicts.levels[METRIC_STAGES[metric]],
    }


# ---------------------------------------------------------------- 瀑布时间轴
#: 时间轴上一帧「距离向一列」的固定行数
WATERFALL_ROWS = 64


def waterfall_column(frame, rows: int = WATERFALL_ROWS) -> np.ndarray | None:
    """一帧的「距离向一列」:每个有效距离门取多普勒向最大值(绝对 dB,同 rd_db 口径),
    下采样(取组内最大)到固定 rows 行,第 0 行是最近的距离。没有 RD 中间量时返回 None。"""
    rd = _get(frame, "rd_log2")
    if rd is None:
        return None
    g = frame.radar.gates
    column = rd[:g].max(axis=1).astype(np.float32) * np.float32(CFAR_DB)
    if g >= rows:
        edges = np.linspace(0, g, rows + 1).astype(np.int64)[:-1]  # g ≥ rows ⇒ 严格递增
        return np.maximum.reduceat(column, edges)
    return column[np.arange(rows) * g // rows]  # 门数不足:按比例重复


# ---------------------------------------------------------------- 步进频解耦
def fine_resolution_m(radar) -> float:
    """细距离分辨率 c / (2 (N−1) |Δf|) = |S_f|·N / (N−1)(m);无步进为 0。"""
    n = int(radar.n_slow)
    if not radar.stepped or n < 2:
        return 0.0
    return abs(float(radar.fine_range_step_m)) * n / (n - 1)


_decouple_cache: tuple | None = None


def decouple_table(frame) -> dict | None:
    """最终检出点第 k 行对应的解耦记录(products["decouple"] 与 CFAR 之后的点同序,
    最终检出点经测角 / 聚类后可能少或多,所以按单元匹配)。

    匹配键:距离门 = round(range_bin),带内多普勒 bin = round(global_bin) mod 子带内 bin 数
    (记录里的小数部分是亚 bin 偏移,整数部分就是 usRangeBin / usDopplerBin)。同一单元的
    多个检测(测角第二峰)共用同一条记录。无步进(没有该键)或没有中间量时返回 None。

    返回 dict(都是 (n_det,) 数组,没有记录的检测为 NaN / period 0 / index −1):
    index、period(展开周期 n)、coarse_range_m(粗距离 rb·S_r,未扣偏置)、global_bin(g)。
    单项缓存:列表逐格取值时同一帧只匹配一次。"""
    global _decouple_cache
    cached = _decouple_cache
    if cached is not None and cached[0] is frame:
        return cached[1]
    dec = _get(frame, "decouple")
    if dec is None:
        return None
    det = frame.detections
    n_per = int(frame.radar.n_doppler)
    cells: dict[tuple[int, int], int] = {}
    if "cell_range_bin" in dec:
        # 采集时记下的整数单元(峰值分组关闭后非峰值单元的亚 bin 偏移可超过半格)
        keys = zip(dec["cell_range_bin"].tolist(), dec["cell_doppler_bin"].tolist())
    else:
        keys = zip(
            np.rint(dec["range_bin"]).astype(np.int64).tolist(),
            (np.rint(dec["global_bin"]).astype(np.int64) % n_per).tolist(),
        )
    for i, key in enumerate(keys):
        cells.setdefault(key, i)
    index = np.full(len(det), -1, np.int64)
    for k, key in enumerate(zip(det.range_bin.tolist(), det.doppler_bin.tolist())):
        index[k] = cells.get(key, -1)
    hit = index >= 0
    period = np.zeros(len(det), np.int64)
    coarse = np.full(len(det), np.nan)
    gbin = np.full(len(det), np.nan)
    period[hit] = np.asarray(dec["period"])[index[hit]]
    coarse[hit] = np.asarray(dec["coarse_range"], np.float64)[index[hit]]
    gbin[hit] = np.asarray(dec["global_bin"], np.float64)[index[hit]]
    out = {
        "index": index,
        "period": period,
        "coarse_range_m": coarse,
        "global_bin": gbin,
    }
    _decouple_cache = (frame, out)
    return out


def decouple_summary(frame) -> tuple[int, int] | None:
    """(有解耦记录的检测数, 其中落在第 0 周期的个数);没有解耦产物时 None。"""
    table = decouple_table(frame)
    if table is None:
        return None
    hit = table["index"] >= 0
    return int(hit.sum()), int((hit & (table["period"] == 0)).sum())


def _solve(wave, range_bin, global_bin, period):
    """解耦示意视图的两条线性关系,用 Python 换算显示坐标。"""
    fine, ratio = float(wave.xFineRangeStep), float(wave.xStepSlopeRatio)
    if fine == 0:
        return 0.0, 0.0
    slow = global_bin + period * wave.ulNumChirpsPerFrame
    distance = (range_bin * wave.xRangeStep - slow * ratio * fine) / (1 - ratio)
    velocity = (slow - distance / fine) * wave.xDopplerStep
    return distance, velocity


def decouple_candidates(config, frame, k: int) -> dict | None:
    """选中检测 k 的「两把尺」数据:选中的周期解与相邻周期(n±1)的候选。

    候选的距离 / 速度调 core 导出的求解器 `vDecoupleSolve`(binding.native)用 n±1 联立求得;
    求解器的距离不含偏置修正,所以把「检测的距离 − 求解器在 n 的距离」当偏置补到候选上。
    注意相邻周期的解联立后**距离几乎不变、速度差 ±N·ΔV**(粗距离方程只经 ρ 耦合到速度),
    它们之所以被排除,是速度超出 ±N·ΔV/2;细尺读数 (g + n·N)·|S_f| 才是每个周期差
    N·|S_f| 的那个量,所以「两把尺」画的是 ruler_m,候选旁另标它的速度。
    没有解耦产物 / 该检测没有记录 / 没有共享库时返回 None。"""
    table = decouple_table(frame)
    if table is None or table["index"][k] < 0:
        return None
    radar, det = frame.radar, frame.detections
    dec = frame.products["decouple"]
    i = int(table["index"][k])
    rb, g = float(dec["range_bin"][i]), float(dec["global_bin"][i])
    n = int(dec["period"][i])
    # 双 PRF 时 B 帧的 S_f / ρ / ΔV 与 A 不同(反向步进时 S_f 还反号):按本帧的 ΔV 认波形。
    wave = config.xWave
    if config.xPair.bEnable:
        step = float(radar.doppler_step_mps)
        alt = config.xAlternateWave
        if abs(step - float(alt.xDopplerStep)) < abs(step - float(wave.xDopplerStep)):
            wave = alt
    r0, _ = _solve(wave, rb, g, n)
    offset = float(det.range_m[k]) - r0
    n_slow = int(radar.n_slow)
    fine = abs(float(radar.fine_range_step_m))
    cands = []
    for m in (n - 1, n + 1):
        r, v = _solve(wave, rb, g, m)
        cands.append(
            {
                "period": m,
                "range_m": r + offset,
                "velocity_mps": v,
                # 细尺读数:第 m 个周期里同一个读数 g 的位置 (g + m·N)·|S_f|
                "ruler_m": (g + m * n_slow) * fine,
            }
        )
    step = float(radar.range_step_m)
    gate = int(det.range_bin[k])
    return {
        "range_m": float(det.range_m[k]),
        "velocity_mps": float(det.velocity_mps[k]),
        "period": n,
        "ruler_m": (g + n * n_slow) * fine,
        "candidates": cands,
        "v_limit_mps": 0.5 * n_slow * float(radar.doppler_step_mps),
        "gate": gate,
        "gate_lo_m": (gate - 0.5) * step,
        "gate_hi_m": (gate + 0.5) * step,
        "limit_m": radar.gates * step,
        "period_m": abs(float(radar.fine_range_step_m)) * int(radar.n_slow),
        "fine_step_m": abs(float(radar.fine_range_step_m)),
        "coarse_step_m": step,
    }


def decouple_scatter(frame, config) -> dict | None:
    """「全部检测」散点:粗距离(m,扣偏置后,与检测距离同基准)、解耦后距离 − 粗距离(m)、展开周期,
    以及每个点对应的检测行号 rows。"""
    table = decouple_table(frame)
    if table is None:
        return None
    det = frame.detections
    bias = float(config.xCalib.xRangeBias) if config is not None else 0.0
    hit = table["index"] >= 0
    coarse = table["coarse_range_m"][hit] - bias
    return {
        "rows": np.flatnonzero(hit),  # 散点第 i 个点是第 rows[i] 个检测(点选用)
        "coarse_m": coarse,
        "delta_m": np.asarray(det.range_m, np.float64)[hit] - coarse,
        "period": table["period"][hit],
    }
