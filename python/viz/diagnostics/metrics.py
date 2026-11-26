"""每个检出点在每一级的诊断指标(列存放,与 ProcessedFrame.detections 同序)与帧级检查。

指标只用 `ProcessedFrame.products` 里 C 导出的量(uiplan.md §5(已删,见 git 历史 fa7d3d4)的 E1 / E2 出口:
fold 候选度量、子带功率、CFAR 逐格噪声与门限、测角快照与峰值功率、点 → 航迹映射与 NIS),
不在 Python 里重写任何处理算法 —— 这里只做比值、换算与对齐。某一级在本帧不可用时该级指标为
NaN,`available` 记 False,`notes` 写原因(界面原样显示)。
"""

from dataclasses import dataclass, field

import numpy as np

from binding import dtypes

#: log2 幅度 → dB(镜像 C 侧 cfar.h 的同名宏)
CFAR_DB = dtypes.CFAR_DB_PER_LOG2
#: int16 绝对值达到它即算饱和
ADC_SATURATION = 32760
ADC_FULL_SCALE = 32768.0
#: 角度等于扫描网格端点的容差(度)
EDGE_TOL_DEG = 1e-4
#: 测角剔除原因编号 → 显示名
DROP_REASONS = {
    dtypes.DIAG_DOA_DROP_GEOM: "几何不可解",
    dtypes.DIAG_DOA_DROP_FOV: "出视场",
    dtypes.DIAG_DOA_DROP_COHERENCE: "相干度不足",
}

NO_PRODUCTS = "该帧没有中间量(诊断未开),不适用"


@dataclass
class DetectionMetrics:
    """形状 (n,) 的列;不适用为 NaN(整型列为 -1,布尔列为 False)。"""

    fold: np.ndarray  # int,检出格选中的 fold;不适用 -1
    fold_margin_db: np.ndarray  # fold 候选度量 最大 / 次大
    empty_suppression_db: np.ndarray  # 占用子带均功率 / 空子带均功率
    cfar_range_margin_db: np.ndarray  # 超门限余量,该向关闭为 NaN
    cfar_doppler_margin_db: np.ndarray
    doa_index: np.ndarray  # int,在 products["doa"] 里的行号;-1 = 无对应
    coherence: np.ndarray  # 峰值功率 / (N·‖s‖²) ∈ [0, 1]
    # 全阵相干度 = |aᴴs|²/(N‖s‖²),在最终发布的方位/俯仰上算,与 [doa] min_coherence 门限同式;
    # 门限在测角级、两基线核验/俯仰分支修正之前判,所以对被改过俯仰的点两者可能略有差别
    coherence_full: np.ndarray
    at_scan_edge: np.ndarray  # 1.0 / 0.0;非直接加权后端为 NaN
    second_peak: np.ndarray  # 1.0 / 0.0(第二峰追加点)
    track_id: np.ndarray  # int,关联到的航迹 id;-1 = 未关联
    track_new: np.ndarray  # bool,本帧起了新航迹
    track_ended: np.ndarray  # bool,关联到的航迹本帧已终止
    track_nis: np.ndarray  # 所关联航迹的 NIS;未关联 / 新建 / 已终止为 NaN
    power_db: np.ndarray  # 检出格 RD 绝对电平(rd_log2 × CFAR_DB)
    available: dict[str, bool] = field(default_factory=dict)  # 级 → 本帧可用
    notes: dict[str, str] = field(default_factory=dict)  # 级 → 不可用的原因

    def __len__(self) -> int:
        return len(self.fold)

    @property
    def cfar_margin_db(self) -> np.ndarray:
        """两向余量的较小值(真正约束检测的那一向);一向关闭时取另一向。"""
        return np.fmin(self.cfar_range_margin_db, self.cfar_doppler_margin_db)


@dataclass
class AdcStats:
    """一帧 ADC 的帧级统计(各 RX 一项)。"""

    saturated: np.ndarray  # (n_rx,) |int16| ≥ ADC_SATURATION 的个数(I、Q 各算一个)
    peak: np.ndarray  # (n_rx,) |int16| 最大值
    rms_dbfs: np.ndarray  # (n_rx,) 复采样 RMS,相对满量程 32768
    dc_db: np.ndarray  # (n_rx,) 复均值模 / RMS
    n_values: int = 0  # 统计的 int16 总个数(各 RX 之和),饱和率的分母;0 = 未记

    @property
    def total_saturated(self) -> int:
        return int(self.saturated.sum())

    @property
    def saturation_rate(self) -> float:
        """饱和 int16 占全部 int16 的比例;没记分母时为 NaN。"""
        return self.total_saturated / self.n_values if self.n_values else float("nan")

    @property
    def rms_spread_db(self) -> float:
        """RX 间 RMS 离散(max − min);全为零功率(NaN)或空时为 NaN。"""
        finite = self.rms_dbfs[np.isfinite(self.rms_dbfs)]
        return float("nan") if finite.size == 0 else float(finite.max() - finite.min())


@dataclass
class FrameChecks:
    """帧级检查(进 KPI,不进逐检测等级)。无中间量时 adc / dcsig 为 None。"""

    adc: AdcStats | None = None
    dcsig: str | None = None  # 「关」/「校准中 n / m」/「就绪」
    dropped: dict[str, int] = field(default_factory=dict)  # 原因 → 测角剔除数
    new_track_ids: list[int] = field(default_factory=list)
    ended_track_ids: list[int] = field(default_factory=list)
    # 距离级帧级检查(ADC / 距离两格与距离屏的 KPI);无中间量时为 NaN / -1
    range_noise_db: float = float("nan")  # 距离像噪底(全帧平均 Σ_rx 的中位数)
    range_peak_bin: int = -1  # 最强距离门
    range_peak_db: float = float("nan")
    rx_spread_db: float = float("nan")  # 各 RX 距离像电平中位数的极差(通道一致性)


def doa_index(frame) -> np.ndarray:
    """最终检出点第 k 行 → products["doa_cloud"] / ["doa"] 的行号(-1 = 无对应)。

    两块板默认关聚类,两者逐行相同(先校验四列逐行相等,成立就用恒等映射);开了聚类时按
    (距离 bin, 多普勒 bin, 方位, 俯仰) 匹配,同值的按出现次序一一配对。"""
    det = frame.detections
    n = len(det)
    out = np.full(n, -1, np.int64)
    products = frame.products
    if products is None or "doa_cloud" not in products:
        return out
    if "pair" in products and not bool(getattr(frame, "clustered", False)):
        # 配对输出含历史救回点,不能按恰好相同的旧 bin 把它当作当前帧测角记录。
        current = products["pair"]["current_index"]
        if len(current) == n:
            return np.asarray(current, np.int64)
    cloud = products["doa_cloud"]
    keys = [
        cloud["range_bin"],
        cloud["doppler_bin"],
        cloud["azimuth_deg"],
        cloud["elevation_deg"],
    ]
    cols = [det.range_bin, det.doppler_bin, det.azimuth_deg, det.elevation_deg]
    if len(keys[0]) == n and all(np.array_equal(a, b) for a, b in zip(cols, keys)):
        return np.arange(n, dtype=np.int64)
    table: dict[tuple, list[int]] = {}
    for i, key in enumerate(zip(*(np.asarray(a).tolist() for a in keys))):
        table.setdefault(key, []).append(i)
    for k, key in enumerate(zip(*(np.asarray(a).tolist() for a in cols))):
        rows = table.get(key)
        if rows:
            out[k] = rows.pop(0)
    return out


def scan_edges(config) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """直接加权后端的扫描网格端点 ((方位首, 方位末), (俯仰首, 俯仰末)),度;
    其它后端(FFT 快路径、music / dml)或未给配置时为 None。

    后端判定同 core:变体为 beam / dbf2d 且 xScan 六项不全为零(bDoaScanConfigured)。
    末端格 = 首 + (点数 − 1)·步长,点数 = round(跨距 / 步长) + 1(core 的 ulDoaScanAxisPoints);
    止点恰在网格上时(两块板的 profile 都是)就是 xScan 的 stop。"""
    if config is None:
        return None
    if int(config.eDoa) not in (dtypes.E_DOA_VARIANT_BEAM, dtypes.E_DOA_VARIANT_DBF2D):
        return None
    s = config.xDoa.xScan
    values = [
        s.xAzStartDeg,
        s.xAzStopDeg,
        s.xAzStepDeg,
        s.xElStartDeg,
        s.xElStopDeg,
        s.xElStepDeg,
    ]
    if not any(values):
        return None

    def axis(start, stop, step):
        span = float(np.float32(stop) - np.float32(start))
        points = int(np.floor(span / float(step) + 0.5)) + 1
        return float(start), float(start) + (points - 1) * float(step)

    return axis(*values[:3]), axis(*values[3:])


def compute_metrics(frame, edges=None, array=None) -> DetectionMetrics:
    """一帧的逐检测指标。`edges` 见 `scan_edges`(None = 非直接加权后端,贴边为 NaN)。
    `array` = (pos_x, pos_z),虚拟通道的阵位(半波长单位);None 时全阵相干度为 NaN。"""
    det, products = frame.detections, frame.products
    n = len(det)
    nan = np.full(n, np.nan)
    m = DetectionMetrics(
        fold=np.full(n, -1, np.int64),
        fold_margin_db=nan.copy(),
        empty_suppression_db=nan.copy(),
        cfar_range_margin_db=nan.copy(),
        cfar_doppler_margin_db=nan.copy(),
        doa_index=np.full(n, -1, np.int64),
        coherence=nan.copy(),
        coherence_full=nan.copy(),
        at_scan_edge=nan.copy(),
        second_peak=nan.copy(),
        track_id=np.full(n, -1, np.int64),
        track_new=np.zeros(n, bool),
        track_ended=np.zeros(n, bool),
        track_nis=nan.copy(),
        power_db=nan.copy(),
        available={"doppler": False, "cfar": False, "doa": False, "track": False},
    )
    if products is None:
        m.notes = {stage: NO_PRODUCTS for stage in m.available}
        return m
    r = np.asarray(det.range_bin, np.intp)
    d = np.asarray(det.doppler_bin, np.intp)
    rd = products["rd_log2"]
    m.power_db = rd[r, d].astype(np.float64) * CFAR_DB
    _doppler(m, frame, r, d)
    _cfar(m, products["cfar"], rd, r, d)
    _doa(m, frame, edges, array)
    _track(m, products["track"])
    if "pair" in products:
        rescued = products["pair"]["current_index"] < 0
        m.fold[rescued] = -1
        for name in (
            "fold_margin_db",
            "empty_suppression_db",
            "cfar_range_margin_db",
            "cfar_doppler_margin_db",
            "power_db",
        ):
            getattr(m, name)[rescued] = np.nan
    return m


def _doppler(m: DetectionMetrics, frame, r, d) -> None:
    p = frame.products
    if "fold_metric" not in p:
        ddm = frame.radar.modulation == "ddm"
        m.notes["doppler"] = (
            "本帧 DDM 诊断无效(子带数超出诊断容量)"
            if ddm
            else "TDM 无 fold 判决,不适用"
        )
        return
    m.available["doppler"] = True
    if len(r) == 0:
        return
    metric = p["fold_metric"][r, d].astype(np.float64)  # (n, n_sub)
    band = p["band_pow"][r, d].astype(np.float64)
    fold = p["fold"][r, d].astype(np.int64)
    n, n_sub = band.shape
    top = np.sort(metric, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        m.fold_margin_db = 10.0 * np.log10(top[:, -1] / top[:, -2])
    occupied = np.zeros((n, n_sub), bool)
    tx = (fold[:, None] + np.arange(frame.radar.n_tx)[None, :]) % n_sub
    occupied[np.arange(n)[:, None], tx] = True
    if n_sub > frame.radar.n_tx:  # 有空子带才有抑制比
        on = (band * occupied).sum(1) / occupied.sum(1)
        off = (band * ~occupied).sum(1) / (~occupied).sum(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            m.empty_suppression_db = 10.0 * np.log10(on / off)
    m.fold = fold


def _cfar(m: DetectionMetrics, cfar: dict, rd, r, d) -> None:
    m.available["cfar"] = True
    cell = rd[r, d].astype(np.float64)
    if cfar["range_on"]:
        noise = cfar["noise_range"][r, d].astype(np.float64)
        threshold = (
            cfar["thr_range_by_range"][r]
            if "thr_range_by_range" in cfar
            else cfar["thr_range"]
        )
        m.cfar_range_margin_db = (cell - noise - threshold) * CFAR_DB
    if cfar["doppler_on"]:
        noise = cfar["noise_doppler"][r, d].astype(np.float64)
        threshold = (
            cfar["thr_doppler_by_range"][r]
            if "thr_doppler_by_range" in cfar
            else cfar["thr_doppler"]
        )
        m.cfar_doppler_margin_db = (cell - noise - threshold) * CFAR_DB


def _doa(m: DetectionMetrics, frame, edges, array=None) -> None:
    doa = frame.products["doa"]
    if doa["state"] == dtypes.DIAG_DOA_STATE_UNSUPPORTED:
        m.notes["doa"] = "该测角变体(music / dml)不出诊断"
        return
    m.available["doa"] = True
    snap = doa["snap"]
    index = doa_index(frame)
    index[index >= len(snap)] = -1
    m.doa_index = index
    rows = index[index >= 0]
    hit = index >= 0
    s = snap[rows]
    snap_pow = (s.real.astype(np.float64) ** 2 + s.imag.astype(np.float64) ** 2).sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        m.coherence[hit] = doa["peak_pow"][rows] / (snap.shape[1] * snap_pow)
    m.second_peak[hit] = doa["second"][rows].astype(np.float64)
    if array is not None:
        det = frame.detections
        m.coherence_full[hit] = _full_coherence(
            s,
            np.asarray(det.azimuth_deg, np.float64)[hit],
            np.asarray(det.elevation_deg, np.float64)[hit],
            array,
        )
    if edges is not None:
        det = frame.detections
        (az_lo, az_hi), (el_lo, el_hi) = edges
        az = np.asarray(det.azimuth_deg, np.float64)[hit]
        el = np.asarray(det.elevation_deg, np.float64)[hit]
        edge = (
            (np.abs(az - az_lo) <= EDGE_TOL_DEG)
            | (np.abs(az - az_hi) <= EDGE_TOL_DEG)
            | (np.abs(el - el_lo) <= EDGE_TOL_DEG)
            | (np.abs(el - el_hi) <= EDGE_TOL_DEG)
        )
        m.at_scan_edge[hit] = edge.astype(np.float64)


def _full_coherence(snap, az_deg, el_deg, array) -> np.ndarray:
    """全阵相干度 |aᴴs|²/(N‖s‖²)(镜像 chain.c 的 prvDoaCoherence;零能量为 0)。
    snap (n, N) 已标定快照;通道数与阵位不符时返回 NaN。"""
    pos_x = np.asarray(array[0], np.float64)
    pos_z = np.asarray(array[1], np.float64)
    n = len(az_deg)
    if snap.shape[1] != len(pos_x) or len(pos_z) != len(pos_x) or len(pos_x) == 0:
        return np.full(n, np.nan)
    el = np.deg2rad(el_deg)
    u = np.sin(np.deg2rad(az_deg)) * np.cos(el)
    v = np.sin(el)
    phase = np.pi * (u[:, None] * pos_x[None, :] + v[:, None] * pos_z[None, :])
    s = snap.astype(np.complex128)
    proj = (s * np.exp(-1j * phase)).sum(1)
    power = (s.real**2 + s.imag**2).sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.abs(proj) ** 2 / (snap.shape[1] * power)
    return np.where(power > 0.0, out, 0.0)


def _track(m: DetectionMetrics, track: dict) -> None:
    pid = np.asarray(track["point_track_id"], np.int64)
    if len(pid) != len(m):
        m.notes["track"] = f"跟踪逐点记录 {len(pid)} 个与检出点 {len(m)} 个不符"
        return
    m.available["track"] = True
    m.track_id = pid
    m.track_new = np.asarray(track["point_new"], bool).copy()
    ids = np.asarray(track["tracks"]["id"], np.int64)
    nis = np.asarray(track["tracks"]["nis"], np.float64)
    ended = set(np.asarray(track["ended_ids"], np.int64).tolist())
    where = {int(t): i for i, t in enumerate(ids)}
    for k, tid in enumerate(pid.tolist()):
        if tid < 0 or m.track_new[k]:
            continue
        i = where.get(tid)
        if i is not None:
            m.track_nis[k] = nis[i]
        elif tid in ended:
            m.track_ended[k] = True


def frame_checks(frame) -> FrameChecks:
    """帧级检查:ADC 饱和 / 各 RX RMS 与 DC、DC 签名状态、测角剔除数、航迹生灭,
    以及距离级的两项(噪底中位、最强距离门、各 RX 电平极差)。"""
    p = frame.products
    if p is None:
        return FrameChecks()
    track = p["track"]
    pid = np.asarray(track["point_track_id"], np.int64)
    new = np.asarray(track["point_new"], bool)
    reasons = np.asarray(p["doa"]["dropped"]["reason"])
    dropped = {}
    for code, count in zip(*np.unique(reasons, return_counts=True)):
        name = DROP_REASONS.get(int(code), f"原因 {int(code)}")
        dropped[name] = int(count)
    checks = FrameChecks(
        adc=adc_stats(p["adc_raw"]),
        dcsig=dcsig_state(p["dcsig"]),
        dropped=dropped,
        new_track_ids=sorted({int(t) for t in pid[new & (pid >= 0)]}),
        ended_track_ids=[int(t) for t in track["ended_ids"]],
    )
    # 实采样板距离 FFT 的后半是镜像:噪底与最强门只在有效距离门里找
    _range_checks(checks, p["range_profile"][:, : frame.radar.gates])
    return checks


def _range_checks(checks: FrameChecks, profile: np.ndarray) -> None:
    """距离像的帧级量:Σ_rx 的噪底中位与峰值、各 RX 电平中位数的极差(功率 → dB)。"""
    power = np.asarray(profile, np.float64)
    if power.ndim != 2 or power.size == 0:
        return
    total = power.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        db = 10.0 * np.log10(np.maximum(total, 1e-30))
        per_rx = 10.0 * np.log10(np.maximum(power, 1e-30))
    checks.range_noise_db = float(np.median(db))
    checks.range_peak_bin = int(np.argmax(db))
    checks.range_peak_db = float(db.max())
    medians = np.median(per_rx, axis=1)
    checks.rx_spread_db = float(np.max(medians) - np.min(medians))


def adc_stats(raw: np.ndarray) -> AdcStats:
    """`adc_raw`((n_chirps, n_rx, n_samples, 2) int16)的各 RX 统计。

    这几个量对 I / Q 两路对称(饱和与峰值逐个 int16 看;RMS = √mean(I² + Q²);复均值模
    = |mean I + j·mean Q|),与哪一路是实部无关,所以直接在原样 int16 上算,不经
    binding.native.adc_complex(定序只在需要区分实虚部的地方解释)。逐 RX 取连续副本再算:
    多轴归约在 AFM761 一帧(768 × 4 × 384 × 2)上要约 10 ms,这样约 2 ms。"""
    n_rx = raw.shape[1]
    saturated = np.zeros(n_rx, np.int64)
    peak = np.zeros(n_rx, np.int64)
    power = np.zeros(n_rx)
    dc_power = np.zeros(n_rx)
    for rx in range(n_rx):
        v = np.ascontiguousarray(raw[:, rx]).reshape(-1, 2)  # (采样数, 2)
        saturated[rx] = np.count_nonzero((v >= ADC_SATURATION) | (v <= -ADC_SATURATION))
        peak[rx] = max(int(v.max()), -int(v.min()))
        flat = v.astype(np.float32).reshape(-1)
        power[rx] = float(np.dot(flat, flat)) / len(v)  # mean(I² + Q²)
        a = v[:, 0].sum(dtype=np.int64) / len(v)
        b = v[:, 1].sum(dtype=np.int64) / len(v)
        dc_power[rx] = a * a + b * b  # |复均值|²
    with np.errstate(divide="ignore", invalid="ignore"):
        rms_dbfs = 10.0 * np.log10(power / ADC_FULL_SCALE**2)
        dc_db = 10.0 * np.log10(dc_power / power)
    return AdcStats(saturated, peak, rms_dbfs, dc_db, int(raw.size))


def dcsig_state(dcsig: dict) -> str:
    if not dcsig["enabled"]:
        return "关"
    if dcsig["ready"]:
        return "就绪"
    return f"校准中 {dcsig['accum_count']} / {dcsig['avg_chirps']}"
