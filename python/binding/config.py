"""UI 配置解析:载入 TOML、阵列 CSV 和缺省参数 JSON,全部使用 Python,不加载原生库。"""

import json
import math
import tomllib
from pathlib import Path

from . import dtypes
from .sensor import MAX_VIRTUAL_ANT

_C_LIGHT = 2.99792458e8
_STEP_TOL = 0.01  # 1%,抓 gross scale 错(3.66x/4.76x 事故远超此),容忍舍入


def _defaults(target, section):
    """从随包 JSON 填入缺省参数;profile.toml 仍负责覆盖显式配置。"""
    values = json.loads(Path(__file__).with_name("defaults.json").read_text())[section]

    def fill(record, fields):
        for name, value in fields.items():
            if isinstance(value, dict):
                fill(getattr(record, name), value)
            elif isinstance(value, list):
                getattr(record, name)[:] = value
            else:
                setattr(record, name, value)

    fill(target, values)
    return target


def valid_range_gates(wave, ingest_spec):
    """按配置确定绘图坐标轴的有效距离门数,不读取采集。"""
    count = int(wave.ulNumAdcSamples)
    return count // 2 if (ingest_spec or {}).get("format") == "afm761" else count


def _req(table: dict, key: str, ctx: str):
    if key not in table:
        raise ValueError(f"profile: {ctx} missing required '{key}'")
    return table[key]


def _derive_steps(
    *,
    start_freq_hz: float,
    freq_slope_hz_per_s: float,
    sample_rate_hz: float,
    idle_time_s: float,
    ramp_end_s: float,
    adc_start_s: float,
    modulation: int,
    num_adc: int,
    num_tx: int,
    num_loops: int,
    num_subbands: int,
    num_chirps: int = 0,
    chirp_step_hz: float = 0.0,
) -> tuple[float, float, float, float]:
    """由波形原始量推 (range_step, doppler_step, fine_range_step, step_slope_ratio)。

    载频逐 chirp 步进 Δf(chirp_step_hz,默认 0 = 无步进)时:
      - 中心频率加 (num_chirps − 1)·Δf/2:慢时间相位按全帧平均载频前进;
      - fine_range_step = c / (2·Δf·num_chirps),带符号(Δf < 0 时为负);
      - step_slope_ratio = Δf·PRF / k,PRF = 1/(idle + ramp_end),k = 调频斜率。
    无步进时后两项为 0。
    """
    sample_time = num_adc / sample_rate_hz
    b_valid = freq_slope_hz_per_s * sample_time
    range_step = _C_LIGHT / (2.0 * b_valid)

    chirp_period = idle_time_s + ramp_end_s
    center_freq = start_freq_hz + b_valid / 2.0 + adc_start_s * freq_slope_hz_per_s
    fine_range_step = 0.0
    step_slope_ratio = 0.0
    if chirp_step_hz != 0.0:
        center_freq += (num_chirps - 1) * chirp_step_hz / 2.0
        fine_range_step = _C_LIGHT / (2.0 * chirp_step_hz * num_chirps)
        step_slope_ratio = chirp_step_hz / chirp_period / freq_slope_hz_per_s

    if modulation == dtypes.E_MOD_DDM:
        fft_n = num_subbands * num_loops
        doppler_step = _C_LIGHT / (2.0 * center_freq * fft_n * chirp_period)
    else:
        doppler_step = _C_LIGHT / (
            2.0 * center_freq * num_loops * num_tx * chirp_period
        )
    return range_step, doppler_step, fine_range_step, step_slope_ratio


def _rel_err_within(stored: float, derived: float, tol: float) -> bool:
    if derived == 0.0:
        return False
    return abs(stored - derived) / abs(derived) < tol


def _load_waveform(root: dict, config: dtypes.ChainCfg) -> None:
    w = root.get("waveform")
    if w is None:
        raise ValueError("profile: missing [waveform] table")

    mod_str = _req(w, "modulation", "[waveform]")
    num_tx = _req(w, "num_tx", "[waveform]")
    num_rx = _req(w, "num_rx", "[waveform]")
    num_adc = _req(w, "num_adc_samples", "[waveform]")
    num_loops = _req(w, "num_doppler_chirps", "[waveform]")
    num_chirps = _req(w, "num_chirps_per_frame", "[waveform]")
    num_subbands = w.get("num_subbands", 0)  # 可选,TDM=0
    start_freq_ghz = _req(w, "start_freq_ghz", "[waveform]")
    freq_slope = _req(w, "freq_slope_mhz_per_us", "[waveform]")
    sample_rate = _req(w, "sample_rate_msps", "[waveform]")
    idle_time = _req(w, "idle_time_us", "[waveform]")
    ramp_end = _req(w, "ramp_end_time_us", "[waveform]")
    adc_start = _req(w, "adc_start_time_us", "[waveform]")
    range_step = _req(w, "range_step", "[waveform]")
    doppler_step = _req(w, "doppler_step", "[waveform]")
    frame_period_ms = _req(w, "frame_period_ms", "[waveform]")
    chirp_step_khz = w.get("chirp_step_khz", 0.0)  # 可选:载频步进(CCM),缺省 0 = 无

    if mod_str == "tdm":
        modulation = dtypes.E_MOD_TDM
    elif mod_str == "ddm":
        modulation = dtypes.E_MOD_DDM
    else:
        raise ValueError(
            f"profile: [waveform] modulation '{mod_str}' unknown (tdm|ddm)"
        )

    if chirp_step_khz != 0.0 and modulation != dtypes.E_MOD_DDM:
        raise ValueError("profile: [waveform] chirp_step_khz 仅限 modulation = ddm")
    if chirp_step_khz == 0.0 and "fine_range_step" in w:
        raise ValueError("profile: [waveform] fine_range_step 需要 chirp_step_khz ≠ 0")

    range_deriv, doppler_deriv, fine_deriv, rho_deriv = _derive_steps(
        start_freq_hz=start_freq_ghz * 1e9,
        freq_slope_hz_per_s=freq_slope * 1e12,  # MHz/us = 1e6/1e-6 Hz*s^-1 = 1e12 Hz/s
        sample_rate_hz=sample_rate * 1e6,
        idle_time_s=idle_time * 1e-6,
        ramp_end_s=ramp_end * 1e-6,
        adc_start_s=adc_start * 1e-6,
        modulation=modulation,
        num_adc=num_adc,
        num_tx=num_tx,
        num_loops=num_loops,
        num_subbands=num_subbands,
        num_chirps=num_chirps,
        chirp_step_hz=chirp_step_khz * 1e3,
    )
    if not _rel_err_within(range_step, range_deriv, _STEP_TOL):
        raise ValueError(
            f"profile: range_step {range_step:.9g} deviates >{_STEP_TOL*100:.0f}% "
            f"from waveform-derived {range_deriv:.9g}"
        )
    if not _rel_err_within(doppler_step, doppler_deriv, _STEP_TOL):
        raise ValueError(
            f"profile: doppler_step {doppler_step:.9g} deviates >{_STEP_TOL*100:.0f}% "
            f"from waveform-derived {doppler_deriv:.9g}"
        )

    # 步进频:fine_range_step 必给并与推导值交叉校验(同 range_step 的红线);
    # ρ 只推导不落 profile。
    fine_step = 0.0
    if chirp_step_khz != 0.0:
        fine_step = _req(w, "fine_range_step", "[waveform] (chirp_step_khz ≠ 0)")
        if not _rel_err_within(fine_step, fine_deriv, _STEP_TOL):
            raise ValueError(
                f"profile: fine_range_step {fine_step:.9g} deviates "
                f">{_STEP_TOL*100:.0f}% from waveform-derived {fine_deriv:.9g}"
            )

    wf = config.xWave
    wf.eModulation = modulation
    wf.ulNumTx = num_tx
    wf.ulNumRx = num_rx
    wf.ulNumAdcSamples = num_adc
    wf.ulNumDopplerChirps = num_loops
    wf.ulNumChirpsPerFrame = num_chirps
    wf.ulNumSubbands = num_subbands
    wf.xRangeStep = range_step
    wf.xDopplerStep = doppler_step
    wf.xFrameTime = frame_period_ms / 1000.0
    wf.xChirpInterval = (idle_time + ramp_end) * 1e-6
    wf.xFineRangeStep = fine_step
    wf.xStepSlopeRatio = rho_deriv if chirp_step_khz != 0.0 else 0.0


#: 窗函数名 ↔ 编号 —— **唯一权威**:profile 解析与参数面板下拉都从这里取。
WINDOW_NAMES = {
    "none": dtypes.E_WINDOW_NONE,
    "hann": dtypes.E_WINDOW_HANN,
    "blackman": dtypes.E_WINDOW_BLACKMAN,
    "hamming": dtypes.E_WINDOW_HAMMING,
}


def _window(table: dict, where: str) -> int:
    win_str = _req(table, "window", where)
    if win_str not in WINDOW_NAMES:
        raise ValueError(
            f"profile: {where} window '{win_str}' unknown "
            f"({'|'.join(WINDOW_NAMES)})"
        )
    return WINDOW_NAMES[win_str]


def _load_range_window(root: dict, config: dtypes.ChainCfg) -> None:
    r = root.get("range")
    if r is None:
        raise ValueError("profile: missing [range] table")
    config.eRangeWindow = _window(r, "[range]")


def _load_doppler_window(root: dict, config: dtypes.ChainCfg) -> None:
    """[doppler] window —— P2 前是硬编码 Hann。距离维与多普勒维选窗是两笔独立的账:
    前者压的是距离旁瓣,后者压的是慢时间旁瓣(强静止杂波泄漏盖住慢速目标)。
    """
    d = root.get("doppler")
    if d is None:
        raise ValueError("profile: missing [doppler] table")
    config.eDopplerWindow = _window(d, "[doppler]")


def _load_dc_signature(root: dict, config: dtypes.ChainCfg) -> None:
    """[range.dc_signature] —— DC 天线耦合签名去除(P1),默认关。

    即使关着也要求给出可用的窗口参数:UI 里能运行期打开这个开关,而 eChainInit 在开启
    状态下会校验它们,填 0 会让重建会话直接失败。
    """
    d = (root.get("range") or {}).get("dc_signature")
    if d is None:
        raise ValueError("profile: missing [range.dc_signature] table")
    dc = config.xDcSig
    dc.bEnable = 1 if _req(d, "enabled", "[range.dc_signature]") else 0
    dc.ulNumPosBins = _req(d, "num_pos_bins", "[range.dc_signature]")
    dc.ulNumNegBins = _req(d, "num_neg_bins", "[range.dc_signature]")
    dc.ulNumAvgChirps = _req(d, "num_avg_chirps", "[range.dc_signature]")


#: DBSCAN 模式名 ↔ 编号 —— **唯一权威**(profile 解析 + 参数面板下拉)。
CLUSTER_MODES = {
    "off": dtypes.E_CLUSTER_OFF,
    "filter": dtypes.E_CLUSTER_FILTER,
    "centroid": dtypes.E_CLUSTER_CENTROID,
}


def _load_track(root: dict, config: dtypes.ChainCfg) -> None:
    """[track] —— 跟踪参数(P7 起进 ChainCfg)。

    默认值从随包 defaults.json 读取,profile 只覆盖显式写出来的键 —— 这样
    默认值只有一份,不会 Python 抄一份、C 一份然后悄悄分叉。目前只开放 P7 的几项;
    其余参数仍只走 C 默认(docs/handbook.md:这组参数未做敏感性扫描、尚未定稿)。
    """
    config.xTrack = _defaults(dtypes.TrackCfg(), "track")
    t = root.get("track")
    if t is None:
        return  # [track] 可选:不写就全用 core 默认
    tk = config.xTrack
    if "group_dispersion" in t:
        tk.bGroupDispersion = 1 if t["group_dispersion"] else 0
    if "vel_unroll" in t:
        tk.bVelUnroll = 1 if t["vel_unroll"] else 0
    if "unroll_alpha" in t:
        tk.xUnrollAlpha = t["unroll_alpha"]
    if "unroll_confidence" in t:
        tk.xUnrollConfidence = t["unroll_confidence"]
    if "gate_gain" in t:
        tk.xGate.xGain = t["gate_gain"]
    if "meas_noise_r" in t:
        # F7:range 量测噪声按"bin 量化"重标(R 维纯 bin 量化、无亚 bin 插值时,
        # 量化误差 ~ U(±ΔR/2) ⇒ 方差 = ΔR²/12)。只开放 r —— 它是本采集上唯一
        # 有实测证据的一项(消融:只改 r 就够,az/el/dop 无可测效果,见
        # docs/archive/afm761_integration_study.md §10);其余三项仍走 core 单源。
        tk.xMeasNoise[0] = t["meas_noise_r"]
    # 场景盒与起航/生命周期计数:core 默认按 SR61(R_max≈47 m、50 ms 帧)标定,
    # 长量程或长帧周期的板要按自己的量程与帧长覆盖(afmccmdualslow,ADR 0016)。
    for key, field in (("scene_x_m", "xSceneX"), ("scene_y_m", "xSceneY")):
        if key in t:
            lo, hi = t[key]
            if not lo < hi:
                raise ValueError(f"[track] {key} 需要 [min, max] 且 min < max")
            setattr(tk, field + "Min", lo)
            setattr(tk, field + "Max", hi)
    for key, field in (
        ("alloc_points", "ulAllocPointsThr"),
        ("hit_points", "ulHitPointsThr"),
        ("det2act", "ulDet2ActThr"),
        ("det2free", "ulDet2FreeThr"),
        ("active2free", "ulActive2FreeThr"),
    ):
        if key in t:
            if not isinstance(t[key], int) or t[key] < 1:
                raise ValueError(f"[track] {key} 必须是正整数")
            setattr(tk, field, t[key])
    # 车载动目标跟踪(2026-11-24,afmccmdual / afmccmdualrev):各开关语义见 track.h,
    # 依据见 profiles/afmccmdual 的 [track] 注释。core 默认全关,不写就逐位不变。
    if "max_tracks" in t:
        if not isinstance(t["max_tracks"], int) or not 1 <= t["max_tracks"] <= 64:
            raise ValueError("[track] max_tracks 必须是 1..64 的整数(MAX_TRACKS)")
        tk.ulMaxTracks = t["max_tracks"]
    if "meas_noise_az" in t:
        if not t["meas_noise_az"] > 0:
            raise ValueError("[track] meas_noise_az 必须 > 0(弧度²)")
        tk.xMeasNoise[1] = t["meas_noise_az"]
    if "static_policy" in t:
        policies = {
            "track": dtypes.E_STATIC_TRACK,
            "suppress": dtypes.E_STATIC_SUPPRESS,
        }
        if t["static_policy"] not in policies:
            raise ValueError("[track] static_policy 只能是 track / suppress")
        tk.eStatic = policies[t["static_policy"]]
    for key, field in (
        ("ego_motion", "bEgoMotion"),
        ("coast", "bCoast"),
        ("gate_noise", "bGateNoise"),
    ):
        if key in t:
            setattr(tk, field, 1 if t[key] else 0)
    for key, field in (
        ("ego_static_mps", "xEgoStaticVel"),
        ("init_tan_vel_mps", "xInitTanVel"),
    ):
        if key in t:
            if not t[key] >= 0:
                raise ValueError(f"[track] {key} 不能为负")
            setattr(tk, field, t[key])


def _load_cluster(root: dict, config: dtypes.ChainCfg) -> None:
    """[cluster] —— DBSCAN 聚类(P6),默认 off。

    ⚠ SDK 3.6 的检测链本身没有聚类环节(clusproc 是 2.x 的),而本仓库的跟踪器自带
    allocation 期聚类 —— 两者职责重叠,开不开要靠 A/B 量。
    """
    c = root.get("cluster")
    if c is None:
        raise ValueError("profile: missing [cluster] table")
    mode = _req(c, "mode", "[cluster]")
    if mode not in CLUSTER_MODES:
        raise ValueError(
            f"profile: [cluster] mode '{mode}' unknown ({'|'.join(CLUSTER_MODES)})"
        )
    config.xCluster.eMode = CLUSTER_MODES[mode]
    config.xCluster.xEpsilon = _req(c, "epsilon_m", "[cluster]")
    config.xCluster.xWeightZ = _req(c, "weight_z", "[cluster]")
    config.xCluster.xWeightVel = _req(c, "weight_vel", "[cluster]")
    config.xCluster.ulMinPoints = _req(c, "min_points", "[cluster]")


def _load_mount(root: dict, config: dtypes.ChainCfg) -> None:
    """[mount] —— 传感器安装姿态(P5)。默认 (0,0) 是精确恒等。

    只有俯仰倾角:gTrack 的 azimTilt 在它自己的 3D 源码里零使用(手册 ch07 §16.4 已
    grep 核实),照抄一个上游没实现的字段只会让人以为它生效了。
    """
    m = root.get("mount")
    if m is None:
        raise ValueError("profile: missing [mount] table")
    config.xMount.xElevTiltDeg = _req(m, "elev_tilt_deg", "[mount]")
    config.xMount.xHeightM = _req(m, "height_m", "[mount]")


def _load_calib(root: dict, config: dtypes.ChainCfg) -> None:
    """[calib] —— 实测标定量。与 [waveform] 的 step 严格分家,理由见 core/types/calib.h:
    step 抄错会等比缩放整个世界(2026-07-17),bias 填错只是整体平移一个常量。
    """
    c = root.get("calib")
    if c is None:
        raise ValueError("profile: missing [calib] table")
    config.xCalib.xRangeBias = _req(c, "range_bias_m", "[calib]")


#: CFAR 噪声估计名 ↔ 编号 —— **唯一权威**(profile 解析 + 参数面板下拉 + Pfa 脚本的报告)。
AVG_MODES = {
    "ca": dtypes.E_CFAR_CA,
    "cago": dtypes.E_CFAR_CAGO,
    "caso": dtypes.E_CFAR_CASO,
    "os": dtypes.E_CFAR_OS,
    "cacc": dtypes.E_CFAR_CACC,
}

#: 峰值分组方案名 ↔ 编号 —— **唯一权威**。
PEAK_GROUP_SCHEMES = {
    "det_matrix": dtypes.E_CFAR_PEAK_GROUP_DET_MATRIX,
    "cfar_peak": dtypes.E_CFAR_PEAK_GROUP_CFAR_PEAK,
}


#: 编号 → 名(报告/UI 按编号打印时用;派生自上一张表,不手抄第二份)
AVG_MODE_NAMES = {mode: name for name, mode in AVG_MODES.items()}


def _load_cfar_dim(table: dict, where: str, dim) -> None:
    """单维 CFAR。门限是**功率域 dB**(与 TI CLI 的 cfarCfg 同口径),不是 log2 单位。

    P3 之前 profile 里写的是 log2 幅度值(1.6),没有物理口径。换算 dB = log2 × 6.0206
    (幅度 log2 +1 ⇒ 功率 ×4),所以 1.6 ⟺ 9.633 dB。dB 才是能和 Pfa 挂钩的量。

    F5 新增 `average_domain`(缺省 "log2",保持历史行为):噪声估计的**平均域**。
    "log2" = 单元(log2 幅度)算术平均 = 功率域**几何**平均(GM-CFAR);"power" =
    功率域**算术**平均后折回 log2 域 = 参考流 input_domain=LINEAR 的乘性门限语义。
    """
    mode_str = _req(table, "average_mode", where)
    if mode_str not in AVG_MODES:
        raise ValueError(
            f"profile: {where} average_mode '{mode_str}' unknown "
            f"({'|'.join(AVG_MODES)})"
        )
    dim.eNoiseMode = AVG_MODES[mode_str]
    dim.xThrDb = _req(table, "threshold_db", where)
    if "enabled" in table:
        if not isinstance(table["enabled"], bool):
            raise ValueError(f"profile: {where} enabled 必须为 bool")
        if table["enabled"] and dim.xThrDb <= 0:
            raise ValueError(f"profile: {where} 启用时 threshold_db 必须为正")
        if not table["enabled"]:
            dim.xThrDb = -abs(dim.xThrDb)
    dim.ulGuardLen = _req(table, "guard_len", where)
    dim.ulWinLen = _req(table, "win_len", where)
    dim.ulOsRank = table.get("os_rank", 0)
    dim.bPeakGrouping = 1 if _req(table, "peak_grouping", where) else 0
    domain = table.get("average_domain", "log2")
    if domain not in ("log2", "power"):
        raise ValueError(
            f"profile: {where} average_domain '{domain}' unknown " f"(log2|power)"
        )
    dim.bPowerMean = 1 if domain == "power" else 0


def _load_cfar(root: dict, config: dtypes.ChainCfg) -> None:
    c = root.get("cfar")
    if c is None:
        raise ValueError("profile: missing [cfar] table")
    r = c.get("range")
    d = c.get("doppler")
    if r is None or d is None:
        raise ValueError("profile: missing [cfar.range] / [cfar.doppler] table")
    _load_cfar_dim(r, "[cfar.range]", config.xCfar.xRange)
    _load_cfar_dim(d, "[cfar.doppler]", config.xCfar.xDoppler)
    scheme_str = _req(c, "peak_group_scheme", "[cfar]")
    if scheme_str not in PEAK_GROUP_SCHEMES:
        raise ValueError(
            f"profile: [cfar] peak_group_scheme '{scheme_str}' unknown "
            f"({'|'.join(PEAK_GROUP_SCHEMES)})"
        )
    config.xCfar.ePeakGroupScheme = PEAK_GROUP_SCHEMES[scheme_str]
    config.xCfar.ulMinRangeBin = _req(c, "min_range_bin", "[cfar]")
    config.xCfar.ulMaxRangeBin = _req(c, "max_range_bin", "[cfar]")
    config.xCfar.ulRangeSkipLow = _req(c, "range_skip_low", "[cfar]")
    config.xCfar.ulRangeSkipHigh = _req(c, "range_skip_high", "[cfar]")
    config.xCfar.bSubBinInterp = 1 if _req(c, "sub_bin_interp", "[cfar]") else 0
    # 步进频解耦的同列爬坡格数(0 = 关),见 core/dpu/cfar/cfar.h ulDecouplePeakClimb
    climb = c.get("decouple_peak_climb", 0)
    if not isinstance(climb, int) or isinstance(climb, bool) or not 0 <= climb <= 8:
        raise ValueError("profile: [cfar] decouple_peak_climb 必须是 0..8 的整数")
    config.xCfar.ulDecouplePeakClimb = climb
    config.xCfar.bCrossPeakGrouping = 1 if c.get("cross_peak_grouping", False) else 0


def _is_integer_token(tok: str) -> bool:
    s = tok.strip(" \t")
    if s and s[0] in "+-":
        s = s[1:]
    return s.isdigit()


def load_array_csv(csv_path: Path, grid_rows: int, grid_cols: int) -> dtypes.Array:
    """格式:idx,pos_x,pos_z,cal_re,cal_im —— '#' 注释/空行/表头跳过,按 idx 直接写槽。"""
    array = dtypes.Array()
    array.ulGridRows = grid_rows
    array.ulGridCols = grid_cols
    filled = 0

    with open(csv_path, "r") as f:
        for line_no, raw_line in enumerate(f, start=1):
            stripped = raw_line.strip(" \t\r\n")
            if not stripped or stripped[0] == "#":
                continue
            fields = stripped.split(",")[:5]
            if len(fields) < 5:
                if not _is_integer_token(fields[0]):
                    continue
                raise ValueError(
                    f"array csv: line {line_no} has {len(fields)} fields (need 5)"
                )
            if not _is_integer_token(fields[0]):
                continue  # 表头行

            idx = int(fields[0])
            pos_x, pos_z, cal_re, cal_im = (float(x) for x in fields[1:5])
            if not (0 <= idx < MAX_VIRTUAL_ANT):
                raise ValueError(
                    f"array csv: line {line_no} idx {idx} out of range [0,{MAX_VIRTUAL_ANT}]"
                )
            array.xPosX[idx] = pos_x
            array.xPosZ[idx] = pos_z
            array.xCalib[idx].xRe = cal_re
            array.xCalib[idx].xIm = cal_im
            filled = max(filled, idx + 1)

    if filled == 0:
        raise ValueError(f"array csv: no antenna rows parsed in {csv_path}")
    array.ulNumVirtual = filled
    return array


#: eDoaVariant 编号 ↔ profile 里的变体名 —— **唯一权威**:profile 解析、auto 提示、
#: 参数面板下拉、顶栏 chip 全从这里取。变体名是 profile 契约的一部分,多抄一份就会漂
#: (例如"UI 里能选 dml、profile 里写 dml 却报 unknown"这类不自洽)。
DOA_VARIANT_NAMES = {
    dtypes.E_DOA_VARIANT_BEAM: "beam",
    dtypes.E_DOA_VARIANT_MUSIC: "music",
    dtypes.E_DOA_VARIANT_DML: "dml",
    dtypes.E_DOA_VARIANT_DBF2D: "dbf2d",
    dtypes.E_DOA_VARIANT_CORR: "corr",
    dtypes.E_DOA_VARIANT_PHASEFIT: "phasefit",
}

#: 变体名 → 编号(由上一张表派生)
DOA_VARIANT_BY_NAME = {name: var for var, name in DOA_VARIANT_NAMES.items()}


def doa_route_recommend(array: dtypes.Array) -> tuple[int, str]:
    """仅供 UI 选择后端标签:规则占位阵走 FFT,其他阵列显示扫描后端。"""
    if array is None or not array.ulNumVirtual:
        return dtypes.E_DOA_VARIANT_BEAM, "未指定阵列,预览 beam"
    rows, cols = int(array.ulGridRows), int(array.ulGridCols)
    occupied = set()
    regular = 0 < rows <= 64 and 0 < cols <= 64
    for index in range(array.ulNumVirtual):
        x, z = float(array.xPosX[index]), float(array.xPosZ[index])
        if not math.isfinite(x) or not math.isfinite(z):
            regular = False
            break
        col, row = round(x), round(z)
        if (
            abs(x - col) > 1e-4
            or abs(z - row) > 1e-4
            or not (0 <= row < rows and 0 <= col < cols)
        ):
            regular = False
            break
        occupied.add((row, col))
    regular = (
        regular
        and {row for row, col in occupied} == set(range(rows))
        and {col for row, col in occupied} == set(range(cols))
    )
    if regular:
        return dtypes.E_DOA_VARIANT_BEAM, "规则占位阵 → beam"
    return dtypes.E_DOA_VARIANT_DBF2D, "非规则阵 → 扫描后端"


def _scan_table(d: dict) -> tuple[str, dict | None]:
    """[doa] 下的扫描网格段 —— 2026-09-12 起规范名 [doa.scan](beam 的网格后端载体),
    旧名 [doa.dbf2d] 仍接受、两个同时给出是配置错(不知道该信谁)。返回 (段名, 表)。"""
    scan, old = d.get("scan"), d.get("dbf2d")
    if scan is not None and old is not None:
        raise ValueError(
            "profile: [doa.scan] and its old name [doa.dbf2d] both present — keep one"
        )
    if scan is not None:
        return "[doa.scan]", scan
    if old is not None:
        return "[doa.dbf2d]", old
    return "[doa.scan]", None


def _load_doa_cfg(d: dict, config: dtypes.ChainCfg) -> None:
    """[doa] 下的估角器共享配置(P4)。四项默认全关,关时与加它们之前逐位相同。

    ⚠ FOV 有独立的 enabled:四个角度全 0 的字面含义是"只接受正前方 0°",不是"不设限"。
    """
    doa = config.xDoa
    fov = d.get("fov")
    if fov is None:
        raise ValueError("profile: missing [doa.fov] table")
    doa.xFov.bEnable = 1 if _req(fov, "enabled", "[doa.fov]") else 0
    doa.xFov.xAzMinDeg = _req(fov, "az_min_deg", "[doa.fov]")
    doa.xFov.xAzMaxDeg = _req(fov, "az_max_deg", "[doa.fov]")
    doa.xFov.xElMinDeg = _req(fov, "el_min_deg", "[doa.fov]")
    doa.xFov.xElMaxDeg = _req(fov, "el_max_deg", "[doa.fov]")

    mp = d.get("multi_peak")
    if mp is None:
        raise ValueError("profile: missing [doa.multi_peak] table")
    doa.xMultiPeak.bEnable = 1 if _req(mp, "enabled", "[doa.multi_peak]") else 0
    doa.xMultiPeak.xThrScale = _req(mp, "threshold_scale", "[doa.multi_peak]")

    doa.bExtMaxVel = 1 if _req(d, "extended_max_velocity", "[doa]") else 0
    doa.bAngleInterp = 1 if _req(d, "angle_interp", "[doa]") else 0
    # 可选:测角相干度门限(0 = 关)。低于它的点视为非单源,丢弃(DDM 逐点路径)。
    doa.xMinCoherence = float(d.get("min_coherence", 0.0))
    if not 0.0 <= doa.xMinCoherence <= 1.0:
        raise ValueError("profile: [doa] min_coherence 须在 0..1")
    # 可选:近处俯仰混叠的帧内修复(ADR 0019;只有 corr 的 DDM 逐点路径消费)。
    repair = d.get("elevation_repair", False)
    if not isinstance(repair, bool):
        raise ValueError("profile: [doa] elevation_repair 须为 true/false")
    doa.bElevationRepair = 1 if repair else 0

    # 扫描网格段:配了就改走直接加权扫描(对阵列几何无要求),不配则 beam 走角度 FFT
    # 快路径 —— 这是同一个估计器的两种求值后端,故 music/dml 下缺段也合法(xScan 保持
    # 零值,它们不消费)。variant=dbf2d 下缺段是配置错:零值网格过不了 eDoaScanValidate,
    # core 会拒,这里提前给出友好报错(见下方 variant 分派)。
    scan_tag, scan = _scan_table(d)
    if scan is not None:
        doa.xScan.xAzStartDeg = _req(scan, "az_start_deg", scan_tag)
        doa.xScan.xAzStopDeg = _req(scan, "az_stop_deg", scan_tag)
        doa.xScan.xAzStepDeg = _req(scan, "az_step_deg", scan_tag)
        doa.xScan.xElStartDeg = _req(scan, "el_start_deg", scan_tag)
        doa.xScan.xElStopDeg = _req(scan, "el_stop_deg", scan_tag)
        doa.xScan.xElStepDeg = _req(scan, "el_step_deg", scan_tag)


def _load_array_and_doa(root: dict, profile_dir: Path, config: dtypes.ChainCfg) -> None:
    a = root.get("array")
    if a is None:
        raise ValueError("profile: missing [array] table")
    csv_file = _req(a, "file", "[array]")
    grid_rows = _req(a, "grid_rows", "[array]")
    grid_cols = _req(a, "grid_cols", "[array]")

    array = load_array_csv(Path(profile_dir) / csv_file, grid_rows, grid_cols)
    config.xArray = array

    d = root.get("doa")
    if d is None:
        raise ValueError("profile: missing [doa] table")
    _load_doa_cfg(d, config)
    variant_str = _req(d, "variant", "[doa]")
    if variant_str == "auto":
        variant, reason = doa_route_recommend(array)
        print(f"config: doa=auto -> {DOA_VARIANT_NAMES[variant]} ({reason})")
    elif variant_str not in DOA_VARIANT_BY_NAME:
        names = "|".join(DOA_VARIANT_NAMES[v] for v in sorted(DOA_VARIANT_NAMES))
        raise ValueError(
            f"profile: [doa] variant '{variant_str}' unknown ({names}|auto)"
        )
    else:
        variant = DOA_VARIANT_BY_NAME[variant_str]

    # 变体名的语义就是"必须按给定网格扫描",与 beam 的"配没配都行"不同。检查放在
    # 分派**之后**:auto 路由到稀疏阵时同样会返回 dbf2d(见 route.c),那条路径也要
    # 有这条友好报错 —— 否则要等 core 抛裸 eStatus 才知道网格没配。
    if variant == dtypes.E_DOA_VARIANT_DBF2D and _scan_table(d)[1] is None:
        raise ValueError(
            "profile: [doa] variant=dbf2d requires a [doa.scan] scan table"
            " (spelled [doa.dbf2d] before 2026-09-12)"
        )
    config.eDoa = variant


def load_ingest_spec(profile_dir: Path) -> dict | None:
    """[ingest] 段 —— 采集 .bin 格式声明(F2 起多板共存,格式不是波形的函数,单独成段)。

    返回 None 表示无该段 = SR61 默认格式(IMRE 复采样)。当前声明集:
      format = "sr61" | "afm761"
      rx_map = [..](afm761 必填,1-based,文件槽位→逻辑 RX;必须是 1..num_rx 的排列,
                   排列性在 ingest 侧校验,这里只知道 toml)
      cp_adc = true|false(afm761 可选,缺省 false):固件 v1.0.1 起的 CP_ADC 格式 ——
                   每条 chirp 记录开头 2 个「采样行」是 CP 控制字(chirp 号 / profile 号 / 通道),
                   ADC 只剩 num_adc − 2 点;详见 ingest.py 模块说明
    """
    profile_dir = Path(profile_dir)
    with open(profile_dir / "profile.toml", "rb") as f:
        ing = tomllib.load(f).get("ingest")
    if ing is None:
        return None
    fmt = ing.get("format")
    if fmt not in ("sr61", "afm761"):
        raise ValueError(f"profile: [ingest] format '{fmt}' unknown (sr61|afm761)")
    rx_map = ing.get("rx_map")
    if fmt == "afm761":
        if not isinstance(rx_map, list) or not all(isinstance(m, int) for m in rx_map):
            raise ValueError("profile: [ingest] afm761 requires integer rx_map list")
    elif rx_map is not None:
        raise ValueError("profile: [ingest] sr61 takes no rx_map")
    cp_adc = ing.get("cp_adc", False)
    if not isinstance(cp_adc, bool):
        raise ValueError("profile: [ingest] cp_adc 必须是 true|false")
    if cp_adc and fmt != "afm761":
        raise ValueError("profile: [ingest] cp_adc 仅限 format = afm761")
    return {
        "format": fmt,
        "rx_map": tuple(rx_map) if rx_map else None,
        "cp_adc": cp_adc,
    }


def load_profile(profile_dir: Path) -> dtypes.ChainCfg:
    """载入 <profile_dir>/profile.toml + <profile_dir>/<array.file> -> ChainCfg(可直接喂 ChainSession)。"""
    profile_dir = Path(profile_dir)
    toml_path = profile_dir / "profile.toml"
    with open(toml_path, "rb") as f:
        root = tomllib.load(f)

    config = dtypes.ChainCfg()
    _load_waveform(root, config)
    _load_calib(root, config)
    _load_mount(root, config)
    _load_cluster(root, config)
    _load_track(root, config)
    _load_range_window(root, config)
    _load_dc_signature(root, config)
    _load_doppler_window(root, config)
    _load_cfar(root, config)
    for key, dim in (
        ("range", config.xCfar.xRange),
        ("doppler", config.xCfar.xDoppler),
    ):
        dim.bOsEdgeScale = bool(
            root.get("cfar", {}).get(key, {}).get("os_edge_scale", False)
        )
    segmented = root.get("cfar", {}).get("segments")
    if segmented:
        ends = segmented.get("end_bins", [])
        ranges = segmented.get("range_threshold_db", [])
        dopplers = segmented.get("doppler_threshold_db", [])
        if len(ends) != 2 or len(ranges) != 3 or len(dopplers) != 3:
            raise ValueError("CFAR 分段需要两个边界和两组三段门限")
        config.xCfar.bSegmented = bool(segmented.get("enabled", True))
        config.xCfar.ulSegmentEnd[:] = ends
        config.xCfar.xRangeThrDb[:] = ranges
        config.xCfar.xDopplerThrDb[:] = dopplers
    supplement = root.get("cfar", {}).get("dominant_supplement", {})
    if supplement:
        config.xCfar.bDominantSupplement = bool(supplement.get("enabled", False))
        for key, field in {
            "start_bin": "ulDominantStart",
            "end_bin": "ulDominantEnd",
            "dominance_threshold_db": "xDominanceThrDb",
            "threshold_db": "xSupplementThrDb",
            "guard_len": "ulSupplementGuard",
            "win_len": "ulSupplementWin",
            "limit": "ulSupplementLimit",
        }.items():
            if config.xCfar.bDominantSupplement or key in supplement:
                setattr(
                    config.xCfar,
                    field,
                    _req(supplement, key, "[cfar.dominant_supplement]"),
                )
    _load_array_and_doa(root, profile_dir, config)
    near = root.get("doa", {}).get("fov", {}).get("elevation_skip_max_range_bin")
    if near is not None:
        if (
            not isinstance(near, int)
            or not 0 <= near < config.xWave.ulNumAdcSamples // 2
        ):
            raise ValueError("elevation_skip_max_range_bin 越界")
        config.xDoa.bNearElevationSkip = 1
        config.xDoa.ulNearElevationMaxBin = near
    pair = root.get("pair", {})
    # 关闭时也保留门限与备用波形,否则 UI 里再开启只会拿到一份全零配置。
    _defaults(config.xPair, "pair")
    if pair:
        config.xPair.bEnable = bool(pair.get("enabled", False))
        config.xPair.bShareTaken = bool(pair.get("share_taken", False))
        config.xPair.bCarryConfirmed = bool(pair.get("carry_confirmed", False))
        config.xPair.bRescueUnfolded = bool(pair.get("rescue_unfolded", False))
        config.xPair.bZeroIdentity = bool(pair.get("zero_identity", False))
        if config.xPair.bEnable and not root.get("ingest", {}).get("cp_adc", False):
            raise ValueError("双 PRF 需要 [ingest] cp_adc=true")
        for key, field in {
            "max_unfold": "ucMaxUnfold",
            "range_gate_m": "xRangeGate",
            "velocity_gate_mps": "xVelocityGate",
            "max_speed_mps": "xMaxSpeed",
            "angle_gate_deg": "xAngleGate",
            "min_coherence": "xMinCoherence",
            "ambiguity_margin": "xAmbiguityMargin",
            "min_overlap_speed_mps": "xMinOverlapSpeed",
            "overlap_range_bins": "xOverlapRangeBins",
            "overlap_doppler_bins": "xOverlapDopplerBins",
            "occupant_range_bins": "xOccupantRangeBins",
            "mask_margin_db": "xMaskMarginDb",
            "azimuth_sigma_deg": "xAzimuthSigma",
            "lateral_extent_m": "xLateralExtent",
        }.items():
            if key in pair:
                if key == "max_unfold" and (
                    not isinstance(pair[key], int) or not 0 <= pair[key] <= 8
                ):
                    raise ValueError("pair.max_unfold 必须在 0..8")
                setattr(config.xPair, field, pair[key])
    base = dict(root["waveform"])
    alternate = base.pop("alternate", None)
    if config.xPair.bEnable and not isinstance(alternate, dict):
        raise ValueError("双 PRF 需要 [waveform.alternate]")
    if isinstance(alternate, dict):
        base.update(alternate)
        other = dtypes.ChainCfg()
        _load_waveform({"waveform": base}, other)
        config.xAlternateWave = other.xWave
    return config
