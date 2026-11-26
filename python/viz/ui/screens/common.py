"""各屏共用的取值函数与列表列声明(照 ccm `gui/screens/common.py`)。

数据取自 ProcessedFrame 本身(检测、航迹、逐级耗时)与 frame.diagnostics(指标、判据、帧级
汇总);这里只取值与格式化,不做计算。各屏的「一句话结论」在各屏文件里声明。
"""

import numpy as np

from ...diagnostics import STAGES, Level
from ..widgets.stagelist import Column


def summary(frame):
    return frame.diagnostics.summary if frame.diagnostics is not None else None


def metric(name: str):
    """逐检测指标列的取值函数(DetectionMetrics 的属性;NaN 显示「—」)。"""

    def value(frame, k):
        diag = frame.diagnostics
        return None if diag is None else float(getattr(diag.metrics, name)[k])

    return value


# ---------------- 列(各屏共用) ----------------
def _level(stage):
    def level(frame, k):
        diag = frame.diagnostics
        return 0 if diag is None else int(diag.verdicts.levels[stage][k])

    return level


def _flagged_text(frame, k):
    """需要看的级;本检测没有任何一级给出诊断时为 None(显示「—」)。"""
    diag = frame.diagnostics
    if diag is None:
        return None
    if all(diag.verdicts.level(s, k) == Level.NA for s in STAGES):
        return None
    stages = diag.verdicts.flagged_stages(k)
    return " · ".join(STAGES[s] for s in stages) if stages else "正常"


def _worst(frame, k):
    diag = frame.diagnostics
    return 0 if diag is None else max(0, int(diag.verdicts.worst(k)))


COL_INDEX = Column("#", lambda f, k: k, "{:d}")
COL_RANGE = Column("距离 m", lambda f, k: float(f.detections.range_m[k]), "{:.2f}")
COL_VELOCITY = Column(
    "速度 m/s", lambda f, k: float(f.detections.velocity_mps[k]), "{:+.3f}"
)
COL_SNR = Column(
    "SNR dB", lambda f, k: float(f.detections.snr_db[k]), "{:.1f}", bar=(0.0, 30.0)
)
COL_AZIMUTH = Column(
    "方位 °", lambda f, k: float(f.detections.azimuth_deg[k]), "{:+.1f}"
)
COL_ELEVATION = Column(
    "俯仰 °", lambda f, k: float(f.detections.elevation_deg[k]), "{:+.1f}"
)
#: 相干度 = 峰值功率 / (N·‖s‖²),由诊断层 metrics 给出;无测角诊断显示「—」。
#: corr 的「相干度」是水平相关谱口径,看不见俯仰错;「全阵相干」与 `[doa] min_coherence` 同口径
COL_COHERENCE = Column(
    "相干度", metric("coherence"), "{:.2f}", level=_level("doa"), bar=(0.0, 1.0)
)
COL_COHERENCE_FULL = Column(
    "全阵相干",
    metric("coherence_full"),
    "{:.2f}",
    level=_level("doa"),
    bar=(0.0, 1.0),
)
COL_FLAGGED = Column("需要看的级", _flagged_text, "{}", level=_worst, align_left=True)

# ---------------- 阶段屏专用列 ----------------
COL_POWER = Column("格电平 dB", metric("power_db"), "{:+.1f}")
COL_FOLD = Column("fold", metric("fold"), "{:.0f}")
COL_FOLD_MARGIN = Column(
    "fold 余量 dB", metric("fold_margin_db"), "{:+.1f}", level=_level("doppler")
)
COL_CFAR_RANGE = Column(
    "距离向余量 dB", metric("cfar_range_margin_db"), "{:+.1f}", level=_level("cfar")
)
COL_CFAR_DOPPLER = Column(
    "多普勒向余量 dB", metric("cfar_doppler_margin_db"), "{:+.1f}", level=_level("cfar")
)
COL_CFAR_MARGIN = Column(
    "CFAR 余量 dB", metric("cfar_margin_db"), "{:+.1f}", level=_level("cfar")
)
COL_NIS = Column("NIS", metric("track_nis"), "{:.2f}", level=_level("track"))


def order_by(attr: str, ascending: bool = True):
    """按某条逐检测指标排列表行序(NaN 一律最后),最可疑的在最上面。

    阶段屏的列表都按「离门限最近 / 最不满足判据」排:余量、相干度取升序。"""

    def order(frame):
        diag = frame.diagnostics
        if diag is None:
            return np.arange(len(frame.detections))
        raw = getattr(diag.metrics, attr, None)
        if raw is None:  # 该级不可用(SR61 没有 fold 指标):保持原序,不抛
            return np.arange(len(frame.detections))
        values = np.asarray(raw, np.float64)
        if not np.isfinite(values).any():
            return np.arange(len(frame.detections))
        return np.argsort(np.where(np.isfinite(values), values, np.inf), kind="stable")

    return order if ascending else lambda frame: order(frame)[::-1]
