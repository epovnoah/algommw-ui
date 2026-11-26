"""帧级汇总:KPI 条与批量统计用的数字(照 ccm `diagnostics/summary.py`)。

处理耗时口径(`processing_ms`):read + range / doppler / cfar / doa / cluster / track / world
+ extract 之和,**不含** "capture"(诊断取中间量)与 "diagnostics"(诊断本身)—— 二者是诊断
开销,「处理耗时」页单列并注明。
"""

from dataclasses import dataclass, field

import numpy as np

from .criteria import STAGES, Level, Verdicts
from .metrics import AdcStats, DetectionMetrics, FrameChecks

#: 帧级判据(没有进「判据」页:这两条是硬件/采集是否正常的硬规则,不是按板标定的)
ADC_SPREAD_WARN_DB = 3.0  # 各 RX RMS 离散超过它 → ADC 格标黄

#: 不计入处理耗时的键(诊断开销)
DIAG_TIMINGS = ("capture", "diagnostics")


def processing_ms(timings_ms: dict) -> float:
    """处理耗时(毫秒),口径见模块文档。"""
    return float(sum(v for k, v in timings_ms.items() if k not in DIAG_TIMINGS))


def _median(values) -> float:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else float("nan")


def _min(values) -> float:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    return float(values.min()) if values.size else float("nan")


@dataclass
class FrameSummary:
    n_detections: int
    n_tracks: int
    flagged: dict[str, int]  # 级 → 需要看或异常的检测个数
    timings_ms: dict[str, float]  # 逐级耗时(含诊断开销的两项,如有)
    processing_ms: float  # 处理耗时,不含诊断开销(口径见模块文档)
    # 各级指标的帧内中位数(该级不可用或无有效值为 NaN)
    cfar_margin_median_db: float = float("nan")
    fold_margin_median_db: float = float("nan")
    fold_margin_min_db: float = float("nan")
    empty_suppression_median_db: float = float("nan")
    coherence_median: float = float("nan")
    nis_median: float = float("nan")  # 逐检测(所关联航迹)的 NIS
    # 计数(阈值取本帧所用判据)
    n_near_threshold: int = 0  # CFAR 余量 < cfar_margin_warn_db
    n_low_fold_margin: int = 0  # fold 余量 < fold_margin_warn_db
    n_low_coherence: int = 0  # 相干度 < coherence_warn
    n_at_edge: int = 0  # 贴扫描边界
    n_unassociated: int = 0  # 未关联到航迹
    # 帧级检查(取自 FrameChecks)
    dropped: dict[str, int] = field(default_factory=dict)  # 测角剔除:原因 → 个数
    new_track_ids: list[int] = field(default_factory=list)
    ended_track_ids: list[int] = field(default_factory=list)
    adc: AdcStats | None = None
    dcsig: str | None = None
    range_noise_db: float = float("nan")
    range_peak_bin: int = -1
    range_peak_db: float = float("nan")
    rx_spread_db: float = float("nan")
    #: ADC / 距离两格的帧级等级(见 `_adc_level` / `_range_level` 的规则)
    adc_level: int = int(Level.NA)
    range_level: int = int(Level.NA)

    @property
    def n_dropped(self) -> int:
        return sum(self.dropped.values())


def summarize(
    metrics: DetectionMetrics,
    verdicts: Verdicts,
    checks: FrameChecks,
    timings_ms: dict,
    n_tracks: int,
) -> FrameSummary:
    m, c = metrics, verdicts.criteria
    with np.errstate(invalid="ignore"):
        near = int((m.cfar_margin_db < c.cfar_margin_warn_db).sum())
        low_fold = int((m.fold_margin_db < c.fold_margin_warn_db).sum())
        low_coherence = int((m.coherence < c.coherence_warn).sum())
    unassociated = int((m.track_id < 0).sum()) if m.available.get("track") else 0
    return FrameSummary(
        n_detections=len(m),
        n_tracks=n_tracks,
        flagged={s: int(verdicts.flagged(s).sum()) for s in STAGES},
        timings_ms=dict(timings_ms),
        processing_ms=processing_ms(timings_ms),
        cfar_margin_median_db=_median(m.cfar_margin_db),
        fold_margin_median_db=_median(m.fold_margin_db),
        fold_margin_min_db=_min(m.fold_margin_db),
        empty_suppression_median_db=_median(m.empty_suppression_db),
        coherence_median=_median(m.coherence),
        nis_median=_median(m.track_nis),
        n_near_threshold=near,
        n_low_fold_margin=low_fold,
        n_low_coherence=low_coherence,
        n_at_edge=int((m.at_scan_edge == 1.0).sum()),
        n_unassociated=unassociated,
        dropped=dict(checks.dropped),
        new_track_ids=list(checks.new_track_ids),
        ended_track_ids=list(checks.ended_track_ids),
        adc=checks.adc,
        dcsig=checks.dcsig,
        range_noise_db=checks.range_noise_db,
        range_peak_bin=checks.range_peak_bin,
        range_peak_db=checks.range_peak_db,
        rx_spread_db=checks.rx_spread_db,
        adc_level=int(_adc_level(checks.adc)),
        range_level=int(_range_level(checks.dcsig)),
    )


def _adc_level(adc: AdcStats | None) -> Level:
    """ADC 格的帧级等级:有削波样本、或各 RX 电平离散 > `ADC_SPREAD_WARN_DB` 时「需要看」。

    没有中间量(诊断关)时不适用。"""
    if adc is None:
        return Level.NA
    if adc.total_saturated > 0:
        return Level.WARN
    spread = adc.rms_spread_db
    return (
        Level.WARN if np.isfinite(spread) and spread > ADC_SPREAD_WARN_DB else Level.OK
    )


def _range_level(dcsig: str | None) -> Level:
    """距离格的帧级等级:DC 签名还在校准(本帧距离像没有补偿)时「需要看」。

    签名关闭是配置选择,不算问题;没有中间量时不适用。"""
    if dcsig is None:
        return Level.NA
    return Level.WARN if dcsig.startswith("校准中") else Level.OK
