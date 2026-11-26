"""诊断与判据(不依赖 Qt):界面的 KPI 条、列表、检查器、各阶段屏与批量脚本共用的「结论」来源。

照 ccm `diagnostics/`:
- metrics:每个检出点在每一级的指标(只用 products 里 C 导出的量)与帧级检查;
- criteria:按阈值分成「正常 / 需要看 / 异常」并给出一句结论;阈值可按板存进 JSON;
- summary:帧级 KPI;
- slices:视图与小图要画的数据切片。

`diagnose(frame, criteria, config) → FrameDiagnostics(metrics, verdicts, summary, checks)`
在处理线程里随帧算好,界面只读;`rejudge` 只按新判据重新分级,不重算指标。
依赖方向:`pipeline → diagnostics → frame / binding`;这里不 import pipeline / ui。
"""

from dataclasses import dataclass

import numpy as np

from .criteria import (
    CRITERIA_PATH,
    DEFAULTS,
    STAGES,
    CriteriaConfig,
    Level,
    Verdicts,
    criteria_for,
    judge,
    load_criteria,
    save_criteria,
)
from .metrics import (
    DetectionMetrics,
    FrameChecks,
    compute_metrics,
    doa_index,
    frame_checks,
    scan_edges,
)
from .slices import display_levels_db
from .summary import FrameSummary, processing_ms, summarize

#: 全链状态条里帧级格(ADC / 距离)与非检测单元的提示:帧级检查进 KPI 与帧级汇总,
#: 状态条上帧级格(ADC / 距离)在无诊断时的提示;有诊断时用 summary.adc_level / range_level
PENDING = "未诊断"


@dataclass
class FrameDiagnostics:
    metrics: DetectionMetrics
    verdicts: Verdicts
    summary: FrameSummary
    checks: FrameChecks


def diagnose(
    frame, criteria: CriteriaConfig | None = None, config=None
) -> FrameDiagnostics:
    """一帧(`frame.ProcessedFrame`)的诊断。criteria 为 None 时用该板默认(criteria_for);
    config(ChainCfg)只用来判定测角后端与扫描网格端点(贴边指标),None 则贴边为 NaN。
    products 为 None(没取中间量)时各级为 NA,不抛。"""
    if criteria is None:
        criteria = criteria_for(frame.radar.profile)
    array = None
    x_array = getattr(config, "xArray", None)  # 测试里的最小配置可以没有阵列
    if x_array is not None:
        n_virtual = int(x_array.ulNumVirtual)
        array = (
            np.array(x_array.xPosX[:n_virtual], np.float64),
            np.array(x_array.xPosZ[:n_virtual], np.float64),
        )
    metrics = compute_metrics(frame, scan_edges(config), array)
    verdicts = judge(metrics, criteria)
    checks = frame_checks(frame)
    summary = summarize(metrics, verdicts, checks, frame.timings_ms, len(frame.tracks))
    return FrameDiagnostics(metrics, verdicts, summary, checks)


def rejudge(
    diagnostics: FrameDiagnostics, criteria: CriteriaConfig
) -> FrameDiagnostics:
    """只换判据:指标与帧级检查原样复用,重新分级并重算汇总。"""
    verdicts = judge(diagnostics.metrics, criteria)
    old = diagnostics.summary
    summary = summarize(
        diagnostics.metrics, verdicts, diagnostics.checks, old.timings_ms, old.n_tracks
    )
    return FrameDiagnostics(diagnostics.metrics, verdicts, summary, diagnostics.checks)


__all__ = [
    "CRITERIA_PATH",
    "DEFAULTS",
    "PENDING",
    "STAGES",
    "CriteriaConfig",
    "DetectionMetrics",
    "FrameChecks",
    "FrameDiagnostics",
    "FrameSummary",
    "Level",
    "Verdicts",
    "compute_metrics",
    "criteria_for",
    "diagnose",
    "display_levels_db",
    "doa_index",
    "frame_checks",
    "judge",
    "load_criteria",
    "processing_ms",
    "rejudge",
    "save_criteria",
    "scan_edges",
    "summarize",
]
