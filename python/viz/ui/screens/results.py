"""场景屏:本帧的点云与航迹是否与场景一致?哪些检测哪一级需要看?

版式是「全出血画布 + 悬浮工具」:点云画布(三维 OpenGL;不可用时退回俯视 / 侧视二维投影,
见 plots/cloud.py)铺满主区,没有卡片描边与内边距;视角 / 余辉 / 视野 / 速度量程 / 坐标系
都叠在画布左上角的悬浮工具条上(世界坐标的 7 个控件折进「世界 ▾」面板),速度色标与图例叠在
左下,右上是选中检测的悬浮卡(含「看它的来路 →」),所以上下文条里只剩屏名与结论。窗口切到
本屏时自动收起检查器、切走时恢复(window.py 的 `_sync_inspector`)。
列表按检测原序,可按列表头部的「各级需要看」芯片筛选。
"""

from ...diagnostics import processing_ms
from .. import theme
from ..plots.cloud import make_cloud_view
from ..state import ViewState
from ..widgets.chips import StageChips
from ..widgets.conclusion import Conclusion
from ..widgets.stagelist import StageList
from .base import Screen
from .common import (
    COL_AZIMUTH,
    COL_COHERENCE,
    COL_COHERENCE_FULL,
    COL_ELEVATION,
    COL_FLAGGED,
    COL_INDEX,
    COL_RANGE,
    COL_SNR,
    COL_VELOCITY,
)


def clauses(frame):
    out = [(f"{len(frame.detections)} 个检测 · {len(frame.tracks)} 条航迹", 0)]
    total = processing_ms(frame.timings_ms)
    period = frame.radar.frame_period_s * 1e3
    if total > period:
        out.insert(0, (f"处理 {total:.1f} ms 超出帧周期 {period:.0f} ms", 2))
    return out


class ResultsScreen(Screen):
    key = "results"
    title = "场景"

    def __init__(self, state: ViewState) -> None:
        self.cloud = make_cloud_view(state)
        self.list = StageList(
            state,
            "检测列表",
            [
                COL_INDEX,
                COL_RANGE,
                COL_VELOCITY,
                COL_SNR,
                COL_AZIMUTH,
                COL_ELEVATION,
                COL_COHERENCE,
                COL_COHERENCE_FULL,
                COL_FLAGGED,
            ],
            obey_filter=True,
        )
        super().__init__(
            state,
            self.cloud,
            self.list,
            Conclusion(state, clauses),
            list_extra=StageChips(state),
        )
        # 全出血:卡片去掉描边与圆角,画布一直铺到主区边
        self.card.setStyleSheet(
            f"#Card {{ background: {theme.PLOT}; border: 0; border-radius: 0; }}"
        )
