"""解耦屏(仅步进频波形,处理链行上位于 CFAR 与测角之间):距离是怎样由「两把尺」定下来的?

版式「漏斗 + 候选表」(views/decouple.py):左上是粗尺 / 细尺 / 放大条(选中检测;分段控件可
切成走动补偿对照,「全部检测」时收起它、让散点独占左列),左下常驻全部检测的修正量散点,右列是
读数格与候选周期表。列表抽屉是本帧检测的展开周期与粗距离(没有逐检测指标,保持检测原序)。
一句话结论:本帧 N 个检测,M 个落在第 0 周期。
"""

from ...diagnostics.slices import decouple_summary, decouple_table
from ..state import ViewState
from ..views.decouple import DecoupleView
from ..views.panels import flatten_card
from ..widgets.conclusion import Conclusion
from ..widgets.stagelist import Column, StageList
from .base import Screen
from .common import COL_INDEX, COL_RANGE, COL_SNR, COL_VELOCITY


def _period(frame, k):
    table = decouple_table(frame)
    if table is None or table["index"][k] < 0:
        return None
    return int(table["period"][k])


def _coarse(frame, k):
    table = decouple_table(frame)
    if table is None or table["index"][k] < 0:
        return None
    return float(table["coarse_range_m"][k])


COL_PERIOD = Column("周期", _period, "{:d}")
COL_COARSE = Column("粗距离 m", _coarse, "{:.2f}")


def clauses(frame):
    summary = decouple_summary(frame)
    if summary is None:
        return [(f"本帧 {len(frame.detections)} 个检测,没有解耦中间量", 0)]
    total, zero = summary
    n = len(frame.detections)
    if total == 0:
        return [(f"本帧 {n} 个检测", 0)]
    return [(f"本帧 {n} 个检测,{zero} 个落在第 0 周期", 0)]


class DecoupleScreen(Screen):
    key = "decouple"
    title = "解耦"

    def __init__(self, state: ViewState) -> None:
        self.view = DecoupleView(state)
        self.list = StageList(
            state,
            "检测(展开周期)",
            [COL_INDEX, COL_RANGE, COL_VELOCITY, COL_SNR, COL_PERIOD, COL_COARSE],
        )
        super().__init__(state, self.view, self.list, Conclusion(state, clauses))
        flatten_card(self)
