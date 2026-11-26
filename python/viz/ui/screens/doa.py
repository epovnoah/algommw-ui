"""测角屏:这个点的方向是谱峰,还是噪声里的一个局部最大?

版式「谱图 + 诊断卡」(views/angle.py):左列方位切片(大)+ 俯仰切片,右列二维空间谱 + 通道相位残差
(含相干度读数);corr 只有一维水平相关谱(俯仰来自垂直基线)。列表按相干度升序(最不像谱峰的在最上面);
溯源页的测角一级仍保留各虚拟通道的相位残差 / 幅度小图与全帧相干度散点。相干度 =
峰值功率 / (通道数 · 快照能量);corr 使用归一化相关谱峰,两种口径不可直接当作同一统计量标定。
"""

import numpy as np

from .. import theme
from ..plots.mini import ChannelBars, DetectionScatter
from ..state import ViewState
from ..views.angle import AngleSpectrumView
from ..views.panels import flatten_card
from ..widgets.conclusion import Conclusion
from ..widgets.stagelist import StageList
from .base import Screen
from .common import (
    COL_AZIMUTH,
    COL_COHERENCE,
    COL_COHERENCE_FULL,
    COL_ELEVATION,
    COL_INDEX,
    COL_POWER,
    COL_RANGE,
    COL_VELOCITY,
    order_by,
    summary,
)

COHERENCE_RANGE = (0.2, 1.02)


def _criteria(frame):
    return frame.diagnostics.verdicts.criteria


def doa_scatter() -> DetectionScatter:
    """全帧相干度散点(模块级不能建 QWidget,屏构造时再建)。"""
    return DetectionScatter(
        "doa",
        lambda f: f.diagnostics.metrics.coherence,
        "相干度 · 本帧全部检测",
        COHERENCE_RANGE,
        lines=[
            (lambda f: _criteria(f).coherence_bad, theme.LEVEL_COLORS[2]),
            (lambda f: _criteria(f).coherence_warn, theme.LEVEL_COLORS[1]),
        ],
    )


def clauses(frame):
    s = summary(frame)
    if s is None:
        return [(f"{len(frame.detections)} 个检测", 0)]
    out = []
    if s.n_low_coherence:
        out.append((f"{s.n_low_coherence} / {s.n_detections} 个低相干", 1))
    if s.n_at_edge:
        out.append((f"{s.n_at_edge} 个贴扫描边界", 1))
    if s.n_dropped:
        reasons = "、".join(f"{k} {v}" for k, v in s.dropped.items())
        out.append((f"测角剔除 {s.n_dropped} 个({reasons})", 1))
    if np.isfinite(s.coherence_median):
        out.append((f"相干度中位 {s.coherence_median:.2f}", 0))
    return out or [(f"{s.n_detections} 个检测", 0)]


class DoaScreen(Screen):
    key = "doa"
    title = "测角"

    def __init__(self, state: ViewState) -> None:
        self.view = AngleSpectrumView(state)
        self.list = StageList(
            state,
            "检测(按相干度)",
            [
                COL_INDEX,
                COL_RANGE,
                COL_VELOCITY,
                COL_AZIMUTH,
                COL_ELEVATION,
                COL_COHERENCE,
                COL_COHERENCE_FULL,
                COL_POWER,
            ],
            order=order_by("coherence"),
        )
        scatter = doa_scatter()
        scatter.on_select = state.select_detection
        super().__init__(
            state,
            self.view,
            self.list,
            Conclusion(state, clauses),
            trace_plots=[ChannelBars(state), scatter],
        )
        flatten_card(self)
