"""多普勒屏:同一目标的多普勒副本是否落在预期的子带里?

TDM 板看 RD 图(零频居中);DDM 板的主图默认是**解调后的非相干探测矩阵**(距离 × 带内多普勒
bin,每格取胜出 fold 的 Σ_t Σ_rx|X|² = core 写进 RdMap、CFAR 消费的那张),分段控件可切到
解调前的原始矩阵(看 4 个 TX 副本各占哪条)。主视图按 `radar.modulation` 在 RD 图与 DDM 矩阵
之间切。版式是「主图 + 边缘剖面」:主图上沿多普勒剖面(选中距离门那一行)、右沿距离剖面(选中
多普勒列)、右上角 fold 余量分布直方图(TDM 板没有 fold,显示空态)。列表按「最可疑」排序:TDM 用格电平升序,DDM 用 fold 余量升序(fold 判错或余量小的排
最前面)。选中距离门的慢时间谱(TX 副本 bin 与空子带 bin 标在谱上)在溯源页的多普勒一级。
"""

import numpy as np
from PySide6.QtWidgets import QGridLayout, QWidget

from ...diagnostics import Level
from .. import theme
from ..plots.mini import SlowTimeSpectrum
from ..state import ViewState
from ..views.ddm import DdmMatrixView
from ..views.marginal import (
    BOTTOM_AXIS_HEIGHT,
    LEFT_AXIS_WIDTH,
    DopplerProfile,
    FoldHistogram,
    RangeProfile,
    SliceSource,
)
from ..views.panels import Panel, flatten_card
from ..views.rdmap import RdMapView
from ..widgets.conclusion import Conclusion
from ..widgets.stagelist import Column, StageList
from .base import Screen
from .common import (
    COL_INDEX,
    COL_RANGE,
    COL_SNR,
    COL_VELOCITY,
    metric,
    order_by,
    summary,
)

_fold_margin = metric("fold_margin_db")


def _is_ddm(frame) -> bool:
    return getattr(frame.radar, "modulation", "tdm") == "ddm"


def _ddm_only(value):
    """DDM 专有的列:TDM 板显示「—」。"""

    def cell(frame, k):
        return value(frame, k) if _is_ddm(frame) else None

    return cell


COL_FOLD = Column("fold", _ddm_only(lambda f, k: float(metric("fold")(f, k))), "{:.0f}")
COL_FOLD_MARGIN = Column("fold 余量 dB", _ddm_only(_fold_margin), "{:+.1f}")


def _order(frame):
    return (order_by("fold_margin_db") if _is_ddm(frame) else order_by("power_db"))(
        frame
    )


def clauses(frame):
    s = summary(frame)
    if s is None or not _is_ddm(frame):
        return [(f"{len(frame.detections)} 个检测(TDM 没有 fold 判决)", 0)]
    out = []
    n = s.flagged.get("doppler", 0)
    if n:
        bad = bool((frame.diagnostics.verdicts.levels["doppler"] >= Level.BAD).any())
        out.append((f"{n} / {s.n_detections} 个 fold 余量偏低", 2 if bad else 1))
    if np.isfinite(s.fold_margin_median_db):
        out.append((f"fold 余量中位 {s.fold_margin_median_db:.1f} dB", 0))
    if np.isfinite(s.empty_suppression_median_db):
        out.append((f"空带抑制中位 {s.empty_suppression_median_db:.1f} dB", 0))
    return out or [(f"{s.n_detections} 个检测", 0)]


class DopplerMain(QWidget):
    """主图 + 边缘剖面:主图按板的调制方式在 RD 图(TDM)与 DDM 矩阵之间切,上沿放选中距离门的
    多普勒剖面(与主图共横轴)、右沿放选中多普勒列的距离剖面(与主图共纵轴)、右上角放 fold
    余量直方图(TDM 板显示「不适用」)。

        ┌ 多普勒剖面(高 130) ─┬ fold 余量分布(宽 250)┐
        │ 主图                │ 距离剖面            │

    这里是**普通 QWidget** 而不是 ResultView:RD 图 / DDM 矩阵 / 三张边缘图都订阅 ViewState
    自己渲染,外层只按 `radar.modulation` 切换主图可见性并把剖面的联动轴接到可见的那张图上。
    把 ResultView 套在 ResultView 里(或套 QStackedWidget)在 PySide6 6.11 的离屏绘制下切屏时
    段错误(实测),普通容器 + setVisible 没有问题。两个主图共用一个读数标签;标题行的分段控件、
    底部的图例随显示的那个视图切换。"""

    PROFILE_HEIGHT = 130
    SIDE_WIDTH = 250

    def __init__(self, state: ViewState) -> None:
        super().__init__()
        self.tdm = RdMapView(state, stage="doppler")
        self.ddm = DdmMatrixView(state)
        self.ddm.readout = self.tdm.readout  # 共用一个读数标签
        self.ddm.hide()
        for widget in self.ddm.controls:
            widget.hide()
        # 主图贴满面板、左轴同宽,上沿剖面的绘图区才与主图对齐;右沿剖面与主图底轴同高
        for view in (self.tdm, self.ddm):
            view.layout().setContentsMargins(0, 0, 0, 0)
            view.plot.getAxis("left").setWidth(LEFT_AXIS_WIDTH)
            view.plot.getAxis("bottom").setHeight(BOTTOM_AXIS_HEIGHT)
        source = SliceSource(self._mode)
        self.top = DopplerProfile(state, source)
        self.right = RangeProfile(state, source)
        self.fold = FoldHistogram(state)
        self.ddm.mode.currentIndexChanged.connect(self._on_ddm_mode)

        map_pane = Panel()
        map_pane.body.addWidget(self.tdm)
        map_pane.body.addWidget(self.ddm)
        top_pane, fold_pane, right_pane = (
            Panel(self.top),
            Panel(self.fold),
            Panel(self.right),
        )
        top_pane.setFixedHeight(self.PROFILE_HEIGHT)
        fold_pane.setFixedSize(self.SIDE_WIDTH, self.PROFILE_HEIGHT)
        right_pane.setFixedWidth(self.SIDE_WIDTH)
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(theme.GAP)
        grid.addWidget(top_pane, 0, 0)
        grid.addWidget(fold_pane, 0, 1)
        grid.addWidget(map_pane, 1, 0)
        grid.addWidget(right_pane, 1, 1)
        grid.setRowStretch(1, 1)
        grid.setColumnStretch(0, 1)

        self.controls = self.tdm.controls + self.ddm.controls
        self.legends = self.tdm.legends
        self.readout = self.tdm.readout
        self._ddm = False
        self._link(self.tdm)
        state.source_changed.connect(self._on_source)
        state.frame_changed.connect(self._on_frame)

    def _mode(self, frame) -> str:
        """剖面取哪个口径:tdm / demod(DDM 解调后)/ raw(DDM 解调前)。"""
        if not _is_ddm(frame):
            return "tdm"
        return "raw" if self.ddm.mode.currentIndex() == 1 else "demod"

    def _link(self, view) -> None:
        self.top.link(view.plot.vb)
        self.right.link(view.plot.vb)

    def _on_ddm_mode(self, _index: int) -> None:
        """DDM 主图换口径:剖面取的矩阵跟着换(隐藏的剖面等显示时再补画)。"""
        frame = self.top.state.frame
        if frame is None:
            return
        for view in (self.top, self.right):
            if view.isVisible():
                view.render_frame(frame)
                view.render_selection(frame, view.state.selection)

    def _on_source(self, source) -> None:
        self._set_ddm(getattr(source, "modulation", "tdm") == "ddm")

    def _on_frame(self, frame) -> None:
        self._set_ddm(_is_ddm(frame))

    def _set_ddm(self, ddm: bool) -> None:
        if ddm == self._ddm:
            return
        self._ddm = ddm
        self.tdm.setVisible(not ddm)
        self.ddm.setVisible(ddm)
        for widget in self.tdm.controls + self.tdm.legends:
            widget.setVisible(not ddm)
        for widget in self.ddm.controls:
            widget.setVisible(ddm)
        self._link(self.ddm if ddm else self.tdm)


class DopplerScreen(Screen):
    key = "doppler"
    title = "多普勒 FFT"

    def __init__(self, state: ViewState) -> None:
        self.main = DopplerMain(state)
        self.list = StageList(
            state,
            "检测(fold 余量 / 格电平)",
            [COL_INDEX, COL_RANGE, COL_VELOCITY, COL_SNR, COL_FOLD, COL_FOLD_MARGIN],
            order=_order,
        )
        super().__init__(
            state,
            self.main,
            self.list,
            Conclusion(state, clauses),
            trace_plots=[SlowTimeSpectrum()],
        )
        flatten_card(self)
