"""ADC 时域:原始采样是否正常(幅度、削波、各 RX 一致)?

小倍数:每路 RX 一张 chirp × 采样幅度图(dBFS,inferno),同时可见(1 / 2 / 4 路按真实 RX 数排:
4 路 2×2,其余竖排),不再分段切 RX。每格左上小标签「RX1 · 饱和 x% · 偏差 y dB」(饱和率取
`AdcStats`,偏差 = 该路 RMS 减各路均值),右下「chirp N × 采样 M」。点任一格即选中那个 chirp,
虚线横贯全部格。选中 chirp 的各 RX 波形是次要视图,挂在滑出栏溯源页的 ADC 一级
(`plots/mini.py` 的 AdcWave,随 `chirp_changed` 重画)。选 chirp 只重画虚线,不重算处理链。
"""

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QGridLayout, QVBoxLayout, QWidget

from ...diagnostics.slices import adc_image_db
from .. import theme
from ..state import ViewState
from .base import ResultView
from .common import heat_item, hover_cell, readout_label, show_heat
from .marginal import Chip
from .panels import Panel

#: 同时摆得下的最大 RX 数(各板 1 / 2 / 4 路)
MAX_RX = 4


class AdcPane(ResultView):
    """一路 RX 的 chirp × 采样热图。选中 chirp 由所在的 `AdcGrid` 统一持有。"""

    title = "ADC 时域"

    def __init__(self, state: ViewState, grid: "AdcGrid", rx: int) -> None:
        super().__init__(state)
        self.grid, self.rx = grid, rx
        self.widget = pg.PlotWidget()
        self.plot = theme.style_plot(self.widget.getPlotItem())
        self.image = heat_item(self.plot)
        self.cursor = pg.InfiniteLine(
            angle=0,
            movable=False,
            pen=theme.pen(theme.ACCENT, 1, Qt.PenStyle.DashLine),
        )
        self.plot.addItem(self.cursor, ignoreBounds=True)
        self.tag = Chip(self.widget, "tl")
        self.size_tag = Chip(self.widget, "br")
        self.pane = Panel(self.widget)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.pane)
        self.widget.scene().sigMouseClicked.connect(self._on_click)
        self.widget.scene().sigMouseMoved.connect(self._on_move)
        self._rect = self._axes = self._image = self._hover = None

    def reset(self, source) -> None:
        self.image.clear()
        self._rect = self._axes = self._image = self._hover = None
        self.tag.set_text("")
        self.size_tag.set_text("")

    def _tag_text(self, frame) -> str:
        text = f"RX{self.rx + 1}"
        diag = frame.diagnostics
        adc = None if diag is None or diag.summary is None else diag.summary.adc
        if adc is None or self.rx >= len(adc.rms_dbfs):
            return text
        per_rx = adc.n_values / max(len(adc.rms_dbfs), 1) if adc.n_values else 0
        if per_rx:
            rate = 100.0 * float(adc.saturated[self.rx]) / per_rx
            text += f" · 饱和 {rate:.2f}%" if rate else " · 饱和 0%"
        mean = float(np.nanmean(adc.rms_dbfs)) if np.isfinite(adc.rms_dbfs).any() else 0
        offset = float(adc.rms_dbfs[self.rx]) - mean
        if np.isfinite(offset):
            offset = 0.0 if abs(offset) < 0.05 else offset  # 不显示「-0.0」
            text += f" · 偏差 {offset:+.1f} dB"
        return text

    def render_frame(self, frame) -> None:
        image = (
            adc_image_db(frame, self.rx) if self.rx < int(frame.radar.n_rx) else None
        )
        if image is None:
            self.image.clear()
            self._image = self._rect = self._axes = None
            self.tag.set_text(f"RX{self.rx + 1} · 该帧没有中间量(诊断未开),不适用")
            self.size_tag.set_text("")
            return
        self._image = image
        n_chirps, n_samples = image.shape
        self.grid.clamp_chirp(n_chirps)
        _, self._rect, self._axes = show_heat(
            self.image, image, np.arange(n_samples), np.arange(n_chirps)
        )
        self.tag.set_text(self._tag_text(frame))
        self.size_tag.set_text(f"chirp {n_chirps} × 采样 {n_samples}")
        self.set_chirp(self.grid.chirp_index)

    def render_selection(self, frame, selection) -> None:
        self.set_chirp(self.grid.chirp_index)

    def set_chirp(self, chirp: int) -> None:
        self.cursor.setValue(float(chirp))

    def set_current(self, current: bool) -> None:
        """列表里点中的那一路:标签边框用强调色。"""
        self.tag.set_accent(current)

    def _on_click(self, event) -> None:
        if self._rect is None or event.button() != Qt.MouseButton.LeftButton:
            return
        cell = hover_cell(self.plot, event.scenePos(), self._rect, self._axes)
        if cell is not None:
            self.grid.select_chirp(cell[0])

    def _on_move(self, event) -> None:
        """ADC 图没有数值刻度:鼠标所在 (chirp, 采样) 的 dBFS 只在读数行显示。"""
        if self._rect is None:
            return
        cell = hover_cell(self.plot, event, self._rect, self._axes)
        if cell is None:
            if self._hover is not None:
                self._hover = None
                self.grid.readout.clear()
            return
        if cell == self._hover:
            return
        self._hover = cell
        row, col = cell
        self.grid.readout.setText(
            f"RX{self.rx + 1} · chirp {row} · 采样 {col} · "
            f"{float(self._image[row, col]):+.1f} dBFS"
        )


class AdcGrid(QWidget):
    """四路(或 1 / 2 路)RX 的热图网格;屏的主视图。

    普通 QWidget(各格 `AdcPane` 才是订阅 ViewState 的 ResultView),持有选中的 chirp、
    `chirp_changed` 信号(AdcWave 订阅)与共用的悬停读数标签。格数随板的真实 RX 数走:
    先建满 MAX_RX 格,多余的藏起来(换板时只重排,不销毁重建)。"""

    chirp_changed = Signal(int)
    title = "ADC 时域"

    def __init__(self, state: ViewState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.state = state
        self.readout = readout_label()
        self.controls: list[QWidget] = []
        self.legends: list[QWidget] = []
        self.chirp_index = 0  #: 选中的 chirp(波形取它;点任一格切换)
        self.panes = [AdcPane(state, self, rx) for rx in range(MAX_RX)]
        self.n_rx = 0
        self._cols = 0
        self.layout_grid = QGridLayout(self)
        self.layout_grid.setContentsMargins(0, 0, 0, 0)
        self.layout_grid.setSpacing(theme.GAP)
        self._arrange(MAX_RX)
        state.source_changed.connect(self._on_source)
        state.frame_changed.connect(self._on_frame)

    # ---- 格数随板走 ----
    def _arrange(self, n_rx: int) -> None:
        n_rx = min(max(int(n_rx), 1), MAX_RX)
        if n_rx == self.n_rx:
            return
        self.n_rx = n_rx
        cols = 2 if n_rx >= 4 else 1
        for pane in self.panes:
            self.layout_grid.removeWidget(pane)
        rows = -(-n_rx // cols)
        for i in range(MAX_RX):
            pane = self.panes[i]
            if i < n_rx:
                self.layout_grid.addWidget(pane, i // cols, i % cols)
                pane.show()
            else:
                pane.hide()
        for r in range(MAX_RX):
            self.layout_grid.setRowStretch(r, 1 if r < rows else 0)
        for c in range(2):
            self.layout_grid.setColumnStretch(c, 1 if c < cols else 0)

    def _on_source(self, source) -> None:
        self._arrange(int(source.n_rx))
        self.chirp_index = 0
        self.chirp_changed.emit(0)

    def _on_frame(self, frame) -> None:
        self._arrange(int(frame.radar.n_rx))

    # ---- chirp 选择 ----
    def clamp_chirp(self, n_chirps: int) -> None:
        self.chirp_index = min(self.chirp_index, n_chirps - 1)

    def select_chirp(self, chirp: int) -> None:
        chirp = int(chirp)
        if chirp == self.chirp_index:
            return
        self.chirp_index = chirp
        for pane in self.panes:
            pane.set_chirp(chirp)
        self.chirp_changed.emit(chirp)

    def set_rx(self, index: int) -> None:
        """列表点中某一路:该格标签用强调边框(各格同时可见,不再切换)。"""
        for pane in self.panes:
            pane.set_current(pane.rx == int(index))
