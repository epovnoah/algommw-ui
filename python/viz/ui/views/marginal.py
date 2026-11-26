"""多图版式用的小件:带描边的面板、压在图内的小标签、边缘剖面图。

八屏版式改造(ADC 的 2×2、距离的上下共轴、多普勒的主图 + 边缘剖面)共用这些:

- `Chip`:压在 PlotWidget 内角落的半透明小标签(不占布局、不吃鼠标、放不下时省略);
- 带框面板 `Panel` 与 `flatten_card` 在 `views/panels.py`(多屏共用,这里不再重复);
- `SliceSource` / `DopplerProfile` / `RangeProfile` / `FoldHistogram`:多普勒屏的上沿剖面、
  右沿剖面与 fold 余量直方图。数据取自帧里已有的 rd_db / DDM 矩阵 / 诊断指标,不重新计算。
"""

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QEvent, Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from ...diagnostics.slices import ddm_raw_matrix_db, rd_db
from .. import theme
from ..state import ViewState
from .base import ResultView, set_title
from .common import rd_column

#: 多普勒屏里主图、上沿剖面共用的左轴宽度;主图、右沿剖面共用的底轴高度(两张图的绘图区才对得齐)
LEFT_AXIS_WIDTH = 52
BOTTOM_AXIS_HEIGHT = 40


class Chip(QLabel):
    """压在 PlotWidget 角上的小标签。anchor:'tl' 左上 / 'tr' 右上 / 'bl' 左下 / 'br' 右下。

    随坐标轴留边(左上避开左轴、右下避开底轴),放不下时从右边省略。"""

    def __init__(self, view: pg.PlotWidget, anchor: str = "tl") -> None:
        super().__init__(view)
        self._view, self._anchor, self._full = view, anchor, ""
        self._plot = view.getPlotItem()
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.set_accent(False)
        self.hide()
        view.installEventFilter(self)

    def set_accent(self, on: bool) -> None:
        """强调边框(当前 / 选中的那一路)。"""
        border = theme.ACCENT if on else theme.LINE
        self.setStyleSheet(
            f"background: rgba(12, 14, 18, 190); color: {theme.TEXT}; "
            f"border: 1px solid {border}; border-radius: 5px; padding: 1px 7px; "
            f"font-size: {theme.FONT_NOTE}px;"
        )

    def eventFilter(self, obj, event) -> bool:
        if event.type() in (QEvent.Type.Resize, QEvent.Type.Show):
            self.place()
        return False

    def set_text(self, text: str) -> None:
        if text == self._full:
            return
        self._full = text
        self.setVisible(bool(text))
        self.place()

    def place(self) -> None:
        if not self._full:
            return
        view = self._view
        left = self._plot.getAxis("left")
        bottom = self._plot.getAxis("bottom")
        x0 = int(left.width()) + 8 if left.isVisible() else 8
        y0 = int(bottom.height()) + 6 if bottom.isVisible() else 6
        room = max(view.width() - x0 - 8, 40)
        self.setText(
            self.fontMetrics().elidedText(
                self._full, Qt.TextElideMode.ElideRight, max(room - 16, 20)
            )
        )
        self.adjustSize()
        x = x0 if self._anchor in ("tl", "bl") else view.width() - 8 - self.width()
        y = 6 if self._anchor in ("tl", "tr") else view.height() - y0 - self.height()
        self.move(max(x, 0), max(y, 0))
        self.raise_()


def plain_plot(title: str = "") -> tuple[pg.PlotWidget, pg.PlotItem]:
    """统一风格的空 PlotWidget(网格、刻度字体、标题压在图内)。"""
    widget = pg.PlotWidget()
    plot = theme.style_plot(widget.getPlotItem())
    set_title(plot, title)
    return widget, plot


def round_range(low: float, high: float, step: float = 5.0) -> tuple[float, float]:
    """量程取整到 step 的倍数:逐帧变的数据不要让刻度每帧重排。"""
    low = float(np.floor(low / step) * step)
    high = float(np.ceil(high / step) * step)
    return low, max(high, low + step)


# ---------------------------------------------------------------- 多普勒屏的剖面
class SliceSource:
    """多普勒屏两条剖面共用的数据:每帧取一次 RD 图 / DDM 原始矩阵(同一帧同一口径不重复取)。

    口径 mode:'tdm' / 'demod'(DDM 解调后,与 RD 图同一矩阵)/ 'raw'(DDM 解调前);
    由屏按板型与 DDM 视图的分段控件给出(`mode_of`)。"""

    def __init__(self, mode_of) -> None:
        self.mode_of = mode_of
        self._frame = None
        self._mode = None
        self._data = None

    def get(self, frame) -> tuple[str, dict | None]:
        mode = self.mode_of(frame)
        if frame is not self._frame or mode != self._mode:
            self._data = ddm_raw_matrix_db(frame) if mode == "raw" else rd_db(frame)
            self._frame, self._mode = frame, mode
        return mode, self._data

    def clear(self) -> None:
        self._frame = self._mode = self._data = None


def selected_cell(frame, selection, data: dict, mode: str):
    """选中单元 → 图像里的 (行, 列);选中在镜像半边、或原始口径没有对应单列时对应项为 None。"""
    n_rows = data["image"].shape[0]
    row = selection.range_bin if 0 <= selection.range_bin < n_rows else None
    col = None
    if mode != "raw":
        col = int(rd_column(selection.doppler_bin, data["image"].shape[1]))
    return row, col


class DopplerProfile(ResultView):
    """多普勒屏上沿剖面:选中距离门那一行的功率随多普勒(与主图共用横轴)。"""

    title = "多普勒剖面"
    KEEP_AUTORANGE = True  # 量程自己定(见 render_frame),横轴由主图联动

    def __init__(self, state: ViewState, source: SliceSource, parent=None) -> None:
        super().__init__(state, parent)
        self.source = source
        self.widget, self.plot = plain_plot("多普勒剖面")
        self.plot.getAxis("left").setWidth(LEFT_AXIS_WIDTH)
        self.plot.setLabel("left", "dB")
        self.curve = self.plot.plot(
            pen=theme.pen(theme.SIGNAL, 1.4), fillLevel=-200, brush=(138, 180, 255, 26)
        )
        self.cursor = pg.InfiniteLine(
            angle=90,
            movable=False,
            pen=theme.pen(theme.ACCENT, 1, Qt.PenStyle.DashLine),
        )
        self.plot.addItem(self.cursor, ignoreBounds=True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.widget)

    def link(self, view_box: pg.ViewBox | None) -> None:
        self.plot.vb.setXLink(view_box)

    def reset(self, source) -> None:
        self.curve.setData([], [])
        self.cursor.setVisible(False)
        self.source.clear()

    def render_frame(self, frame) -> None:
        self._draw(frame, self.state.selection)

    def render_selection(self, frame, selection) -> None:
        self._draw(frame, selection)

    def _draw(self, frame, selection) -> None:
        mode, data = self.source.get(frame)
        if data is None or selection is None:
            self.curve.setData([], [])
            self.cursor.setVisible(False)
            set_title(self.plot, "多普勒剖面:该帧没有中间量,不适用")
            return
        row, col = selected_cell(frame, selection, data, mode)
        if row is None:
            self.curve.setData([], [])
            self.cursor.setVisible(False)
            set_title(self.plot, "多普勒剖面:选中在镜像半边,不画")
            return
        x = data["bin"] if mode == "raw" else data["doppler"]
        y = data["image"][row]
        self.curve.setData(x, y)
        low, high = round_range(
            float(np.nanmin(y)), float(np.nanmax(y)) + 1.0, step=5.0
        )
        self.plot.setYRange(low, high, padding=0)
        if col is None:
            self.cursor.setVisible(False)
        else:
            self.cursor.setValue(float(x[col]))
            self.cursor.setVisible(True)
        set_title(
            self.plot,
            f"多普勒剖面 @ {float(data['range_m'][row]):.1f} m"
            + ("(解调前)" if mode == "raw" else ""),
        )


class RangeProfile(ResultView):
    """多普勒屏右沿剖面:选中多普勒列的功率随距离(竖向曲线,与主图共用纵轴)。"""

    title = "距离剖面"
    KEEP_AUTORANGE = True

    def __init__(self, state: ViewState, source: SliceSource, parent=None) -> None:
        super().__init__(parent=parent, state=state)
        self.source = source
        self.widget, self.plot = plain_plot("距离剖面")
        self.plot.hideAxis("left")
        self.plot.getAxis("bottom").setHeight(BOTTOM_AXIS_HEIGHT)
        self.plot.setLabel("bottom", "dB")
        self.curve = self.plot.plot(pen=theme.pen(theme.SIGNAL, 1.4))
        self.cursor = pg.InfiniteLine(
            angle=0,
            movable=False,
            pen=theme.pen(theme.ACCENT, 1, Qt.PenStyle.DashLine),
        )
        self.plot.addItem(self.cursor, ignoreBounds=True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.widget)

    def link(self, view_box: pg.ViewBox | None) -> None:
        self.plot.vb.setYLink(view_box)

    def reset(self, source) -> None:
        self.curve.setData([], [])
        self.cursor.setVisible(False)
        self.source.clear()

    def render_frame(self, frame) -> None:
        self._draw(frame, self.state.selection)

    def render_selection(self, frame, selection) -> None:
        self._draw(frame, selection)

    def _draw(self, frame, selection) -> None:
        mode, data = self.source.get(frame)
        if data is None or selection is None:
            self.curve.setData([], [])
            self.cursor.setVisible(False)
            set_title(self.plot, "距离剖面:不适用")
            return
        if mode == "raw":
            self.curve.setData([], [])
            self.cursor.setVisible(False)
            set_title(self.plot, "距离剖面:解调前口径无单列剖面")
            return
        row, col = selected_cell(frame, selection, data, mode)
        if col is None:
            return
        image = data["image"]
        x = image[:, col]
        self.curve.setData(x, data["range_m"])
        low, high = round_range(float(np.nanmin(x)), float(np.nanmax(x)) + 1.0)
        self.plot.setXRange(low, high, padding=0)
        if row is None:
            self.cursor.setVisible(False)
        else:
            self.cursor.setValue(float(data["range_m"][row]))
            self.cursor.setVisible(True)
        axis = data["doppler"]
        where = (
            f"bin {int(axis[col])}"
            if data["doppler_unit"] == "bin"
            else f"{float(axis[col]):+.2f} {data['doppler_unit']}"
        )
        set_title(self.plot, f"距离剖面 @ {where}")


class FoldHistogram(ResultView):
    """多普勒屏右上角:本帧全部检测的 fold 余量分布(数据取诊断指标 `fold_margin_db`)。"""

    title = "fold 余量分布"
    KEEP_AUTORANGE = True
    BINS = 14

    def __init__(self, state: ViewState, parent=None) -> None:
        super().__init__(state, parent)
        self.head = QLabel()
        self.head.setStyleSheet(
            f"color: {theme.TEXT}; font-size: {theme.FONT_NOTE}px; font-weight: 600;"
            " padding: 3px 8px 0 8px;"
        )
        self.median = QLabel()
        self.median.setStyleSheet(
            f"color: {theme.DIM}; font-size: {theme.FONT_NOTE}px; padding: 3px 8px 0 0;"
        )
        self.widget, self.plot = plain_plot()
        self.plot.setMenuEnabled(False)
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.setLabel("bottom", "dB")
        self.bars = pg.BarGraphItem(x=[], height=[], width=1, brush=theme.SIGNAL)
        self.plot.addItem(self.bars)
        #: 空态文字(没有可画的数据时盖在图上,图本身隐藏)
        self.empty = QLabel()
        self.empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty.setWordWrap(True)
        self.empty.setStyleSheet(
            f"color: {theme.DIM}; font-size: {theme.FONT_NOTE}px; padding: 8px;"
        )
        self.head.setText("fold 余量分布")
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.head)
        row.addStretch(1)
        row.addWidget(self.median)
        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)
        column.addLayout(row)
        column.addWidget(self.widget, 1)
        column.addWidget(self.empty, 1)
        self.empty.hide()

    def _show_empty(self, text: str) -> None:
        self.empty.setText(text)
        self.empty.show()
        self.widget.hide()
        self.median.clear()

    def reset(self, source) -> None:
        self.bars.setOpts(x=[], height=[])

    def render_frame(self, frame) -> None:
        if getattr(frame.radar, "modulation", "tdm") != "ddm":
            self._show_empty("TDM 板没有 fold,不适用")
            return
        diag = frame.diagnostics
        margins = None if diag is None else np.asarray(diag.metrics.fold_margin_db)
        if margins is not None:
            margins = margins[np.isfinite(margins)]
        if margins is None or margins.size == 0:
            self._show_empty("本帧没有 fold 余量(诊断未开或没有检测)")
            return
        self.empty.hide()
        self.widget.show()
        low = min(0.0, float(np.floor(margins.min() * 2) / 2))
        high = max(float(np.ceil(margins.max() * 2) / 2), low + 1.0)
        counts, edges = np.histogram(margins, bins=self.BINS, range=(low, high))
        width = float(edges[1] - edges[0])
        self.bars.setOpts(x=edges[:-1] + width / 2, height=counts, width=width * 0.9)
        self.plot.setXRange(low, high, padding=0.02)
        self.plot.setYRange(0, max(int(counts.max()), 1), padding=0.08)
        self.median.setText(
            f"中位 {float(np.median(margins)):.1f} dB · {margins.size} 个"
        )
