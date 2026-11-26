"""溯源页:选中检测在每一级留下的痕迹,从上到下一条竖直时间线(设计稿的 `.steps`)。

每级一行:状态圆点(当前级实心强调色)、级名 + 状态形状、一行等宽事实、可选的一张 64 px 小图(检查器只有 300 px 宽,行距 12);
点某一行跳到那一级的屏。各屏原来放在右栏 / 主区下半的次要视图(CFAR 切片、测角相位残差、
距离像局部、ADC 波形……)都挂在对应那一级下面:默认只露第一张 64 px 小图,点行右侧的
「⤢」放大(180 px、带标题,并露出该级其余的图)。
级的等级与事实取自 `diagnostics.chain.trace_facts`,只读帧与诊断,不做计算。
"""

from collections.abc import Sequence

from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ...diagnostics.chain import NA, trace_facts
from .. import theme
from ..plots.mini import MiniPlot
from ..state import Selection, ViewState
from ..views.base import ResultView

_GUTTER = 20  # 左侧留给圆点与连线


class _Header(QFrame):
    clicked = Signal()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class TraceRow(QWidget):
    chosen = Signal(str)  # 点行:跳到该级的屏
    enlarged_changed = Signal()

    def __init__(
        self, key: str, label: str, plots: Sequence[QWidget], last: bool
    ) -> None:
        super().__init__()
        self.key, self.label, self.last = key, label, last
        self.level = NA
        self.current = False
        self.plots = list(plots)
        self.enlarged = False
        column = QVBoxLayout(self)
        column.setContentsMargins(_GUTTER, 0, 0, 12)
        column.setSpacing(2)
        self.header = _Header()
        self.header.setCursor(Qt.CursorShape.PointingHandCursor)
        head = QHBoxLayout(self.header)
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(8)
        self.name = QLabel(label)
        self.name.setStyleSheet(f"color: {theme.TEXT}; font-weight: 500;")
        self.mark = QLabel("")
        head.addWidget(self.name)
        head.addWidget(self.mark)
        head.addStretch(1)
        self.zoom = QToolButton()
        self.zoom.setText("⤢")
        self.zoom.setCheckable(True)
        self.zoom.setToolTip("放大这一级的图")
        self.zoom.setCursor(Qt.CursorShape.PointingHandCursor)
        self.zoom.setStyleSheet(
            f"QToolButton {{ border: 0; color: {theme.DIM}; font-size: {theme.FONT_BODY}px; "
            f"min-width: {theme.MIN_HIT - 8}px; min-height: {theme.MIN_HIT - 8}px; }}"
            f"QToolButton:hover {{ color: {theme.TEXT}; }}"
            f"QToolButton:checked {{ color: {theme.TEXT}; }}"
        )
        self.zoom.setVisible(bool(self.plots))
        self.zoom.toggled.connect(self._on_zoom)
        head.addWidget(self.zoom)
        self.header.clicked.connect(lambda: self.chosen.emit(self.key))
        self.fact = QLabel("—")
        self.fact.setWordWrap(True)
        self.fact.setStyleSheet(
            f"color: {theme.DIM}; font-family: {theme.MONO}; font-size: {theme.FONT_TICK}px;"
        )
        column.addWidget(self.header)
        column.addWidget(self.fact)
        for plot in self.plots:
            if isinstance(plot, MiniPlot):
                plot.set_enlarged(False)
            column.addWidget(plot)
        self._apply_enlarged()

    def _on_zoom(self, on: bool) -> None:
        self.enlarged = on
        self._apply_enlarged()
        self.enlarged_changed.emit()

    def _apply_enlarged(self) -> None:
        """收起时只露第一张图(64 px);放大时全露(180 px)。没有数据的级(不适用)不露图。"""
        for i, plot in enumerate(self.plots):
            if isinstance(plot, MiniPlot):
                plot.set_enlarged(self.enlarged)
            plot.setVisible((self.enlarged or i == 0) and self.level != NA)
        self.zoom.setVisible(bool(self.plots) and self.level != NA)

    def shown_plots(self) -> list[QWidget]:
        if self.level == NA:
            return []
        return [p for i, p in enumerate(self.plots) if self.enlarged or i == 0]

    def show_state(self, level: int, text: str, current: bool) -> None:
        if level != self.level:
            self.level = level
            self.mark.setText(theme.LEVEL_SHAPES[level] if level != NA else "")
            self.mark.setStyleSheet(
                f"color: {theme.LEVEL_COLORS[level]}; font-size: 10px;"
            )
            self._apply_enlarged()
        if text != self.fact.text():
            self.fact.setText(text)
        if current != self.current:
            self.current = current
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        x = 5.5
        if not self.last:
            painter.setPen(QPen(QColor(theme.LINE), 1))
            painter.drawLine(QPointF(x, 16), QPointF(x, self.height()))
        color = QColor(theme.ACCENT if self.current else theme.LEVEL_COLORS[self.level])
        painter.setPen(QPen(color, 2))
        painter.setBrush(color if self.current else QColor(theme.GROUND))
        painter.drawEllipse(QPointF(x, 10), 4.5, 4.5)


class TracePage(ResultView):
    title = "溯源"
    KEEP_AUTORANGE = True  # 小图随选中检测变化

    screen_chosen = Signal(str)

    def __init__(
        self,
        state: ViewState,
        tabs: Sequence[tuple[str, str]],
        extras: dict[str, Sequence[QWidget]],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(state, parent)
        self.heading = QLabel("溯源")
        self.heading.setStyleSheet(
            f"color: {theme.TEXT}; font-size: 13px; font-weight: 600;"
        )
        self.hint = QLabel("它在每一级留下的痕迹,从上到下")
        self.hint.setStyleSheet(f"color: {theme.DIM};")
        content = QWidget()
        column = QVBoxLayout(content)
        column.setContentsMargins(10, 8, 8, 8)
        column.setSpacing(0)
        column.addWidget(self.heading)
        column.addWidget(self.hint)
        column.addSpacing(8)
        self.rows: dict[str, TraceRow] = {}
        keys = [(k, label) for k, label in tabs if k != "results"]
        for i, (key, label) in enumerate(keys):
            row = TraceRow(key, label, extras.get(key, ()), last=i == len(keys) - 1)
            row.chosen.connect(self.screen_chosen.emit)
            row.enlarged_changed.connect(self._rerender)
            column.addWidget(row)
            self.rows[key] = row
        column.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(content)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)
        self._current = ""

    def set_current(self, key: str) -> None:
        """当前所在的屏对应的那一级:圆点实心强调色。"""
        self._current = key
        for k, row in self.rows.items():
            row.current = k == key
            row.update()

    def set_stepped(self, stepped: bool) -> None:
        """「解耦」一行只在步进频波形里出现。"""
        if "decouple" in self.rows:
            self.rows["decouple"].setVisible(stepped)
            keys = [k for k, r in self.rows.items() if not r.isHidden()]
            for k, row in self.rows.items():
                row.last = k == keys[-1]

    def reset(self, source) -> None:
        for row in self.rows.values():
            row.show_state(NA, "—", row.key == self._current)
        self.heading.setText("溯源")

    def _rerender(self) -> None:
        if self.state.frame is not None and self.state.selection is not None:
            self.render_selection(self.state.frame, self.state.selection)

    def render_frame(self, frame) -> None:
        if self.state.selection is not None:
            self.render_selection(frame, self.state.selection)

    def render_selection(self, frame, selection: Selection) -> None:
        k = selection.detection
        if k is None or k >= len(frame.detections):
            self.heading.setText("溯源")
            self.hint.setText("非检测单元:在主图或列表里选一个检测")
            for row in self.rows.values():
                row.show_state(NA, "—", row.key == self._current)
            return
        self.heading.setText(f"检测 #{k} 的来路")
        self.hint.setText("它在每一级留下的痕迹,从上到下")
        facts = trace_facts(frame, k)
        for key, row in self.rows.items():
            level, text = facts.get(key, (NA, "—"))
            row.show_state(level, text, key == self._current)
            for plot in row.shown_plots():
                if isinstance(plot, MiniPlot):
                    plot.update_detection(frame, k)
