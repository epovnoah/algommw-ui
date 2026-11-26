"""处理链:整条链常在,嵌在顶栏中间 —— 既是导航,也是全链摘要。

每级是一个紧凑按钮,一行内「状态形状 + 级名 + 等宽小字实时数」(● 正常 / ▲ 需要看 / ■ 异常,
与颜色双编码;取数口径见 `diagnostics.chain`),当前级底部一条 2 px 强调色下划线并带浅底;
级间用弱色的 `›` 分隔。点一级切到该级的屏;「解耦」一级只在波形带载频步进时出现。

各级**按内容宽度**从左排起,不等分铺满;实时数的宽度按一句最长样例定死,所以数字逐帧变化时
各级位置不动。链区放不下时逐档省略实时数 —— 先只留当前级的,再全部省略,最后才裁 —— 链区的
最小宽度是 0,绝不撑宽窗口。每帧只改文字与形状颜色(没变的不动),不整体重设样式表。
"""

from collections.abc import Sequence

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QWidget,
)

from ...diagnostics.chain import DASH, NA, chain_cells
from .. import theme
from ..state import ViewState
from ..views.common import ClipLabel

#: 屏 key → 计入该级耗时的 timings_ms 键(提示里显示)
_TIMING_KEYS = {
    "adc": ("read",),
    "rangefft": ("range",),
    "doppler": ("doppler",),
    "cfar": ("cfar",),
    "decouple": ("pair",),
    "doa": ("doa",),
    "track": ("cluster", "track"),
    "results": ("world", "extract"),
}
#: 只在有步进的波形里出现的级
CONDITIONAL = {"decouple"}
#: 每级实时数的最长样例:按它定死实时数的宽度(超出的部分被裁,不挪动别的级)
_VALUE_SAMPLES = {
    "adc": "饱和 0.0%",
    "rangefft": "峰 00.0 m",
    "doppler": "余量 -0.0 dB",
    "cfar": "000 检测",
    "decouple": "细分辨 0.00 m",
    "doa": "相干 0.00",
    "track": "00 航迹",
    "results": "000 点",
}
_SEPARATOR_WIDTH = 12
_MARK_WIDTH = 12


class StageButton(QPushButton):
    def __init__(self, key: str, label: str) -> None:
        super().__init__()
        self.key, self.label = key, label
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        row = QHBoxLayout(self)
        row.setContentsMargins(6, 0, 6, 2)  # 下边留 2 px:下划线画在按钮自己的底边框上
        row.setSpacing(4)
        self.mark = QLabel("")
        self.mark.setFixedWidth(_MARK_WIDTH)
        self.name = QLabel(label)
        self.value = ClipLabel(DASH)
        sample = _VALUE_SAMPLES.get(key, "0000")
        self.value.setFixedWidth(
            QFontMetrics(theme.mono_font(theme.FONT_TICK)).horizontalAdvance(sample) + 2
        )
        self.value.setStyleSheet(
            f"color: {theme.DIM}; font-family: {theme.MONO}; font-size: {theme.FONT_TICK}px;"
        )
        self.mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        for widget in (self.mark, self.name, self.value):
            widget.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            row.addWidget(widget)
        self._level: int | None = None
        self._value_shown = True
        self._style()
        self.toggled.connect(lambda _on: self._style())

    def set_value_shown(self, shown: bool) -> None:
        if shown != self._value_shown:
            self._value_shown = shown
            self.value.setVisible(shown)

    def width_for(self, with_value: bool) -> int:
        """放得下「标记 + 级名(+ 实时数)」所需的宽度。"""
        row = self.layout()
        margins = row.contentsMargins()
        width = margins.left() + margins.right() + _MARK_WIDTH
        width += row.spacing() + self.name.sizeHint().width()
        if with_value:
            width += row.spacing() + self.value.width()
        return width

    def sizeHint(self) -> QSize:
        return QSize(self.width_for(self._value_shown), 36)

    def minimumSizeHint(self) -> QSize:
        return QSize(self.width_for(False), 0)

    def _style(self) -> None:
        on = self.isChecked()
        self.setStyleSheet(
            f"StageButton {{ background: {theme.RAISED if on else 'transparent'}; "
            f"border: 0; border-radius: 0; text-align: left; "
            f"border-bottom: 2px solid {theme.ACCENT if on else 'transparent'}; }}"
            f"StageButton:hover {{ border-bottom-color: "
            f"{theme.ACCENT if on else theme.LINE_HOVER}; }}"
        )
        self.name.setStyleSheet(
            f"color: {theme.TEXT if on else theme.DIM}; font-size: {theme.FONT_BODY}px;"
        )

    def show_cell(self, level: int, value: str, tip: str) -> None:
        if level != self._level:
            self._level = level
            self.mark.setText(theme.LEVEL_SHAPES[level] if level != NA else "")
            self.mark.setStyleSheet(
                f"color: {theme.LEVEL_COLORS[level]}; font-size: 10px;"
            )
        if value != self.value.text():
            self.value.setText(value)
        if tip != self.toolTip():
            self.setToolTip(tip)


class ChainRow(QWidget):
    """处理链(嵌在顶栏里)。切屏由窗口完成(Ctrl+1…9 的动作在窗口),这里只发信号。"""

    screen_chosen = Signal(str)

    def __init__(
        self,
        state: ViewState,
        tabs: Sequence[tuple[str, str]],
        available: set[str],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.state = state
        self.setObjectName("Chain")
        # 宽度完全由顶栏分配、不看各级的 sizeHint:窄了裁,不撑宽窗口
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        self.items: dict[str, StageButton] = {}
        self._separators: dict[str, QLabel] = {}
        self._current = ""
        group = QButtonGroup(self)
        group.setExclusive(True)
        for key, label in tabs:
            item = StageButton(key, label)
            item.setParent(self)
            item.setEnabled(key in available)
            item.clicked.connect(lambda _=False, k=key: self.screen_chosen.emit(k))
            group.addButton(item)
            self.items[key] = item
            sep = QLabel("›", self)
            sep.setAlignment(Qt.AlignmentFlag.AlignCenter)
            sep.setStyleSheet(
                f"color: {theme.LINE_HOVER}; font-size: {theme.FONT_BODY}px;"
            )
            sep.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            self._separators[key] = sep
            if key in CONDITIONAL:
                item.hide()
        state.frame_changed.connect(self.render_frame)
        state.source_changed.connect(self.reset)
        self._relayout()

    def minimumSizeHint(self) -> QSize:
        return QSize(0, 0)

    def sizeHint(self) -> QSize:
        return QSize(0, 36)

    def visible_keys(self) -> list[str]:
        """当前显示的级(顺序即处理顺序;isHidden 不依赖窗口是否已显示)。"""
        return [k for k, item in self.items.items() if not item.isHidden()]

    def set_current(self, key: str) -> None:
        item = self.items.get(key)
        if item is not None and not item.isChecked():
            item.setChecked(True)
        if key != self._current:
            self._current = key
            self._relayout()

    def reset(self, radar=None) -> None:
        for key in CONDITIONAL:
            if key in self.items and radar is not None:
                self.items[key].setVisible(radar.stepped)
        for item in self.items.values():
            item.show_cell(NA, DASH, item.label)
        self._relayout()

    def render_frame(self, frame) -> None:
        cells = chain_cells(frame)
        for key, item in self.items.items():
            level, value = cells[key]
            ms = sum(frame.timings_ms.get(k, 0.0) for k in _TIMING_KEYS.get(key, ()))
            tip = f"{item.label} · {ms:.1f} ms" if ms else item.label
            item.show_cell(level, value, tip)

    # ---- 排布:按内容宽度从左排起,放不下时逐档省略实时数 ----
    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._relayout()

    def _relayout(self) -> None:
        shown = [k for k in self.visible_keys()]
        gaps = _SEPARATOR_WIDTH * max(len(shown) - 1, 0)

        def total(mode: int) -> int:
            # mode 0 = 各级都带实时数;1 = 只有当前级带;2 = 都不带
            return gaps + sum(
                self.items[k].width_for(mode == 0 or (mode == 1 and k == self._current))
                for k in shown
            )

        mode = 0
        while mode < 2 and total(mode) > self.width():
            mode += 1
        height = self.height()
        x = 0
        for i, key in enumerate(shown):
            item = self.items[key]
            with_value = mode == 0 or (mode == 1 and key == self._current)
            item.set_value_shown(with_value)
            width = item.width_for(with_value)
            item.setGeometry(x, 0, width, height)
            x += width
            sep = self._separators[key]
            if i < len(shown) - 1:
                sep.setGeometry(x, 0, _SEPARATOR_WIDTH, height)
                sep.show()
                x += _SEPARATOR_WIDTH
            else:
                sep.hide()
        for key, item in self.items.items():
            if key not in shown:
                self._separators[key].hide()
