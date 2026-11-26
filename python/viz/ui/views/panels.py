"""复合主视图的面板件:带圆角描边的面板 `Panel`(设计稿的 `.pn`),四角可压小标签(`.chip`)。

多张图并排的屏(解耦、测角)把主视图写成一个复合 QWidget,里面摆几块 Panel;屏卡片本身
改成透明无框(`flatten_card`),描边由各 Panel 自己画,面板之间留 GAP。标签是叠层:不进布局、
不吃鼠标,所以不占图的面积,也不会因文字长短把窗口撑宽(宽度封顶在面板宽度以内)。
"""

from PySide6.QtCore import QEvent, Qt
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget

from .. import theme

_PAD = 8
_CHIP_STYLE = (
    "background: rgba(12, 14, 18, 190); color: {text}; border: 1px solid {line};"
    "border-radius: 5px; padding: 1px 7px; font-size: {size}px;"
)


class Panel(QFrame):
    """PLOT 底、LINE_SOFT 1px 描边、8 圆角的面板;`child` 铺满,左上 / 右上可压标签。"""

    def __init__(self, child: QWidget | None = None, name: str = "Panel") -> None:
        super().__init__()
        self.setObjectName(name)
        self.setStyleSheet(
            f"#{name} {{ background: {theme.PLOT}; border: 1px solid {theme.LINE_SOFT}; "
            f"border-radius: {theme.RADIUS_CARD}px; }}"
        )
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(1, 1, 1, 1)
        self.body.setSpacing(0)
        if child is not None:
            self.body.addWidget(child)
        style = _CHIP_STYLE.format(
            text=theme.TEXT, line=theme.LINE, size=theme.FONT_NOTE
        )
        self.chips: dict[str, QLabel] = {}
        for side in ("left", "right"):
            chip = QLabel(self)
            chip.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            chip.setStyleSheet(style)
            chip.hide()
            self.chips[side] = chip

    def set_chip(self, side: str, text: str) -> None:
        """设左上(`left`)/ 右上(`right`)标签;空串即隐藏。文字可带富文本着色。"""
        chip = self.chips[side]
        if chip.text() != text:
            chip.setText(text)
        chip.setVisible(bool(text))
        self._place()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._place()

    def event(self, event: QEvent) -> bool:
        done = super().event(event)
        if event.type() == QEvent.Type.Show:
            self._place()
        return done

    def _place(self) -> None:
        room = max(self.width() - 2 * _PAD, 40)
        left, right = self.chips["left"], self.chips["right"]
        for chip in (left, right):
            chip.setMaximumWidth(room)
            chip.adjustSize()
        if not left.isHidden() and not right.isHidden():
            # 两块放不下时右边的让出一部分(图例比标题次要)
            over = left.width() + right.width() + _PAD - room
            if over > 0:
                right.setMaximumWidth(max(right.width() - over, 60))
                right.adjustSize()
        left.move(_PAD, 6)
        right.move(max(self.width() - _PAD - right.width(), _PAD), 6)
        for chip in (left, right):
            if not chip.isHidden():
                chip.raise_()


def flatten_card(screen) -> None:
    """让屏的主图卡片透明无框:复合主视图里的 Panel 各自带描边,否则外面多套一层框。"""
    screen.card.setStyleSheet("#Card { background: transparent; border: none; }")
