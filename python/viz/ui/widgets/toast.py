"""浮在画布上的提示条:取代状态栏,出现在顶栏下方居中。

状态栏永远占一行、每帧重写一句与别处重复的话;提示条只在有事件时出现(落盘、改参、失败),
几秒后自己消失。它**不在布局里**(绝对定位的子控件),显隐不会让画布重排;位置在顶栏
正下方,压在屏的上下文条上、不压列表与主图的数据区。
`showMessage / currentMessage / clearMessage` 与 QStatusBar 同名,调用方不用改。
"""

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QLabel, QWidget

from .. import theme

DEFAULT_TIMEOUT_MS = 8000


class Toast(QLabel):
    """top = 离父控件顶边的距离(窗口按顶栏的高度更新);follow 非空时叠在它下面。"""

    def __init__(self, parent: QWidget, top: int = 110, follow: "Toast | None" = None):
        super().__init__(parent)
        self.top = top
        self.follow = follow
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.clearMessage)
        self._color = theme.TEXT
        self._apply_style()
        self.hide()

    def _apply_style(self) -> None:
        self.setStyleSheet(
            f"background: {theme.RAISED}; color: {self._color}; "
            f"border: 1px solid {theme.LINE_HOVER}; border-radius: {theme.RADIUS_CONTROL}px; "
            f"padding: 6px 14px; font-size: {theme.FONT_NOTE}px;"
        )

    def set_color(self, color: str) -> None:
        self._color = color
        self._apply_style()

    def showMessage(self, text: str, timeout: int = DEFAULT_TIMEOUT_MS) -> None:
        self.setText(text)
        self._timer.stop()
        if not text:
            self.hide()
            return
        self.adjustSize()
        self.reposition()
        self.show()
        self.raise_()
        if timeout > 0:
            self._timer.start(timeout)

    def currentMessage(self) -> str:
        return self.text()

    def clearMessage(self) -> None:
        self._timer.stop()
        self.setText("")
        self.hide()

    def reposition(self) -> None:
        parent = self.parentWidget()
        if parent is None:
            return
        self.setMaximumWidth(max(parent.width() - 64, 100))
        self.adjustSize()
        y = self.top
        if self.follow is not None and self.follow.isVisible():
            y = self.follow.y() + self.follow.height() + 6
        self.move((parent.width() - self.width()) // 2, y)
