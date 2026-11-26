"""检查器容器(右侧可停靠栏):默认展开,可收起 / 再滑出,把主区挤窄(不是浮层遮挡)。

内容固定宽 `theme.SIDE_WIDTH`(300),容器的宽度在 0 与它之间做动画,所以滑动期间内容不会重排。
关闭时内容真的 hide,里面的视图因此不渲染(ResultView 只在可见时渲染),不占回放帧时间。
开合状态由窗口记进 data/ui_state.json;无记录时按展开。类名沿用「滑出栏」。
"""

from PySide6.QtCore import Property, QEasingCurve, QPropertyAnimation, Signal
from PySide6.QtWidgets import QHBoxLayout, QWidget

from .. import theme

DURATION_MS = 180


class SlideOut(QWidget):
    opened_changed = Signal(bool)

    def __init__(
        self, content: QWidget, parent: QWidget | None = None, open_: bool = True
    ) -> None:
        super().__init__(parent)
        self.setObjectName("SlideOut")
        self.setStyleSheet(f"#SlideOut {{ border-left: 1px solid {theme.LINE_SOFT}; }}")
        self.content = content
        content.setFixedWidth(theme.SIDE_WIDTH)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(content)
        self._open = open_
        self._width = theme.SIDE_WIDTH + 1 if open_ else 0  # +1:左边线
        self.setFixedWidth(self._width)
        content.setVisible(open_)
        self._anim = QPropertyAnimation(self, b"panelWidth", self)
        self._anim.setDuration(DURATION_MS)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._anim.finished.connect(self._on_finished)

    def _get_width(self) -> int:
        return self._width

    def _set_width(self, width: int) -> None:
        self._width = int(width)
        self.setFixedWidth(self._width)

    panelWidth = Property(int, _get_width, _set_width)

    def is_open(self) -> bool:
        """目标状态(动画途中也是最终要到的状态)。"""
        return self._open

    def set_open(self, open_: bool, animate: bool = True) -> None:
        if open_ == self._open and (not animate or self._anim.state() == 0):
            return
        self._open = open_
        self._anim.stop()
        target = theme.SIDE_WIDTH + 1 if open_ else 0  # +1:左边线
        if open_:
            self.content.show()
        if animate:
            self._anim.setStartValue(self._width)
            self._anim.setEndValue(target)
            self._anim.start()
        else:
            self._set_width(target)
            self._on_finished()
        self.opened_changed.emit(open_)

    def _on_finished(self) -> None:
        if not self._open:
            self.content.hide()
