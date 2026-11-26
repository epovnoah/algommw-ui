"""分段控件:一组互斥的文字按钮(设计稿的 `.seg`),承载一屏的模式 / 叠加选项。

接口仿 QComboBox 的子集(currentIndex / setCurrentIndex / currentText / count / currentIndexChanged),
视图原来用下拉框的地方换成它不用改逻辑。选项少(2~5 个)才用它;选项多的放命令面板。
"""

from collections.abc import Sequence

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QButtonGroup, QFrame, QHBoxLayout, QPushButton, QWidget

from .. import theme

HEIGHT = theme.MIN_HIT


class Segmented(QFrame):
    currentIndexChanged = Signal(int)

    def __init__(
        self, items: Sequence[str], tip: str = "", parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.setObjectName("Seg")
        self.setFixedHeight(HEIGHT)
        self.setStyleSheet(
            f"#Seg {{ background: {theme.SURFACE}; border: 1px solid {theme.LINE}; "
            f"border-radius: {theme.RADIUS_CONTROL}px; }}"
            f"#Seg QPushButton {{ min-height: {theme.MIN_HIT - 6}px; padding: 0 10px; "
            f"background: transparent; border: 0; border-radius: 5px; color: {theme.DIM}; "
            f"font-size: {theme.FONT_BODY}px; }}"
            f"#Seg QPushButton:hover {{ color: {theme.TEXT}; }}"
            f"#Seg QPushButton:checked {{ background: {theme.LINE}; color: {theme.TEXT}; }}"
        )
        if tip:
            self.setToolTip(tip)
        self._row = QHBoxLayout(self)
        self._row.setContentsMargins(2, 2, 2, 2)
        self._row.setSpacing(0)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._buttons: list[QPushButton] = []
        self._index = -1
        self.set_items(items)

    def set_items(self, items: Sequence[str]) -> None:
        """换一组选项(通道数随板变);不发信号,当前项保持在范围内。"""
        for button in self._buttons:
            self._group.removeButton(button)
            self._row.removeWidget(button)
            button.deleteLater()
        self._buttons = []
        for i, text in enumerate(items):
            button = QPushButton(text)
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(lambda _=False, k=i: self.setCurrentIndex(k))
            self._group.addButton(button)
            self._row.addWidget(button)
            self._buttons.append(button)
        self._index = min(max(self._index, 0), len(self._buttons) - 1)
        if self._buttons:
            self._buttons[self._index].setChecked(True)

    def count(self) -> int:
        return len(self._buttons)

    def currentIndex(self) -> int:
        return self._index

    def currentText(self) -> str:
        return self._buttons[self._index].text() if self._buttons else ""

    def setCurrentIndex(self, index: int) -> None:
        index = min(max(int(index), 0), len(self._buttons) - 1)
        if index < 0:
            return
        self._buttons[index].setChecked(True)  # 点同一项时按钮会被点灭:这里恢复
        if index != self._index:
            self._index = index
            self.currentIndexChanged.emit(index)

    def set_enabled_item(self, index: int, enabled: bool, tip: str = "") -> None:
        self._buttons[index].setEnabled(enabled)
        self._buttons[index].setToolTip(tip)
