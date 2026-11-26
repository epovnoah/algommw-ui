"""命令面板(Ctrl/⌘+K):输入几个字就能切屏、跳帧、跳异常帧、调速度、开面板页。

命令由窗口按当前状态现建(`provider(query)` 返回 Command 列表):静态命令按子串过滤,
输入纯数字时额外给出「跳到帧 N」。每条命令把自己的快捷键写在右边,边用边学。
↑ ↓ 选择、Enter 执行、Esc 或点外面关闭。
"""

from collections.abc import Callable
from dataclasses import dataclass

from PySide6.QtCore import QEvent, Qt
from PySide6.QtWidgets import (
    QDialog,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .. import theme


@dataclass(frozen=True)
class Command:
    title: str
    run: Callable[[], None]
    hint: str = ""  # 快捷键 / 补充说明,显示在右侧
    keywords: str = ""  # 只参与匹配的别名


def filter_commands(commands: list[Command], query: str) -> list[Command]:
    """所有空白分隔的词都出现在 标题 + 别名 里(忽略大小写)才保留,保持原顺序。"""
    words = query.lower().split()
    if not words:
        return commands
    return [
        c
        for c in commands
        if all(w in f"{c.title} {c.keywords}".lower() for w in words)
    ]


class CommandPalette(QDialog):
    WIDTH = 560

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent, Qt.WindowType.Popup)
        self.setObjectName("Palette")
        self.setFixedWidth(self.WIDTH)
        self.setStyleSheet(
            f"#Palette {{ background: {theme.SURFACE}; border: 1px solid {theme.LINE_HOVER}; "
            f"border-radius: {theme.RADIUS_CARD}px; }}"
            f"QLineEdit {{ background: {theme.GROUND}; color: {theme.TEXT}; "
            f"border: 1px solid {theme.LINE}; border-radius: {theme.RADIUS_CONTROL}px; "
            f"padding: 7px 10px; font-size: {theme.FONT_BODY}px; }}"
            f"QListWidget {{ background: transparent; border: 0; outline: 0; }}"
            f"QListWidget::item {{ padding: 8px 10px; border-radius: 8px; }}"
            f"QListWidget::item:selected {{ background: {theme.ACCENT_BG}; "
            f"color: {theme.TEXT}; }}"
        )
        self.provider: Callable[[str], list[Command]] = lambda _q: []
        self._shown: list[Command] = []
        self.edit = QLineEdit()
        self.edit.setPlaceholderText("输入命令、屏名或帧号…")
        self.edit.installEventFilter(self)
        self.edit.textChanged.connect(self._refill)
        self.list = QListWidget()
        self.list.setFixedHeight(300)
        self.list.itemActivated.connect(lambda _item: self._run())
        self.list.itemClicked.connect(lambda _item: self._run())
        column = QVBoxLayout(self)
        column.setContentsMargins(10, 10, 10, 10)
        column.setSpacing(8)
        column.addWidget(self.edit)
        column.addWidget(self.list)

    def open_palette(self, provider: Callable[[str], list[Command]]) -> None:
        self.provider = provider
        self.edit.clear()
        self._refill("")
        parent = self.parentWidget()
        if parent is not None:
            top_left = parent.mapToGlobal(parent.rect().topLeft())
            self.move(
                top_left.x() + (parent.width() - self.WIDTH) // 2, top_left.y() + 70
            )
        self.show()
        self.edit.setFocus()

    def _refill(self, query: str) -> None:
        self._shown = self.provider(query)
        self.list.clear()
        for command in self._shown:
            item = QListWidgetItem(
                f"{command.title}    ·    {command.hint}"
                if command.hint
                else command.title
            )
            self.list.addItem(item)
        if self._shown:
            self.list.setCurrentRow(0)

    def _run(self) -> None:
        row = self.list.currentRow()
        if 0 <= row < len(self._shown):
            command = self._shown[row]
            self.close()
            command.run()

    def eventFilter(self, obj, event) -> bool:
        if obj is self.edit and event.type() == QEvent.Type.KeyPress:
            key = event.key()
            if key in (Qt.Key.Key_Down, Qt.Key.Key_Up):
                step = 1 if key == Qt.Key.Key_Down else -1
                row = min(max(self.list.currentRow() + step, 0), self.list.count() - 1)
                self.list.setCurrentRow(row)
                return True
            if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                self._run()
                return True
        return super().eventFilter(obj, event)
