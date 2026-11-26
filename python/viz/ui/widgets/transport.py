"""传输条(窗口底部,总高 34 px):播放 / 暂停圆钮 · 上一帧 / 下一帧 · 帧计数 · 瀑布时间轴 · 倍速。

位置固定、与屏无关。动作(播放 / 单步)由窗口建好传进来 —— 快捷键挂在窗口上,这里只负责
把它们摆成按钮。「跳到上 / 下一个异常帧」与「落盘」不占底栏:前者是快捷键 [ ] 与命令面板项,
后者在命令面板与滑出栏的「参数」页。倍速是一个按钮,点开列出全部档位。
"""

from collections.abc import Mapping

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QAction, QColor, QPainter
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QToolButton,
    QWidget,
)

from .. import theme
from . import icons
from .timeline import HEIGHT as TIMELINE_HEIGHT
from .timeline import Timeline

PLAY_SIZE = 26
STEP_SIZE = 22
HEIGHT = 34  # 边距 4 + 播放钮 26 + 边距 4;顶上那条 1 px 分隔线画在上边距里
#: 倍速档的按钮短名(其余原样)
SHORT_SPEED = {"最快(逐帧)": "最快", "1× 实时": "1×"}


class TransportBar(QWidget):
    speed_chosen = Signal(str)  # SPEEDS 的键

    def __init__(
        self,
        actions: Mapping[str, QAction],
        speeds: list[str],
        default_speed: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("Transport")
        self.setFixedHeight(HEIGHT)
        self.setStyleSheet("QToolButton { border: 0; }")
        row = QHBoxLayout(self)
        row.setContentsMargins(theme.GUTTER, 4, theme.GUTTER, 4)
        row.setSpacing(8)

        step_back = self._icon_button(actions["step_back"])
        row.addWidget(step_back)
        play = self._icon_button(actions["play"], PLAY_SIZE)
        play.setIconSize(QSize(12, 12))
        play.setStyleSheet(
            f"QToolButton {{ background: {theme.TEXT}; border-radius: {PLAY_SIZE // 2}px; }}"
            f"QToolButton:disabled {{ background: {theme.LINE}; }}"
        )
        row.addWidget(play)
        row.addWidget(self._icon_button(actions["step"]))

        self.frame_label = QLabel("0 / 0")
        self.frame_label.setMinimumWidth(74)
        self.frame_label.setStyleSheet(
            f"color: {theme.TEXT}; font-family: {theme.MONO}; font-size: {theme.FONT_TICK}px;"
        )
        row.addWidget(self.frame_label)

        self.timeline = Timeline()
        self.timeline.setFixedHeight(TIMELINE_HEIGHT)
        self.timeline.setToolTip(
            "距离随时间的瀑布图,同时是时间轴(拖动 / 点击跳帧)。\n"
            "只含已处理过的帧,列随播放与跳转逐步长出来。\n"
            "上沿色条:异常帧(有「异常」级检测 = 红;需要看的检测数在已处理帧的前 10% = 琥珀)。\n"
            "上一个 / 下一个异常帧:[ 与 ]。"
        )
        row.addWidget(self.timeline, 1)

        self._speeds = list(speeds)
        self.speed_button = QPushButton()
        self.speed_button.setProperty("role", "ghost")
        self.speed_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.speed_button.setMinimumWidth(56)
        self.speed_button.setFixedHeight(STEP_SIZE)
        self.speed_button.setToolTip("播放速度")
        self.speed_button.setStyleSheet(
            f"min-height: {STEP_SIZE}px; padding: 0 8px; "
            f"font-family: {theme.MONO}; font-size: {theme.FONT_NOTE}px;"
        )
        self.speed_button.clicked.connect(self._pick_speed)
        row.addWidget(self.speed_button)
        self._speed = default_speed
        self.set_speed(default_speed)

    @staticmethod
    def _icon_button(action: QAction, size: int = STEP_SIZE) -> QToolButton:
        button = QToolButton()
        button.setDefaultAction(action)
        button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        button.setIconSize(QSize(14, 14))
        button.setFixedSize(size, size)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        return button

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(0, 0, self.width(), 1, QColor(theme.LINE_SOFT))

    def _pick_speed(self) -> None:
        menu = QMenu(self)
        for key in self._speeds:
            action = menu.addAction(key)
            action.setCheckable(True)
            action.setChecked(key == self._speed)
            action.triggered.connect(lambda _=False, k=key: self.speed_chosen.emit(k))
        pos = self.speed_button.mapToGlobal(self.speed_button.rect().topLeft())
        pos.setY(pos.y() - menu.sizeHint().height())  # 弹在按钮上方(按钮贴着窗口底)
        menu.exec(pos)

    def set_speed(self, key: str) -> None:
        """只改显示,不发信号。"""
        self._speed = key
        self.speed_button.setText(SHORT_SPEED.get(key, key))

    def speed_text(self) -> str:
        return self.speed_button.text()


def play_icons() -> tuple:
    """(播放图标, 暂停图标),深底上的字色,给窗口随播放状态切换(圆钮是浅色底)。"""
    color = theme.GROUND
    return icons.play(color), icons.pause(color)
