"""场景屏画布上的悬浮件:工具条、「世界 ▾」面板、速度色标、选中卡(三维点云与二维投影共用)。

画布铺满主区,这些部件作为画布(`PointCloudView` / `ProjectionView`)的子部件叠在上面、不进布局:
- `HudBar`:左上角的半透明悬浮工具条,里面是分段控件与「世界 ▾」按钮;
- `WorldButton`:点开一张小面板,放世界坐标的 7 个控件(控件对象与行为不变,只是挪了位置);
- `RampLegend`:速度色标的渐变条(左下图例的一部分,由屏的叠层条承载);
- `SelectionCard`:右上角的选中检测卡 —— `#序号 · 距离 · 速度` + 距离像小曲线 + 「看它的来路 →」;
- `place_hud`:按画布尺寸摆放工具条、选中卡与说明小标签。
GL 画布上叠 QWidget 在 cocoa 与 offscreen 下都能显示、点击(合成在窗口后备存储里),见 guiprobe 的
gl_point_cloud。
"""

import numpy as np
from PySide6.QtCore import QEvent, QPoint, Qt, Signal
from PySide6.QtGui import QColor, QLinearGradient, QPainter
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .. import theme
from ..state import ViewState
from ..views.common import ClipLabel
from ..widgets.seg import Segmented
from .mini import RangeLocal

HUD_PAD = 8  # 悬浮件离画布边的距离
HUD_GAP = 6  # 工具条内部、工具条与说明标签之间的间距
HUD_FILL = "rgba(12, 14, 18, 200)"  # 悬浮件的半透明深底
CARD_WIDTH = 240
SPARK_HEIGHT = 38


def chip_style(name: str, radius: int = 5) -> str:
    """叠层小标签 / 悬浮卡的统一底:半透明深底、LINE 描边。"""
    return (
        f"#{name} {{ background: {HUD_FILL}; border: 1px solid {theme.LINE}; "
        f"border-radius: {radius}px; }}"
    )


def translucent(seg: Segmented) -> Segmented:
    """把分段控件的实心底换成悬浮件的半透明深底(其余样式不动)。"""
    seg.setStyleSheet(seg.styleSheet().replace(theme.SURFACE, HUD_FILL))
    return seg


def lut_color(lut: np.ndarray, t: float) -> QColor:
    """色表(n×4,浮点 0~1 或字节 0~255)在 t∈[0,1] 处的颜色。"""
    row = np.asarray(lut)[int(round(t * (len(lut) - 1)))][:3].astype(float)
    if lut.dtype.kind == "f":
        row = row * 255
    return QColor(*(int(round(v)) for v in row))


class HudBar(QWidget):
    """左上角悬浮工具条:一行分段控件 / 按钮,26 px 高;内容显隐变化时发 `content_changed`。"""

    content_changed = Signal()

    def __init__(self, widgets: list[QWidget], parent: QWidget) -> None:
        super().__init__(parent)
        self.setObjectName("HudBar")
        self.setFixedHeight(theme.MIN_HIT)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(HUD_GAP)
        for widget in widgets:
            if isinstance(widget, Segmented):
                translucent(widget)
            row.addWidget(widget)

    def event(self, event: QEvent) -> bool:
        done = super().event(event)
        if event.type() == QEvent.Type.LayoutRequest:
            self.content_changed.emit()
        return done


class WorldButton(QPushButton):
    """「世界 ▾」:点开小面板,里面是世界坐标的全部控件(跟随 / 范围 / 点集 / 高度 / 配准 / L / 视轴)。
    面板是 Popup 窗口,点它外面自动收起;控件挂在面板里,对象与信号不变。"""

    def __init__(self, controls: list[QWidget], parent: QWidget | None = None) -> None:
        super().__init__(f"世界 ▾  {len(controls)} 项", parent)
        self.setObjectName("WorldButton")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(theme.MIN_HIT)
        self.setStyleSheet(
            f"#WorldButton {{ background: {HUD_FILL}; border: 1px solid {theme.LINE}; "
            f"border-radius: {theme.RADIUS_CONTROL}px; color: {theme.TEXT}; padding: 0 10px; "
            f"font-size: {theme.FONT_BODY}px; }}"
            f"#WorldButton:hover {{ border-color: {theme.LINE_HOVER}; }}"
        )
        self.panel = QFrame(self, Qt.WindowType.Popup)
        self.panel.setObjectName("WorldPanel")
        self.panel.setStyleSheet(
            f"#WorldPanel {{ background: {theme.SURFACE}; border: 1px solid {theme.LINE}; "
            f"border-radius: {theme.RADIUS_CARD}px; }}"
        )
        grid = QGridLayout(self.panel)
        grid.setContentsMargins(8, 8, 8, 8)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)
        segs = [c for c in controls if isinstance(c, Segmented)]
        spins = [c for c in controls if not isinstance(c, Segmented)]
        for row, seg in enumerate(segs):
            grid.addWidget(seg, row, 0, 1, 2)
        for column, spin in enumerate(spins):
            grid.addWidget(spin, len(segs), column)
        self.clicked.connect(self.toggle_panel)

    def toggle_panel(self) -> None:
        if self.panel.isVisible():
            self.panel.hide()
            return
        self.panel.adjustSize()
        self.panel.move(self.mapToGlobal(QPoint(0, self.height() + 4)))
        self.panel.show()


class RampBar(QWidget):
    """速度色标的渐变条:接近(蓝)→ 静止(黄)→ 远离(红),取点云用的同一张色表。"""

    def __init__(self, lut: np.ndarray) -> None:
        super().__init__()
        self.lut = lut
        self.setFixedSize(96, 8)

    def paintEvent(self, _event) -> None:
        gradient = QLinearGradient(0, 0, self.width(), 0)
        for t in np.linspace(0.0, 1.0, 9):
            gradient.setColorAt(float(t), lut_color(self.lut, float(t)))
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(gradient)
        painter.drawRoundedRect(self.rect(), 3, 3)


class RampLegend(QWidget):
    """左下图例里的速度色标左半:「径向速度 −5(接近)」+ 渐变条;右半文字由视图的 `legend` 标签接着写。"""

    def __init__(self, lut: np.ndarray) -> None:
        super().__init__()
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        self.label = QLabel()
        self.label.setStyleSheet(f"color: {theme.DIM}; font-size: {theme.FONT_NOTE}px;")
        row.addWidget(self.label)
        row.addWidget(RampBar(lut), 0, Qt.AlignmentFlag.AlignVCenter)

    def set_limit(self, limit_mps: float) -> None:
        self.label.setText(f"径向速度 −{limit_mps:g}(接近)")


class SelectionCard(QFrame):
    """右上角悬浮的选中检测卡,宽 240:有选中检测时显示,没有时隐藏。点「看它的来路 →」
    = 打开检查器的溯源页(检查器在场景屏默认收起,见 window 的 `_sync_inspector`)。"""

    changed = Signal()  # 显隐 / 尺寸变了,画布据此重新摆位

    def __init__(self, state: ViewState, parent: QWidget) -> None:
        super().__init__(parent)
        self.state = state
        self.setObjectName("SelectionCard")
        self.setStyleSheet(chip_style("SelectionCard", theme.RADIUS_CARD))
        self.setFixedWidth(CARD_WIDTH)
        column = QVBoxLayout(self)
        column.setContentsMargins(10, 8, 10, 8)
        column.setSpacing(2)
        self.title = ClipLabel()
        self.title.setStyleSheet(
            f"color: {theme.ACCENT}; font-family: {theme.MONO}; font-size: {theme.FONT_NOTE}px;"
        )
        self.note = QLabel("选中检测")
        self.note.setStyleSheet(f"color: {theme.DIM}; font-size: 11px;")
        self.spark = RangeLocal(SPARK_HEIGHT)
        self.spark.set_enlarged(False)  # 安静档:不画标题,只留一条走势线
        self.spark.widget.setFixedHeight(SPARK_HEIGHT)
        self.spark.plot.hideAxis("left")
        self.spark.plot.hideAxis("bottom")
        self.button = QPushButton("看它的来路 →")
        self.button.setProperty("role", "trace")
        self.button.setFixedHeight(22)
        self.button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.button.clicked.connect(self._open_trace)
        for widget in (self.title, self.note, self.spark):
            column.addWidget(widget)
        column.addSpacing(4)
        column.addWidget(self.button)
        self.hide()

    def _open_trace(self) -> None:
        from ..drawer import Drawer

        self.state.request_panel(Drawer.TRACE)

    def show_selection(self, frame, k: int | None) -> None:
        """第 k 个检测的卡;k 为 None(选中的是没有检测的单元)或越界时隐藏。"""
        det = frame.detections
        if k is None or not 0 <= k < len(det):
            if not self.isHidden():
                self.hide()
                self.changed.emit()
            return
        self.title.setText(
            f"#{k} · {det.range_m[k]:.2f} m · {det.velocity_mps[k]:+.2f} m/s"
        )
        self.spark.update_detection(frame, k)
        if self.isHidden():
            self.adjustSize()
            self.show()
            self.changed.emit()
        self.raise_()


def place_hud(
    view: QWidget, bar: HudBar, card: SelectionCard, notice: QWidget | None = None
) -> None:
    """工具条贴左上、选中卡贴右上、说明小标签在工具条下面;工具条宽度让给选中卡。"""
    room = view.width() - 2 * HUD_PAD
    if card.isVisibleTo(view):
        room -= card.width() + HUD_GAP
    bar.setFixedWidth(max(min(bar.sizeHint().width(), room), 0))
    bar.move(HUD_PAD, HUD_PAD)
    bar.raise_()
    card.adjustSize()
    card.move(view.width() - card.width() - HUD_PAD, HUD_PAD)
    card.raise_()
    if notice is not None:
        notice.move(HUD_PAD, HUD_PAD + theme.MIN_HIT + HUD_GAP)
        notice.adjustSize()
        notice.raise_()
