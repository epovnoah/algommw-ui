"""传输栏用的图标:QPainter 现画,不带图片资源(深色主题,单色)。

图标只在建窗时画一次(带 devicePixelRatio,Retina 不糊),之后按状态换 QIcon,不逐帧重画。
"""

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QApplication

SIZE = 16


def _draw(paint, color: str) -> QIcon:
    screen = QApplication.primaryScreen()
    ratio = screen.devicePixelRatio() if screen is not None else 1.0
    pixmap = QPixmap(int(SIZE * ratio), int(SIZE * ratio))
    pixmap.setDevicePixelRatio(ratio)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(QColor(color), 1.6)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(QColor(color))
    paint(painter)
    painter.end()
    return QIcon(pixmap)


def _triangle(p: QPainter, pts: list[tuple[float, float]]) -> None:
    path = QPainterPath(QPointF(*pts[0]))
    for x, y in pts[1:]:
        path.lineTo(x, y)
    path.closeSubpath()
    p.drawPath(path)


def play(color: str) -> QIcon:
    return _draw(lambda p: _triangle(p, [(4.5, 2.5), (13, 8), (4.5, 13.5)]), color)


def pause(color: str) -> QIcon:
    def paint(p: QPainter) -> None:
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(QRectF(3.5, 2.5, 3.2, 11), 0.8, 0.8)
        p.drawRoundedRect(QRectF(9.3, 2.5, 3.2, 11), 0.8, 0.8)

    return _draw(paint, color)


def _chevron(p: QPainter, forward: bool) -> None:
    p.setBrush(Qt.BrushStyle.NoBrush)
    x0, x1 = (4.5, 10.5) if forward else (10.5, 4.5)
    p.drawPolyline([QPointF(x0, 3), QPointF(x1, 8), QPointF(x0, 13)])


def step_back(color: str) -> QIcon:
    return _draw(lambda p: _chevron(p, False), color)


def step_forward(color: str) -> QIcon:
    return _draw(lambda p: _chevron(p, True), color)
