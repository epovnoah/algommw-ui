"""瀑布时间轴:底部一条 22 px 的控件,画整段采集的「距离 × 时间」瀑布图 —— 拖它就是在拖数据。

一帧一列、纵轴距离(近的在下),随 `add_frames` 到来的帧逐列填入;还没处理的帧留底色。
每列是 `diagnostics.slices.waterfall_column`(每个有效距离门取多普勒向最大值,下采样到 64 行),
色表同热图(inferno)。**色阶只在已填列数首次达 16 列、以及之后每翻一倍时重算**(取已填列的
稳健分位):逐帧重算会让整条图的颜色不停闪。当前帧是一条 2 px 浅色竖线;图上沿一行 2 px
高的色条标「异常帧」(有「异常」级检测:红;需要看的检测数高于已处理帧的前 10%:琥珀)。

它是 QAbstractSlider,窗口里仍按滑块用(setRange / setValue / valueChanged / blockSignals),
拖动或点图即跳帧。概览只含**已处理过**的帧:后台按需处理、不预扫整段采集,所以列随播放与
跳转逐步长出来(未处理的帧是空白,悬停提示「未处理」)。

绘制:整段数据烘成一张 QImage(帧数 × 64,一次贴图,不是上千个图元);数据变化后最多每
200 ms 重烘一次,播放时每帧只重画那一条播放头。
"""

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QPointF, QRect, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QAbstractSlider, QSizePolicy, QToolTip, QWidget

from ...diagnostics import STAGES, Level, display_levels_db, processing_ms
from ...diagnostics.slices import WATERFALL_ROWS, waterfall_column
from .. import theme

HEIGHT = 22
STRIP_HEIGHT = 2
_BAKE_MS = 200
_RADIUS = 5
#: 色阶首次计算的列数下限:不足时先用已有的列,满了这个数再更新一次,之后每翻倍更新
LEVELS_MIN_COLUMNS = 16
#: 未处理的帧(底色)
EMPTY_COLOR = "#12081F"


class Timeline(QAbstractSlider):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setOrientation(Qt.Orientation.Horizontal)
        self.setFixedHeight(HEIGHT)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._lut = pg.colormap.get(theme.POWER_CMAP).getLookupTable(
            0.0, 1.0, 256, alpha=False
        )
        self._alloc(0)
        self._image: QImage | None = None
        self._strip: QImage | None = None
        self._buffers: tuple = ()  # QImage 引用的 numpy 缓冲,须与图同寿
        self._dirty = True
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(_BAKE_MS)
        self._timer.timeout.connect(self.update)
        self.rangeChanged.connect(lambda _lo, hi: self._alloc(hi + 1))
        self._head = 0.0
        self.valueChanged.connect(self._on_value)

    # ---- 数据 ----
    def _alloc(self, n: int) -> None:
        if n == getattr(self, "_n", -1):
            return
        self._n = n
        self._cols = np.full((WATERFALL_ROWS, n), np.nan, np.float32)
        self._ms = np.full(n, np.nan)
        self._flagged = np.zeros(n, np.int32)
        self._bad = np.zeros(n, bool)
        self._known = np.zeros(n, bool)
        self._levels: tuple[float, float] | None = None
        self._levels_columns = 0  # 上次算色阶时已填的列数
        self._dirty = True
        self.update()

    def clear(self) -> None:
        """清空概览(范围与当前值不动):改参后缓存作废,旧列不能和新结果混在一起。"""
        self._n = -1  # 同样长度也要真清空
        self._alloc(max(self.maximum() - self.minimum() + 1, 1))

    def reset(self, n_frames: int) -> None:
        """换会话:范围改成 0 … n_frames-1、值回到 0、概览清空。不发 valueChanged
        (窗口的跳帧槽挂在它上面,换会话时不该触发一次跳转)。"""
        self.blockSignals(True)
        self.setRange(0, max(n_frames - 1, 0))
        self.setValue(0)
        self.blockSignals(False)
        self.clear()

    def add_frames(self, frames) -> None:
        """记下新处理出的帧(顺序往前的一批);只改数组,重烘由定时器合并。"""
        for frame in frames:
            k = int(frame.index)
            if not 0 <= k < self._n:
                continue
            self._known[k] = True
            self._ms[k] = processing_ms(frame.timings_ms)
            column = waterfall_column(frame)
            if column is not None:
                self._cols[:, k] = column
            diag = frame.diagnostics
            if diag is not None:
                self._flagged[k] = sum(diag.summary.flagged.values())
                self._bad[k] = any(
                    bool((diag.verdicts.levels[s] >= Level.BAD).any()) for s in STAGES
                )
        self._dirty = True
        if not self._timer.isActive():
            self._timer.start()

    def filled_columns(self) -> int:
        return int(np.isfinite(self._cols[0]).sum()) if self._n else 0

    def levels(self) -> tuple[float, float] | None:
        """当前色阶;只在首次有列、首次满 LEVELS_MIN_COLUMNS 列、之后每翻倍时更新。"""
        count = self.filled_columns()
        if count == 0:
            return self._levels
        last = self._levels_columns
        if (
            self._levels is None
            or (count >= LEVELS_MIN_COLUMNS > last)
            or (last >= LEVELS_MIN_COLUMNS and count >= 2 * last)
        ):
            values = self._cols[:, np.isfinite(self._cols[0])]
            self._levels = display_levels_db(values, ceiling_db=None)
            self._levels_columns = count
        return self._levels

    def anomalous(self) -> np.ndarray:
        """异常帧掩码(口径见模块文档)。"""
        mask = self._bad & self._known
        counts = self._flagged[self._known]
        if counts.size:
            mask |= self._known & (self._flagged > np.percentile(counts, 90))
        return mask

    def step_anomaly(self, direction: int) -> int | None:
        """当前帧之后(direction=1)或之前(-1)最近的异常帧号;没有则 None。"""
        hits = np.flatnonzero(self.anomalous())
        if direction > 0:
            hits = hits[hits > self.value()]
            return int(hits[0]) if hits.size else None
        hits = hits[hits < self.value()]
        return int(hits[-1]) if hits.size else None

    def _on_value(self, value: int) -> None:
        """播放头移动只重画旧、新两条窄带,不重画整条图。"""
        old, self._head = self._head, self._x_of(value)
        for x in (old, self._head):
            self.update(QRect(int(x) - 4, 0, 8, self.height()))

    def refresh_head(self) -> None:
        """窗口在 blockSignals 下设值后调:信号被挡住时 valueChanged 不会触发上面的重画。"""
        self._on_value(self.value())

    # ---- 几何 ----
    def _frame_at(self, x: float) -> int:
        n = max(self.maximum() - self.minimum() + 1, 1)
        return self.minimum() + min(max(int(x / max(self.width(), 1) * n), 0), n - 1)

    def _x_of(self, value: int) -> float:
        n = max(self.maximum() - self.minimum() + 1, 1)
        return (value - self.minimum() + 0.5) / n * self.width()

    # ---- 鼠标 ----
    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.setSliderDown(True)
            self.setValue(self._frame_at(event.position().x()))
            event.accept()

    def mouseMoveEvent(self, event) -> None:
        k = self._frame_at(event.position().x())
        if self.isSliderDown():
            self.setValue(k)
        elif 0 <= k < self._n:
            if self._known[k]:
                text = (
                    f"帧 {k + 1} · {self._ms[k]:.1f} ms · "
                    f"{self._flagged[k]} 个需要看"
                    + (" · 有异常级" if self._bad[k] else "")
                )
            else:
                text = f"帧 {k + 1} · 未处理"
            QToolTip.showText(event.globalPosition().toPoint(), text, self)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.setSliderDown(False)
            event.accept()

    # ---- 绘制 ----
    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._head = self._x_of(self.value())

    def _bake(self) -> None:
        """瀑布图与异常色条各烘成一张 QImage(帧数 × 行数 / 帧数 × 1)。"""
        n = max(self._n, 1)
        rgb = np.empty((WATERFALL_ROWS, n, 4), np.uint8)
        empty = QColor(EMPTY_COLOR)
        rgb[...] = (empty.blue(), empty.green(), empty.red(), 255)
        levels = self.levels()
        known = np.isfinite(self._cols)
        if levels is not None and known.any():
            low, high = levels
            index = np.clip((self._cols - low) / max(high - low, 1e-6) * 255, 0, 255)
            index = np.nan_to_num(index).astype(np.uint8)
            colors = self._lut[index]  # (行, 列, 3) RGB
            bgr = colors[..., ::-1]
            rgb[..., :3] = np.where(known[..., None], bgr, rgb[..., :3])
        wf = np.ascontiguousarray(rgb[::-1])  # 近距离在下
        strip = np.zeros((1, n, 4), np.uint8)
        anomalous = self.anomalous()
        for mask, color in (
            (anomalous, theme.WARN),
            (self._bad & self._known, theme.BAD),
        ):
            c = QColor(color)
            strip[0, mask[:n]] = (c.blue(), c.green(), c.red(), 255)
        self._buffers = (wf, strip)
        self._image = QImage(
            wf.data, n, WATERFALL_ROWS, n * 4, QImage.Format.Format_RGB32
        )
        self._strip = QImage(strip.data, n, 1, n * 4, QImage.Format.Format_ARGB32)

    def paintEvent(self, _event) -> None:
        # 数据变了最多每 200 ms 烘一次(定时器在跑时沿用旧图);播放头照常每帧重画
        if self._image is None or (self._dirty and not self._timer.isActive()):
            self._bake()
            self._dirty = False
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        clip = QPainterPath()
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        clip.addRoundedRect(rect, _RADIUS, _RADIUS)
        painter.setClipPath(clip)
        painter.drawImage(QRectF(0, 0, self.width(), self.height()), self._image)
        painter.drawImage(QRectF(0, 0, self.width(), STRIP_HEIGHT), self._strip)
        x = self._x_of(self.value())
        painter.setPen(QPen(QColor(theme.TEXT), 2))
        painter.drawLine(QPointF(x, 0), QPointF(x, self.height()))
        painter.setClipping(False)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(theme.LINE), 1))
        painter.drawRoundedRect(rect, _RADIUS, _RADIUS)
