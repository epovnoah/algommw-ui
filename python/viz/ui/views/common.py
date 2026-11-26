"""视图共用的小部件与显示名表,以及各二维图共用的坐标换算。

约定:所有热图的横轴 / 纵轴都是**物理坐标**(米、度、bin、采样序号),数组本身按
[行 = 纵轴, 列 = 横轴] 存放;显示时才把行序换成图上的 y 方向,坐标换算只在这里写一份。
"""

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, QSize, Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from ...diagnostics.slices import display_levels_db
from .. import theme

#: 逐级耗时的键(`ProcessedFrame.timings_ms`)→ 显示名,顺序即处理顺序。
#: "read" / "extract" 由 FrameProcessor 计时,其余由 ChainSession.process_frame 计时。
TIMING_STAGES = {
    "read": "读取 ADC",
    "range": "距离 FFT",
    "doppler": "多普勒 FFT",
    "cfar": "CFAR",
    "doa": "测角",
    "pair": "双 PRF 配对",
    "cluster": "聚类",
    "track": "跟踪",
    "world": "坐标变换",
    "extract": "结果提取",
}


def rd_column(doppler_bin, n_doppler: int):
    """core 的无符号多普勒 bin → `FrameResult.rd_db`(多普勒轴已 fftshift)里的列号。

    标量进标量出,数组进数组出(散点一次算全部检测时用数组形态)。"""
    bins = np.asarray(doppler_bin, np.int64)
    columns = (bins + n_doppler // 2) % n_doppler
    return int(columns) if columns.ndim == 0 else columns


class ClipLabel(QLabel):
    """文字再长也不撑宽窗口的标签:最小宽度 0,放不下时从右边裁掉(与 `Conclusion` 同一约定)。

    QLabel 的 minimumSizeHint 等于整段文字的宽度,读数 / 图例每帧换字时,所在布局的最小宽度
    跟着变,QMainWindow 会被顶宽(只涨不缩,越过屏幕)。sizeHint 不动:够宽时仍按文字宽度排。"""

    def minimumSizeHint(self) -> QSize:
        return QSize(0, super().minimumSizeHint().height())


def readout_label() -> QLabel:
    """悬停读数标签(屏把它放在底部一行右侧):等宽小字,次要文字色。"""
    label = ClipLabel()
    label.setStyleSheet(
        f"color: {theme.DIM}; font-family: {theme.MONO}; font-size: {theme.FONT_TICK}px;"
    )
    return label


def legend_label() -> QLabel:
    """图例标签(屏把它放在底部一行):说明文字,次要文字色,内容用富文本着色。"""
    label = ClipLabel()
    label.setStyleSheet(f"color: {theme.DIM}; font-size: {theme.FONT_NOTE}px;")
    return label


def vbox(
    widget: QWidget, *children: QWidget, stretch: tuple[int, ...] | None = None
) -> QVBoxLayout:
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(2, 2, 2, 2)
    layout.setSpacing(2)
    for i, child in enumerate(children):
        layout.addWidget(child, stretch[i] if stretch else 0)
    return layout


# ---------------------------------------------------------------- 图像坐标
def heat_levels(image_db: np.ndarray) -> tuple[float, float]:
    """热图的自动色阶,每帧按数据重算(不提供手动调整)。

    口径是**绝对电平**:下限 = 噪声底(中位数)再低 3 dB,上限 = 99.95% 分位,让最强的少数
    单元饱和、目标落在色阶中段;上限**不封 0 dB** —— RTI / RD 图 / 子带功率在 +50~+130 dB,
    dBFS 在负区间,按「相对本帧峰值」封顶会把整张图压成一个颜色。
    """
    return display_levels_db(image_db, ceiling_db=None)


def power_colormap() -> pg.ColorMap:
    """热图的彩色 LUT(inferno:cold = 弱、亮黄 = 强)。

    ⚠ 必须显式设:`ImageItem` 的默认 LUT 是**灰度**,以前是 `ColorBarItem.setImageItem()`
    顺带把 colormap 灌进图里的,去掉 colorbar 后不设就成了黑白热图。
    """
    return pg.colormap.get(theme.POWER_CMAP)


def levels_note(levels: tuple[float, float]) -> str:
    """去掉 colorbar 后,把自动色阶写进图标题:「色阶 48~93 dB」。"""
    return f"　色阶 {levels[0]:.0f}~{levels[1]:.0f} dB"


def axis_rect(x: np.ndarray, y: np.ndarray) -> QRectF:
    """一维坐标数组(单元中心的坐标,可不等距)在图上占的矩形。

    图像的一个像素画一个单元,单元中心在坐标值上,所以矩形两端各退半个格。"""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    dx = (x[-1] - x[0]) / max(len(x) - 1, 1) if len(x) > 1 else 1.0
    dy = (y[-1] - y[0]) / max(len(y) - 1, 1) if len(y) > 1 else 1.0
    return QRectF(x[0] - dx / 2, y[0] - dy / 2, len(x) * dx, len(y) * dy)


def nearest_index(axis: np.ndarray, value: float) -> int:
    """坐标值 → 最近的格序号(裁剪到合法范围)。"""
    axis = np.asarray(axis, float)
    return int(np.clip(np.abs(axis - value).argmin(), 0, len(axis) - 1))


def image_axes(shape: tuple[int, int], rect: QRectF) -> tuple[np.ndarray, np.ndarray]:
    """(行数, 列数) 与矩形 → 每个单元的 (纵轴坐标, 横轴坐标)。行 0 在矩形底边。"""
    rows, cols = shape
    x = rect.left() + (np.arange(cols) + 0.5) * rect.width() / max(cols, 1)
    y = rect.top() + (np.arange(rows) + 0.5) * rect.height() / max(rows, 1)
    return y, x


def heat_item(plot: pg.PlotItem) -> pg.ImageItem:
    """建热图图元并挂到 plot。彩色 LUT 一并设好 —— `ImageItem` 默认是灰度。"""
    item = pg.ImageItem()
    plot.addItem(item)
    item.setColorMap(power_colormap())
    return item


def show_heat(item: pg.ImageItem, image: np.ndarray, x, y):
    """把 image 按自动色阶画进 item,并按物理坐标 (x, y) 定位;返回 (levels, rect, axes)。

    setRect 必须在 setImage 之后:先设时按 1×1 像素换算,图像会被放大几十倍。"""
    levels = heat_levels(image)
    item.setImage(image, autoLevels=False, levels=levels)
    rect = axis_rect(x, y)
    item.setRect(rect)
    return levels, rect, image_axes(image.shape, rect)


class Crosshair:
    """不可拖动的十字线,标出选中单元(不叫 cursor:会遮住 QWidget.cursor())。"""

    def __init__(
        self, plot: pg.PlotItem, vertical: bool = True, horizontal: bool = True
    ) -> None:
        dash = theme.pen(theme.ACCENT, 1, Qt.PenStyle.DashLine)
        self.v = (
            pg.InfiniteLine(angle=90, movable=False, pen=dash) if vertical else None
        )
        self.h = (
            pg.InfiniteLine(angle=0, movable=False, pen=dash) if horizontal else None
        )
        for line in (self.v, self.h):
            if line is not None:
                line.setZValue(20)
                plot.addItem(line, ignoreBounds=True)

    def set(self, x: float | None = None, y: float | None = None) -> None:
        if self.v is not None and x is not None:
            self.v.setValue(float(x))
        if self.h is not None and y is not None:
            self.h.setValue(float(y))


def hover_cell(plot: pg.PlotItem, event, rect: QRectF, axes) -> tuple[int, int] | None:
    """鼠标事件 → 图像里的 (行, 列);不在图像上或越界时 None。axes 见 `image_axes`。"""
    if not plot.sceneBoundingRect().contains(event):
        return None
    point = plot.vb.mapSceneToView(event)
    if not rect.contains(point):
        return None
    y_axis, x_axis = axes
    row = nearest_index(y_axis, point.y())
    col = nearest_index(x_axis, point.x())
    return row, col
