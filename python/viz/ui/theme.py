"""界面主题:设计令牌、深色 Qt 调色板、全局样式表与 pyqtgraph 全局选项。

视觉规格是 docs/uidesign.html:这里的常量就是它 `:root` 的 CSS 变量,命名照搬(GROUND = --ground …)。
克制:中性深色底,**只有一个强调色**(薄荷青 ACCENT)用于「当前 / 选中」;状态用颜色 + 形状
双编码(● 正常 / ▲ 需要看 / ■ 异常);数字一律等宽字体,文字用无衬线。字体本机没装就自然回退,
不打包字体。

色图选择(感知均匀、色盲友好,避免 jet 造成的虚假边界):
- 功率 / dB 热图:inferno;
- 有正负的量(实采样 ADC、误差):CET-D1 发散色图;相位:CET-C1 循环色图;
- 点云径向速度:CET-R3(中点为黄;CET-D1 中点是白,0 速度点在深色背景上不醒目)。
只做深色主题(亮色主题与 `--theme` 已删)。
"""

import pyqtgraph as pg
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QColor, QFont, QPalette, QPixmapCache
from PySide6.QtWidgets import QApplication, QLabel, QWidget

# ---- 设计令牌(docs/uidesign.html 的 :root) ----
GROUND = "#0C0E12"  # 窗口底
PLOT = "#101318"  # 图底(绘图区、列表底)
SURFACE = "#14171D"  # 面(按钮、胶囊、分段控件底)
RAISED = "#1B1F27"  # 抬升面(悬停、激活)
LINE = "#262B35"  # 线
LINE_SOFT = "#1E222A"  # 弱线(卡片描边、分隔)
LINE_HOVER = "#3A4150"  # 悬停时的线
TEXT = "#E8EAEE"
DIM = "#98A0AE"  # 次要文字
ACCENT = "#6EE7D8"  # 强调:只用于当前 / 选中
ACCENT_BG = "#173A3D"  # 强调色的淡底(选中行、悬停),约 15% 强调色叠在面上
OK = "#5BD08F"
WARN = "#F5B544"
BAD = "#F0616D"

# ---- 字体与尺寸(界面字体 Inter / PingFang SC 在 apply() 里设;等宽见 MONO)----
MONO = '"JetBrains Mono", "Menlo", monospace'
FONT_TITLE = 14  # 屏名(上下文条里)
FONT_BODY = 13
FONT_NOTE = 12  # 说明
FONT_TICK = 12  # 刻度与数字
RADIUS_CONTROL = 7  # 按钮与输入
RADIUS_CARD = 8  # 图卡片
MIN_HIT = 26  # 桌面上可点区域的最小高度
GUTTER = 8  # 窗口左右边距:顶栏 / 屏 / 传输条对齐同一条竖线
GAP = 6  # 屏内纵向间距:上下文条 / 主图卡片 / 列表抽屉
PLOT_TITLE_HEIGHT = 20  # 图标题行高(pyqtgraph 定死 30 px,9pt 一行字用不了)
SIDE_WIDTH = 300  # 检查器(可停靠右栏)宽度
GRID_ALPHA = 0.07

POWER_CMAP = "inferno"
VELOCITY_CMAP = "CET-R3"  # 蓝 = 接近、黄 = 静止、红 = 远离

# 通道与类别色(深色背景上可区分)
RX_COLORS = ("#4fc3f7", "#ffb74d", "#81c784", "#f06292")
TX_COLORS = ("#64b5f6", "#ffd54f", "#a5d6a7", "#ce93d8")
EMPTY_BAND_COLOR = "#5f6670"
SUBBAND_COLORS = ("#e57373", "#64b5f6", "#ffd54f", "#81c784", "#ba68c8", "#4db6ac")

DETECTION = OK  # 检测点:通过判决(设计稿里的绿色空心圆)
TRACK = TEXT  # 航迹空心方块与 id 文字
NOISE = "#E3C567"  # CFAR 噪声估计
THRESHOLD = "#F8870E"  # CFAR 门限(inferno 的橙)
SIGNAL = "#8AB4FF"  # 主曲线(设计稿里小图的蓝)

# 判据等级(diagnostics.Level:-1 不适用 / 0 正常 / 1 需要看 / 2 异常)
LEVEL_COLORS = {-1: "#5B6270", 0: OK, 1: WARN, 2: BAD}
#: 等级形状:与颜色一起双编码(绿 / 琥珀在色觉缺陷下接近,不能只靠色相)
LEVEL_SHAPES = {-1: "○", 0: "●", 1: "▲", 2: "■"}

# CFAR 叠加三色:仅距离向通过 / 仅多普勒向通过 / 两者都通过(RGBA)
CFAR_RANGE_ONLY = (79, 195, 247, 150)
CFAR_DOPPLER_ONLY = (255, 213, 79, 150)
CFAR_BOTH = (255, 82, 82, 230)

PIXMAP_CACHE_KB = 512 * 1024


def stylesheet() -> str:
    """全局样式表:提示、滚动条、输入、表格、页签。各部件自己的样式只管形状,色值都从这里的令牌来。

    按钮的几种长相用 `role` 属性选(`setProperty("role", "pill")`),不在每个部件里各写一份:
    pill(胶囊,带边框的面)/ ghost(幽灵,次要文字)/ trace(强调描边)。"""
    return f"""
    QToolTip {{ background: {RAISED}; color: {TEXT}; border: 1px solid {LINE};
        padding: 4px 8px; font-size: {FONT_NOTE}px; }}
    QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
    QScrollBar::handle:vertical {{ background: {LINE}; border-radius: 4px; min-height: 28px; }}
    QScrollBar::handle:vertical:hover {{ background: {LINE_HOVER}; }}
    QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 0; }}
    QScrollBar::handle:horizontal {{ background: {LINE}; border-radius: 4px; min-width: 28px; }}
    QScrollBar::handle:horizontal:hover {{ background: {LINE_HOVER}; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
    QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
    QPushButton[role="pill"], QPushButton[role="ghost"], QPushButton[role="trace"] {{
        min-height: {MIN_HIT}px; padding: 0 10px; border-radius: {RADIUS_CONTROL}px;
        font-size: {FONT_BODY}px; }}
    QPushButton[role="pill"] {{ background: {SURFACE}; border: 1px solid {LINE}; color: {TEXT}; }}
    QPushButton[role="pill"]:hover {{ border-color: {LINE_HOVER}; }}
    QPushButton[role="ghost"] {{ background: transparent; border: 1px solid {LINE}; color: {DIM}; }}
    QPushButton[role="ghost"]:hover {{ border-color: {LINE_HOVER}; color: {TEXT}; }}
    QPushButton[role="trace"] {{ background: transparent; border: 1px solid {ACCENT}; color: {ACCENT}; }}
    QPushButton[role="trace"]:hover {{ background: rgba(110, 231, 216, 0.08); }}
    QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{
        background: {PLOT}; color: {TEXT}; border: 1px solid {LINE};
        border-radius: {RADIUS_CONTROL}px; padding: 2px 8px; min-height: {MIN_HIT - 6}px; }}
    QComboBox QAbstractItemView {{ background: {SURFACE}; color: {TEXT};
        selection-background-color: {ACCENT_BG}; selection-color: {TEXT}; }}
    QTableView, QTreeWidget {{ background: {PLOT}; border: 0; alternate-background-color: {PLOT};
        selection-background-color: {ACCENT_BG}; selection-color: {TEXT}; }}
    QHeaderView::section {{ background: {PLOT}; color: {DIM}; border: 0;
        border-bottom: 1px solid {LINE}; padding: 4px 8px; font-size: {FONT_TICK}px; }}
    QMenu {{ background: {SURFACE}; color: {TEXT}; border: 1px solid {LINE}; padding: 4px; }}
    QMenu::item {{ padding: 6px 18px; border-radius: 6px; }}
    QMenu::item:selected {{ background: {RAISED}; }}
    """


def apply(app: QApplication) -> None:
    """全局主题:Fusion 风格 + 深色调色板 + 全局样式表,与 pyqtgraph 的深色画布一致。

    同时把 Qt 位图缓存上限从默认 10 MB 调到 512 MB:坐标轴开了 DeviceCoordinateCache
    (见 views.base),带网格的坐标轴缓存位图覆盖整个数据区,Retina 屏上单根就有数 MB,
    默认上限下缓存被反复挤掉、每帧重绘全部刻度文字(ccm 实测 2 倍缩放回放:主线程占用
    92% → 68%,帧率 17 → 48 帧/s)。
    """
    QPixmapCache.setCacheLimit(PIXMAP_CACHE_KB)
    app.setStyle("Fusion")
    font = QFont()
    font.setFamilies(["Inter", "PingFang SC", "Helvetica Neue"])
    # 用点数而不是像素:pyqtgraph 的参数树等第三方控件按 font.pointSize() 做算术,
    # 像素字体的点数是 -1。点数按屏幕逻辑 DPI 折算,使字高恰为规格的 FONT_BODY 像素
    # (macOS 是 72 dpi,1 pt = 1 px;离屏平台是 96 dpi)
    screen = app.primaryScreen()
    dpi = screen.logicalDotsPerInch() if screen is not None else 72.0
    font.setPointSizeF(FONT_BODY * 72.0 / max(dpi, 1.0))
    app.setFont(font)
    colors = QPalette()
    roles = {
        QPalette.ColorRole.Window: GROUND,
        QPalette.ColorRole.WindowText: TEXT,
        QPalette.ColorRole.Base: PLOT,
        QPalette.ColorRole.AlternateBase: SURFACE,
        QPalette.ColorRole.ToolTipBase: RAISED,
        QPalette.ColorRole.ToolTipText: TEXT,
        QPalette.ColorRole.Text: TEXT,
        QPalette.ColorRole.Button: SURFACE,
        QPalette.ColorRole.ButtonText: TEXT,
        QPalette.ColorRole.BrightText: "#ffffff",
        QPalette.ColorRole.Highlight: ACCENT_BG,
        QPalette.ColorRole.HighlightedText: TEXT,
        QPalette.ColorRole.PlaceholderText: DIM,
    }
    for role, color in roles.items():
        colors.setColor(role, QColor(color))
    for role in (
        QPalette.ColorRole.Text,
        QPalette.ColorRole.ButtonText,
        QPalette.ColorRole.WindowText,
    ):
        colors.setColor(QPalette.ColorGroup.Disabled, role, QColor("#6b717b"))
    app.setPalette(colors)
    app.setStyleSheet(stylesheet())


def configure_pyqtgraph() -> None:
    pg.setConfigOptions(
        imageAxisOrder="row-major",  # 图像数组按 [行=y, 列=x],与 numpy 矩阵习惯一致
        background=PLOT,
        foreground=DIM,
        antialias=False,  # 大量点的曲线关抗锯齿
    )


def pen(color: str, width: float = 1.0, style=None):
    p = pg.mkPen(color, width=width)
    if style is not None:
        p.setStyle(style)
    return p


def mono_font(pixel_size: int = FONT_TICK) -> QFont:
    font = QFont()
    font.setFamilies(["JetBrains Mono", "Menlo"])
    font.setStyleHint(QFont.StyleHint.Monospace)
    font.setPixelSize(pixel_size)
    return font


def style_plot(plot: pg.PlotItem, title: str | None = None) -> pg.PlotItem:
    """统一坐标网格与坐标轴:网格线几乎看不见(alpha 0.07),刻度文字用次要文字色 + 等宽 12 px。

    不开自动降采样与 clipToView:两者都按「当前视图范围」裁剪数据,而视图初始是默认的
    0~1,数据落在别的量级时会被抽空,自动缩放从此看不到数据。"""
    plot.showGrid(x=True, y=True, alpha=GRID_ALPHA)
    tick_font = mono_font()
    for name in ("left", "bottom", "top", "right"):
        axis = plot.getAxis(name)
        axis.setTickFont(tick_font)
        axis.setPen(pg.mkPen(LINE))
        axis.setTextPen(pg.mkPen(DIM))
        axis.labelStyle = {"color": DIM, "font-size": f"{FONT_NOTE}px"}
    if title:
        set_plot_title(plot, title)
    return plot


class _TitleChip(QLabel):
    """压在图内左上角的标题小标签:不占布局、不吃鼠标、放不下时从右边省略。

    图标题以前独占一行(PLOT_TITLE_HEIGHT),现在是图上的叠层,数据区因此多出一行。"""

    def __init__(self, view: QWidget, plot: pg.PlotItem) -> None:
        super().__init__(view)
        self._plot = plot
        self._full = ""
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setStyleSheet(
            f"background: rgba(12, 14, 18, 190); color: {TEXT}; border: 1px solid {LINE};"
            f"border-radius: 5px; padding: 1px 7px; font-size: {FONT_NOTE}px;"
        )
        view.installEventFilter(self)

    def eventFilter(self, obj, event) -> bool:
        if event.type() in (QEvent.Type.Resize, QEvent.Type.Show):
            self.place()
        return False

    def show_text(self, text: str) -> None:
        self._full = text
        self.setVisible(bool(text))
        self.place()

    def place(self) -> None:
        view = self.parentWidget()
        if view is None or not self._full:
            return
        left = self._plot.getAxis("left")
        x = int(left.width()) + 8 if left.isVisible() else 8
        room = max(view.width() - x - 8, 40)
        metrics = self.fontMetrics()
        pad = 16
        self.setText(
            metrics.elidedText(
                self._full, Qt.TextElideMode.ElideRight, max(room - pad, 20)
            )
        )
        self.adjustSize()
        self.move(x, 6)
        self.raise_()


def set_plot_title(plot: pg.PlotItem, text: str) -> None:
    """设图标题:画成图内左上角的小标签,标题行收成 0 高。

    拿不到所在视图(图还没放进视图)时退回旧做法:标题行收到 PLOT_TITLE_HEIGHT。"""
    view = plot.getViewWidget()
    if view is None:
        plot.setTitle(text, size="9pt")
        plot.titleLabel.setMaximumHeight(PLOT_TITLE_HEIGHT)
        plot.layout.setRowFixedHeight(0, PLOT_TITLE_HEIGHT)
        return
    # 文字仍记在 titleLabel 上(探针与调试读它),只是行高收成 0、不再占位
    plot.setTitle(text, size="9pt")
    plot.titleLabel.setMaximumHeight(0)
    plot.titleLabel.hide()
    plot.layout.setRowFixedHeight(0, 0)
    chip = getattr(plot, "_algommw_chip", None)
    if chip is None:
        chip = plot._algommw_chip = _TitleChip(view, plot)
    chip.show_text(text)
