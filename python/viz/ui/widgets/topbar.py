"""顶栏(一行,高 40 px):品牌 · 板与采集胶囊 · 处理链 · 每帧耗时 · 参数 · 检查器 · 搜索。

处理链(`widgets/chain.py` 的 ChainRow)嵌在中间,吃掉剩余宽度;窄了它自己省略实时数。
低频项(落盘开关等)不占顶栏,收进命令面板与检查器的「参数」页。胶囊点开「打开…」弹层
(换板 / 换采集);「每帧 X / Y ms」= 处理耗时 / 帧周期(文字后一条迷你预算条),超出周期标红;
「检查器」钮开合右侧栏(与 Ctrl+反斜杠同一动作),打开时用强调色描边;最右的钮只写快捷键。
"""

from PySide6.QtCore import QRectF, QSize, Qt, Signal
from PySide6.QtGui import QAction, QColor, QFontMetrics, QPainter
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget

from .. import theme
from .chain import ChainRow
from .pill import fit_pill

HEIGHT = 40
BUDGET_BAR = QSize(52, 4)


class _BudgetBar(QWidget):
    """52×4 的迷你预算条:填充 = 处理耗时 / 帧周期(超周期时满格标红)。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(BUDGET_BAR)
        self._fraction = 0.0
        self._bad = False

    def set_fraction(self, fraction: float, bad: bool) -> None:
        fraction = min(max(fraction, 0.0), 1.0)
        # 按像素取整:没变的不重画(每帧都会调)
        if (
            round(fraction * BUDGET_BAR.width())
            != round(self._fraction * BUDGET_BAR.width())
            or bad != self._bad
        ):
            self._fraction, self._bad = fraction, bad
            self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(theme.LINE))
        rect = QRectF(self.rect())
        painter.drawRoundedRect(rect, 2, 2)
        if self._fraction > 0:
            painter.setBrush(QColor(theme.BAD if self._bad else theme.DIM))
            painter.drawRoundedRect(
                QRectF(0, 0, rect.width() * self._fraction, rect.height()), 2, 2
            )


class TopBar(QWidget):
    palette_requested = Signal()
    params_requested = Signal()
    inspector_requested = Signal()

    def __init__(
        self,
        open_action: QAction,
        palette_hint: str,
        chain: ChainRow,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("TopBar")
        self.setFixedHeight(HEIGHT)
        self.setStyleSheet(f"#TopBar {{ border-bottom: 1px solid {theme.LINE_SOFT}; }}")
        row = QHBoxLayout(self)
        # 下边留 1 px 给分隔线;处理链按钮顶到上下沿,所以上下边距为 0
        row.setContentsMargins(theme.GUTTER, 0, theme.GUTTER, 1)
        row.setSpacing(10)
        brand = QLabel("algommw")
        brand.setStyleSheet(
            f"color: {theme.TEXT}; font-size: {theme.FONT_BODY}px; font-weight: 600; "
            "letter-spacing: 0.5px;"
        )
        row.addWidget(brand)

        self.source_button = QPushButton()
        self.source_button.setProperty("role", "pill")
        self.source_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.source_button.setToolTip("换板 / 换采集目录(新会话:缓存与航迹从零开始)")
        self.source_button.setFixedHeight(theme.MIN_HIT)
        inner = QHBoxLayout(self.source_button)
        inner.setContentsMargins(10, 0, 10, 0)
        inner.setSpacing(8)
        self.source_name = QLabel("打开…")
        self.source_detail = QLabel("")
        self.source_detail.setStyleSheet(f"color: {theme.DIM};")
        for label in (self.source_name, self.source_detail):
            label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        inner.addWidget(self.source_name)
        inner.addWidget(self.source_detail)
        self.source_button.clicked.connect(open_action.trigger)
        row.addWidget(self.source_button)

        self.chain = chain
        row.addWidget(chain, 1)

        self.budget = QLabel("")
        self._budget_bad: bool | None = None
        self._style_budget(False)
        # 数字位数变了不能挪动右边的钮和处理链:按最宽的一句预留宽度
        self.budget.setMinimumWidth(
            QFontMetrics(theme.mono_font(theme.FONT_NOTE)).horizontalAdvance(
                "每帧 000.0 / 000 ms"
            )
        )
        self.budget.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        row.addWidget(self.budget)
        self.budget_bar = _BudgetBar()
        self.budget_bar.hide()
        row.addWidget(self.budget_bar)

        self.params_button = self._ghost("参数")
        self.params_button.clicked.connect(self.params_requested.emit)
        row.addWidget(self.params_button)
        self.inspector_button = self._ghost("检查器")
        self.inspector_button.setToolTip("开合右侧检查器  Ctrl+\\")
        self.inspector_button.clicked.connect(self.inspector_requested.emit)
        row.addWidget(self.inspector_button)
        self.palette_button = self._ghost(palette_hint)
        self.palette_button.setToolTip("跳到帧 · 切换屏 · 搜索命令")
        self.palette_button.clicked.connect(self.palette_requested.emit)
        row.addWidget(self.palette_button)
        self._inspector_open: bool | None = None
        self.set_inspector_open(False)

    @staticmethod
    def _ghost(text: str) -> QPushButton:
        button = QPushButton(text)
        button.setProperty("role", "ghost")
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        return button

    def set_inspector_open(self, open_: bool) -> None:
        """检查器开着时「检查器」钮用强调色描边。"""
        if open_ == self._inspector_open:
            return
        self._inspector_open = open_
        self.inspector_button.setStyleSheet(
            f"QPushButton {{ border-color: {theme.ACCENT}; color: {theme.TEXT}; }}"
            if open_
            else ""
        )

    def _style_budget(self, bad: bool) -> None:
        if bad != self._budget_bad:
            self._budget_bad = bad
            self.budget.setStyleSheet(
                f"color: {theme.BAD if bad else theme.DIM}; font-family: {theme.MONO}; "
                f"font-size: {theme.FONT_NOTE}px;"
            )

    def set_source(self, profile: str, mode: str, capture: str, n_frames: int) -> None:
        self.source_name.setText(f"{profile.upper()} · {mode}")
        self.source_detail.setText(f"{capture} · {n_frames} 帧")
        fit_pill(self.source_button)

    def set_budget(self, total_ms: float | None, period_ms: float | None) -> None:
        """每帧处理耗时 / 帧周期;None 清空。"""
        if total_ms is None or period_ms is None:
            self.budget.setText("")
            self.budget_bar.hide()
            return
        bad = total_ms > period_ms
        self._style_budget(bad)
        self.budget.setText(f"每帧 {total_ms:.1f} / {period_ms:.0f} ms")
        self.budget_bar.set_fraction(
            total_ms / period_ms if period_ms > 0 else 1.0, bad
        )
        self.budget_bar.show()

    def budget_text(self) -> str:
        return self.budget.text()
