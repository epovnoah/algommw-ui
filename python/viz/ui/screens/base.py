"""屏的统一骨架:一屏只讲一件事(设计稿 docs/uidesign.html 的 `main`)。

    屏名 · 一句话结论 ……………………………… [分段控件…] [列表 N ▴]     ← 30 px 上下文条
    ┌──────────────────────────────────────────────────────┐
    │ 一张主图(圆角 8 的卡片,内边距 0,占满剩余空间)       │
    │ [图例…]                                  [悬停读数]  │ ← 压在卡片上的半透明小标签
    └──────────────────────────────────────────────────────┘

- 上下文条:屏名 `heading`(14 px)、一句话结论(`widgets/conclusion.py`,单行、放不下就裁)、
  弹性空白、主视图的分段控件 `controls`、列表开关 `list_button`(「列表 N ▴ / ▾」);
- 图例 `legends` 与悬停读数 `readout` 不占布局:图例压在卡片左下角、读数压在右下角(半透明底,
  鼠标穿透,随卡片尺寸与内容定位),主图因此多出一行;
- 列表默认收起,点开从主图下方展开(占主区约 40% 高),再点收起;类属性 `list_docked = True` 的屏
  改为默认展开、固定高 250 px(检测表是这一屏的主角时用);
- 溯源不再有「看它的来路」按钮:右侧检查器常驻,选中检测后它的「溯源」页自己更新;
- 各屏原来的次要视图(CFAR 切片、测角相位残差……)不在主区,屏把它们声明在 `trace_plots`,
  由窗口挂进溯源页对应那一级(widgets/trace.py)。

具体的屏仍只声明部件(见 screens/cfar.py),不写布局代码。主视图可选地带三个属性:
`controls`(上下文条里的分段控件)、`legends`(图例)、`readout`(悬停读数标签)。
"""

from PySide6.QtCore import QEvent, QObject, QRect, Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .. import theme
from ..state import ViewState
from ..widgets.conclusion import Conclusion

#: 主图卡片与列表抽屉的高度比(开着列表时列表约占主区 40%)
CARD_STRETCH, LIST_STRETCH = 3, 2
#: 上下文条高度、固定列表(list_docked)高度
CONTEXT_HEIGHT = 30
DOCKED_LIST_HEIGHT = 250
#: 叠层小标签离卡片边的距离、两块叠层之间的最小间距
_OVERLAY_PAD, _OVERLAY_GAP = 8, 8


class _Overlay(QFrame):
    """压在主图卡片上的半透明小标签条:内容(图例 / 读数)变化时发 `content_changed`,
    由屏据此重新定位。鼠标穿透,不挡主图的悬停与点选。"""

    content_changed = Signal()

    def __init__(self, name: str, widgets: list[QWidget], parent: QWidget) -> None:
        super().__init__(parent)
        self.setObjectName(name)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setStyleSheet(
            f"#{name} {{ background: rgba(12, 14, 18, 190); border-radius: 5px; }}"
        )
        self.widgets = widgets
        row = QHBoxLayout(self)
        row.setContentsMargins(8, 2, 8, 2)
        row.setSpacing(12)
        for widget in widgets:
            widget.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            row.addWidget(widget)
        self.hide()

    def has_content(self) -> bool:
        """有一个没被藏起来、且(是标签时)有字的部件。"""
        for widget in self.widgets:
            if widget.isHidden():
                continue
            text = getattr(widget, "text", None)
            if callable(text) and not text():
                continue
            return True
        return False

    def event(self, event: QEvent) -> bool:
        done = super().event(event)
        if event.type() == QEvent.Type.LayoutRequest:
            self.content_changed.emit()
        return done


class Screen(QWidget):
    key = ""
    title = ""
    list_label = "列表"  # 上下文条里按钮的名字(ADC 是各 RX 的统计表,跟踪屏是航迹表)
    #: True = 列表抽屉默认展开、固定高 DOCKED_LIST_HEIGHT;False = 默认收起,展开时占主区约 40%
    list_docked = False

    def __init__(
        self,
        state: ViewState,
        main: QWidget,
        listing: QWidget,
        conclusion: Conclusion,
        trace_plots=(),
        list_extra: QWidget | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.state = state
        self.main, self.list, self.conclusion = main, listing, conclusion
        self.trace_plots = list(trace_plots)
        self._stale = True

        column = QVBoxLayout(self)
        column.setContentsMargins(theme.GUTTER, 6, theme.GUTTER, 6)
        column.setSpacing(theme.GAP)

        # ---- 上下文条 ----
        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(12)
        self.heading = QLabel(self.title)
        self.heading.setStyleSheet(
            f"color: {theme.TEXT}; font-size: {theme.FONT_TITLE}px; font-weight: 600;"
        )
        head.addWidget(self.heading)
        head.addWidget(conclusion, 1, Qt.AlignmentFlag.AlignVCenter)
        self.controls = list(getattr(main, "controls", []))
        for widget in self.controls:
            head.addWidget(widget)
        self.list_button = QPushButton()
        self.list_button.setProperty("role", "ghost")
        self.list_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.list_button.clicked.connect(self.toggle_list)
        # 行数的位数变了不能把左边的控件推走:按四位数预留宽度(点云上限 1024)
        widest = self.list_button.fontMetrics().horizontalAdvance(
            f"{self.list_label}  8888  ▾"
        )
        self.list_button.setMinimumWidth(widest + 24)
        head.addWidget(self.list_button)
        self.context_bar = QWidget()
        self.context_bar.setFixedHeight(CONTEXT_HEIGHT)
        self.context_bar.setLayout(head)
        column.addWidget(self.context_bar)

        # ---- 主图卡片 ----
        self.card = QFrame()
        self.card.setObjectName("Card")
        self.card.setStyleSheet(
            f"#Card {{ background: {theme.PLOT}; border: 1px solid {theme.LINE_SOFT}; "
            f"border-radius: {theme.RADIUS_CARD}px; }}"
        )
        card_layout = QVBoxLayout(self.card)
        card_layout.setContentsMargins(0, 0, 0, 0)
        card_layout.addWidget(main)
        column.addWidget(self.card, CARD_STRETCH)

        # ---- 叠层:图例(左下)与悬停读数(右下),压在卡片上、不在布局里 ----
        self.legends = list(getattr(main, "legends", []))
        self.readout = getattr(main, "readout", None)
        self.legend_strip = _Overlay("LegendStrip", self.legends, self.card)
        self.readout_strip = _Overlay(
            "ReadoutStrip",
            [self.readout] if self.readout is not None else [],
            self.card,
        )
        for strip in (self.legend_strip, self.readout_strip):
            strip.content_changed.connect(self._place_overlays)
        self.card.installEventFilter(self)

        # ---- 列表抽屉 ----
        self.drawer = QFrame()
        self.drawer.setObjectName("ListDrawer")
        self.drawer.setStyleSheet(
            f"#ListDrawer {{ background: {theme.PLOT}; border: 1px solid {theme.LINE_SOFT}; "
            f"border-radius: {theme.RADIUS_CARD}px; }}"
        )
        drawer_layout = QVBoxLayout(self.drawer)
        drawer_layout.setContentsMargins(6, 4, 6, 4)
        drawer_layout.setSpacing(4)
        drawer_head = QHBoxLayout()
        self.caption = QLabel(getattr(listing, "caption", self.list_label))
        self.caption.setStyleSheet(
            f"color: {theme.DIM}; font-size: {theme.FONT_NOTE}px;"
        )
        drawer_head.addWidget(self.caption)
        drawer_head.addStretch(1)
        if list_extra is not None:
            drawer_head.addWidget(list_extra)
        drawer_layout.addLayout(drawer_head)
        drawer_layout.addWidget(listing, 1)
        if self.list_docked:
            self.drawer.setFixedHeight(DOCKED_LIST_HEIGHT)
            column.addWidget(self.drawer, 0)
        else:
            self.drawer.hide()
            column.addWidget(self.drawer, LIST_STRETCH)
        if hasattr(listing, "caption_changed"):
            listing.caption_changed.connect(self._on_caption)
        self._count = 0
        self._set_list_text()

        state.frame_changed.connect(self._on_frame)
        state.source_changed.connect(self._on_source)

    # ---- 叠层定位 ----
    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if obj is self.card and event.type() == QEvent.Type.Resize:
            self._place_overlays()
        return super().eventFilter(obj, event)

    def _place_overlays(self) -> None:
        """图例贴左下、读数贴右下;两者共用卡片宽度,图例让给读数它放不下的部分。"""
        width, height = self.card.width(), self.card.height()
        avail = width - 2 * _OVERLAY_PAD
        strips = [s for s in (self.legend_strip, self.readout_strip) if s.has_content()]
        for strip in (self.legend_strip, self.readout_strip):
            if strip not in strips and not strip.isHidden():
                strip.hide()
        if not strips or avail <= 0:
            return
        hints = {s: s.layout().sizeHint() for s in strips}
        widths = {s: hints[s].width() for s in strips}
        if len(strips) == 2:
            room = avail - _OVERLAY_GAP
            if sum(widths.values()) > room:
                # 读数是悬停时才有的短句,先保它;图例吃剩下的,至少留一半
                widths[self.readout_strip] = min(
                    widths[self.readout_strip],
                    max(room // 2, room - widths[self.legend_strip]),
                )
                widths[self.legend_strip] = max(room - widths[self.readout_strip], 0)
        else:
            only = strips[0]
            widths[only] = min(widths[only], avail)
        for strip in strips:
            h = hints[strip].height()
            if strip is self.legend_strip:
                x = _OVERLAY_PAD
            else:
                x = width - _OVERLAY_PAD - widths[strip]
            rect = QRect(x, height - _OVERLAY_PAD - h, widths[strip], h)
            if strip.geometry() != rect:
                strip.setGeometry(rect)
            if strip.isHidden():
                strip.show()
                strip.raise_()

    # ---- 列表抽屉 ----
    @property
    def list_open(self) -> bool:
        return not self.drawer.isHidden()

    def set_list_open(self, open_: bool) -> None:
        self.drawer.setVisible(open_)
        self._set_list_text()

    def toggle_list(self) -> None:
        self.set_list_open(not self.list_open)

    def _set_list_text(self) -> None:
        arrow = "▾" if self.list_open else "▴"
        self.list_button.setText(f"{self.list_label}  {self._count}  {arrow}")

    def _on_caption(self, caption: str, _rows: int) -> None:
        self.caption.setText(caption)

    # ---- 随帧更新列表行数(只在屏可见时) ----
    def _on_source(self, _source) -> None:
        self._count = 0
        self._set_list_text()

    def _on_frame(self, frame) -> None:
        if not self.isVisible():
            self._stale = True
            return
        self._refresh_count(frame)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._stale and self.state.frame is not None:
            self._refresh_count(self.state.frame)

    def _refresh_count(self, frame) -> None:
        self._stale = False
        self._place_overlays()  # 图例随帧换字:保险起见每帧对一次(没变就不动)
        count = len(self.list.rows_for(frame))
        if count != self._count:
            self._count = count
            self._set_list_text()
