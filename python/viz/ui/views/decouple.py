"""解耦屏的主视图(仅步进频波形):距离是怎样由「两把尺」定下来的。

两种显示(分段控件):

- **选中检测**(默认):设计稿的「两把尺」。上方粗尺(0 到有效距离上限,每 50 m 一刻度,选中检测
  所在的粗距离门用强调色小方块标出)只能说「在第 N 格里」;中间细尺按模糊周期 |S_f|·N 分段(第 0、
  第 1… 周期),细尺读数在每个周期里重复 —— 选中的解是实心强调色圆点(标「R m ✓」),相邻周期
  (n±1)的候选是虚线空心圆(标细尺读数与它联立出的速度:超出 ±N·ΔV/2 就是被排除的原因);
  下方放大条把选中检测所在的 1 个粗距离门按细距离步长分格,点落在对应格里;右侧大号等宽字给出
  距离与径向速度。没有选中检测时提示「选一个检测」并自动选中信噪比最高者。
- **全部检测**:散点,横轴粗距离、纵轴解耦后距离 − 粗距离(m),颜色按展开周期。

数据全部来自 `diagnostics.slices`(decouple_candidates / decouple_scatter);这里只画。
候选的求解调 core 导出的 `vDecoupleSolve`,所以需要共享库;没有时只画选中的解。
"""

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QAbstractItemView,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from binding.motion import MotionPreview

from ...diagnostics.slices import (
    decouple_candidates,
    decouple_scatter,
    decouple_table,
)
from .. import theme
from ..state import ViewState
from ..widgets.seg import Segmented
from .base import ResultView, set_title
from .common import ClipLabel
from .panels import Panel

MODES = ("选中检测", "全部检测", "走动补偿")
COARSE_TICK_M = 50.0
MIN_HEIGHT = 300


def period_color(period: int) -> str:
    """展开周期 → 颜色(同一周期各帧同色;周期可为负)。"""
    return theme.SUBBAND_COLORS[int(period) % len(theme.SUBBAND_COLORS)]


class Rulers(QWidget):
    """「两把尺」图:QPainter 现画,数据由 set_view 给(None = 提示文字)。"""

    def __init__(self) -> None:
        super().__init__()
        self.view: dict | None = None
        self.message = "选一个检测"
        # 只防三级放大互相压住:再矮,粗尺 / 细尺 / 放大条的纵向间距就不够画虚线了
        self.setMinimumHeight(MIN_HEIGHT)

    def set_view(self, view: dict | None, message: str = "") -> None:
        self.view, self.message = view, message
        self.update()

    # ---- 绘制 ----
    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        mono, sans = theme.mono_font(12), QFont()
        sans.setPixelSize(theme.FONT_NOTE)
        if self.view is None:
            painter.setPen(QColor(theme.DIM))
            painter.setFont(sans)
            painter.drawText(
                QRectF(0, 0, w, h), int(Qt.AlignmentFlag.AlignCenter), self.message
            )
            return
        v = self.view
        x0, x1 = 250.0, max(w - 48.0, 340.0)
        limit = v["limit_m"]

        def x_of(meters: float) -> float:
            return x0 + meters / limit * (x1 - x0)

        # 竖向:粗尺贴上沿,放大条贴下沿(底部留一行刻度字),细尺居中;随面板高度伸缩
        y_coarse = 44.0
        zoom_bottom = h - 30.0
        zoom_top = zoom_bottom - min(max(0.17 * h, 56.0), 110.0)
        y_fine = (y_coarse + zoom_top) / 2 + 6

        def text(x, y, string, color, font, align=Qt.AlignmentFlag.AlignLeft):
            """在 (x, y) 写一行字;align 为 AlignHCenter 时 x 是文字中心,否则是左端。"""
            painter.setFont(font)
            painter.setPen(QColor(color))
            left = x - 200 if align == Qt.AlignmentFlag.AlignHCenter else x
            painter.drawText(
                QRectF(left, y - 10, 400, 20),
                int(align | Qt.AlignmentFlag.AlignVCenter),
                string,
            )

        center = Qt.AlignmentFlag.AlignHCenter
        # ---- 粗尺 ----
        text(24, y_coarse - 4, "粗尺 · 快时间", theme.DIM, sans)
        text(
            24,
            y_coarse + 16,
            f"{v['coarse_step_m']:.2f} m / 格 · 0 – {limit:.0f} m",
            theme.DIM,
            mono,
        )
        painter.setPen(QPen(QColor(theme.LINE_HOVER), 2))
        painter.drawLine(QPointF(x0, y_coarse), QPointF(x1, y_coarse))
        painter.setPen(QPen(QColor(theme.LINE_HOVER), 1))
        tick = 0.0
        while tick <= limit + 1e-6:
            painter.drawLine(
                QPointF(x_of(tick), y_coarse - 8), QPointF(x_of(tick), y_coarse + 8)
            )
            text(x_of(tick), y_coarse + 24, f"{tick:g}", theme.DIM, mono, center)
            tick += COARSE_TICK_M
        gate_x = x_of((v["gate_lo_m"] + v["gate_hi_m"]) / 2)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(theme.ACCENT))
        painter.drawRoundedRect(QRectF(gate_x - 4, y_coarse - 14, 8, 28), 2, 2)
        text(gate_x, y_coarse - 28, f"第 {v['gate']} 格", theme.ACCENT, mono, center)

        # ---- 粗尺的搜索窗投到细尺 ----
        half = v["period_m"] / 2
        win_lo, win_hi = max(v["range_m"] - half, 0.0), min(v["range_m"] + half, limit)
        painter.setPen(QPen(QColor(theme.ACCENT), 1, Qt.PenStyle.DashLine))
        painter.drawLine(
            QPointF(gate_x, y_coarse + 40), QPointF(x_of(win_lo), y_fine - 44)
        )
        painter.drawLine(
            QPointF(gate_x, y_coarse + 40), QPointF(x_of(win_hi), y_fine - 44)
        )
        fill = QColor(theme.ACCENT)
        fill.setAlphaF(0.06)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(fill)
        painter.drawRoundedRect(
            QRectF(x_of(win_lo), y_fine - 44, x_of(win_hi) - x_of(win_lo), 88), 8, 8
        )
        text(
            (x_of(win_lo) + x_of(win_hi)) / 2,
            y_fine - 54,
            "粗尺给出的搜索窗(±半个周期)",
            theme.DIM,
            sans,
            center,
        )

        # ---- 细尺 ----
        text(24, y_fine - 4, "细尺 · 慢时间", theme.DIM, sans)
        text(
            24,
            y_fine + 16,
            f"{v['fine_step_m']:.2f} m / 格 · 每 {v['period_m']:.0f} m 重复",
            theme.DIM,
            mono,
        )
        painter.setPen(QPen(QColor(theme.LINE_HOVER), 2))
        painter.drawLine(QPointF(x0, y_fine), QPointF(x1, y_fine))
        painter.setPen(QPen(QColor(theme.LINE_HOVER), 1))
        n_periods = int(limit // v["period_m"]) + 1
        for n in range(n_periods + 1):
            edge = n * v["period_m"]
            if edge <= limit + 1e-6:
                painter.drawLine(
                    QPointF(x_of(edge), y_fine - 16), QPointF(x_of(edge), y_fine + 16)
                )
        for n in range(n_periods):
            mid = (n + 0.5) * v["period_m"]
            if mid <= limit:
                text(x_of(mid), y_fine + 32, f"第 {n} 周期", theme.DIM, sans, center)
        # 候选(虚线空心圆)
        for cand in v["candidates"]:
            ruler = cand["ruler_m"]
            if not 0.0 <= ruler <= limit:
                continue
            painter.setPen(QPen(QColor(theme.DIM), 2, Qt.PenStyle.DashLine))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QPointF(x_of(ruler), y_fine), 8, 8)
            text(x_of(ruler), y_fine - 36, f"{ruler:.2f} m", theme.DIM, mono, center)
            text(
                x_of(ruler),
                y_fine - 22,
                f"v {cand['velocity_mps']:+.1f} m/s ✗",
                theme.DIM,
                mono,
                center,
            )
        # 选中的解(实心)
        sol_x = x_of(v["ruler_m"])
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(theme.ACCENT))
        painter.drawEllipse(QPointF(sol_x, y_fine), 9, 9)
        text(sol_x, y_fine - 22, f"{v['range_m']:.2f} m ✓", theme.ACCENT, mono, center)

        # ---- 放大条:一个粗距离门按细距离步长分格 ----
        fine = v["fine_step_m"]
        lo = min(v["gate_lo_m"], v["range_m"] - 0.5 * fine)
        hi = max(v["gate_hi_m"], v["range_m"] + 0.5 * fine)
        bx0, bx1 = x0 + 40.0, x0 + 40.0 + (x1 - x0) * 0.50
        painter.setPen(QPen(QColor(theme.LINE_HOVER), 1, Qt.PenStyle.DashLine))
        painter.drawLine(QPointF(sol_x - 6, y_fine + 44), QPointF(bx0, zoom_top))
        painter.drawLine(QPointF(sol_x + 6, y_fine + 44), QPointF(bx1, zoom_top))
        painter.setPen(QPen(QColor(theme.LINE), 1))
        painter.setBrush(QColor(theme.SURFACE))
        painter.drawRoundedRect(
            QRectF(bx0, zoom_top, bx1 - bx0, zoom_bottom - zoom_top), 10, 10
        )

        def zx(meters: float) -> float:
            return bx0 + (meters - lo) / (hi - lo) * (bx1 - bx0)

        cells = max(int(round((v["gate_hi_m"] - v["gate_lo_m"]) / fine)), 1)
        edges = [
            v["gate_lo_m"] + i * (v["gate_hi_m"] - v["gate_lo_m"]) / cells
            for i in range(cells + 1)
        ]
        hit = next(
            (i for i in range(cells) if edges[i] <= v["range_m"] < edges[i + 1]), None
        )
        if hit is not None:
            tint = QColor(theme.ACCENT)
            tint.setAlphaF(0.10)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(tint)
            painter.drawRect(
                QRectF(
                    zx(edges[hit]),
                    zoom_top + 1,
                    zx(edges[hit + 1]) - zx(edges[hit]),
                    zoom_bottom - zoom_top - 2,
                )
            )
        painter.setPen(QPen(QColor(theme.LINE), 1))
        for edge in edges[1:-1]:
            painter.drawLine(
                QPointF(zx(edge), zoom_top), QPointF(zx(edge), zoom_bottom)
            )
        for edge in edges:
            text(zx(edge), zoom_bottom + 16, f"{edge:.1f}", theme.DIM, mono, center)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(theme.ACCENT))
        painter.drawEllipse(
            QPointF(zx(v["range_m"]), (zoom_top + zoom_bottom) / 2), 9, 9
        )

        # ---- 放大条左侧的说明 ----
        mid = (zoom_top + zoom_bottom) / 2
        text(24, mid - 18, "选中格放大", theme.TEXT, sans)
        text(24, mid + 2, f"粗尺只说「在第 {v['gate']} 格」", theme.DIM, sans)
        text(24, mid + 20, f"细尺把这一格分成 {cells} 份", theme.DIM, sans)


class _SidePanel(Panel):
    """右列:上是选中检测的读数格(距离 / 径向速度 / 周期 / 细分辨,等宽大号字),
    下是「候选周期」表。表里只放 `decouple_candidates` 真实算出来的量:选中的周期解与相邻周期
    (n±1)的候选,各自的距离、细尺读数、联立出的速度;没有「一致度」这类评分,所以不列。"""

    WIDTH = 340
    HEADERS = ("周期", "距离 m", "细尺 m", "速度 m/s", "状态")

    def __init__(self) -> None:
        super().__init__(name="DecoupleSide")
        self.setFixedWidth(self.WIDTH)
        self.body.setContentsMargins(0, 0, 0, 0)
        self.body.setSpacing(0)
        grid = QGridLayout()
        grid.setContentsMargins(14, 10, 14, 10)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(8)
        self.values: dict[str, ClipLabel] = {}
        for i, (key, name) in enumerate(
            (
                ("range", "距离"),
                ("velocity", "径向速度"),
                ("period", "周期"),
                ("fine", "细分辨"),
            )
        ):
            cell = QVBoxLayout()
            cell.setSpacing(0)
            caption = ClipLabel(name)
            caption.setStyleSheet(f"color: {theme.DIM}; font-size: 11px;")
            value = ClipLabel("—")
            value.setStyleSheet(
                f"color: {theme.ACCENT if key in ('range', 'velocity') else theme.TEXT};"
                f" font-family: {theme.MONO}; font-size: 20px;"
            )
            cell.addWidget(caption)
            cell.addWidget(value)
            grid.addLayout(cell, i // 2, i % 2)
            self.values[key] = value
        self.body.addLayout(grid)
        head = ClipLabel("候选周期")
        head.setStyleSheet(
            f"color: {theme.TEXT}; font-size: {theme.FONT_NOTE}px; font-weight: 600;"
            f" padding: 6px 14px; border-top: 1px solid {theme.LINE_SOFT};"
        )
        self.body.addWidget(head)
        self.table = QTableWidget(0, len(self.HEADERS))
        self.table.setHorizontalHeaderLabels(self.HEADERS)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(26)
        self.table.setShowGrid(False)
        self.table.setFrameShape(QTableWidget.Shape.NoFrame)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        header.setHighlightSections(False)
        self.table.setStyleSheet(f"QTableView {{ font-size: {theme.FONT_NOTE}px; }}")
        self.body.addWidget(self.table)
        self.note = ClipLabel()
        self.note.setStyleSheet(
            f"color: {theme.DIM}; font-size: 11px; padding: 6px 14px 10px 14px;"
        )
        self.body.addWidget(self.note)
        self.body.addStretch(1)

    def clear(self, message: str = "") -> None:
        for value in self.values.values():
            value.setText("—")
        self.table.setRowCount(0)
        self.note.setText(message)

    def show_detection(self, frame, k: int | None, view: dict | None) -> None:
        if k is None:
            self.clear("选一个检测")
            return
        det = frame.detections
        table = decouple_table(frame)
        has_record = table is not None and table["index"][k] >= 0
        period = int(table["period"][k]) if has_record else None
        self.values["range"].setText(f"{det.range_m[k]:.2f} m")
        self.values["velocity"].setText(f"{det.velocity_mps[k]:+.2f} m/s")
        self.values["period"].setText("—" if period is None else f"第 {period} 周期")
        self.values["fine"].setText(
            f"{abs(float(frame.radar.fine_range_step_m)):.2f} m"
        )
        rows = [
            (
                "—" if period is None else f"{period}",
                f"{det.range_m[k]:.2f}",
                "—" if view is None else f"{view['ruler_m']:.2f}",
                f"{det.velocity_mps[k]:+.2f}",
                "● 选中",
                True,
                False,
            )
        ]
        note = ""
        if view is None:
            note = "没有候选数据(无解耦记录或无共享库),只列选中检测"
        else:
            limit = view["v_limit_mps"]
            for cand in sorted(view["candidates"], key=lambda c: c["period"]):
                rows.append(
                    (
                        f"{cand['period']}",
                        f"{cand['range_m']:.2f}",
                        f"{cand['ruler_m']:.2f}",
                        f"{cand['velocity_mps']:+.2f}",
                        "○ 排除",
                        False,
                        abs(cand["velocity_mps"]) > limit,
                    )
                )
            note = f"候选速度超出 ±{limit:.2f} m/s 即被排除"
        self.table.setRowCount(len(rows))
        for r, (*cells, selected, over) in enumerate(rows):
            for c, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setTextAlignment(
                    int(
                        (
                            Qt.AlignmentFlag.AlignLeft
                            if c in (0, 4)
                            else Qt.AlignmentFlag.AlignRight
                        )
                        | Qt.AlignmentFlag.AlignVCenter
                    )
                )
                if selected:
                    item.setBackground(QColor(theme.ACCENT_BG))
                    if c == 4:
                        item.setForeground(QColor(theme.ACCENT))
                else:
                    warn = over and c == 3  # 速度超出范围:被排除的原因
                    item.setForeground(QColor(theme.WARN if warn else theme.DIM))
                self.table.setItem(r, c, item)
        self.note.setText(note)


class DecoupleView(ResultView):
    """复合主视图:左列 = 上「两把尺」(或走动补偿对照)+ 下「全部检测」修正量散点,
    右列 = 读数格 + 候选周期表。三种模式只改左上面板与左下面板的显隐。"""

    title = "解耦"
    KEEP_AUTORANGE = True  # 散点的范围随帧变

    def __init__(self, state: ViewState, parent: QWidget | None = None) -> None:
        super().__init__(state, parent)
        self.mode = Segmented(MODES, tip="两把尺 / 修正量 / 选中目标的离线聚焦对照")
        self.mode.currentIndexChanged.connect(self._on_mode)
        self.controls = [self.mode]
        self.rulers = Rulers()
        self.widget = pg.PlotWidget()
        self.plot = theme.style_plot(self.widget.getPlotItem())
        self.plot.setLabel("bottom", "粗距离", units="m")
        self.plot.setLabel("left", "解耦后距离 − 粗距离", units="m")
        for name in ("left", "bottom"):  # 修正量只有分米级:别让坐标轴自己换成 mm
            self.plot.getAxis(name).enableAutoSIPrefix(False)
        self.points = pg.ScatterPlotItem(size=8, pen=None)
        self.ring = pg.ScatterPlotItem(
            symbol="o", size=16, pen=theme.pen(theme.ACCENT, 2), brush=None
        )
        self.plot.addItem(self.points)
        self.plot.addItem(self.ring)
        self.points.sigClicked.connect(self._on_point)
        self.motion_widget = pg.PlotWidget()
        self.motion_plot = theme.style_plot(self.motion_widget.getPlotItem())
        self.motion_plot.setLabel("left", "相对补偿前能量", units="倍")
        self.motion_plot.setLabel("bottom", "选中目标的候选速度")
        self.motion_widget.hide()
        self._motion_models = {}

        # ---- 版式:左(宽)上下两块,右(340)一整列 ----
        self.top_panel = Panel(name="DecoupleTop")
        self.top_panel.body.addWidget(self.rulers)
        self.top_panel.body.addWidget(self.motion_widget)
        self.scatter_panel = Panel(self.widget, name="DecoupleScatter")
        self.side = _SidePanel()
        left = QVBoxLayout()
        left.setSpacing(theme.GAP)
        left.addWidget(self.top_panel, 13)
        left.addWidget(self.scatter_panel, 10)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(theme.GAP)
        row.addLayout(left, 1)
        row.addWidget(self.side)
        self._scatter: dict | None = None
        self._periods = None  # 散点里出现的展开周期(图例用)
        self._update_legends()

    def sizeHint(self) -> QSize:
        # 几块 PlotWidget 的 sizeHint 叠起来会比屏里的列表抽屉还高,抽屉展开时按比例分不到空间
        return QSize(640, 420)

    @property
    def all_mode(self) -> bool:
        return self.mode.currentIndex() == 1

    @property
    def motion_mode(self) -> bool:
        return self.mode.currentIndex() == 2

    # ---- ResultView ----
    def reset(self, source) -> None:
        self.rulers.set_view(None, "选一个检测")
        self.points.clear()
        self.ring.clear()
        self._scatter = None
        self.motion_plot.clear()
        self.side.clear("选一个检测")
        self._periods = None
        self._update_legends()

    def _on_mode(self, _index: int) -> None:
        # 全部检测:左上面板收起,散点独占左列;其余模式上下两块都在
        self.top_panel.setVisible(not self.all_mode)
        self.rulers.setVisible(not self.all_mode and not self.motion_mode)
        self.motion_widget.setVisible(self.motion_mode)
        if self.motion_mode:  # 走动补偿的图自带标题
            self.top_panel.set_chip("left", "")
        self._update_legends()
        if self.state.frame is not None:
            self._render(self.state.frame, self.state.selection)

    def render_frame(self, frame) -> None:
        data = decouple_scatter(frame, self.state.config)
        self._scatter = data
        if data is None:
            self.points.clear()
            self.ring.clear()
            set_title(self.plot, "该帧没有解耦中间量,不适用")
            self._periods = None
            self._update_legends()
            return
        self.points.setData(
            x=data["coarse_m"],
            y=data["delta_m"],
            brush=[pg.mkBrush(period_color(p)) for p in data["period"]],
            data=data["rows"],
        )
        set_title(
            self.plot,
            f"本帧全部检测 · 细尺修正量 vs 粗距离门({len(data['rows'])} 个)",
        )
        self._periods = data["period"]
        self._update_legends()

    def render_selection(self, frame, selection) -> None:
        k = selection.detection
        self._render_ring(selection)
        if k is None and not self.all_mode and not self.motion_mode:
            det = frame.detections
            if len(det):
                self.state.select_detection(int(np.argmax(det.snr_db)))  # 回到本方法
            else:
                self.rulers.set_view(None, "本帧没有检测")
                self.side.clear("本帧没有检测")
            return
        view = None
        if k is not None and self.state.config is not None:
            try:
                view = decouple_candidates(self.state.config, frame, k)
            except (ImportError, OSError):
                view = None
        self.side.show_detection(frame, k, view)
        if self.motion_mode:
            self._render_motion(frame, k)
            return
        if self.all_mode:
            return
        message = "该帧没有解耦中间量,或选中的检测没有解耦记录" if view is None else ""
        self.rulers.set_view(view, message)
        self.top_panel.set_chip(
            "left", "" if k is None else f"三级放大 · #{k}" if view else "三级放大"
        )

    def _render_motion(self, frame, k) -> None:
        self.motion_plot.clear()
        products = frame.products or {}
        if k is None or "motion_range_fft" not in products:
            set_title(
                self.motion_plot, "选中一个检测；旧缓存帧需要重新处理才能查看补偿"
            )
            return
        profile = int(products["motion_profile"])
        # 预览模型按(板, profile)缓存:换板后 B 帧的步进符号/PRI 可能不同。
        board = (self.state.source.profile if self.state.source else "") or "afmccmdual"
        if (board, profile) not in self._motion_models:
            self._motion_models[(board, profile)] = MotionPreview(profile, board)
        distance = float(frame.detections.range_m[k])
        velocity = float(frame.detections.velocity_mps[k])
        gain = self._motion_models[(board, profile)].gain_db(
            products["motion_range_fft"], distance, velocity
        )
        if not np.isfinite(gain):
            set_title(self.motion_plot, "该检测的相干能量不足，无法比较")
            return
        # 基线固定为同一 R/v 的线性匹配积分；柱高保留正值，低速近零增益也能看见两柱。
        ratio = 10 ** (gain / 10)
        self.motion_plot.addItem(
            pg.BarGraphItem(
                x=[0, 1],
                height=[1, ratio],
                width=0.55,
                brushes=[pg.mkBrush(theme.DIM), pg.mkBrush(theme.ACCENT)],
            )
        )
        self.motion_plot.setXRange(-0.6, 1.6, padding=0)
        self.motion_plot.setYRange(0, max(1, ratio) * 1.2, padding=0)
        self.motion_plot.getAxis("bottom").setTicks(
            [[(0, "原线性模型"), (1, "走动补偿")]]
        )
        set_title(
            self.motion_plot,
            f"R {distance:.2f} m · v {velocity:+.2f} m/s · 能量变化 {gain:+.2f} dB"
            "；候选局部聚焦，不代表测角精度提升",
        )

    # ---- 内部 ----
    def _render_ring(self, selection) -> None:
        data = self._scatter
        k = selection.detection
        if data is None or k is None or k not in data["rows"]:
            self.ring.clear()
            return
        i = int(np.flatnonzero(data["rows"] == k)[0])
        self.ring.setData(x=[data["coarse_m"][i]], y=[data["delta_m"][i]])

    def _on_point(self, _item, points, _event) -> None:
        if len(points):
            self.state.select_detection(int(points[0].data()))

    def _update_legends(self) -> None:
        """散点的周期图例常驻在散点面板右上;左上面板的尺图例随模式换。"""
        periods = self._periods
        if periods is None or not len(periods):
            self.scatter_panel.set_chip("right", "")
        else:
            parts = [
                f"<span style='color:{period_color(p)}'>●</span> 第 {p} 周期"
                for p in sorted({int(p) for p in periods})
            ]
            self.scatter_panel.set_chip("right", "　".join(parts))
        self.top_panel.set_chip(
            "right",
            (
                ""
                if self.motion_mode
                else f"<span style='color:{theme.ACCENT}'>●</span> 选中的解　"
                "<span>◌</span> 被排除的候选"
            ),
        )
