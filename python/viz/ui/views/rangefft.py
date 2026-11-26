"""距离 FFT:目标在哪些距离门?DC 签名去除做了什么?各通道是否一致?

共轴上下堆叠:上 RTI(距离 × chirp,Σ_rx),下选中 chirp 的距离像(对照全帧平均;按 RX 切换时
同时给出 DC 签名去除前 / 后)。两张图共用距离轴(setXLink,缩放平移联动),各画一条竖向游标标出选中
的距离门;RTI 上另有一条横向虚线标选中的 chirp。分段控件只剩通道(Σ / RX1..),chirp 选择框同时
作用于两张图(RTI 本身是 Σ_rx,不随通道变)。点 RTI 任一处即在该距离门选中单元并把距离像切到
那个 chirp;点距离像按横坐标选中距离门。实采样板只画前半有效距离门。
"""

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QSpinBox, QVBoxLayout, QWidget

from ...diagnostics.slices import range_axis_m, range_profiles_db, rti_db
from .. import theme
from ..state import ViewState
from ..widgets.seg import HEIGHT as SEG_HEIGHT
from ..widgets.seg import Segmented
from .base import ResultView, set_title
from .common import (
    Crosshair,
    heat_item,
    hover_cell,
    legend_label,
    levels_note,
    nearest_index,
    readout_label,
    show_heat,
)
from .marginal import LEFT_AXIS_WIDTH
from .panels import Panel


class RangeView(ResultView):
    title = "距离像"

    def __init__(self, state: ViewState, parent: QWidget | None = None) -> None:
        super().__init__(state, parent)
        self.rx = Segmented(["Σ", "RX1", "RX2", "RX3", "RX4"], tip="距离像的曲线通道")
        self.rx.currentIndexChanged.connect(self._rerender)
        self.chirp = QSpinBox()
        self.chirp.setPrefix("chirp ")
        self.chirp.setFixedHeight(SEG_HEIGHT)
        self.chirp.setToolTip(
            "选中哪一个 chirp(距离像取它、RTI 上画虚线);点 RTI 也会切到它"
        )
        self.chirp.valueChanged.connect(self._rerender)
        self.readout = readout_label()
        self.dcsig = legend_label()
        self.controls = [self.rx, self.chirp]
        self.legends = [self.dcsig]

        # 上:RTI;下:距离像。左轴定宽,两张图的绘图区才横向对齐、距离轴联动才看得出共轴
        self.rti_widget = pg.PlotWidget()
        self.rtp = theme.style_plot(self.rti_widget.getPlotItem())
        self.rtp.setLabel("left", "chirp")
        self.rtp.hideAxis("bottom")  # 距离轴只在下图画一份
        self.rti = heat_item(self.rtp)
        self.chirp_line = pg.InfiniteLine(
            angle=0,
            movable=False,
            pen=theme.pen(theme.ACCENT, 1, Qt.PenStyle.DotLine),
        )
        self.rtp.addItem(self.chirp_line, ignoreBounds=True)
        self.rti_cursor = Crosshair(self.rtp, horizontal=False)
        self.rti_widget.scene().sigMouseClicked.connect(self._on_click)
        self.rti_widget.scene().sigMouseMoved.connect(self._on_move)

        self.profile_widget = pg.PlotWidget()
        self.pp = theme.style_plot(self.profile_widget.getPlotItem())
        self.pp.setLabel("bottom", "距离", units="m")
        self.pp.setLabel("left", "功率", units="dB")
        self.pp.addLegend(offset=(-10, 10))
        self.chirp_curve = self.pp.plot(
            pen=theme.pen(theme.SIGNAL, 1.2), name="选中 chirp"
        )
        self.mean_curve = self.pp.plot(
            pen=theme.pen(theme.DIM, 1, Qt.PenStyle.DashLine),
            name="全帧平均",
        )
        self.before_curve = self.pp.plot(
            pen=theme.pen(theme.THRESHOLD, 1, Qt.PenStyle.DotLine),
            name="DC 签名去除前",
        )
        self.cursor = Crosshair(self.pp, horizontal=False)
        self.profile_widget.scene().sigMouseClicked.connect(self._on_profile_click)
        for plot in (self.rtp, self.pp):
            plot.getAxis("left").setWidth(LEFT_AXIS_WIDTH)
        self.pp.vb.setXLink(self.rtp.vb)  # 距离轴联动

        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(theme.GAP)
        column.addWidget(Panel(self.rti_widget), 155)
        column.addWidget(Panel(self.profile_widget), 100)
        self._rect = None
        self._axes = None
        self._image = None
        self._levels = None  #: 本帧 RTI 的自动色阶(标题里显示,render_selection 复用)
        self._hover = None

    # ---- ResultView ----
    def reset(self, source) -> None:
        for curve in (self.chirp_curve, self.mean_curve, self.before_curve):
            curve.setData([], [])
        self.rti.clear()
        self._rect = self._axes = self._image = self._hover = self._levels = None
        self.readout.clear()
        self.dcsig.clear()

    def _sync_chirp_range(self, frame) -> None:
        products = frame.products
        if products is None or "range_power" not in products:
            return
        last = products["range_power"].shape[0] - 1
        if self.chirp.maximum() != last:
            self.chirp.blockSignals(True)
            self.chirp.setRange(0, last)
            self.chirp.blockSignals(False)

    def render_frame(self, frame) -> None:
        self._sync_chirp_range(frame)
        self._render_profile(frame)
        image = rti_db(frame)
        if image is None:
            self.rti.clear()
            self._image = self._rect = self._axes = None
            set_title(self.rtp, "该帧没有中间量(诊断未开),不适用")
            return
        self._image = image
        n_chirps = image.shape[0]
        levels, self._rect, self._axes = show_heat(
            self.rti, image, range_axis_m(frame.radar), np.arange(n_chirps)
        )
        self._levels = levels
        self.chirp_line.setValue(float(self.chirp.value()))
        self._set_rti_title()

    def _set_rti_title(self, gate: int | None = None, range_m: float = 0.0) -> None:
        if self._image is None:
            return
        text = f"RTI · 距离 × chirp(Σ_rx,{self._image.shape[0]} chirp)"
        if gate is not None:
            text += f" · 距离门 {gate}({range_m:.2f} m)"
        set_title(self.rtp, text + (levels_note(self._levels) if self._levels else ""))

    def render_selection(self, frame, selection) -> None:
        axis = range_axis_m(frame.radar)
        if not 0 <= selection.range_bin < len(axis):
            return  # 选中的是镜像半边(实采样)的距离门:不画
        x = float(axis[selection.range_bin])
        self.cursor.set(x=x)
        self.rti_cursor.set(x=x)
        self._set_rti_title(selection.range_bin, x)

    # ---- 曲线与交互 ----
    def _render_profile(self, frame) -> None:
        x = range_axis_m(frame.radar)
        profiles = range_profiles_db(frame, self.chirp.value())
        if profiles is None:
            for curve in (self.chirp_curve, self.mean_curve, self.before_curve):
                curve.setData([], [])
            return
        rx = self.rx.currentIndex()
        if rx == 0:  # Σ_rx:与 RTI 同口径
            self.chirp_curve.setData(x, profiles["chirp_db"])
            self.mean_curve.setData(x, profiles["mean_sum_db"])
            self.before_curve.setData([], [])
        else:
            i = min(rx - 1, profiles["mean_db"].shape[0] - 1)
            self.chirp_curve.setData(x, profiles["mean_db"][i])
            self.mean_curve.setData(x, profiles["mean_sum_db"])
            if profiles["dcsig_active"]:
                self.before_curve.setData(x, profiles["mean_before_db"][i])
            else:
                self.before_curve.setData([], [])
        self.dcsig.setText(
            "DC 签名:已去除" if profiles["dcsig_active"] else "DC 签名:关 / 未就绪"
        )
        set_title(self.pp, "距离像(dB):选中 chirp 对照全帧平均")

    def _rerender(self) -> None:
        """通道 / chirp 变了:只重画距离像与 RTI 上的 chirp 虚线(RTI 本身不随它们变)。"""
        if self.state.frame is not None:
            self._render_profile(self.state.frame)
            self.chirp_line.setValue(float(self.chirp.value()))

    def _on_profile_click(self, event) -> None:
        """点距离像:按横坐标选中距离门(多普勒保持)。"""
        frame = self.state.frame
        if frame is None or event.button() != Qt.MouseButton.LeftButton:
            return
        if not self.pp.sceneBoundingRect().contains(event.scenePos()):
            return
        x_axis = range_axis_m(frame.radar)
        x = self.pp.vb.mapSceneToView(event.scenePos()).x()
        self.state.select_cell(
            nearest_index(x_axis, x),
            self.state.selection.doppler_bin if self.state.selection else 0,
        )

    def _on_click(self, event) -> None:
        if self._rect is None or event.button() != Qt.MouseButton.LeftButton:
            return
        cell = hover_cell(self.rtp, event.scenePos(), self._rect, self._axes)
        if cell is None:
            return
        row, col = cell
        self.chirp.setValue(int(row))  # 距离像跟着切到这一 chirp
        x_axis = range_axis_m(self.state.frame.radar)
        self.state.select_cell(
            nearest_index(x_axis, x_axis[col]),
            self.state.selection.doppler_bin if self.state.selection else 0,
        )

    def _on_move(self, event) -> None:
        if self._rect is None or self._image is None:
            return
        cell = hover_cell(self.rtp, event, self._rect, self._axes)
        if cell is None:
            self._hover = None
            self.readout.clear()
            return
        if cell == self._hover:
            return
        self._hover = cell
        row, col = cell
        x_axis = range_axis_m(self.state.frame.radar)
        self.readout.setText(
            f"chirp {row} · {x_axis[col]:.2f} m · {float(self._image[row, col]):+.1f} dB"
        )
