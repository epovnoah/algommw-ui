"""选中检测的小图:溯源检查器的各行与阶段屏的侧栏共用(照 ccm `gui/plots/mini.py`)。

每个小图是一个 QWidget,update_detection(frame, k) 用第 k 个检测的数据重画;数据全部来自帧与
诊断结果,小图本身不做计算。坐标随选中检测变化,所以保持自动缩放。

小图按级分工:多普勒屏用 `SlowTimeSpectrum`,CFAR 屏与检查器用
`CfarSlices`,检查器的测角级用 `ChannelBars`(测角屏主区的同内容大图是 `ChannelResidual`),
跟踪屏与检查器用 `NisBars` / `GateMap`,距离屏用 `RangeLocal`。检查器通过 `STAGE_PLOTS` 挂对应的一张(见 widgets/inspector.py)。
"""

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget

from ...diagnostics.slices import (
    adc_waveforms,
    cfar_profiles,
    ddm_slow_time_db,
    ddm_tx_bins,
    doa_spectrum,
    gates,
    range_axis_m,
    range_profiles_db,
    track_table,
)
from .. import theme
from ..views.base import set_title


def mini_plot(height: int = 112) -> tuple[pg.PlotWidget, pg.PlotItem]:
    widget = pg.PlotWidget()
    widget.setMinimumHeight(height)
    widget.setMaximumHeight(int(height * 1.5))
    item = theme.style_plot(widget.getPlotItem())
    item.hideButtons()
    item.setMenuEnabled(False)
    for name in ("left", "bottom"):
        item.getAxis(name).setStyle(tickLength=-3)
    return widget, item


def _row(parent: QWidget, *widgets: QWidget) -> QHBoxLayout:
    """横排放进 parent(布局创建时即带父部件,子部件随即归 parent 所有)。"""
    row = QHBoxLayout(parent)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(4)
    for w in widgets:
        row.addWidget(w)
    return row


COMPACT_HEIGHT = 64  # 溯源页里小图的高度
ENLARGED_HEIGHT = 180  # 点「放大」后


class MiniPlot(QWidget):
    """小图基类:子类实现 update_detection(frame, k)。"""

    def update_detection(self, frame, k: int | None) -> None:
        raise NotImplementedError

    def set_enlarged(self, enlarged: bool) -> None:
        """溯源页里的两档高度:64 px 的小图(不画标题,说明在行的事实文字里)/ 放大档带标题。
        改完要让下一次 update_detection 重画标题(清掉「标题没变」的记忆)。"""
        for widget in self.findChildren(pg.PlotWidget):
            plot = widget.getPlotItem()
            plot._algommw_quiet = not enlarged
            height = ENLARGED_HEIGHT if enlarged else COMPACT_HEIGHT
            widget.setMinimumHeight(height)
            widget.setMaximumHeight(height)
            # 64 px 的小图像一条走势线:不要左轴与轴标题(数值在放大档里看),只留横轴刻度
            plot.showAxis("left", enlarged)
            plot.getAxis("bottom").showLabel(enlarged)
            if not enlarged:
                plot.setTitle(None)
            plot._algommw_title = None


class DetectionScatter(MiniPlot):
    """本帧全部检测的「SNR – 某项指标」散点,按该级判据着色,可点选;选中的检测加青色圈。

    value(frame) 返回 (n,) 的指标;lines 为参考线 [(值或 frame→值 的函数, 颜色)],
    如判据阈值。
    """

    def __init__(
        self,
        stage: str,
        value,
        title: str,
        y_range: tuple[float, float],
        lines=(),
        height: int = 150,
    ) -> None:
        super().__init__()
        self.stage, self.value = stage, value
        self.widget, self.plot = mini_plot(height)
        self.plot.setYRange(*y_range, padding=0.02)
        self.plot.setLabel("bottom", "SNR", units="dB")
        set_title(self.plot, title)
        self.lines = []
        for level, color in lines:
            line = pg.InfiniteLine(
                pos=0 if callable(level) else level,
                angle=0,
                movable=False,
                pen=theme.pen(color, 1, Qt.PenStyle.DashLine),
            )
            self.plot.addItem(line, ignoreBounds=True)
            self.lines.append((level, line))
        self.points = pg.ScatterPlotItem(size=6, pen=None)
        self.ring = pg.ScatterPlotItem(
            symbol="o", size=14, pen=theme.pen(theme.ACCENT, 2), brush=None
        )
        self.plot.addItem(self.points)
        self.plot.addItem(self.ring)
        self.points.sigClicked.connect(self._on_click)
        self.on_select = None  # 由所在面板设置:点选某个检测时调用
        _row(self, self.widget)

    def update_detection(self, frame, k: int | None) -> None:
        det, diag = frame.detections, frame.diagnostics
        if diag is None or len(det) == 0:
            self.points.clear()
            self.ring.clear()
            return
        for level, line in self.lines:
            if callable(level):
                line.setValue(level(frame))
        values = np.asarray(self.value(frame), float)
        y_lo, y_hi = self.plot.vb.viewRange()[1]
        shown = np.clip(values, y_lo, y_hi)  # 超出量程的点画在边上
        levels = diag.verdicts.levels[self.stage]
        brushes = [
            pg.mkBrush(theme.LEVEL_COLORS[int(lv)] if lv >= 1 else theme.SIGNAL)
            for lv in levels
        ]
        self.points.setData(
            x=det.snr_db, y=shown, brush=brushes, data=np.arange(len(det))
        )
        if k is None:
            self.ring.clear()
        else:
            self.ring.setData(x=[det.snr_db[k]], y=[shown[k]])

    def _on_click(self, _item, points, _event) -> None:
        if len(points) and self.on_select is not None:
            self.on_select(int(points[0].data()))


# ---------------------------------------------------------------- 各级专用小图
PHASE_LIMIT_DEG = 180.0


class _Profile:
    """功率 / 噪声 / 门限三线的小图(CFAR 两个方向、距离像都用它)。"""

    def __init__(self, height: int) -> None:
        self.widget, self.plot = mini_plot(height)
        self.power = self.plot.plot(pen=theme.pen(theme.SIGNAL, 1.2), name="功率")
        self.noise = self.plot.plot(
            pen=theme.pen(theme.NOISE, 1, Qt.PenStyle.DashLine), name="噪声"
        )
        self.threshold = self.plot.plot(
            pen=theme.pen(theme.THRESHOLD, 1.2), name="门限"
        )
        self.cursor = pg.InfiniteLine(
            angle=90,
            movable=False,
            pen=theme.pen(theme.ACCENT, 1, Qt.PenStyle.DashLine),
        )
        self.plot.addItem(self.cursor, ignoreBounds=True)

    def show(self, x, power, noise, threshold, cursor: float, title: str) -> None:
        self.power.setData(x, power)
        if noise is None:
            self.noise.clear()
            self.threshold.clear()
        else:
            self.noise.setData(x, noise)
            self.threshold.setData(x, threshold)
        self.cursor.setValue(float(cursor))
        set_title(self.plot, title)


class CfarSlices(MiniPlot):
    """CFAR:选中单元附近的距离向与整条多普勒向的功率 / 噪声 / 门限切片。"""

    def __init__(self, height: int = 112, window: int = 24) -> None:
        super().__init__()
        self.span = window
        self.range_plot, self.doppler_plot = _Profile(height), _Profile(height)
        _row(self, self.range_plot.widget, self.doppler_plot.widget)

    def update_detection(self, frame, k: int | None) -> None:
        profiles = None if k is None else cfar_profiles(frame, k, self.span)
        if profiles is None:
            for part in (self.range_plot, self.doppler_plot):
                part.show([], [], None, None, 0.0, "该帧没有 CFAR 中间量")
            return
        r, d = profiles["range"], profiles["doppler"]
        self.range_plot.show(
            r["axis"],
            r["power_db"],
            r["noise_db"],
            r["threshold_db"],
            r["cursor"],
            "距离向:功率 / 噪声 / 门限(m)",
        )
        self.doppler_plot.show(
            d["axis"],
            d["power_db"],
            d["noise_db"],
            d["threshold_db"],
            d["cursor"],
            f"多普勒向({d['unit']})",
        )


class SlowTimeSpectrum(MiniPlot):
    """DDM:选中距离门的完整慢时间谱,竖线为 TX 副本(彩)与空子带(灰点线)。"""

    def __init__(self, height: int = 112) -> None:
        super().__init__()
        self.widget, self.plot = mini_plot(height)
        self.curve = self.plot.plot(pen=theme.pen(theme.SIGNAL, 1.2))
        self.lines: list[pg.InfiniteLine] = []
        _row(self, self.widget)

    def update_detection(self, frame, k: int | None) -> None:
        data = None
        if k is not None:
            data = ddm_slow_time_db(frame, int(frame.detections.range_bin[k]))
        if data is None:
            self.curve.setData([], [])
            self._clear_lines()
            set_title(self.plot, "TDM 无慢时间谱,不适用")
            return
        self.curve.setData(data["bin"], data["power_db"])
        self._clear_lines()
        bins = ddm_tx_bins(frame, k) if k is not None else None
        if bins is not None:
            for i, value in enumerate(bins["tx_bins"]):
                self._add_line(value, theme.TX_COLORS[i % len(theme.TX_COLORS)])
            for value in bins["empty_bins"]:
                self._add_line(value, theme.EMPTY_BAND_COLOR, Qt.PenStyle.DotLine)
        set_title(self.plot, "选中距离门的慢时间谱(竖线:TX 副本 / 空子带)")

    def _add_line(self, value: float, color: str, style=None) -> None:
        line = pg.InfiniteLine(
            pos=float(value), angle=90, movable=False, pen=theme.pen(color, 1.2, style)
        )
        self.plot.addItem(line, ignoreBounds=True)
        self.lines.append(line)

    def _clear_lines(self) -> None:
        for line in self.lines:
            self.plot.removeItem(line)
        self.lines = []


class ChannelBars(MiniPlot):
    """测角:各虚拟通道去掉估计方向后的相位残差与幅度(按 TX 分组着色)。

    探针按当前处理参数懒建(换会话 / 改参数后 `reset_probe` 重建);没有探针时显示原因。"""

    def __init__(self, state=None, height: int = 104) -> None:
        super().__init__()
        self.state = state
        self.probe = None
        self.phase_plot = _Profile(height)
        self.phase_plot.noise.clear()
        self.phase_plot.threshold.clear()
        self.phase_plot.power.setPen(theme.pen(theme.SIGNAL, 1.2))
        self.amp_plot = _Profile(height)
        self.amp_plot.noise.clear()
        self.amp_plot.threshold.clear()
        self.amp_plot.power.setPen(theme.pen(theme.DETECTION, 1.2))
        _row(self, self.phase_plot.widget, self.amp_plot.widget)
        self._reset_probe()
        if state is not None:
            state.config_changed.connect(lambda *_: self._reset_probe())

    def _reset_probe(self) -> None:
        from ..views.angle import make_probe

        config = getattr(self.state, "config", None) if self.state is not None else None
        self.probe = make_probe(config)

    def reset_probe(self) -> None:
        self._reset_probe()

    def update_detection(self, frame, k: int | None) -> None:
        spec = None if k is None else doa_spectrum(frame, k, self.probe)
        if spec is None:
            for part in (self.phase_plot, self.amp_plot):
                part.show([], [], None, None, 0.0, "没有测角谱(探针不可用或该点无记录)")
            return
        index = np.arange(len(spec["phase_deg"]))
        self.phase_plot.show(
            index,
            spec["phase_deg"],
            None,
            None,
            0.0,
            f"相位残差(°):相干度 {spec['coherence']:.2f}",
        )
        self.phase_plot.plot.setYRange(-PHASE_LIMIT_DEG, PHASE_LIMIT_DEG, padding=0)
        self.amp_plot.show(
            index, spec["amplitude_db"], None, None, 0.0, "幅度(相对最强通道 dB)"
        )
        self.amp_plot.plot.setYRange(-20.0, 1.0, padding=0)


class ChannelResidual(QWidget):
    """测角屏右下的「通道相位残差」面板:上为各虚拟通道去掉估计方向后的相位残差(柱),下为幅度
    (相对最强通道,柱);与检查器里的 `ChannelBars` 同一份数据(`doa_spectrum` 的 phase_deg /
    amplitude_db),只是在主区里占满面板。谱由测角屏的主视图取好后交给 `show_spec`,这里不再建探针。"""

    def __init__(self) -> None:
        super().__init__()
        self.phase_widget = pg.PlotWidget()
        self.phase = theme.style_plot(self.phase_widget.getPlotItem())
        self.phase.setLabel("left", "相位残差", units="°")
        self.phase.setYRange(-PHASE_LIMIT_DEG, PHASE_LIMIT_DEG, padding=0)
        self.phase.addLine(y=0, pen=theme.pen(theme.LINE_HOVER, 1))
        self.phase_bars = pg.BarGraphItem(x=[], height=[], width=0.7)
        self.phase.addItem(self.phase_bars)
        self.amp_widget = pg.PlotWidget()
        self.amp = theme.style_plot(self.amp_widget.getPlotItem())
        self.amp.setLabel("left", "幅度", units="dB")
        self.amp.setLabel("bottom", "虚拟通道")
        self.amp.setYRange(-20.0, 1.0, padding=0)
        self.amp_bars = pg.BarGraphItem(x=[], height=[], width=0.7)
        self.amp.addItem(self.amp_bars)
        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)
        column.addWidget(self.phase_widget, 3)
        column.addWidget(self.amp_widget, 2)

    def show_spec(self, spec: dict | None, message: str = "") -> None:
        if spec is None:
            self.phase_bars.setOpts(x=[], height=[])
            self.amp_bars.setOpts(x=[], height=[])
            set_title(self.phase, message or "没有测角谱")
            set_title(self.amp, "")
            return
        index = np.arange(len(spec["phase_deg"]))
        n = len(index)
        self.phase_bars.setOpts(
            x=index, y0=0.0, height=spec["phase_deg"], brush=pg.mkBrush(theme.SIGNAL)
        )
        # 幅度是相对最强通道的负 dB:从 0 往下画,-inf(零幅度通道)压在量程下沿
        amplitude = np.maximum(np.nan_to_num(spec["amplitude_db"], neginf=-20.0), -20.0)
        self.amp_bars.setOpts(
            x=index, y0=0.0, height=amplitude, brush=pg.mkBrush(theme.DETECTION)
        )
        for plot in (self.phase, self.amp):
            plot.setXRange(-0.7, n - 0.3, padding=0)
        set_title(self.phase, f"通道相位残差 · {n} 路")
        set_title(self.amp, "幅度(相对最强通道)")


class NisBars(MiniPlot):
    """跟踪:本帧各航迹的 NIS(柱),选中的检测所属航迹为青色;两条判据线。"""

    def __init__(self, height: int = 112) -> None:
        super().__init__()
        self.widget, self.plot = mini_plot(height)
        self.plot.setMouseEnabled(False, False)
        self.bars = pg.BarGraphItem(x=[], height=[], width=0.6)
        self.plot.addItem(self.bars)
        self.lines: list[pg.InfiniteLine] = []
        _row(self, self.widget)

    def update_detection(self, frame, k: int | None) -> None:
        table = track_table(frame)
        if table is None or not len(table["id"]):
            self.bars.setOpts(x=[], height=[])
            self._clear_lines()
            set_title(self.plot, "该帧没有跟踪中间量")
            return
        criteria = frame.diagnostics.verdicts.criteria
        nis = np.where(np.isfinite(table["nis"]), table["nis"], 0.0)
        own = -1
        if k is not None:
            own = int(frame.diagnostics.metrics.track_id[k])
        brushes = [
            theme.ACCENT if int(tid) == own else theme.DIM for tid in table["id"]
        ]
        self.bars.setOpts(x=np.arange(len(nis)), y0=0.0, height=nis, brushes=brushes)
        self.plot.setXRange(-0.6, len(nis) - 0.4, padding=0)
        self._clear_lines()
        for value, color in (
            (criteria.nis_warn, theme.LEVEL_COLORS[1]),
            (criteria.nis_bad, theme.LEVEL_COLORS[2]),
        ):
            line = pg.InfiniteLine(
                pos=value,
                angle=0,
                movable=False,
                pen=theme.pen(color, 1, Qt.PenStyle.DashLine),
            )
            self.plot.addItem(line, ignoreBounds=True)
            self.lines.append(line)
        set_title(self.plot, f"NIS(本帧 {len(nis)} 条航迹;虚线 = 判据)")

    def _clear_lines(self) -> None:
        for line in self.lines:
            self.plot.removeItem(line)
        self.lines = []


class GateMap(MiniPlot):
    """跟踪:量测空间(方位 × 距离)里的检测点与选中航迹的矩形门。"""

    def __init__(self, height: int = 140) -> None:
        super().__init__()
        self.widget, self.plot = mini_plot(height)
        self.plot.setLabel("bottom", "方位", units="°")
        self.plot.setLabel("left", "距离", units="m")
        self.points = pg.ScatterPlotItem(
            size=5, pen=None, brush=pg.mkBrush(theme.DETECTION)
        )
        self.gate = pg.PlotDataItem(pen=theme.pen(theme.NOISE, 1.2))
        self.plot.addItem(self.points)
        self.plot.addItem(self.gate)
        _row(self, self.widget)

    def update_detection(self, frame, k: int | None) -> None:
        det = frame.detections
        self.points.setData(x=det.azimuth_deg, y=det.range_m)
        data = gates(frame)
        own = -1
        if k is not None and frame.diagnostics is not None:
            own = int(frame.diagnostics.metrics.track_id[k])
        if data is None or own < 0 or own not in data["id"]:
            self.gate.setData([], [])
            set_title(self.plot, "选中点未关联到航迹(或该帧无跟踪中间量)")
            return
        row = int(np.flatnonzero(data["id"] == own)[0])
        center, half = data["center"][row], data["half_width"][row]
        rng, az = center[0], center[1]  # 量测空间:[距离 m, 方位 °, 俯仰 °, 径向速度]
        range_half, az_half = half[0], half[1]
        if not (np.isfinite(rng) and np.isfinite(az)):
            self.gate.setData([], [])
            set_title(self.plot, f"航迹 {own}:本帧新建,没有预测量测")
            return
        x = [az - az_half, az + az_half, az + az_half, az - az_half, az - az_half]
        y = [
            rng - range_half,
            rng - range_half,
            rng + range_half,
            rng + range_half,
            rng - range_half,
        ]
        self.gate.setData(x, y)
        set_title(
            self.plot,
            f"航迹 {own} 的门:±{range_half:.1f} m、±{az_half:.2f}°(量测空间)",
        )


class RangeLocal(MiniPlot):
    """距离像:选中检测所在距离门附近 ±span 个 bin 的功率(全帧平均 Σ_rx)。"""

    def __init__(self, height: int = 112, span: int = 16) -> None:
        super().__init__()
        self.span = span
        self.widget, self.plot = mini_plot(height)
        self.plot.setLabel("bottom", "距离", units="m")
        self.curve = self.plot.plot(pen=theme.pen(theme.SIGNAL, 1.2))
        self.cursor = pg.InfiniteLine(
            angle=90,
            movable=False,
            pen=theme.pen(theme.ACCENT, 1, Qt.PenStyle.DashLine),
        )
        self.plot.addItem(self.cursor, ignoreBounds=True)
        _row(self, self.widget)

    def update_detection(self, frame, k: int | None) -> None:
        profiles = range_profiles_db(frame, 0)
        if profiles is None or k is None:
            self.curve.setData([], [])
            set_title(self.plot, "该帧没有距离像中间量")
            return
        axis = range_axis_m(frame.radar)
        r = int(frame.detections.range_bin[k])
        lo, hi = max(0, r - self.span), min(len(axis), r + self.span + 1)
        self.curve.setData(axis[lo:hi], profiles["mean_sum_db"][lo:hi])
        self.cursor.setValue(float(axis[r]))
        set_title(self.plot, f"距离像局部 ±{self.span} bin(全帧平均 Σ_rx)")


class AdcWave(MiniPlot):
    """ADC:选中 chirp 的各 RX 时域波形(实部实线、虚部点线;实采样只有实部)。

    chirp 由 ADC 屏主图的点选决定(`AdcView.chirp_index`),它随 `chirp_changed` 重画。
    波形是帧级的,与选中哪个检测无关。"""

    def __init__(self, view, state, height: int = 112) -> None:
        super().__init__()
        self.view, self.state = view, state
        self.widget, self.plot = mini_plot(height)
        self.plot.setLabel("bottom", "快时间采样")
        self.waves: list[tuple[pg.PlotDataItem, pg.PlotDataItem]] = []
        for i, color in enumerate(theme.RX_COLORS):
            real = self.plot.plot(pen=theme.pen(color, 1.2))
            imag = self.plot.plot(pen=theme.pen(color, 1, Qt.PenStyle.DotLine))
            self.waves.append((real, imag))
        _row(self, self.widget)
        view.chirp_changed.connect(self._on_chirp)

    def _on_chirp(self, _chirp: int) -> None:
        if self.isVisible() and self.state.frame is not None:
            self.update_detection(self.state.frame, None)

    def update_detection(self, frame, k: int | None) -> None:
        waves = adc_waveforms(frame, None, self.view.chirp_index)
        if waves is None:
            for real, imag in self.waves:
                real.clear()
                imag.clear()
            set_title(self.plot, "该帧没有中间量(诊断未开),不适用")
            return
        x, real, imag = waves["sample"], waves["real"], waves["imag"]
        for i, (real_curve, imag_curve) in enumerate(self.waves):
            if i < real.shape[0]:
                real_curve.setData(x, real[i])
                if waves["real_sampling"]:
                    imag_curve.clear()
                else:
                    imag_curve.setData(x, imag[i])
            else:
                real_curve.clear()
                imag_curve.clear()
        kind = "实采样" if waves["real_sampling"] else "复采样(实线 I / 点线 Q)"
        set_title(self.plot, f"chirp {self.view.chirp_index} · 各 RX 波形 · {kind}")
