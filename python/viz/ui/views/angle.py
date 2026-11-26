"""测角谱:选中检测的空间谱峰是否尖锐、相干度多高?

版式「谱图 + 诊断卡」:左列(约 60%)上是过峰的方位切片(大)、下是俯仰切片;右列(约 40%)
上是二维空间谱(AFM761 是直接加权的方位 × 俯仰网格;SR61 是 FFT 后端的归一化空间频率网格),
下是「通道相位残差」(相位与幅度柱状图,右上角标相干度)。探针(`binding.native.DoaProbe`)
用与处理链同一组参数初始化,换会话或改处理参数时重建;没有共享库或测角后端不导出谱时
显示原因而不是报错。corr 只显示水平相关谱,俯仰来自垂直基线,不画俯仰谱与二维谱;
1D 后端(MUSIC)同理没有俯仰维。
"""

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, QSize
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget

from binding import dtypes

from ...diagnostics.slices import doa_spectrum
from .. import theme
from ..plots.mini import ChannelResidual
from ..state import ViewState
from ..views.base import ResultView, set_title
from .common import axis_rect, heat_item
from .panels import Panel

#: 谱的相对电平下限(dB):低于它的画在下限,避免旁瓣把色阶拉平
FLOOR_DB = -20.0


def _name(label: str) -> str:
    """轴名去掉单位尾巴(「方位 °」→「方位」),用在图标题里。"""
    return label.removesuffix(" °")


def make_probe(config):
    """无数据 UI 不创建原生测角探针,保留谱图的空态。"""
    return None


class AngleSpectrumView(ResultView):
    title = "测角谱"

    def __init__(self, state: ViewState, parent: QWidget | None = None) -> None:
        super().__init__(state, parent)
        self.probe = make_probe(state.config)
        state.config_changed.connect(self._on_config)

        # 二维空间谱(右上):色阶固定 FLOOR_DB~0,写在图内标题里,不再带 colorbar
        self.image_widget = pg.PlotWidget()
        self.ip = theme.style_plot(self.image_widget.getPlotItem(), "二维空间谱")
        self.image = heat_item(self.ip)
        self.peak = pg.ScatterPlotItem(
            symbol="x", size=11, pen=theme.pen(theme.DETECTION, 2), brush=None
        )
        self.peak.setZValue(10)
        self.ip.addItem(self.peak)

        # 方位切片(左上,大)
        self.az_widget = pg.PlotWidget()
        self.az_plot = theme.style_plot(self.az_widget.getPlotItem())
        self.az_plot.setLabel("left", "功率", units="dB")
        self.az_plot.setYRange(FLOOR_DB, 0.5, padding=0)
        self.az_curve = self.az_plot.plot(pen=theme.pen(theme.SIGNAL, 1.4))
        self.az_peak = pg.InfiniteLine(
            angle=90, movable=False, pen=theme.pen(theme.DETECTION, 1)
        )
        self.az_plot.addItem(self.az_peak, ignoreBounds=True)

        # 俯仰切片(左下):横轴是俯仰,与方位切片同一读法
        self.el_widget = pg.PlotWidget()
        self.el_plot = theme.style_plot(self.el_widget.getPlotItem())
        self.el_plot.setLabel("left", "功率", units="dB")
        self.el_plot.setYRange(FLOOR_DB, 0.5, padding=0)
        self.el_curve = self.el_plot.plot(pen=theme.pen(theme.SIGNAL, 1.4))
        self.el_peak = pg.InfiniteLine(
            angle=90, movable=False, pen=theme.pen(theme.DETECTION, 1)
        )
        self.el_plot.addItem(self.el_peak, ignoreBounds=True)

        # 通道相位残差(右下),相干度读数压在面板右上
        self.residual = ChannelResidual()

        # 四块面板:左列 方位 7 : 俯仰 3,右列 二维谱 5 : 残差 4;左右 3 : 2
        # (不用 GraphicsLayoutWidget:离屏绘制会崩,每张图仍是独立 PlotWidget)
        self.az_panel = Panel(self.az_widget, "AngleAz")
        self.el_panel = Panel(self.el_widget, "AngleEl")
        self.uv_panel = Panel(self.image_widget, "AngleUv")
        self.res_panel = Panel(self.residual, "AngleRes")
        left, right = QVBoxLayout(), QVBoxLayout()
        for column, panels in (
            (left, ((self.az_panel, 7), (self.el_panel, 3))),
            (right, ((self.uv_panel, 5), (self.res_panel, 4))),
        ):
            column.setSpacing(theme.GAP)
            for panel, stretch in panels:
                column.addWidget(panel, stretch)
        row = self.row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(theme.GAP)
        row.addLayout(left, 3)
        row.addLayout(right, 2)
        set_title(self.az_plot, "方位切片")
        set_title(self.el_plot, "俯仰切片")
        self._grid: tuple | None = None
        self._rect = QRectF()

    def sizeHint(self) -> QSize:
        # 几块 PlotWidget 的 sizeHint 叠起来会比屏里的列表抽屉还高,抽屉展开时按比例分不到空间
        return QSize(640, 420)

    # ---- ResultView ----
    def reset(self, source) -> None:
        self.probe = make_probe(self.state.config)
        self._grid = None
        self._clear("选一个检测看空间谱")

    def _on_config(self, config) -> None:
        self.probe = make_probe(config)
        self._grid = None
        # 参数(换板/换 profile/改配置)变了就只作废已画的谱,不拿新探针去重画旧帧:
        # 换数据源时 `state.frame` 还是上一条源的帧,重画会报形状不符,画出来的也不属于当前源。
        # 新参数后的第一帧到达时(或下次显示本屏时)自然重画。
        self._shown = None

    def render_selection(self, frame, selection) -> None:
        if selection.detection is None:
            self._clear("选一个检测看空间谱(非检测单元没有测角记录)")
            return
        spec = doa_spectrum(frame, selection.detection, self.probe)
        if spec is None:
            reason = (
                "测角探针不可用(共享库缺失或后端不导出谱)"
                if self.probe is None or not getattr(self.probe, "available", False)
                else "该帧没有中间量,或此检测没有测角记录"
            )
            self._clear(reason)
            return
        corr = spec["backend"] == dtypes.DIAG_PROBE_CORR
        one_d = spec["power_db"].shape[0] == 1
        self._set_layout(has_elevation=not corr and not one_d)
        self.residual.show_spec(spec)
        self._set_coherence(frame, spec["coherence"], selection.detection)
        if corr:
            self.az_plot.setXRange(
                float(spec["axis_col"][0]), float(spec["axis_col"][-1]), padding=0
            )
            self.az_plot.setLabel("bottom", spec["col_label"])
            self.az_curve.setData(
                spec["axis_col"], np.maximum(spec["azimuth_cut"], FLOOR_DB)
            )
            self.az_peak.setValue(float(spec["axis_col"][spec["peak_col"]]))
            set_title(
                self.az_plot,
                f"水平相关谱 · 方位 {spec['azimuth_deg']:+.1f}° · "
                f"俯仰 {spec['elevation_deg']:+.1f}°(垂直基线)",
            )
            return
        self._set_grid(spec)
        image = np.maximum(spec["power_db"], FLOOR_DB)
        self.image.setImage(image, autoLevels=False, levels=(FLOOR_DB, 0.0))
        # setRect 必须在 setImage 之后(见 views/adc.py 同处注释)
        self.image.setRect(self._rect)
        r, c = int(spec["peak_row"]), int(spec["peak_col"])
        if spec["backend"] == dtypes.DIAG_PROBE_DIRECT:
            x, y = spec["azimuth_deg"], spec["elevation_deg"]
        else:  # FFT 后端的两轴是归一化空间频率:峰位取网格坐标
            x, y = float(spec["axis_col"][c]), float(spec["axis_row"][r])
        self.peak.setData(x=[x], y=[y])
        self.az_curve.setData(
            spec["axis_col"], np.maximum(spec["azimuth_cut"], FLOOR_DB)
        )
        self.el_curve.setData(
            spec["axis_row"], np.maximum(spec["elevation_cut"], FLOOR_DB)
        )
        self.az_peak.setValue(x)
        self.el_peak.setValue(y)
        set_title(
            self.az_plot,
            f"过峰的{_name(spec['col_label'])}切片 · 方位 {spec['azimuth_deg']:+.1f}°"
            + ("(1D 后端:只有方位一维)" if one_d else ""),
        )
        set_title(
            self.el_plot,
            f"过峰的{_name(spec['row_label'])}切片 · 俯仰 {spec['elevation_deg']:+.1f}°",
        )
        set_title(self.ip, f"二维空间谱 · 色阶 {FLOOR_DB:.0f}~0 dB(相对峰值)")

    # ---- 内部 ----
    def _set_layout(self, has_elevation: bool) -> None:
        # corr 的主图就是一条相关谱,保留基线俯仰的文字读数,不展示虚构的俯仰切片与二维谱;
        # 1D 后端同样没有俯仰维。两块收起后方位切片、通道残差各自占满所在的列。
        self.el_panel.setVisible(has_elevation)
        self.uv_panel.setVisible(has_elevation)

    def _set_coherence(self, frame, coherence: float | None, k: int | None) -> None:
        """相干度读数压在残差面板右上;按判据线着色(与结论、列表同口径)。
        k 为选中检测的下标,用来取诊断里的全阵相干度(与 min_coherence 门限同口径)。"""
        if coherence is None:
            self.res_panel.set_chip("right", "")
            return
        color = theme.TEXT
        if frame.diagnostics is not None:
            criteria = frame.diagnostics.verdicts.criteria
            if coherence < criteria.coherence_bad:
                color = theme.LEVEL_COLORS[2]
            elif coherence < criteria.coherence_warn:
                color = theme.LEVEL_COLORS[1]
        text = (
            f"相干度 <span style='color:{color}; font-family:{theme.MONO}'>"
            f"{coherence:.2f}</span>"
        )
        full = self._full_coherence(frame, k)
        if full is not None:
            text += f" · 全阵 <span style='font-family:{theme.MONO}'>{full:.2f}</span>"
        self.res_panel.set_chip("right", text)

    @staticmethod
    def _full_coherence(frame, k: int | None) -> float | None:
        diag = frame.diagnostics
        if diag is None or k is None:
            return None
        column = diag.metrics.coherence_full
        if not 0 <= k < len(column) or not np.isfinite(column[k]):
            return None
        return float(column[k])

    def _set_grid(self, spec) -> None:
        """网格变了才重设坐标范围(固定量程,坐标轴不必每帧重绘)。"""
        rows, cols = np.asarray(spec["axis_row"]), np.asarray(spec["axis_col"])
        key = (
            float(rows[0]),
            float(rows[-1]),
            len(rows),
            float(cols[0]),
            float(cols[-1]),
            len(cols),
        )
        if self._grid == key:
            return
        self._grid = key
        self._rect = axis_rect(cols, rows)
        self.ip.setRange(
            xRange=(float(cols[0]), float(cols[-1])),
            yRange=(float(rows[0]), float(rows[-1])),
            padding=0,
        )
        self.ip.setLabel("bottom", spec["col_label"])
        self.ip.setLabel("left", spec["row_label"])
        self.az_plot.setXRange(float(cols[0]), float(cols[-1]), padding=0)
        self.az_plot.setLabel("bottom", spec["col_label"])
        self.el_plot.setXRange(float(rows[0]), float(rows[-1]), padding=0)
        self.el_plot.setLabel("bottom", spec["row_label"])

    def _clear(self, title: str) -> None:
        self.image.clear()
        self.peak.clear()
        self.az_curve.setData([], [])
        self.el_curve.setData([], [])
        self.residual.show_spec(None, title)
        self._set_coherence(None, None, None)
        set_title(self.ip, "二维空间谱")
        set_title(self.az_plot, title)
        set_title(self.el_plot, "俯仰切片")
