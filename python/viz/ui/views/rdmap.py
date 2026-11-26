"""RD 平面:本帧的距离-多普勒功率图,叠加检测点与(可选)CFAR 判决 / DDM fold 分区。

多普勒屏(TDM)与 CFAR 屏共用这个视图:标记按 `stage` 那一级的等级着色,色标可拖;
DDM 板还能把每个 RD 单元选中的 fold 画成类别图。点图上任一处即选中那个单元。
"""

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget

from binding import dtypes

from ...diagnostics import STAGES, Level
from ...diagnostics.slices import doppler_coord, fold_map, rd_db
from .. import theme
from ..state import ViewState
from ..views.base import ResultView, set_title
from ..widgets.seg import Segmented
from .common import (
    Crosshair,
    heat_item,
    hover_cell,
    legend_label,
    levels_note,
    nearest_index,
    rd_column,
    readout_label,
    show_heat,
    vbox,
)

OVERLAY_MODES = ("无叠加", "CFAR 判决", "fold 分区")
#: CFAR 叠加的分类 → (RGBA, 图例文字)
CFAR_CLASSES = (
    ((180, 120, 240, 140), "主导多普勒补点候选"),
    (theme.CFAR_BOTH, "两向都过"),
    (theme.CFAR_RANGE_ONLY, "仅距离向"),
    (theme.CFAR_DOPPLER_ONLY, "仅多普勒向"),
    ((95, 102, 112, 90), "skip / FOV 外"),
)


def cfar_overlay_rgba(flags: np.ndarray) -> np.ndarray:
    """CFAR 逐格标志位 → RGBA 叠加图(两向都过 / 仅一向 / skip)。

    某一向关闭时该向视为通过(core 的约定),所以只需看两个 PASS 位;skip 优先。"""
    range_pass = (flags & dtypes.DIAG_CFAR_RANGE_PASS) != 0
    doppler_pass = (flags & dtypes.DIAG_CFAR_DOPPLER_PASS) != 0
    skip = (flags & dtypes.DIAG_CFAR_SKIP) != 0
    rgba = np.zeros(flags.shape + (4,), np.uint8)
    rgba[range_pass & ~doppler_pass] = theme.CFAR_RANGE_ONLY
    rgba[doppler_pass & ~range_pass] = theme.CFAR_DOPPLER_ONLY
    rgba[range_pass & doppler_pass] = theme.CFAR_BOTH
    rgba[(flags & dtypes.DIAG_CFAR_SUPPLEMENT) != 0] = (180, 120, 240, 140)
    rgba[skip] = 0  # 叠加图只标「过门限」的单元,skip 区留空
    return rgba


def fold_colormap(n_sub: int) -> pg.ColorMap:
    """fold 分区图的类别色:子带色循环取,步长与 6 个候选 fold 一一对应。"""
    colors = [
        pg.mkColor(theme.SUBBAND_COLORS[i % len(theme.SUBBAND_COLORS)])
        for i in range(max(n_sub, 1))
    ]
    return pg.ColorMap(np.linspace(0, 1, len(colors)), colors)


class RdMapView(ResultView):
    title = "RD 图"

    def __init__(
        self, state: ViewState, stage: str = "cfar", parent: QWidget | None = None
    ) -> None:
        super().__init__(state, parent)
        self.stage = stage
        self.widget = pg.PlotWidget()
        self.plot = theme.style_plot(self.widget.getPlotItem())
        self.plot.setLabel("bottom", "多普勒")
        self.plot.setLabel("left", "距离", units="m")
        self.image = heat_item(self.plot)
        self.overlay = pg.ImageItem()
        self.overlay.setZValue(5)
        self.plot.addItem(self.overlay)
        self.markers = pg.ScatterPlotItem(
            symbol="x", size=9, pen=theme.pen(theme.DETECTION, 1.6), brush=None
        )
        self.markers.setZValue(11)
        self.plot.addItem(self.markers)
        self.crosshair = Crosshair(self.plot)
        self.widget.scene().sigMouseClicked.connect(self._on_click)
        self.widget.scene().sigMouseMoved.connect(self._on_move)

        self.mode = Segmented(list(OVERLAY_MODES), tip="叠加")
        self.mode.currentIndexChanged.connect(self._on_mode)
        self.legend = legend_label()
        self.readout = readout_label()
        self.controls = [self.mode]
        self.legends = [self.legend]
        vbox(self, self.widget)
        self._rect = None
        self._axes = None
        self._data = None
        self._frame = None
        self._hover = None
        self._pens = {
            level: theme.pen(color if level >= Level.WARN else theme.DETECTION, 1.6)
            for level, color in theme.LEVEL_COLORS.items()
        }
        self._update_legend()

    @property
    def overlay_mode(self) -> str:
        return OVERLAY_MODES[self.mode.currentIndex()]

    # ---- ResultView ----
    def reset(self, source) -> None:
        self.image.clear()
        self.overlay.clear()
        self.markers.clear()
        self._rect = self._axes = self._data = self._hover = None
        self._frame = None
        self.readout.clear()

    def render_frame(self, frame) -> None:
        data = rd_db(frame)
        self._frame = frame
        if data is None:
            self.image.clear()
            self.overlay.clear()
            self.markers.clear()
            set_title(self.plot, "该帧没有中间量(诊断未开),不适用")
            return
        self._data = data
        image, axis = data["image"], data["doppler"]
        levels, self._rect, self._axes = show_heat(
            self.image, image, axis, data["range_m"]
        )
        unit = data["doppler_unit"]
        self.plot.setLabel(
            "bottom", "多普勒速度" if unit == "m/s" else "多普勒 bin(折叠)"
        )
        self._render_markers(frame)
        self._render_overlay(frame)
        n_det = len(frame.detections)
        set_title(
            self.plot,
            f"RD 图(绝对 dB)· 检测 {n_det} · 零频居中"
            + ("· 折叠基带(DDM)" if unit == "bin" else "")
            + levels_note(levels),
        )

    def render_selection(self, frame, selection) -> None:
        if self._data is None:
            return
        self.crosshair.set(
            x=doppler_coord(frame.radar, selection.doppler_bin),
            y=(
                float(frame.detections.range_m[selection.detection])
                if selection.detection is not None
                else selection.range_bin * frame.radar.range_step_m
            ),
        )

    # ---- 绘制 ----
    def _render_markers(self, frame) -> None:
        det = frame.detections
        diag = frame.diagnostics
        if diag is None or not len(det):
            self.markers.clear()
            return
        levels = diag.verdicts.levels[self.stage]
        if np.all(levels == Level.NA):  # 本级不适用(如 TDM 的 fold 判决):按最差的级着色
            levels = np.max(
                np.vstack([diag.verdicts.levels[s] for s in STAGES]), axis=0
            )
        axis = np.asarray(self._data["doppler"])
        columns = rd_column(np.asarray(det.doppler_bin, int), len(axis))
        self.markers.setData(
            x=axis[columns],
            y=det.range_m,
            pen=[self._pens[int(level)] for level in levels],
        )

    def _render_overlay(self, frame) -> None:
        mode = self.overlay_mode
        products = frame.products
        if mode == "CFAR 判决" and products is not None and "cfar" in products:
            flags = products["cfar"]["flags"][: frame.radar.gates][
                :, self._column_order(frame)
            ]
            self.overlay.setImage(
                cfar_overlay_rgba(flags), autoLevels=False, levels=(0, 255)
            )
            self.overlay.setRect(self._rect)
        elif mode == "fold 分区":
            folds = fold_map(frame)
            if folds is None:
                self.overlay.clear()
            else:
                n_sub = (
                    int(products["band_pow"].shape[2])
                    if products is not None and "band_pow" in products
                    else 6
                )
                self.overlay.setImage(
                    folds.astype(np.float32),
                    autoLevels=False,
                    levels=(0, max(n_sub - 1, 1)),
                    lut=fold_colormap(n_sub).getLookupTable(nPts=256),
                )
                self.overlay.setRect(self._rect)
        else:
            self.overlay.clear()

    @staticmethod
    def _column_order(frame) -> np.ndarray:
        """RD 图的多普勒列序(fftshift 后),叠加掩码按同一列序重排。"""
        return np.fft.fftshift(np.arange(frame.radar.n_doppler))

    # ---- 交互 ----
    def _cell_at(self, event) -> tuple[int, int] | None:
        if self._rect is None:
            return None
        return hover_cell(self.plot, event, self._rect, self._axes)

    def _on_click(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        cell = self._cell_at(event.scenePos())
        if cell is not None:
            self.state.select_cell(cell[0], self._doppler_bin(cell[1]))

    def _doppler_bin(self, column: int) -> int:
        """RD 图里的列号 → core 的无符号多普勒 bin。"""
        n = len(self._data["doppler"])
        return (int(column) + n // 2) % n

    def _on_move(self, event) -> None:
        cell = self._cell_at(event)
        if cell is None or self._frame is None or self._data is None:
            self._hover = None
            self.readout.clear()
            return
        if cell == self._hover:
            return
        self._hover = cell
        row, col = cell
        assert self._data is not None
        frame, data = self._frame, self._data
        power = float(data["image"][row, col])
        text = (
            f"{data['range_m'][row]:.2f} m　"
            f"{data['doppler'][col]:+.2f} {data['doppler_unit']}　{power:+.1f} dB"
        )
        products = frame.products
        if products is not None and "cfar" in products:
            flags = int(products["cfar"]["flags"][row, self._column_order(frame)[col]])
            text += "　" + _flags_text(flags)
        if products is not None and "fold" in products:
            text += (
                f"　fold {int(products['fold'][row, self._column_order(frame)[col]])}"
            )
        self.readout.setText(text)

    def _on_mode(self) -> None:
        self._update_legend()
        if self.state.frame is not None:
            self.render_frame(self.state.frame)

    def _update_legend(self) -> None:
        parts = [f"<span style='color:{theme.DETECTION}'>✕ 检测</span>"]
        if self.overlay_mode == "CFAR 判决":
            for rgb, name in CFAR_CLASSES[:3]:
                parts.append(
                    f"<span style='color:rgb({rgb[0]},{rgb[1]},{rgb[2]})'>▮ {name}</span>"
                )
        elif self.overlay_mode == "fold 分区":
            parts.append("▮ 颜色 = 该单元选中的 fold")
        self.legend.setText("　".join(parts))


def _flags_text(flags: int) -> str:
    if flags & dtypes.DIAG_CFAR_SKIP:
        return "skip(不参与)"
    passed = []
    if flags & dtypes.DIAG_CFAR_SUPPLEMENT:
        passed.append("主导多普勒补点候选")
    if flags & dtypes.DIAG_CFAR_RANGE_PASS:
        passed.append("距离向过")
    if flags & dtypes.DIAG_CFAR_DOPPLER_PASS:
        passed.append("多普勒向过")
    return " · ".join(passed) if passed else "未过门限"
