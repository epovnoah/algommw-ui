"""DDM 平面:解调后的非相干探测矩阵,以及解调前 4 个 TX 副本落在哪个子带。

上方主图有两种口径,下拉切换:

- **解调后(非相干和)**:距离 × 带内多普勒 bin = 每格取胜出 fold 后的 Σ_t Σ_rx |X|²,
  就是 core 写进 RdMap、CFAR 消费的那张矩阵(与 rd_db 同源)。一个目标只出现一次,
  速度要按该格的 fold 展开(全局 bin = fold·perSub + D),读数里直接给展开后的 m/s。
- **原始(解调前)**:距离 × 全局慢时间 bin(子带 × 带内 bin,afm761 = 6 × 128 = 768),
  竖线标出选中检测的 4 个 TX 副本 bin(TX 色)与空子带 bin(灰);这里能看到同一个目标的
  多个副本各占一条。

TDM 板没有这些量,视图显示一句说明。
"""

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget

from ...diagnostics.slices import (
    ddm_raw_matrix_db,
    ddm_tx_bins,
    doppler_coord,
    fold_map,
    rd_db,
)
from .. import theme
from ..state import ViewState
from ..views.base import ResultView, set_title
from ..views.common import (
    heat_item,
    hover_cell,
    levels_note,
    readout_label,
    show_heat,
    vbox,
)
from ..widgets.seg import Segmented


class DdmMatrixView(ResultView):
    title = "DDM 矩阵"

    def __init__(self, state: ViewState, parent: QWidget | None = None) -> None:
        super().__init__(state, parent)
        self.widget = pg.PlotWidget()
        self.plot = theme.style_plot(self.widget.getPlotItem())
        self.plot.setLabel("bottom", "全局慢时间 bin(子带 × 子带内 bin)")
        self.plot.setLabel("left", "距离", units="m")
        self.image = heat_item(self.plot)
        self.mode = Segmented(["解调后", "原始"])
        self.mode.setToolTip(
            "解调后 = 每格取胜出 fold 的 Σ_t Σ_rx|X|²,即 CFAR 的输入(距离 × 带内多普勒 bin);\n"
            "原始 = 解调前的全局慢时间 bin(距离 × 子带 × 带内 bin),同一目标的多个 TX 副本各占一条"
        )
        self.mode.currentIndexChanged.connect(self._on_mode)
        #: 子带分隔线与 TX / 空子带标记,只在换会话或换选中时重建
        self.lines: list[pg.InfiniteLine] = []
        self.readout = readout_label()
        self.controls = [self.mode]

        vbox(self, self.widget)
        self.widget.scene().sigMouseClicked.connect(self._on_click)
        self.widget.scene().sigMouseMoved.connect(self._on_move)

        self._rect = None
        self._axes = None
        self._data = None
        self._frame = None
        self._hover = None
        self.demod = None  #: 解调后(非相干和):距离 × 带内多普勒 bin
        self.raw = None  #: 解调前:距离 × 全局慢时间 bin
        self._folds = None  #: 每格的胜出 fold(读数里把 bin 展开成速度)
        self._demod = True  #: 当前主图画的是哪一种口径

    # ---- ResultView ----
    def reset(self, source) -> None:
        self.image.clear()
        self._clear_lines()
        self._rect = self._axes = self._data = self._hover = None
        self._frame = self.demod = self.raw = self._folds = None
        self.readout.clear()

    def render_frame(self, frame) -> None:
        self._frame = frame
        self.demod = rd_db(frame)
        self.raw = ddm_raw_matrix_db(frame)
        self._folds = fold_map(frame)
        if self.demod is None or self.raw is None:
            self.image.clear()
            self._clear_lines()
            self._data = None
            set_title(self.plot, "TDM 板没有 DDM 矩阵,不适用")
            return
        self._show_mode()

    def _on_mode(self, _index: int) -> None:
        """换口径只重画:两张矩阵已备好,色阶每帧按数据重算。"""
        self._show_mode()

    def _show_mode(self) -> None:
        """按当前口径重画主图 —— 只重画:两张矩阵在 render_frame 里已经备好。"""
        if self._frame is None or self.demod is None or self.raw is None:
            return
        raw_mode = self.mode.currentIndex() == 1
        self._demod = not raw_mode
        data = self.raw if raw_mode else self.demod
        image = data["image"]
        self._data = data
        levels, rect, axes = show_heat(
            self.image,
            image,
            data["bin"] if raw_mode else data["doppler"],
            data["range_m"],
        )
        # 两种口径的横轴跨度差 ns 倍(128 vs 768 bin):换口径要重新适配视野,同口径内不动
        span_changed = (
            self._rect is None
            or abs(rect.left() - self._rect.left()) > 1e-6
            or abs(rect.width() - self._rect.width()) > 1e-6
        )
        self._rect = rect
        self._axes = axes
        self._clear_lines()
        if raw_mode:
            self.plot.setLabel("bottom", "全局慢时间 bin(子带 × 子带内 bin)")
            self._render_subband_lines(data)
            set_title(
                self.plot,
                f"DDM 原始矩阵(解调前):{data['n_sub']} 子带 × {data['n_per_sub']} bin"
                f" = {image.shape[1]} 全局慢时间 bin" + levels_note(levels),
            )
        else:
            self.plot.setLabel("bottom", "多普勒 bin(零频居中;速度按 fold 展开)")
            set_title(
                self.plot,
                f"DDM 解调后非相干矩阵:距离 × {image.shape[1]} 带内多普勒 bin"
                "(每格取胜出 fold 的 Σ_t Σ_rx|X|²,即 CFAR 的输入)"
                + levels_note(levels),
            )
        if span_changed:
            self.plot.autoRange(padding=0.01)
        self._render_selection_lines(self._frame)

    def render_selection(self, frame, selection) -> None:
        if self._data is None:
            return
        self._render_selection_lines(frame)

    # ---- 绘制 ----
    def _render_subband_lines(self, data) -> None:
        self._clear_lines()
        n_per, n_sub = int(data["n_per_sub"]), int(data["n_sub"])
        for k in range(1, n_sub):
            line = pg.InfiniteLine(
                pos=k * n_per - 0.5,
                angle=90,
                movable=False,
                pen=theme.pen(theme.LINE, 1),
            )
            self.plot.addItem(line, ignoreBounds=True)
            self.lines.append(line)

    def _render_selection_lines(self, frame) -> None:
        """选中检测的标记:解调后口径标它的带内 bin;原始口径标 4 个 TX 副本与空子带 bin。"""
        for line in self.lines:
            if getattr(line, "_algommw_tx", False):
                self.plot.removeItem(line)
        self.lines = [ln for ln in self.lines if not getattr(ln, "_algommw_tx", False)]
        if self.state.selection is None or self.state.selection.detection is None:
            return
        if self._demod:
            line = pg.InfiniteLine(
                pos=doppler_coord(frame.radar, self.state.selection.doppler_bin),
                angle=90,
                movable=False,
                pen=theme.pen(theme.ACCENT, 1, Qt.PenStyle.DashLine),
            )
            line._algommw_tx = True
            self.plot.addItem(line, ignoreBounds=True)
            self.lines.append(line)
            return
        bins = ddm_tx_bins(frame, self.state.selection.detection)
        if bins is None:
            return
        for i, value in enumerate(bins["tx_bins"]):
            color = theme.TX_COLORS[i % len(theme.TX_COLORS)]
            line = pg.InfiniteLine(
                pos=float(value), angle=90, movable=False, pen=theme.pen(color, 1.4)
            )
            line._algommw_tx = True
            self.plot.addItem(line, ignoreBounds=True)
            self.lines.append(line)
        for value in bins["empty_bins"]:
            line = pg.InfiniteLine(
                pos=float(value),
                angle=90,
                movable=False,
                pen=theme.pen(theme.EMPTY_BAND_COLOR, 1, Qt.PenStyle.DotLine),
            )
            line._algommw_tx = True
            self.plot.addItem(line, ignoreBounds=True)
            self.lines.append(line)

    def _clear_lines(self) -> None:
        for line in self.lines:
            self.plot.removeItem(line)
        self.lines = []

    # ---- 交互 ----
    def _column_bin(self, col: int) -> int:
        """图像的列 → 无符号多普勒 bin。解调后是零频居中的列序,原始是全局 bin(取带内)。"""
        n_per = int(self.raw["n_per_sub"]) if self.raw else 1
        if self._demod:
            return (col + n_per // 2) % n_per
        return col % n_per

    def _on_click(self, event) -> None:
        """点主图:选中该距离门与该列对应的多普勒 bin(解调后取带内 bin,原始取后 n_per 位)。"""
        if self._rect is None or self._data is None:
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        cell = hover_cell(self.plot, event.scenePos(), self._rect, self._axes)
        if cell is None:
            return
        row, col = cell
        self.state.select_cell(row, self._column_bin(col))

    def _on_move(self, event) -> None:
        if self._rect is None or self._data is None:
            return
        cell = hover_cell(self.plot, event, self._rect, self._axes)
        if cell is None:
            self._hover = None
            self.readout.clear()
            return
        if cell == self._hover:
            return
        self._hover = cell
        row, col = cell
        data = self._data
        assert data is not None
        level = float(data["image"][row, col])
        range_m = float(data["range_m"][row])
        doppler_bin = self._column_bin(col)
        radians = self._global_bin(row, col, doppler_bin)
        if radians is None:
            self.readout.setText(f"{range_m:.2f} m　bin {col}　{level:+.1f} dB")
            return
        global_bin, fold, velocity = radians
        self.readout.setText(
            f"{range_m:.2f} m　bin {doppler_bin}　fold {fold} → 全局 {global_bin}"
            f"　{velocity:+.2f} m/s　{level:+.1f} dB"
        )

    def _global_bin(self, row: int, col: int, doppler_bin: int):
        """(行, 显示列, 带内 bin) → (全局 bin, fold, 速度 m/s)。fold_map 的列已是显示列序。"""
        frame, folds = self._frame, self._folds
        if not self._demod or frame is None or folds is None:
            return None
        radar = frame.radar
        n_per = int(self.raw["n_per_sub"]) if self.raw else int(radar.n_doppler)
        n_fft = n_per * int(self.raw["n_sub"]) if self.raw else n_per
        fold = int(folds[row, col])
        global_bin = fold * n_per + doppler_bin
        signed = global_bin if global_bin < n_fft // 2 else global_bin - n_fft
        return global_bin, fold, signed * float(radar.doppler_step_mps)
