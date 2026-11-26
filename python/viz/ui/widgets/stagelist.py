"""阶段列表:只列本级相关的列,按本级指标排序(最差在前);选中一行即在各屏选中该检测
(照 ccm `gui/widgets/stage_list.py`)。住在屏的列表抽屉里(默认收起,点「检测列表」展开)。

列是声明:Column(表头, 取值函数, 格式, 等级函数)。取值与等级函数的参数是 (帧, 检测序号),
数据从帧与其诊断结果取,列表本身不做计算。模型 / 视图实现,换帧时一次格式化好全部文字,
只生成可见的行。
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QAbstractTableModel, QModelIndex, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QStyledItemDelegate,
    QTableView,
    QWidget,
)

from ...diagnostics import STAGES
from .. import theme
from ..state import Selection, ViewState
from ..views.base import ResultView
from ..views.common import vbox

_RIGHT = int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
_LEFT = int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
ROLE_BAR = int(Qt.ItemDataRole.UserRole) + 1  # 行内条形的填充比例 0~1(无则 None)
ROLE_LEVEL = int(Qt.ItemDataRole.UserRole) + 2  # 本行最差等级(无等级列时 None)
ROW_HEIGHT = 26
_SHAPE_STRIP = 20  # 首列左侧留给严重度形状的宽度
_BAR_W, _BAR_H, _BAR_GAP = 44, 4, 8


@dataclass(frozen=True)
class Column:
    header: str
    value: Callable[[object, int], object]  # 返回 None 显示「—」
    fmt: str = "{}"
    level: Callable[[object, int], int] | None = None  # diagnostics.Level,>=1 时着色
    align_left: bool = False
    bar: tuple[float, float] | None = None  # 值域 (下, 上):给了就在数字右边画行内条形

    def text(self, frame, k: int) -> str:
        value = self.value(frame, k)
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            return "—"
        return self.fmt.format(value)

    def fraction(self, frame, k: int) -> float | None:
        """行内条形的填充比例(0~1);没给 bar 或值无效时为 None。"""
        if self.bar is None:
            return None
        value = self.value(frame, k)
        if value is None or not np.isfinite(value):
            return None
        low, high = self.bar
        return float(min(max((value - low) / (high - low), 0.0), 1.0))


class TableModel(QAbstractTableModel):
    def __init__(self, columns: Sequence[Column]) -> None:
        super().__init__()
        self.columns = list(columns)
        self.rows: list[int] = []
        self.text: list[list[str]] = []
        self.colors: list[list[QColor | None]] = []
        self.bars: list[list[float | None]] = []
        self.levels: list[int | None] = []
        #: 有等级列的表才在首列画严重度形状(ADC 各 RX 统计这类表没有「需要看」的概念)
        self.has_levels = any(c.level is not None for c in self.columns)

    def set_rows(self, frame, rows: Sequence[int]) -> None:
        self.beginResetModel()
        self.rows = [int(k) for k in rows] if frame is not None else []
        self.text, self.colors, self.bars, self.levels = [], [], [], []
        for k in self.rows:
            self.text.append([c.text(frame, k) for c in self.columns])
            colors, worst = [], None
            for c in self.columns:
                level = c.level(frame, k) if c.level is not None else 0
                colors.append(QColor(theme.LEVEL_COLORS[level]) if level >= 1 else None)
                if c.level is not None:
                    worst = level if worst is None else max(worst, level)
            self.colors.append(colors)
            self.levels.append(worst)
            self.bars.append([c.fraction(frame, k) for c in self.columns])
        self.endResetModel()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.rows)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.columns)

    def headerData(
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ):
        if orientation != Qt.Orientation.Horizontal:
            return None
        if role == Qt.ItemDataRole.DisplayRole:
            return self.columns[section].header
        if role == Qt.ItemDataRole.TextAlignmentRole:  # 表头与数值同侧对齐
            left = self.columns[section].align_left or section == 0
            return _LEFT if left else _RIGHT
        return None

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        if role == Qt.ItemDataRole.DisplayRole:
            return self.text[index.row()][index.column()]
        if role == Qt.ItemDataRole.TextAlignmentRole:
            left = self.columns[index.column()].align_left or index.column() == 0
            return _LEFT if left else _RIGHT
        if role == Qt.ItemDataRole.ForegroundRole:
            return self.colors[index.row()][index.column()]
        if role == ROLE_BAR:
            return self.bars[index.row()][index.column()]
        if role == ROLE_LEVEL:
            return self.levels[index.row()]
        return None


class RowDelegate(QStyledItemDelegate):
    """首列左侧画严重度形状(● ▲ ■ ○,与颜色双编码);带 bar 的列在数字右边画 4 px 条形。
    文字仍由基类画,这里只收窄它的绘制区,所以选中底色、对齐、前景色都照旧。"""

    def paint(self, painter: QPainter, option, index) -> None:
        bar = index.data(ROLE_BAR)
        level = index.data(ROLE_LEVEL) if index.column() == 0 else None
        shifted = option.__class__(option)
        if level is not None:
            shifted.rect = shifted.rect.adjusted(_SHAPE_STRIP, 0, 0, 0)
        if bar is not None:
            shifted.rect = shifted.rect.adjusted(0, 0, -(_BAR_W + _BAR_GAP + 6), 0)
        super().paint(painter, shifted, index)
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if level is not None:
            painter.setPen(QColor(theme.LEVEL_COLORS[level]))
            font = painter.font()
            if font.pixelSize() > 0:  # 表格样式表按像素定字号
                font.setPixelSize(max(int(font.pixelSize() * 0.8), 8))
            else:
                font.setPointSizeF(font.pointSizeF() * 0.8)
            painter.setFont(font)
            strip = option.rect.adjusted(6, 0, 0, 0)
            strip.setWidth(_SHAPE_STRIP - 6)
            painter.drawText(
                strip, int(Qt.AlignmentFlag.AlignCenter), theme.LEVEL_SHAPES[level]
            )
        if bar is not None:
            x = option.rect.right() - _BAR_W - 6
            y = option.rect.center().y() - _BAR_H / 2
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(theme.LINE))
            painter.drawRoundedRect(QRectF(x, y, _BAR_W, _BAR_H), 2, 2)
            warn = index.data(Qt.ItemDataRole.ForegroundRole)
            painter.setBrush(warn if warn is not None else QColor(theme.DIM))
            painter.drawRoundedRect(QRectF(x, y, _BAR_W * bar, _BAR_H), 2, 2)
        painter.restore()


class FillTable(QTableView):
    """列宽撑满视口、不留右侧空白:先按内容算「自然宽度」,再按比例放大到视口宽。"""

    def __init__(self) -> None:
        super().__init__()
        self._natural: list[int] = []

    def set_natural(self, widths: list[int]) -> None:
        self._natural = widths
        self._fill()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._fill()

    def _fill(self) -> None:
        total = sum(self._natural)
        width = self.viewport().width()
        if not total or width <= 0:
            return
        scale = width / total
        used = 0
        last = len(self._natural) - 1
        for i, natural in enumerate(self._natural):
            w = width - used if i == last else int(natural * scale)
            self.setColumnWidth(i, w)
            used += w


def make_table(model: TableModel) -> FillTable:
    """列表的统一外观与行为:32 px 行、严重度形状 / 行内条形(RowDelegate)、列宽撑满视口。"""
    table = FillTable()
    table.setModel(model)
    table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.verticalHeader().setVisible(False)
    table.verticalHeader().setDefaultSectionSize(ROW_HEIGHT)
    table.setItemDelegate(RowDelegate(table))
    table.setShowGrid(False)
    table.setFrameShape(QAbstractItemView.Shape.NoFrame)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
    table.horizontalHeader().setStretchLastSection(False)
    table.setStyleSheet(f"QTableView {{ font-size: {theme.FONT_NOTE}px; }}")
    return table


def fit_columns(table: FillTable, model: TableModel) -> None:
    """按首批行的内容定自然列宽,再给形状条 / 行内条形 / 左对齐的标签列留出位置,最后撑满。"""
    table.resizeColumnsToContents()
    widths = []
    for i, column in enumerate(model.columns):
        extra = 16
        if i == 0 and model.has_levels:
            extra += _SHAPE_STRIP
        if column.bar is not None:
            extra += _BAR_W + _BAR_GAP
        width = table.columnWidth(i) + extra
        widths.append(max(width, 140) if column.align_left else width)
    table.set_natural(widths)


class StageList(ResultView):
    """order(frame) 返回行序(检测序号数组,None = 检测原序);obey_filter=True 时只列
    ViewState.stage_filter 那一级需要看的检测。行数 / 说明变了发 `caption_changed`,
    屏据此更新「检测列表 N」按钮与抽屉头。"""

    caption_changed = Signal(str, int)  # (抽屉头说明, 行数)

    def __init__(
        self,
        state: ViewState,
        title: str,
        columns: Sequence[Column],
        order: Callable[[object], np.ndarray] | None = None,
        obey_filter: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(state, parent)
        self.title = title
        self.base_title, self.order, self.obey_filter = title, order, obey_filter
        self.model = TableModel(columns)
        self.table = make_table(self.model)
        self.table.selectionModel().selectionChanged.connect(self._on_row)
        vbox(self, self.table)
        self._sized = False
        self.caption = title
        if obey_filter:
            state.filter_changed.connect(self._on_filter)

    def rows_for(self, frame) -> np.ndarray:
        n = len(frame.detections)
        if self.order is not None:
            rows = np.asarray(self.order(frame), int)
        else:
            rows = np.arange(n)
        stage = self.state.stage_filter if self.obey_filter else None
        if stage is not None and frame.diagnostics is not None:
            flagged = frame.diagnostics.verdicts.flagged(stage)
            rows = rows[flagged[rows]]
        return rows

    def reset(self, source) -> None:
        self.model.set_rows(None, [])
        self._sized = False
        self.caption_changed.emit(self.caption, 0)

    def render_frame(self, frame) -> None:
        self._refresh(frame)

    def _on_filter(self, _stage) -> None:
        if self.state.frame is not None:
            self._refresh(self.state.frame)

    def _refresh(self, frame) -> None:
        rows = self.rows_for(frame)
        self.table.selectionModel().blockSignals(True)
        self.model.set_rows(frame, rows)
        self.table.selectionModel().blockSignals(False)
        stage = self.state.stage_filter if self.obey_filter else None
        self.caption = (
            f"{self.base_title} · 只看「{STAGES[stage]}」需要看的"
            if stage is not None
            else self.base_title
        )
        self.caption_changed.emit(self.caption, len(rows))
        if not self._sized and len(rows):
            fit_columns(self.table, self.model)
            self._sized = True
        if self.state.selection is not None:
            self.render_selection(frame, self.state.selection)

    def render_selection(self, frame, selection: Selection) -> None:
        selection_model = self.table.selectionModel()
        selection_model.blockSignals(True)
        if selection.detection is None or selection.detection not in self.model.rows:
            self.table.clearSelection()
        else:
            row = self.model.rows.index(selection.detection)
            self.table.selectRow(row)
            self.table.scrollTo(self.model.index(row, 0))
        selection_model.blockSignals(False)
        self.table.viewport().update()

    def _on_row(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if rows:
            self.state.select_detection(self.model.rows[rows[0].row()])
