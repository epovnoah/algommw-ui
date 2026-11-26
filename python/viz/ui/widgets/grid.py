"""通用列表:行不是「检测」的屏用它(ADC 屏列 RX、跟踪屏列航迹)。

与 widgets/stagelist 共用同一套列声明与模型(`Column` / `TableModel`),差别只在
「行是什么」由调用方给:`rows_for(frame)` 返回行键数组,`Column.value(frame, key)` 用行键
取值,`on_select(key)` 把点选接到 ViewState(航迹 id、通道号……)。`key_of(frame, selection)`
把全局选中映射回行键,用来高亮当前行;返回 None 则不高亮。
"""

from collections.abc import Callable, Sequence

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget

from ..state import ViewState
from ..views.base import ResultView
from ..views.common import vbox
from .stagelist import Column, TableModel, fit_columns, make_table


class GridTable(ResultView):
    caption_changed = Signal(str, int)  # (抽屉头说明, 行数),同 StageList

    def __init__(
        self,
        state: ViewState,
        title: str,
        columns: Sequence[Column],
        rows_for: Callable[[object], np.ndarray],
        on_select: Callable[[int], None] | None = None,
        key_of: Callable[[object, object], int | None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(state, parent)
        self.title = title
        self.rows_for, self.on_select, self.key_of = rows_for, on_select, key_of
        self.model = TableModel(columns)
        self.table = make_table(self.model)
        self.table.selectionModel().selectionChanged.connect(self._on_row)
        vbox(self, self.table)
        self._sized = False
        self.caption = title

    def reset(self, source) -> None:
        self.model.set_rows(None, [])
        self._sized = False
        self.caption_changed.emit(self.caption, 0)

    def render_frame(self, frame) -> None:
        rows = np.asarray(self.rows_for(frame), int)
        self.table.selectionModel().blockSignals(True)
        self.model.set_rows(frame, rows)
        self.table.selectionModel().blockSignals(False)
        self.caption_changed.emit(self.caption, len(rows))
        if not self._sized and len(rows):
            fit_columns(self.table, self.model)
            self._sized = True
        if self.state.selection is not None and self.key_of is not None:
            self.render_selection(frame, self.state.selection)

    def render_selection(self, frame, selection) -> None:
        key = self.key_of(frame, selection) if self.key_of is not None else None
        selection_model = self.table.selectionModel()
        selection_model.blockSignals(True)
        if key is None or key not in self.model.rows:
            self.table.clearSelection()
        else:
            row = self.model.rows.index(key)
            self.table.selectRow(row)
            self.table.scrollTo(self.model.index(row, 0))
        selection_model.blockSignals(False)
        self.table.viewport().update()

    def _on_row(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if rows and self.on_select is not None:
            self.on_select(self.model.rows[rows[0].row()])
