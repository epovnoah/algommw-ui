"""检测明细表:本帧全部检测的 Point_t 字段;选中一行即在各屏选中该检测(照 ccm
`gui/views/table.py`)。

用模型 / 视图(QTableView + QAbstractTableModel)而不是 QTableWidget:换帧时只重置模型,
文字在换帧时一次格式化好、绘制时查表,只生成可见的行;列宽只在换会话后的第一帧算一次。
各屏的列表用 widgets/stagelist(按级指标排序 + 分级着色),这张全字段表按需取用。
"""

import numpy as np
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PySide6.QtWidgets import QAbstractItemView, QHeaderView, QTableView, QWidget

from ..state import Selection, ViewState
from .base import ResultView
from .common import vbox

#: (表头, 取值函数 (Detections, 行) → 数, 格式)
COLUMNS = [
    ("#", lambda d, k: k, "{:d}"),
    ("距离 bin", lambda d, k: int(d.range_bin[k]), "{:d}"),
    ("多普勒 bin", lambda d, k: int(d.doppler_bin[k]), "{:d}"),
    ("距离 m", lambda d, k: float(d.range_m[k]), "{:.2f}"),
    ("速度 m/s", lambda d, k: float(d.velocity_mps[k]), "{:+.3f}"),
    # 双 PRF:已配对 / 单帧(速度为单帧折叠值,高速目标会错)/ 救回,附展开周期;其余板为 —
    ("配对", lambda d, k: d.pair_label(k), "{}"),
    ("SNR dB", lambda d, k: float(d.snr_db[k]), "{:.1f}"),
    ("噪声 dB", lambda d, k: float(d.noise_db[k]), "{:.1f}"),
    ("方位 °", lambda d, k: float(d.azimuth_deg[k]), "{:+.2f}"),
    ("俯仰 °", lambda d, k: float(d.elevation_deg[k]), "{:+.2f}"),
    ("x m", lambda d, k: float(d.xyz[k, 0]), "{:+.2f}"),
    ("y m", lambda d, k: float(d.xyz[k, 1]), "{:.2f}"),
    ("z m", lambda d, k: float(d.xyz[k, 2]), "{:+.2f}"),
]
_ALIGN = int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)


def _text(value, fmt: str) -> str:
    if isinstance(value, float) and not np.isfinite(value):
        return "—"
    return fmt.format(value)


class DetectionTableModel(QAbstractTableModel):
    def __init__(self) -> None:
        super().__init__()
        #: 换帧时一次格式化好;data() 每帧被调用数千次,只做查表
        self.text: list[list[str]] = []

    def set_frame(self, frame) -> None:
        self.beginResetModel()
        self.text = []
        if frame is not None:
            det = frame.detections
            self.text = [
                [_text(get(det, k), fmt) for _, get, fmt in COLUMNS]
                for k in range(len(det))
            ]
        self.endResetModel()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.text)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(COLUMNS)

    def headerData(
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ):
        if (
            role == Qt.ItemDataRole.DisplayRole
            and orientation == Qt.Orientation.Horizontal
        ):
            return COLUMNS[section][0]
        return None

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        if role == Qt.ItemDataRole.DisplayRole:
            return self.text[index.row()][index.column()]
        if role == Qt.ItemDataRole.TextAlignmentRole:
            return _ALIGN
        return None


class DetectionTableView(ResultView):
    title = "检测明细"

    def __init__(self, state: ViewState, parent: QWidget | None = None) -> None:
        super().__init__(state, parent)
        self.model = DetectionTableModel()
        self.table = QTableView()
        self.table.setModel(self.model)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(22)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Interactive
        )
        self.table.selectionModel().selectionChanged.connect(self._on_row)
        self._columns_sized = False
        vbox(self, self.table)

    def reset(self, source) -> None:
        self.model.set_frame(None)
        self._columns_sized = False

    def render_frame(self, frame) -> None:
        self.table.selectionModel().blockSignals(True)
        self.model.set_frame(frame)
        self.table.selectionModel().blockSignals(False)
        if not self._columns_sized and self.model.rowCount():
            self.table.resizeColumnsToContents()  # 只在换会话后的第一帧按内容定一次列宽
            self._columns_sized = True

    def render_selection(self, frame, selection: Selection) -> None:
        selection_model = self.table.selectionModel()
        selection_model.blockSignals(True)
        k = selection.detection
        if k is None or k >= self.model.rowCount():
            self.table.clearSelection()
        else:
            self.table.selectRow(k)
            self.table.scrollTo(self.model.index(k, 0))
        selection_model.blockSignals(False)
        self.table.viewport().update()

    def _on_row(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if rows:
            self.state.select_detection(rows[0].row())
