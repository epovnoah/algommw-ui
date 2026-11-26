"""跟踪屏的两个视图:俯视地图 `TrackView` 与航迹栏 `TrackLanes`。

`TrackView`:世界系俯视(x-y),检测点按所属航迹着色,未关联的点画暗色;航迹是空心方块 + id,
近 N 帧轨迹线取自本视图自己累积的历史(不依赖播放器缓存)。门限在量测空间(距离 / 方位),
世界系里画会跑偏,所以放在溯源页跟踪一级的 GateMap 小图里。
`TrackLanes`:左侧常驻的航迹栏,每条航迹一行 —— id(与地图方块同色)、状态徽标(新建 / 确认)、
龄期(帧数)、NIS 小条;按 NIS 降序,点一行 = 选中那条航迹(与地图联动)。
"""

from collections import defaultdict, deque

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ...diagnostics.slices import track_table
from .. import theme
from ..state import Selection, ViewState
from ..views.base import ResultView, set_title
from ..views.common import ClipLabel

#: 轨迹线保留的帧数
TRAIL_FRAMES = 30
DIM = 0.35


def track_color(track_id: int) -> pg.mkColor:
    """航迹 id 的固定颜色(同一 id 各帧同色,便于肉眼跟)。"""
    return pg.intColor(int(track_id) % 12, hues=12, values=1, maxValue=230)


class TrackView(ResultView):
    title = "航迹俯视"

    def __init__(self, state: ViewState, parent: QWidget | None = None) -> None:
        super().__init__(state, parent)
        self.widget = pg.PlotWidget()
        self.plot = theme.style_plot(self.widget.getPlotItem())
        self.plot.setLabel("bottom", "x", units="m")
        self.plot.setLabel("left", "y", units="m")
        self.plot.setAspectLocked(True)
        self.points = pg.ScatterPlotItem(size=7, pen=None)
        self.marks = pg.ScatterPlotItem(
            symbol="s", size=13, pen=theme.pen(theme.TRACK, 1.3), brush=None
        )
        self.selected_point = pg.ScatterPlotItem(
            symbol="o", size=15, pen=theme.pen(theme.ACCENT, 2), brush=None
        )
        self.selected_track = pg.ScatterPlotItem(
            symbol="s", size=22, pen=theme.pen(theme.ACCENT, 2), brush=None
        )
        for item, z in (
            (self.points, 5),
            (self.marks, 6),
            (self.selected_point, 7),
            (self.selected_track, 8),
        ):
            item.setZValue(z)
            self.plot.addItem(item)
        self.labels: list[pg.TextItem] = []
        self.trail_curves: dict[int, pg.PlotDataItem] = {}
        # 检测 / 航迹计数:压在图内左上角、标题小标签下面的小标签,不占布局
        self.hint = ClipLabel(self)
        self.hint.setObjectName("TrackHint")
        self.hint.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.hint.setStyleSheet(
            f"#TrackHint {{ background: rgba(12, 14, 18, 190); color: {theme.TEXT}; "
            f"border: 1px solid {theme.LINE}; border-radius: 5px; padding: 1px 7px; "
            f"font-size: {theme.FONT_NOTE}px; }}"
        )
        self.hint.hide()
        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.addWidget(self.widget)
        self._trails: dict[int, deque] = defaultdict(lambda: deque(maxlen=TRAIL_FRAMES))
        self.points.sigClicked.connect(self._on_point)
        self.marks.sigClicked.connect(self._on_track)
        state.track_changed.connect(self._on_track_changed)

    def _on_track_changed(self, _track_id) -> None:
        """列表 / 航迹栏里选中航迹:只重画选中框(不可见时记脏,显示时补画)。"""
        if self.isVisible() and self.state.frame is not None:
            self.render_selection(self.state.frame, self.state.selection)
        else:
            self._dirty = True

    def _place_hint(self) -> None:
        left = self.plot.getAxis("left")
        x = int(left.width()) + 8 if left.isVisible() else 8
        self.hint.adjustSize()
        self.hint.move(x, 6 + theme.PLOT_TITLE_HEIGHT + 4)
        self.hint.raise_()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if not self.hint.isHidden():
            self._place_hint()

    # ---- ResultView ----
    def reset(self, source) -> None:
        self._trails.clear()
        for curve in self.trail_curves.values():
            self.plot.removeItem(curve)
        self.trail_curves = {}
        for item in (self.points, self.marks, self.selected_point, self.selected_track):
            item.clear()
        for label in self.labels:
            self.plot.removeItem(label)
        self.labels = []
        self.hint.hide()

    def accumulate(self, frame) -> None:
        """每帧记下各航迹的世界系位置,画近 N 帧轨迹用(与是否可见无关)。"""
        tracks = frame.tracks
        for track_id, xyz in zip(tracks.track_id, tracks.xyz):
            self._trails[int(track_id)].append((float(xyz[0]), float(xyz[1])))

    def render_frame(self, frame) -> None:
        det, tracks = frame.detections, frame.tracks
        ids = _point_track_ids(frame)
        brushes = []
        for track_id in ids:
            if track_id < 0:
                color = pg.mkColor(theme.DETECTION)
                color.setAlphaF(DIM)
            else:
                color = track_color(track_id)
            brushes.append(color)
        self.points.setData(
            x=det.xyz[:, 0],
            y=det.xyz[:, 1],
            brush=brushes,
            data=np.arange(len(det)),
        )
        self.marks.setData(
            x=tracks.xyz[:, 0],
            y=tracks.xyz[:, 1],
            data=np.asarray(tracks.track_id, int),
        )
        self._render_labels(tracks)
        self._render_trails()
        self.hint.setText(f"检测 {len(det)} · 航迹 {len(tracks)}")
        self.hint.show()
        self._place_hint()
        set_title(self.plot, f"俯视 x–y(世界系)· 近 {TRAIL_FRAMES} 帧")

    def render_selection(self, frame, selection: Selection | None) -> None:
        det = frame.detections
        if selection is None or selection.detection is None:
            self.selected_point.clear()
        else:
            k = selection.detection
            self.selected_point.setData(x=[det.xyz[k, 0]], y=[det.xyz[k, 1]])
        track_id = self.state.track
        if track_id is None:
            self.selected_track.clear()
            return
        hit = np.flatnonzero(frame.tracks.track_id == track_id)
        if len(hit):
            self.selected_track.setData(
                x=[frame.tracks.xyz[hit[0], 0]], y=[frame.tracks.xyz[hit[0], 1]]
            )
        else:  # 航迹本帧不在输出里(已终止):只留轨迹线
            self.selected_track.clear()

    # ---- 绘制 ----
    def _render_labels(self, tracks) -> None:
        for label in self.labels:
            self.plot.removeItem(label)
        self.labels = []
        for track_id, xyz in zip(tracks.track_id, tracks.xyz):
            label = pg.TextItem(
                str(int(track_id)), color=theme.TRACK, anchor=(-0.3, -0.3)
            )
            label.setPos(float(xyz[0]), float(xyz[1]))
            self.plot.addItem(label, ignoreBounds=True)
            self.labels.append(label)

    def _render_trails(self) -> None:
        live = set()
        for track_id, points in self._trails.items():
            curve = self.trail_curves.get(track_id)
            if curve is None:
                color = track_color(track_id)
                color.setAlphaF(0.55)
                curve = self.plot.plot(pen=pg.mkPen(color, width=1))
                self.trail_curves[track_id] = curve
            curve.setData(x=[p[0] for p in points], y=[p[1] for p in points])
            live.add(track_id)
        for track_id in set(self.trail_curves) - live:
            self.plot.removeItem(self.trail_curves.pop(track_id))

    # ---- 交互 ----
    def _on_point(self, _item, points, _event) -> None:
        if len(points):
            self.state.select_detection(int(points[0].data()))

    def _on_track(self, _item, points, _event) -> None:
        if len(points):
            self.state.select_track(int(points[0].data()))


def _point_track_ids(frame) -> np.ndarray:
    """最终检出点 → 航迹 id(-1 = 未关联);没有跟踪中间量时全为 -1。"""
    products = frame.products
    if products is None or "track" not in products:
        return np.full(len(frame.detections), -1, np.int32)
    ids = np.asarray(products["track"]["point_track_id"], np.int32)
    if len(ids) != len(frame.detections):
        return np.full(len(frame.detections), -1, np.int32)
    return ids


def track_rows(frame) -> np.ndarray:
    """列表行序:航迹按 NIS 从大到小(NaN 在后),没有中间量时按 id。"""
    table = track_table(frame)
    if table is None:
        return np.arange(len(frame.tracks), dtype=int)
    nis = np.where(np.isfinite(table["nis"]), table["nis"], -np.inf)
    return np.argsort(-nis, kind="stable")


# ---------------------------------------------------------------- 航迹栏
LANE_HEIGHT = 30
#: track_table 的 state:1 = 待确认(新建)、2 = 已确认
_STATE_BADGE = {1: "新建", 2: "确认"}


class _Meter(QWidget):
    """NIS 小条:占宽 = NIS / 判据上限,颜色随等级(正常 / 需要看 / 异常)。"""

    def __init__(self) -> None:
        super().__init__()
        self.fraction, self.color = 0.0, QColor(theme.DIM)
        self.setFixedSize(54, 3)

    def show_value(self, fraction: float, color: str) -> None:
        fraction = 0.0 if not np.isfinite(fraction) else min(max(fraction, 0.0), 1.0)
        if fraction != self.fraction or QColor(color) != self.color:
            self.fraction, self.color = fraction, QColor(color)
            self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(theme.LINE))
        painter.drawRoundedRect(QRectF(self.rect()), 1.5, 1.5)
        painter.setBrush(self.color)
        painter.drawRoundedRect(
            QRectF(0, 0, self.width() * self.fraction, self.height()), 1.5, 1.5
        )


class _Lane(QFrame):
    """航迹栏的一行:#id | 状态徽标 + 龄 | NIS + 小条。点一下发 `chosen(id)`。"""

    chosen = Signal(int)

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("Lane")
        self.setFixedHeight(LANE_HEIGHT)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.track_id = -1
        self.selected = None
        row = QHBoxLayout(self)
        row.setContentsMargins(10, 0, 10, 0)
        row.setSpacing(8)
        self.id_label = QLabel()
        self.id_label.setFixedWidth(38)
        self.badge = QLabel()
        self.age = QLabel()
        self.age.setStyleSheet(
            f"color: {theme.DIM}; font-family: {theme.MONO}; font-size: 10.5px;"
        )
        self.nis = QLabel()
        self.nis.setStyleSheet(
            f"color: {theme.TEXT}; font-family: {theme.MONO}; font-size: 11px;"
        )
        self.meter = _Meter()
        right = QVBoxLayout()
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(2)
        right.addWidget(self.nis, 0, Qt.AlignmentFlag.AlignRight)
        right.addWidget(self.meter, 0, Qt.AlignmentFlag.AlignRight)
        row.addWidget(self.id_label)
        row.addWidget(self.badge)
        row.addWidget(self.age, 1)
        row.addLayout(right)
        self.set_selected(False)

    def set_selected(self, selected: bool) -> None:
        if selected == self.selected:
            return
        self.selected = selected
        fill = theme.ACCENT_BG if selected else "transparent"
        self.setStyleSheet(
            f"#Lane {{ background: {fill}; border: 0; "
            f"border-bottom: 1px solid {theme.LINE_SOFT}; }}"
        )

    def show_track(self, track_id: int, state: int, age: int, nis, meter) -> None:
        self.track_id = track_id
        color = track_color(track_id).name()
        self.id_label.setText(f"#{track_id}")
        self.id_label.setStyleSheet(
            f"color: {color}; font-family: {theme.MONO}; font-size: {theme.FONT_NOTE}px;"
        )
        text = _STATE_BADGE.get(state, "—")
        edge = theme.ACCENT if state == 1 else theme.LINE_HOVER
        tone = theme.ACCENT if state == 1 else theme.DIM
        self.badge.setText(text)
        self.badge.setStyleSheet(
            f"color: {tone}; border: 1px solid {edge}; border-radius: 4px; "
            f"padding: 0 5px; font-size: 10.5px;"
        )
        self.age.setText(f"龄 {age}")
        self.nis.setText("NIS —" if not np.isfinite(nis) else f"NIS {nis:.1f}")
        self.meter.show_value(*meter)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self.track_id >= 0:
            self.chosen.emit(self.track_id)
        super().mousePressEvent(event)


class TrackLanes(ResultView):
    """跟踪屏左侧常驻的航迹栏(按 NIS 降序);点一行选中航迹,选中行高亮。"""

    title = "航迹栏"
    caption = "航迹(按 NIS)"

    def __init__(self, state: ViewState, parent: QWidget | None = None) -> None:
        super().__init__(state, parent)
        head = QHBoxLayout()
        head.setContentsMargins(10, 6, 10, 6)
        name = QLabel("航迹")
        name.setStyleSheet(f"color: {theme.TEXT}; font-weight: 600;")
        order = QLabel("按 NIS 降序")
        order.setStyleSheet(f"color: {theme.DIM}; font-size: {theme.FONT_NOTE}px;")
        head.setSpacing(8)
        head.addWidget(name)
        head.addWidget(order)
        head.addStretch(1)
        self.empty = ClipLabel()
        self.empty.setStyleSheet(
            f"color: {theme.DIM}; font-size: {theme.FONT_NOTE}px; padding: 10px;"
        )
        self.empty.setWordWrap(False)
        self.lanes: list[_Lane] = []
        body = QWidget()
        self.column = QVBoxLayout(body)
        self.column.setContentsMargins(0, 0, 0, 0)
        self.column.setSpacing(0)
        self.column.addWidget(self.empty)
        self.column.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(body)
        note = ClipLabel("暗色点未关联;方块 = 航迹,颜色同 id")
        note.setStyleSheet(f"color: {theme.DIM}; font-size: 11.5px; padding: 8px 10px;")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addLayout(head)
        outer.addWidget(scroll, 1)
        outer.addWidget(note)
        state.track_changed.connect(self._on_track_changed)

    def rows_for(self, frame) -> np.ndarray:
        """行键 = track_table 的行号,行序按 NIS 降序(屏据此数航迹)。"""
        return track_rows(frame)

    def reset(self, source) -> None:
        for lane in self.lanes:
            lane.hide()
        self.empty.setText("")
        self.empty.hide()

    def render_frame(self, frame) -> None:
        table = track_table(frame)
        if table is None:
            self._show_rows(frame, None, [])
            return
        self._show_rows(frame, table, list(track_rows(frame)))

    def _show_rows(self, frame, table, rows) -> None:
        diag = frame.diagnostics
        limit = diag.verdicts.criteria.nis_bad if diag is not None else None
        warn = diag.verdicts.criteria.nis_warn if diag is not None else None
        while len(self.lanes) < len(rows):
            lane = _Lane()
            lane.chosen.connect(self.state.select_track)
            self.column.insertWidget(len(self.lanes) + 1, lane)
            self.lanes.append(lane)
        for k, lane in enumerate(self.lanes):
            if k >= len(rows):
                lane.hide()
                continue
            r = int(rows[k])
            nis = float(table["nis"][r])
            scale = limit if limit else 10.0
            level = 0
            if warn and np.isfinite(nis):
                level = 2 if limit and nis >= limit else 1 if nis >= warn else 0
            lane.show_track(
                int(table["id"][r]),
                int(table["state"][r]),
                int(table["age"][r]),
                nis,
                (nis / scale, theme.LEVEL_COLORS[level] if warn else theme.DIM),
            )
            lane.show()
        if table is None:
            self.empty.setText("该帧没有跟踪中间量")
        else:
            self.empty.setText("" if len(rows) else "本帧没有航迹")
        self.empty.setVisible(bool(self.empty.text()))
        self._mark_selected()

    def render_selection(self, frame, selection) -> None:
        self._mark_selected()

    def _on_track_changed(self, _track_id) -> None:
        if self.isVisible():
            self._mark_selected()

    def _mark_selected(self) -> None:
        track_id = self.state.track
        for lane in self.lanes:
            lane.set_selected(not lane.isHidden() and lane.track_id == track_id)
