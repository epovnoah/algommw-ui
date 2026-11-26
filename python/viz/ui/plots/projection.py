"""点云的二维投影:俯视(x-y)与侧视(y-z)(照 ccm `gui/plots/projection.py`)。

坐标系与 core 的世界系一致:x 横向、y 雷达正前方、z 高度。两个方向等比例锁定 ——
否则高度方向会按数据自动放大,零点几米的测角误差看起来像大片散布。
检测点按径向速度用 CET-R3 着色(蓝 = 接近、黄 = 静止、红 = 远离;检出点速度远离为正),
航迹画成空心方块 + id 文字;选中检测加青色圈、选中航迹加青色方框;点选检测 / 航迹联动
ViewState。场景屏的主视图是三维点云(`plots/cloud.py`);OpenGL 不可用(`gl_available()`
为假,如离屏测试的 offscreen)时退回这两个投影,悬浮工具条 / 选中卡 / 速度色标与三维版一致。
"""

import numpy as np
import pyqtgraph as pg
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from .. import theme
from ..state import Selection, ViewState
from ..views.base import ResultView
from ..views.common import legend_label
from ..widgets.seg import Segmented
from .scenehud import HudBar, RampLegend, SelectionCard, chip_style, place_hud

#: 速度着色的量程档位(m/s),超出按端色;固定量程让颜色跨帧可比(照 ccm)。
#: ±5 看行人与慢车;双 PRF 展开后的车速常到 10–20 m/s,±5 会把它们全涂成端色。
VELOCITY_LIMITS_MPS = (5.0, 15.0, 30.0)
_DIM_ALPHA = 46  # KPI 条筛选某一级时,其余点的不透明度(0~255)


def gl_available() -> bool:
    """当前 Qt 平台能否创建 OpenGL 视图;offscreen / minimal 平台不支持 QOpenGLWidget。"""
    if QGuiApplication.platformName() in ("offscreen", "minimal"):
        return False
    try:
        import pyqtgraph.opengl  # noqa: F401  依赖 PyOpenGL
    except Exception:  # noqa: BLE001 -- 缺 PyOpenGL 等任何原因都按不可用处理
        return False
    return True


def velocity_colors(velocity: np.ndarray, lut: np.ndarray, limit: float) -> np.ndarray:
    """径向速度(m/s,远离为正)→ (n, 4) uint8 RGBA。负(接近)取色图低端(蓝)。"""
    t = (np.clip(np.nan_to_num(velocity) / limit, -1, 1) + 1) / 2
    return lut[np.round(t * (len(lut) - 1)).astype(int)].copy()


def velocity_limit_seg(state: ViewState) -> Segmented:
    """速度色标档位开关,与 ViewState 双向同步(三维点云与二维投影各放一个)。"""
    seg = Segmented(
        [f"±{v:g}" for v in VELOCITY_LIMITS_MPS],
        tip="速度色标量程 m/s:超出量程的点按端色",
    )
    seg.setCurrentIndex(VELOCITY_LIMITS_MPS.index(state.velocity_limit_mps))
    seg.currentIndexChanged.connect(
        lambda i: state.set_velocity_limit(VELOCITY_LIMITS_MPS[i])
    )
    state.velocity_limit_changed.connect(
        lambda v: seg.setCurrentIndex(VELOCITY_LIMITS_MPS.index(v))
    )
    return seg


class Projection:
    """一个二维投影:雷达位置、检测点、航迹、选中标记;点击检测点 / 航迹即选中它。"""

    def __init__(
        self,
        state: ViewState,
        title: str,
        axes: tuple[int, int],
        labels: tuple[str, str],
        radar_symbol: str,
    ) -> None:
        self.state, self.axes = state, axes
        self.widget = pg.PlotWidget()
        self.plot = theme.style_plot(self.widget.getPlotItem(), title)
        self.plot.setLabel("bottom", labels[0], units="m")
        self.plot.setLabel("left", labels[1], units="m")
        self.plot.setAspectLocked(True)
        radar_mark = pg.ScatterPlotItem(
            x=[0], y=[0], symbol=radar_symbol, size=12, pen=None, brush=theme.TEXT
        )
        self.points = pg.ScatterPlotItem(size=7, pen=None)
        self.tracks = pg.ScatterPlotItem(
            symbol="s", size=13, pen=theme.pen(theme.TRACK, 1.5), brush=None
        )
        self.selected = pg.ScatterPlotItem(
            symbol="o", size=17, pen=theme.pen(theme.ACCENT, 2), brush=None
        )
        self.selected_track = pg.ScatterPlotItem(
            symbol="s", size=21, pen=theme.pen(theme.ACCENT, 2), brush=None
        )
        items = (
            radar_mark,
            self.points,
            self.tracks,
            self.selected,
            self.selected_track,
        )
        for z, item in enumerate(items):
            item.setZValue(10 + z)
            self.plot.addItem(item)
        #: 航迹 id 文字:图元常驻、按需增补,每帧只改文字与位置(不每帧删建)
        self.labels: list[pg.TextItem] = []
        self.points.sigClicked.connect(self._on_point)
        self.tracks.sigClicked.connect(self._on_track)

    def set_points(self, xyz: np.ndarray, brushes: list) -> None:
        a, b = self.axes
        self.points.setData(
            x=xyz[:, a], y=xyz[:, b], brush=brushes, data=np.arange(len(xyz))
        )

    def set_tracks(self, xyz: np.ndarray, track_ids: np.ndarray) -> None:
        a, b = self.axes
        self.tracks.setData(x=xyz[:, a], y=xyz[:, b], data=np.asarray(track_ids))
        while len(self.labels) < len(track_ids):
            label = pg.TextItem(color=theme.TRACK, anchor=(-0.25, 1.1))
            label.setZValue(16)
            self.plot.addItem(label, ignoreBounds=True)
            self.labels.append(label)
        for i, label in enumerate(self.labels):
            if i < len(track_ids):
                label.setText(str(int(track_ids[i])))
                label.setPos(float(xyz[i, a]), float(xyz[i, b]))
            label.setVisible(i < len(track_ids))

    def set_selected(self, xyz: np.ndarray | None) -> None:
        if xyz is None:
            self.selected.clear()
        else:
            self.selected.setData(x=[xyz[self.axes[0]]], y=[xyz[self.axes[1]]])

    def set_selected_track(self, xyz: np.ndarray | None) -> None:
        if xyz is None:
            self.selected_track.clear()
        else:
            self.selected_track.setData(x=[xyz[self.axes[0]]], y=[xyz[self.axes[1]]])

    def clear(self) -> None:
        for item in (self.points, self.tracks, self.selected, self.selected_track):
            item.clear()
        for label in self.labels:
            label.setVisible(False)

    def _on_point(self, _item, points, _event) -> None:
        if len(points):
            self.state.select_detection(int(points[0].data()))

    def _on_track(self, _item, points, _event) -> None:
        if len(points):
            self.state.select_track(int(points[0].data()))


class ProjectionView(ResultView):
    """场景屏主视图的二维版:俯视(x-y)与侧视(y-z)两个投影,分段控件二选一,一次只显示一张。
    notice 非空时在悬浮工具条下面叠一个说明小标签(三维点云退回二维时注明原因)。两张的数据每帧
    都更新(点很少,几乎不花时间),切换时不用补画。画布铺满,控件都是悬浮件。"""

    title = "点云投影"

    def __init__(
        self, state: ViewState, parent: QWidget | None = None, notice: str = ""
    ) -> None:
        super().__init__(state, parent)
        self.lut = pg.colormap.get(theme.VELOCITY_CMAP).map(
            np.linspace(0, 1, 256), mode="byte"
        )
        self.top = Projection(state, "", (0, 1), ("x 横向", "y 前向"), "t1")
        self.side = Projection(state, "", (1, 2), ("y 前向", "z 高度"), "t2")
        self.mode = Segmented(["俯视", "侧视"], tip="视角")
        self.mode.currentIndexChanged.connect(self._on_mode)
        self.limit_seg = velocity_limit_seg(state)
        self.controls = []  # 控件在悬浮工具条上,上下文条不放
        self.ramp = RampLegend(self.lut)
        self.legend = legend_label()
        self.legends = [self.ramp, self.legend]
        self.side.widget.hide()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.top.widget, 1)
        layout.addWidget(self.side.widget, 1)
        self.toolbar = HudBar([self.mode, self.limit_seg], self)
        self.card = SelectionCard(state, self)
        self.notice = None
        if notice:
            self.notice = QLabel(notice, self)
            self.notice.setObjectName("SceneNotice")
            self.notice.setStyleSheet(
                chip_style("SceneNotice")
                + f"#SceneNotice {{ color: {theme.DIM}; padding: 1px 7px; "
                f"font-size: {theme.FONT_NOTE}px; }}"
            )
        self.toolbar.content_changed.connect(self._place_hud)
        self.card.changed.connect(self._place_hud)
        self._xyz = np.zeros((0, 3))
        self._has_targets = False
        self._update_legend()
        state.track_changed.connect(self._on_track)
        state.filter_changed.connect(self._on_filter)
        state.velocity_limit_changed.connect(self._on_velocity_limit)

    def _place_hud(self) -> None:
        place_hud(self, self.toolbar, self.card, self.notice)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._place_hud()

    def _on_velocity_limit(self, _limit: float) -> None:
        self._update_legend()
        if self.state.frame is not None:
            self.render_frame(self.state.frame)

    def _on_mode(self, index: int) -> None:
        self.top.widget.setVisible(index == 0)
        self.side.widget.setVisible(index == 1)
        self.relock_ranges()
        self._lock_ranges()

    def _update_legend(self) -> None:
        limit = self.state.velocity_limit_mps
        self.ramp.set_limit(limit)
        self.legend.setText(
            f"+{limit:g} m/s(远离)"
            f"　<span style='color:{theme.TRACK}'>□ 航迹</span>"
            f"　<span style='color:{theme.ACCENT}'>◎ 选中</span>"
        )

    def reset(self, source) -> None:
        self._xyz = np.zeros((0, 3))
        self._has_targets = False
        for projection in (self.top, self.side):
            projection.clear()

    def render_frame(self, frame) -> None:
        det, tracks = frame.detections, frame.tracks
        self._has_targets = bool(len(det) or len(tracks))
        self._xyz = np.asarray(det.xyz, float).reshape(-1, 3)
        colors = velocity_colors(
            det.velocity_mps, self.lut, self.state.velocity_limit_mps
        )
        stage = self.state.stage_filter
        if stage is not None and frame.diagnostics is not None and len(colors):
            colors[~frame.diagnostics.verdicts.flagged(stage), 3] = _DIM_ALPHA
        brushes = [pg.mkBrush(*(int(v) for v in c)) for c in colors]
        track_xyz = np.asarray(tracks.xyz, float).reshape(-1, 3)
        for projection in (self.top, self.side):
            projection.set_points(self._xyz, brushes)
            projection.set_tracks(track_xyz, tracks.track_id)
        self._render_track(frame)

    def _plot_items(self) -> list[pg.PlotItem]:
        # 选中卡里的距离像小图要随选中检测自动缩放,不能被「首次有数据后锁范围」锁死
        return [i for i in super()._plot_items() if i is not self.card.spark.plot]

    def _lock_ranges(self) -> None:
        # 雷达原点常驻图中,不能把双 PRF 预热空帧误认为已有目标并锁在原点附近。
        if self._has_targets:
            super()._lock_ranges()

    def render_selection(self, frame, selection: Selection) -> None:
        k = selection.detection
        xyz = self._xyz[k] if k is not None and k < len(self._xyz) else None
        for projection in (self.top, self.side):
            projection.set_selected(xyz)
        self.card.show_selection(frame, k)

    def _render_track(self, frame) -> None:
        tid = self.state.track
        xyz = None
        if tid is not None:
            hit = np.flatnonzero(frame.tracks.track_id == tid)
            if len(hit):
                xyz = np.asarray(frame.tracks.xyz[hit[0]], float)
        for projection in (self.top, self.side):
            projection.set_selected_track(xyz)

    def _on_track(self, _track_id) -> None:
        if self.isVisible() and self.state.frame is not None:
            self._render_track(self.state.frame)
        else:
            self._dirty = True

    def _on_filter(self, _stage) -> None:
        if self.isVisible() and self.state.frame is not None:
            self._render(self.state.frame, self.state.selection)
        else:
            self._dirty = True
