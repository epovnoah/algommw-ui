"""三维点云:场景屏的主视图(照 ccm `gui/plots/pointcloud3d.py`)。

- 画布铺满主区(全出血);视角 / 余辉 / 视野 / 速度量程 / 坐标系的分段控件叠成左上角的悬浮
  工具条(`toolbar`,见 plots/scenehud.py),上下文条里不放控件(`controls` 为空);世界坐标的
  7 个控件折进工具条上的「世界 ▾」面板(`world_button`);右上角悬浮选中卡(`card`),左下图例
  带速度渐变条,右下读数里有「双击复位视角」;
- 视角:透视 / 俯视 / 侧视 / 正视。后三种用极小视场角的远距离透视近似正交投影,缩放、旋转照常可用;
- 着色:径向速度,CET-R3(蓝 = 接近、黄 = 静止、红 = 远离),量程 ±5 / ±15 / ±30 m/s 档位
  (与二维投影共用 `ViewState.velocity_limit_mps`);取检出点的
  `Detections.velocity_mps`(core 的 xDoppler,远离为正);
- 余辉:叠加最近 1 / 5 / 10 帧,或全量累积本轮截至当前帧的检测点;全量模式保持固定透明度,
  不随帧龄消失。历史点可点选:不换帧、不改当前选中,记为 `ViewState.history_pick`
  (信息页「选中累积点」显示其字段),点空白处或当前帧的点即取消;
- 视野:「聚焦」按 90% 分位距离取景(检测少于 20 个时取全部),「全部」包含远处离群点;
- 地面画距离环与视场边界(`RadarInfo.fov_azimuth_deg`;未启用视场时画 ±70°,图例注明)。
- 航迹:比检测点大的空心标记(线框立方体)+ id 文字 + 近 10 帧轨迹线。轨迹取自 accumulate
  累积的历史(与余辉同一份),只连同一跟踪起点(track_epoch)的帧。航迹不按 v_r 着色(用统一的航迹色,
  与检测点区分开;v_r 与检出点同为远离为正);
- 选中检测、选中航迹各加一个不做深度测试的青色光晕;点击选点(投影矩阵把点映射到屏幕后取
  最近的检测或航迹),双击复位视角;KPI 条筛选某一级时其余点变淡。
- 坐标系:雷达(以上各项)/ 世界(动平台:多普勒推算轨迹,逐帧点云拼成地图,见 plots/world.py)。
  世界坐标下工具条换成建图的控件,雷达坐标的场景、余辉与航迹隐藏,点选只认本帧检测。
OpenGL 不可用的平台(离屏测试的 offscreen)由 `make_cloud_view` 换成二维投影
(`projection.ProjectionView`,行为不变)。`GLTextItem` 用 QPainter 画,`renderToArray`
里看不到,窗口截图里有。
"""

from typing import NamedTuple

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont, QGuiApplication, QVector3D
from PySide6.QtWidgets import QVBoxLayout, QWidget

from ...frame import EMPTY3, Detections
from .. import theme
from ..state import HistoryPick, Selection, ViewState
from ..views.base import ResultView
from ..views.common import legend_label, readout_label
from ..widgets.seg import Segmented
from .projection import (
    ProjectionView,
    gl_available,
    velocity_colors,
    velocity_limit_seg,
)
from .scenehud import (
    HudBar,
    RampLegend,
    SelectionCard,
    WorldButton,
    place_hud,
)
from .world import WorldLayer

FRAMES = {"radar": "雷达", "world": "世界"}
MODES = {"persp": "透视", "top": "俯视", "side": "侧视", "front": "正视"}
TRAILS = {"1": "余辉 1", "5": "5", "10": "10", "all": "全量"}
FITS = {"focus": "聚焦", "all": "全部"}
#: 视场未启用(fov_azimuth_deg 为 None)时画的方位扇区,度(照 ccm 的 ±70°)
DEFAULT_FOV_DEG = (-70.0, 70.0)
ORTHO_FOV_DEG = 1.0
PERSP_FOV_DEG = 55.0
# 视角 → (方位角, 仰角),pyqtgraph 的相机位置 = 中心 + 距离 × (cos仰·cos方位, cos仰·sin方位, sin仰)
CAMERA = {
    "persp": (-118.0, 28.0),
    "top": (-90.0, 90.0),
    "side": (0.0, 0.0),
    "front": (-90.0, 0.0),
}
#: 航迹轨迹线覆盖的帧数(含当前帧)
TRACK_TRAIL_FRAMES = 10
PICK_RADIUS_PX = 12.0
DIM_ALPHA = 0.18  # KPI 条筛选某一级时,其余点的不透明度

# 航迹的空心标记:线框立方体的 12 条棱(每条两个端点),单位半边长
_CORNERS = np.array(
    [[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], np.float32
)
_EDGES = [
    (i, j)
    for i in range(8)
    for j in range(i + 1, 8)
    if np.count_nonzero(_CORNERS[i] != _CORNERS[j]) == 1
]
CUBE_SEGMENTS = _CORNERS[np.array(_EDGES).ravel()]  # (24, 3)


class _Past(NamedTuple):
    """accumulate 记下的一帧,余辉、航迹轨迹线与历史点点选共用。"""

    xyz: np.ndarray
    velocity: np.ndarray
    track_id: np.ndarray
    track_xyz: np.ndarray
    track_epoch: int
    frame_id: int  # 采集帧号(usFrameIdx)
    detections: Detections  # 该帧检出点的列(点选历史点时取字段)


def nice_step(extent: float) -> float:
    raw = extent / 5
    magnitude = 10 ** np.floor(np.log10(raw))
    return float(next(m * magnitude for m in (1, 2, 5, 10) if m * magnitude >= raw))


def scene_extent(ranges: np.ndarray, include_all: bool) -> float:
    ranges = np.asarray(ranges, float)
    ranges = ranges[np.isfinite(ranges)]
    if ranges.size == 0:
        return 20.0
    reach = (
        ranges.max() if include_all or ranges.size < 20 else np.percentile(ranges, 90)
    )
    return float(max(15.0, reach * 1.2))


def make_cloud_view(state: ViewState) -> ResultView:
    """结果屏主视图:OpenGL 可用时为三维点云,否则退回二维投影并注明原因。"""
    if gl_available():
        return PointCloudView(state)
    platform = QGuiApplication.platformName()
    return ProjectionView(
        state,
        notice=f"三维视图需要 OpenGL,当前 Qt 平台({platform})不支持,以下为二维投影",
    )


def _make_gl_view():
    """创建带点选信号的 GLViewWidget;延迟导入 pyqtgraph.opengl,没有 OpenGL 的平台不加载它。"""
    import pyqtgraph.opengl as gl

    class GLView(gl.GLViewWidget):
        clicked = Signal(float, float)  # 左键单击(未拖动)的位置,逻辑像素
        reset_requested = Signal()  # 双击

        def mousePressEvent(self, ev):
            self._press = ev.position()
            super().mousePressEvent(ev)

        def mouseReleaseEvent(self, ev):
            press = getattr(self, "_press", None)
            if press is not None and ev.button() == Qt.MouseButton.LeftButton:
                moved = ev.position() - press
                if abs(moved.x()) + abs(moved.y()) < 4:
                    self.clicked.emit(ev.position().x(), ev.position().y())
            super().mouseReleaseEvent(ev)

        def mouseDoubleClickEvent(self, ev):
            self.reset_requested.emit()

    return GLView()


def _rgba(color: str, alpha: float) -> tuple[float, float, float, float]:
    c = QColor(color)
    c.setAlphaF(alpha)
    return c.getRgbF()


class PointCloudView(ResultView):
    title = "三维点云"
    KEEP_AUTORANGE = (
        True  # 没有 pyqtgraph 二维坐标轴;相机只在换源 / 改视角 / 改视野时复位
    )

    def __init__(self, state: ViewState, parent: QWidget | None = None) -> None:
        super().__init__(state, parent)
        self.mode, self.trail, self.fit = "persp", 1, "focus"
        self.lut = pg.colormap.get(theme.VELOCITY_CMAP).map(
            np.linspace(0, 1, 256), mode="float"
        )
        self.mode_seg = Segmented(list(MODES.values()), tip="视角")
        self.trail_seg = Segmented(
            list(TRAILS.values()),
            tip="1 / 5 / 10 帧:近期余辉;全量累积:本轮截至当前帧的全部检测点",
        )
        self.fit_seg = Segmented(
            list(FITS.values()), tip="视野:聚焦 = 90% 分位距离取景;全部 = 含远处离群点"
        )
        self.mode_seg.currentIndexChanged.connect(
            lambda i: self._set_mode(list(MODES)[i])
        )
        self.trail_seg.currentIndexChanged.connect(
            lambda i: self._set_trail(list(TRAILS)[i])
        )
        self.fit_seg.currentIndexChanged.connect(lambda i: self._set_fit(list(FITS)[i]))
        self.radar_controls = [
            self.mode_seg,
            self.trail_seg,
            self.fit_seg,
            velocity_limit_seg(state),
        ]
        #: 左下图例:速度渐变条(雷达坐标)+ 接着写的文字标签(航迹 / 选中 / 视场;世界坐标换成世界图例)
        self.ramp = RampLegend(self.lut)
        self.legend = legend_label()
        self.legends = [self.ramp, self.legend]
        self.readout = readout_label()

        self.gl = _make_gl_view()
        self._build_gl_items()
        self.world_mode = False
        self._radar_camera: dict | None = None
        self.world = WorldLayer(self.gl)
        self.world.changed.connect(self._rerender)
        self.frame_seg = Segmented(
            list(FRAMES.values()),
            tip="坐标系:雷达 = 以雷达为原点看单帧;"
            "世界 = 动平台上用多普勒推算车辆轨迹,把逐帧点云拼成地图",
        )
        self.frame_seg.currentIndexChanged.connect(
            lambda i: self._set_frame(list(FRAMES)[i] == "world")
        )
        # 控件都叠在画布上(悬浮工具条),上下文条里一个不放;坐标系开关放最左:雷达控件显隐、
        # 「世界 ▾」出现时它的位置不动,切回去不用挪鼠标。世界坐标的 7 个控件折进「世界 ▾」面板
        self.controls = []
        self.world_button = WorldButton(self.world.controls)
        self.world_button.setVisible(False)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.gl)
        self.toolbar = HudBar(
            [self.frame_seg, *self.radar_controls, self.world_button], self
        )
        self.card = SelectionCard(state, self)
        self.toolbar.content_changed.connect(self._place_hud)
        self.card.changed.connect(self._place_hud)

        self._history: dict[int, _Past] = {}
        #: 余辉点(trail_points.pos 的行)→ (处理序号, 该帧检测行号),与画出的顺序一致
        self._trail_rows = np.zeros((0, 2), np.int64)
        self._readout_base = ""
        self._xyz = EMPTY3
        self._track_xyz = EMPTY3
        self._track_id = np.zeros(0, np.int32)
        self._fov = DEFAULT_FOV_DEG
        self._fov_enabled = False
        self._fit_pending = True
        self.extent = 20.0
        self._update_legend()
        self._build_scene()
        self._reset_camera()
        state.track_changed.connect(self._on_track)
        state.filter_changed.connect(self._on_filter)
        state.frames_processed.connect(self.accumulate_frames)
        state.velocity_limit_changed.connect(self._on_velocity_limit)
        state.history_pick_changed.connect(lambda _pick: self._render_pick())

    def _on_velocity_limit(self, _limit: float) -> None:
        self._update_legend()
        self._rerender()

    def _place_hud(self) -> None:
        place_hud(self, self.toolbar, self.card)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._place_hud()

    # ---------------- OpenGL 图元(常驻,每帧只 setData) ----------------
    def _build_gl_items(self) -> None:
        import pyqtgraph.opengl as gl

        self.gl.setBackgroundColor(theme.PLOT)
        self.gl.clicked.connect(self._on_click)
        self.gl.reset_requested.connect(self._reset_camera)
        grey = (0.78, 0.8, 0.83)
        self.ground = gl.GLLinePlotItem(
            pos=EMPTY3, mode="lines", color=(*grey, 0.16), width=1
        )
        self.fov_edges = gl.GLLinePlotItem(
            pos=EMPTY3, mode="lines", color=(*grey, 0.35), width=1
        )
        self.wall = gl.GLLinePlotItem(
            pos=EMPTY3, mode="lines", color=(*grey, 0.16), width=1
        )
        self.axes = gl.GLLinePlotItem(pos=EMPTY3, mode="lines", width=2)
        self.trail_points = gl.GLScatterPlotItem(
            pos=EMPTY3, size=4, glOptions="translucent"
        )
        self.track_lines = gl.GLLinePlotItem(
            pos=EMPTY3, mode="lines", color=_rgba(theme.TRACK, 0.55), width=1.5
        )
        self.track_marks = gl.GLLinePlotItem(
            pos=EMPTY3, mode="lines", color=_rgba(theme.TRACK, 0.9), width=1.5
        )
        self.halo = gl.GLScatterPlotItem(
            pos=EMPTY3,
            size=22,
            color=_rgba(theme.ACCENT, 0.55),
            glOptions="additive",
        )
        self.track_halo = gl.GLScatterPlotItem(
            pos=EMPTY3,
            size=34,
            color=_rgba(theme.ACCENT, 0.45),
            glOptions="additive",
        )
        #: 点中的历史点:比当前选中更大更淡的光晕,两者同时出现时也分得开
        self.pick_halo = gl.GLScatterPlotItem(
            pos=EMPTY3,
            size=30,
            color=_rgba(theme.ACCENT, 0.35),
            glOptions="additive",
        )
        self.points = gl.GLScatterPlotItem(pos=EMPTY3, size=8, glOptions="translucent")
        for item in (
            self.ground,
            self.fov_edges,
            self.wall,
            self.axes,
            self.trail_points,
            self.track_lines,
            self.track_marks,
            self.halo,
            self.track_halo,
            self.pick_halo,
            self.points,
        ):
            self.gl.addItem(item)
        self.ring_labels: list = []
        #: 航迹 id 文字:按需增补,每帧只改文字与位置,多余的隐藏(不每帧删建)
        self.track_labels: list = []

    def _build_scene(self) -> None:
        """按场景尺度与视场画地面距离环、视场边界、竖直网格(侧视 / 正视用)与坐标轴。"""
        import pyqtgraph.opengl as gl

        ext, step = self.extent, nice_step(self.extent)
        lo, hi = self._fov
        az = np.radians(np.linspace(lo, hi, int(np.ceil((hi - lo) / 3.5)) + 1))
        rings, segments = np.arange(step, ext * 1.0001, step), []
        for r in rings:
            arc = np.stack([r * np.sin(az), r * np.cos(az), np.zeros_like(az)], axis=1)
            segments.append(np.repeat(arc, 2, axis=0)[1:-1])
        for a in np.radians([a for a in (-60, -30, 0, 30, 60) if lo <= a <= hi]):
            segments.append(
                np.array([[0, 0, 0], [ext * np.sin(a), ext * np.cos(a), 0]])
            )
        self.ground.setData(pos=np.concatenate(segments) if segments else EMPTY3)
        edges = [
            [0, 0, 0],
            [ext * np.sin(np.radians(lo)), ext * np.cos(np.radians(lo)), 0],
        ]
        edges += [
            [0, 0, 0],
            [ext * np.sin(np.radians(hi)), ext * np.cos(np.radians(hi)), 0],
        ]
        self.fov_edges.setData(pos=np.array(edges, float))
        height = ext * 0.6
        wall = []
        for z in np.arange(-np.floor(height / step) * step, height + 1e-9, step):
            wall += [[0, 0, z], [0, ext, z], [-ext, 0, z], [ext, 0, z]]
        for t in np.arange(0, ext + 1e-9, step):
            wall += [[0, t, -height], [0, t, height]]
        for t in np.arange(-np.floor(ext / step) * step, ext + 1e-9, step):
            wall += [[t, 0, -height], [t, 0, height]]
        self.wall.setData(pos=np.array(wall, float))
        length = step * 0.6
        axes = [[0, 0, 0], [length, 0, 0], [0, 0, 0], [0, length, 0]]
        axes += [[0, 0, 0], [0, 0, length]]
        colors = [[0.9, 0.45, 0.45, 1]] * 2 + [[0.5, 0.78, 0.52, 1]] * 2
        colors += [[0.39, 0.71, 0.96, 1]] * 2
        self.axes.setData(pos=np.array(axes, float), color=np.array(colors))
        for label in self.ring_labels:
            self.gl.removeItem(label)
        font = QFont()
        font.setPointSize(9)
        self.ring_labels = []
        for r in rings:
            label = gl.GLTextItem(
                pos=(1.5 * step / 10, r, 0),
                text=f"{r:g} m",
                color=(200, 204, 212, 150),
                font=font,
            )
            self.gl.addItem(label)
            self.ring_labels.append(label)
        self._apply_mode_visibility()

    def _apply_mode_visibility(self) -> None:
        radar = not self.world_mode
        flat = self.mode in ("persp", "top")
        for item in (self.ground, self.fov_edges, *self.ring_labels):
            item.setVisible(radar and flat)
        self.wall.setVisible(radar and not flat)
        for item in (
            self.axes,
            self.points,
            self.trail_points,
            self.track_lines,
            self.track_marks,
            self.track_halo,
            self.pick_halo,
        ):
            item.setVisible(radar)
        if not radar:  # 回到雷达坐标时由 _draw_tracks 按本帧航迹重新显示
            for label in self.track_labels:
                label.setVisible(False)

    def _set_frame(self, world: bool) -> None:
        """切坐标系:换标题行控件与图元,各自的相机分开记(切回雷达时恢复原视角)。"""
        if world == self.world_mode:
            return
        self.world_mode = world
        for widget in self.radar_controls:
            widget.setVisible(not world)
        self.world_button.setVisible(world)
        if world:
            opts = self.gl.opts
            self._radar_camera = {
                "pos": QVector3D(opts["center"]),
                "distance": opts["distance"],
                "elevation": opts["elevation"],
                "azimuth": opts["azimuth"],
                "fov": opts["fov"],
            }
            self.world.enter()
        else:
            self.world.scene.set_visible(False)
            if self._radar_camera is not None:
                camera = dict(self._radar_camera)
                self.gl.setCameraParams(fov=camera.pop("fov"))
                self.gl.setCameraPosition(**camera)
        self._apply_mode_visibility()
        self._update_legend()
        self._rerender()

    def _reset_camera(self) -> None:
        if self.world_mode:
            self.world.reset_camera()
            return
        azimuth, elevation = CAMERA[self.mode]
        ext = self.extent
        if self.mode == "persp":
            self.gl.setCameraParams(fov=PERSP_FOV_DEG)
            self.gl.setCameraPosition(
                pos=QVector3D(0, ext * 0.42, 0),
                distance=ext * 1.25,
                elevation=elevation,
                azimuth=azimuth,
            )
            return
        # 近似正交:视场角很小、距离很远;按水平视场取景,宽度覆盖 ±70° 扇区或高度方向的 ±0.6 倍尺度
        aspect = max(self.gl.width(), 1) / max(self.gl.height(), 1)
        half_width = {
            "top": max(ext * 0.97, ext * 0.55 * aspect),
            "side": max(ext * 0.55, ext * 0.65 * aspect),
            "front": max(ext * 0.97, ext * 0.65 * aspect),
        }[self.mode]
        distance = half_width * 1.05 / np.tan(np.radians(ORTHO_FOV_DEG / 2))
        center = {
            "top": (0, ext * 0.5, 0),
            "side": (0, ext * 0.5, 0),
            "front": (0, 0, 0),
        }
        self.gl.setCameraParams(fov=ORTHO_FOV_DEG)
        self.gl.setCameraPosition(
            pos=QVector3D(*center[self.mode]),
            distance=distance,
            elevation=elevation,
            azimuth=azimuth,
        )

    # ---------------- 控件 ----------------
    def _set_mode(self, mode: str) -> None:
        self.mode = mode
        self._apply_mode_visibility()
        self._reset_camera()
        self._rerender()

    def _set_trail(self, trail: str) -> None:
        self.trail = None if trail == "all" else int(trail)
        self._fit_pending = True
        self._rerender()

    def _set_fit(self, fit: str) -> None:
        self.fit = fit
        self._fit_pending = True
        self._rerender()

    def _rerender(self) -> None:
        if self.state.frame is not None:
            self._render(self.state.frame, self.state.selection)

    def _update_legend(self) -> None:
        self.ramp.setVisible(not self.world_mode)
        if self.world_mode:
            self.legend.setText(self.world.legend_html())
            return
        limit = self.state.velocity_limit_mps
        self.ramp.set_limit(limit)
        lim = f"{limit:g}"
        lo, hi = self._fov
        fov = (
            f"视场 {lo:g}° ~ {hi:g}°"
            if self._fov_enabled
            else f"视场 ±{hi:g}°(视场未启用)"
        )
        dim = theme.DIM
        self.legend.setText(
            f"+{lim} m/s(远离)"
            f"　<span style='color:{theme.TRACK}'>□ 航迹</span>"
            f"　<span style='color:{theme.ACCENT}'>◎ 选中</span>"
            f"　<span style='color:{dim}'>∠ {fov}</span>"
        )

    # ---------------- 数据 ----------------
    def _colors(self, velocity: np.ndarray) -> np.ndarray:
        return velocity_colors(
            velocity, self.lut, self.state.velocity_limit_mps
        ).astype(np.float32)

    def reset(self, source) -> None:
        self._history.clear()
        self._trail_rows = np.zeros((0, 2), np.int64)
        self._xyz, self._track_xyz = EMPTY3, EMPTY3
        self._track_id = np.zeros(0, np.int32)
        fov = source.fov_azimuth_deg
        self._fov_enabled = fov is not None
        self._fov = tuple(fov) if fov is not None else DEFAULT_FOV_DEG
        self.world.reset(source.frame_period_s, self._fov)
        self._update_legend()
        for item in (
            self.points,
            self.trail_points,
            self.track_lines,
            self.track_marks,
            self.halo,
            self.track_halo,
            self.pick_halo,
        ):
            item.setData(pos=EMPTY3)
        for label in self.track_labels:
            label.setVisible(False)
        self._fit_pending = True
        self._build_scene()
        self._reset_camera()

    def accumulate_frames(self, frames) -> None:
        # 跳帧时播放器只显示批次末帧,但累积层必须收齐中间帧。新一轮处理从
        # track_epoch 开始,清掉旧参数/旧递推的结果;缓存回看不会触发这里。
        if frames and frames[0].index == frames[0].track_epoch:
            self._history.clear()
            self._fit_pending = True
        self.world.feed(frames)  # 世界坐标的推算与地图也要收齐中间帧
        for frame in frames:
            self.accumulate(frame)

    def accumulate(self, frame) -> None:
        # 按帧号覆盖,回看/重显不重复累加。只保留轻量点云与航迹数组,
        # 不持有 ProcessedFrame 的 RD/频谱等大块数据;切到全量时可立即补全。
        self.world.add_shown(frame)
        det, tracks = frame.detections, frame.tracks
        self._history[frame.index] = _Past(
            np.array(det.xyz, dtype=np.float32, copy=True).reshape(-1, 3),
            np.array(det.velocity_mps, dtype=np.float32, copy=True),
            np.array(tracks.track_id, copy=True),
            np.array(tracks.xyz, dtype=np.float32, copy=True).reshape(-1, 3),
            frame.track_epoch,
            int(frame.fi),
            det,
        )

    def render_frame(self, frame) -> None:
        if self.world_mode:
            self.world.render(frame)
            self.readout.setText(self.world.readout_html(frame))
            self.legend.setText(self.world.legend_html())  # 随「压平 / 实测」变
            return
        det, tracks = frame.detections, frame.tracks
        self._xyz = np.asarray(det.xyz, float).reshape(-1, 3)
        self._track_xyz = np.asarray(tracks.xyz, float).reshape(-1, 3)
        self._track_id = np.asarray(tracks.track_id)
        colors = self._colors(det.velocity_mps)
        stage = self.state.stage_filter
        if stage is not None and frame.diagnostics is not None and len(colors):
            colors[~frame.diagnostics.verdicts.flagged(stage), 3] = DIM_ALPHA
        self._draw_trail(frame.index)
        if self._fit_pending:
            fit_xyz = self._xyz
            if self.trail is None:
                fit_xyz = np.concatenate([self.trail_points.pos, self._xyz])
            if len(fit_xyz):
                ranges = np.linalg.norm(fit_xyz, axis=1)
                self.extent = scene_extent(ranges, self.fit == "all")
                self._fit_pending = False
                self._build_scene()
                self._reset_camera()
        outside = (
            int((np.linalg.norm(self._xyz, axis=1) > self.extent * 1.02).sum())
            if len(self._xyz)
            else 0
        )
        self._readout_base = (
            (
                f"全量累积 {len(self._xyz) + len(self.trail_points.pos)} 点 · "
                if self.trail is None
                else ""
            )
            + (f"视野外 {outside} 个检测(视野选「全部」)· " if outside else "")
            + "拖动旋转 · 滚轮缩放 · 双击复位视角"
        )
        self.points.setData(pos=self._xyz, color=colors)
        self._draw_tracks(frame)
        self._render_track()
        self._render_pick()

    def _draw_trail(self, index: int) -> None:
        older = [
            i
            for i in self._history
            if i < index and (self.trail is None or index - self.trail < i)
        ]
        if not older:
            self.trail_points.setData(pos=EMPTY3)
            self._trail_rows = np.zeros((0, 2), np.int64)
            return
        positions, colors, rows = [], [], []
        for i in older:
            past = self._history[i]
            n = len(past.xyz)
            rows.append(np.c_[np.full(n, i), np.arange(n)])
            c = self._colors(past.velocity)
            c[:, 3] = (
                0.55 if self.trail is None else 0.55 - 0.4 * (index - i) / self.trail
            )
            positions.append(past.xyz)
            colors.append(c)
        self.trail_points.setData(
            pos=np.concatenate(positions), color=np.concatenate(colors)
        )
        self._trail_rows = np.concatenate(rows).astype(np.int64)

    def _draw_tracks(self, frame) -> None:
        """航迹:线框立方体标记、近 10 帧轨迹线(同一跟踪起点的历史帧)与 id 文字。"""
        import pyqtgraph.opengl as gl

        xyz, ids = self._track_xyz, self._track_id
        half = self.extent / 60
        marks = xyz[:, None, :] + CUBE_SEGMENTS[None] * half
        self.track_marks.setData(pos=marks.reshape(-1, 3))
        past = sorted(
            (
                (i, p)
                for i, p in self._history.items()
                if frame.index - TRACK_TRAIL_FRAMES < i <= frame.index
                and p.track_epoch == frame.track_epoch
            ),
            key=lambda item: item[0],
        )
        segments = []
        for tid in ids:
            path = [
                p.track_xyz[p.track_id == tid][0] for _, p in past if tid in p.track_id
            ]
            if len(path) >= 2:
                segments.append(np.repeat(np.array(path), 2, axis=0)[1:-1])
        self.track_lines.setData(pos=np.concatenate(segments) if segments else EMPTY3)
        font = QFont()
        font.setPointSize(10)
        while len(self.track_labels) < len(ids):
            label = gl.GLTextItem(color=QColor(theme.TRACK), font=font)
            self.gl.addItem(label)
            self.track_labels.append(label)
        offset = np.array([half * 1.4, 0.0, half * 1.4])
        for i, label in enumerate(self.track_labels):
            if i < len(ids):
                label.setData(pos=xyz[i] + offset, text=str(int(ids[i])))
            label.setVisible(i < len(ids))

    def render_selection(self, frame, selection: Selection) -> None:
        k = selection.detection
        rows = self.world.current_xyz if self.world_mode else self._xyz
        xyz = rows[k] if k is not None and k < len(rows) else None
        self.halo.setData(pos=EMPTY3 if xyz is None else xyz[None, :])
        self.card.show_selection(frame, k)

    def _render_track(self) -> None:
        tid = self.state.track
        hit = np.flatnonzero(self._track_id == tid) if tid is not None else []
        self.track_halo.setData(pos=self._track_xyz[hit[:1]] if len(hit) else EMPTY3)

    def _render_pick(self) -> None:
        """历史点的光晕与读数。该点已不在余辉里(回看到它之前、改余辉档位、重算清空)就取消。"""
        if self.world_mode:  # 世界坐标不显示余辉,读数归推算与地图
            return
        pick = self.state.history_pick
        hit = (
            np.flatnonzero(
                (self._trail_rows[:, 0] == pick.index)
                & (self._trail_rows[:, 1] == pick.row)
            )
            if pick is not None
            else np.zeros(0, np.int64)
        )
        if pick is not None and not len(hit):
            self.state.set_history_pick(None)  # 信号回到这里,按「无选中」再画一次
            return
        self.pick_halo.setData(
            pos=self.trail_points.pos[hit[:1]] if len(hit) else EMPTY3
        )
        if pick is None:
            self.readout.setText(self._readout_base)
            return
        self.readout.setText(
            f"累积点 · 帧 {pick.frame_id} #{pick.row} · {pick.range_m:.2f} m · "
            f"{pick.velocity_mps:+.2f} m/s({pick.pair_label})· 方位 {pick.azimuth_deg:+.1f}° · "
            f"SNR {pick.snr_db:.1f} dB · 详情见信息页 · 点空白处取消"
        )

    def _history_pick(self, k: int) -> HistoryPick:
        """余辉第 k 个点 → 它所在帧的检测字段(取值,之后该帧被裁出余辉也不影响已显示的信息)。"""
        index, row = (int(v) for v in self._trail_rows[k])
        past = self._history[index]
        det = past.detections
        return HistoryPick(
            index=index,
            frame_id=past.frame_id,
            row=row,
            range_bin=int(det.range_bin[row]),
            doppler_bin=int(det.doppler_bin[row]),
            range_m=float(det.range_m[row]),
            velocity_mps=float(det.velocity_mps[row]),
            snr_db=float(det.snr_db[row]),
            noise_db=float(det.noise_db[row]),
            azimuth_deg=float(det.azimuth_deg[row]),
            elevation_deg=float(det.elevation_deg[row]),
            xyz=tuple(float(v) for v in past.xyz[row]),
            pair_label=det.pair_label(row),
        )

    def _on_track(self, _track_id) -> None:
        if self.isVisible() and self.state.frame is not None:
            self._render_track()
        else:
            self._dirty = True

    def _on_filter(self, _stage) -> None:
        if self.isVisible() and self.state.frame is not None:
            self._render(self.state.frame, self.state.selection)
        else:
            self._dirty = True

    # ---------------- 点选 ----------------
    def screen_positions(self, xyz: np.ndarray) -> np.ndarray:
        """世界系点在 GL 视图上的屏幕坐标(逻辑像素),(n, 2);相机后方的点为 NaN。"""
        w, h = self.gl.width(), self.gl.height()
        matrix = self.gl.projectionMatrix((0, 0, w, h), (0, 0, w, h))
        matrix = matrix * self.gl.viewMatrix()
        m = np.array(matrix.copyDataTo(), float).reshape(4, 4)
        homogeneous = np.c_[xyz, np.ones(len(xyz))] @ m.T
        with np.errstate(divide="ignore", invalid="ignore"):
            ndc = homogeneous[:, :3] / homogeneous[:, 3:4]
        screen = np.stack([(ndc[:, 0] + 1) / 2 * w, (1 - ndc[:, 1]) / 2 * h], axis=1)
        screen[homogeneous[:, 3] <= 0] = np.nan
        return screen

    def _on_click(self, x: float, y: float) -> None:
        """屏幕上最近的检测、航迹或余辉历史点(PICK_RADIUS_PX 以内);距离相同时依次优先
        当前检测、航迹、历史点。点中历史点只记 history_pick(不换帧、不改当前选中);
        点中当前帧的点 / 航迹或点空白处,都取消历史点选中。"""
        best = None  # (屏幕距离, 类别, 行号)
        candidates = (
            [("detection", self.world.current_xyz)]
            if self.world_mode
            else [
                ("detection", self._xyz),
                ("track", self._track_xyz),
                ("history", self.trail_points.pos),
            ]
        )
        for kind, xyz in candidates:
            if not len(xyz):
                continue
            screen = self.screen_positions(xyz)
            distance = np.hypot(screen[:, 0] - x, screen[:, 1] - y)
            if np.all(np.isnan(distance)):
                continue
            k = int(np.nanargmin(distance))
            if distance[k] <= PICK_RADIUS_PX and (
                best is None or distance[k] < best[0]
            ):
                best = (float(distance[k]), kind, k)
        if best is None:
            self.state.set_history_pick(None)
            return
        _, kind, k = best
        if kind == "history":
            self.state.set_history_pick(self._history_pick(k))
            return
        self.state.set_history_pick(None)
        if kind == "detection":
            self.state.select_detection(k)
        else:
            self.state.select_track(int(self._track_id[k]))
