"""三维点云的「世界」坐标:动平台上只用雷达推算车辆轨迹,把逐帧点云拼成地图。

算法在 `viz/world/`(纯 numpy):多普勒自运动拟合给车速与转向的预测,每帧再把本帧静止点对到
前几帧的局部地图上修正航向,视轴角随之在线自适应。这里是界面的两层:
- `WorldScene`:GL 图元 —— 地图分块散点、本帧点、轨迹、车辆线框、视场扇形、跟随网格与相机;
- `WorldLayer`:标题行的控件(视角 / 累积范围 / 静止点 / 压平 / 配准 / L / 视轴初值)、
  推算与地图的会话、图例与读数。`PointCloudView` 持有它,坐标系切到「世界」时显示。

回放口径:帧时刻 = 帧序号 × 帧周期;推算与地图只吃按顺序新处理出的帧(新一轮处理从航迹起点
清零重来,缓存回看不重复累加)。显示的是**截至当前帧**的地图:回看旧帧时地图、车辆与轨迹都回到
那一帧。地图按 CHUNK_SECONDS 分块,封好的块只上传一次;「最近 30 s」按块取整(最多多显示一块)。
不用 GLMeshItem:本机 OpenGL 的 mesh 路径有崩溃记录。
"""

import math
from typing import Any, NamedTuple

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QColor, QVector3D
from PySide6.QtWidgets import QDoubleSpinBox

from ...frame import EMPTY3
from ...world.ego import estimate_ego_motion
from ...world.mapping import MOVING, WorldMap
from ...world.odometry import OdometryParams, Pose, RadarOdometry
from .. import theme
from ..widgets.seg import Segmented

RECENT_S = 30.0
CHUNK_SECONDS = 2.0
HEIGHT_RANGE_M = (-2.0, 4.0)  # 相对雷达高度,色带两端的初值(攒够点前用)
HEIGHT_FIT_POINTS = 3000  # 地图攒够这么多点后,按实测高度的分位数定一次色带范围
HEIGHT_FIT_PCT = (5.0, 95.0)
HEIGHT_MIN_SPAN_M = 3.0
MAP_POINT_SIZE = 4.0  # 同雷达坐标的余辉点
CURRENT_POINT_SIZE = 8.0  # 同雷达坐标的本帧检测
GROUND_BELOW_RADAR_M = 0.5  # 网格与轨迹画在雷达下方这么多
GRID_SPACING_M = 10.0
GRID_SIZE_M = 200.0
FOLLOW_LOOKAHEAD_M = 15.0
FOLLOW_DISTANCE_M = 70.0
FOLLOW_ELEVATION_DEG = 28.0
FIT_MARGIN_M = 40.0  # 全图取景时轨迹外留的边(地图点在轨迹两侧)
FIT_ELEVATION_DEG = 60.0
CAMERA_FOV_DEG = 55.0
FOV_RANGE_M = 50.0
VEHICLE_X_M = (-0.95, 0.95)  # 车身坐标(后轴中心为原点):x 右、y 前、z 上(z=0 为雷达高度)
VEHICLE_REAR_M = -1.0
VEHICLE_FRONT_M = 0.1  # 车头 = 雷达位置再往前这么多
VEHICLE_Z_M = (-0.5, 1.0)

_RAMP_STOPS = np.array([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
_RAMP_RGB = np.array(
    [[0.25, 0.45, 1.0], [0.20, 0.85, 0.45], [1.0, 0.88, 0.25], [1.0, 0.35, 0.25]]
)  # 蓝 → 绿 → 黄 → 红


def _rgba(color: str, alpha: float) -> tuple[float, float, float, float]:
    c = QColor(color)
    c.setAlphaF(alpha)
    return c.getRgbF()


GREY = (0.78, 0.8, 0.83)
STATIC_COLOR = _rgba(theme.TEXT, 0.95)
MOVING_COLOR = _rgba(theme.WARN, 0.95)
TRAJECTORY_COLOR = _rgba(theme.SIGNAL, 0.9)
VEHICLE_COLOR = _rgba(theme.TEXT, 0.85)
FOV_COLOR = (*GREY, 0.35)
GRID_COLOR = (200, 204, 212, 28)  # GLGridItem 取 0..255


def fit_height_range(u: np.ndarray) -> tuple[float, float]:
    """实测高度的 5%–95% 分位(向外取整到 1 m,至少 3 m 宽)作色带两端。

    固定 −2~+4 m 在实采上不够用:俯仰偏低时 88% 的点低于 −2 m,色带只剩红蓝两端。"""
    lo, hi = np.percentile(np.asarray(u, np.float64), HEIGHT_FIT_PCT)
    lo, hi = math.floor(lo), math.ceil(hi)
    if hi - lo < HEIGHT_MIN_SPAN_M:
        mid = (lo + hi) / 2
        lo, hi = mid - HEIGHT_MIN_SPAN_M / 2, mid + HEIGHT_MIN_SPAN_M / 2
    return float(lo), float(hi)


def height_colors(
    u: np.ndarray, alpha: float = 0.8, span: tuple[float, float] = HEIGHT_RANGE_M
) -> np.ndarray:
    """(N,) 相对雷达的高度 → (N, 4) float32 颜色;span 是色带两端。"""
    lo, hi = span
    norm = np.clip((np.asarray(u, np.float64) - lo) / (hi - lo), 0.0, 1.0)
    colors = np.empty((len(norm), 4), np.float32)
    for channel in range(3):
        colors[:, channel] = np.interp(norm, _RAMP_STOPS, _RAMP_RGB[:, channel])
    colors[:, 3] = alpha
    return colors


def flatten(points: np.ndarray) -> np.ndarray:
    """压平显示:高度置 0(雷达高度的水平面)。"""
    out = np.array(points, np.float32, copy=True).reshape((-1, 3))
    out[:, 2] = 0.0
    return out


def body_to_world(xyz, heading_deg: float) -> np.ndarray:
    """车身坐标(x 右、y 前、z 上)→ 世界偏移,航向从初始方向顺时针。"""
    h = math.radians(heading_deg)
    c, s = math.cos(h), math.sin(h)
    xyz = np.asarray(xyz, np.float64).reshape((-1, 3))
    out = np.empty_like(xyz)
    out[:, 0] = xyz[:, 0] * c + xyz[:, 1] * s
    out[:, 1] = -xyz[:, 0] * s + xyz[:, 1] * c
    out[:, 2] = xyz[:, 2]
    return out


def vehicle_lines(pose: Pose, params: OdometryParams) -> np.ndarray:
    """车辆线框盒(12 条棱,(24, 3)),后端在后轴后 1 m,前端在雷达前 0.1 m。"""
    ys = (VEHICLE_REAR_M, params.rear_axle_m + VEHICLE_FRONT_M)
    corners = np.array(
        [(x, y, z) for z in VEHICLE_Z_M for y in ys for x in VEHICLE_X_M], np.float64
    )  # 序号 = x + 2y + 4z
    edges = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7)]
    edges += [(0, 4), (1, 5), (2, 6), (3, 7)]
    local = corners[np.array(edges).reshape(-1)]
    world = body_to_world(local, pose.heading_deg) + (pose.e, pose.n, 0.0)
    return world.astype(np.float32)


def fov_lines(
    pose: Pose, params: OdometryParams, fov_deg: tuple[float, float]
) -> np.ndarray:
    """视场扇形(两条边 + 圆弧),从雷达位置出发,朝向 = 航向 + 视轴角。"""
    lo, hi = fov_deg
    angles = np.radians(np.linspace(lo, hi, 25))
    arc = np.stack(
        [FOV_RANGE_M * np.sin(angles), FOV_RANGE_M * np.cos(angles), 0 * angles], 1
    )
    segments = [np.zeros(3), arc[0], np.zeros(3), arc[-1]]
    for a, b in zip(arc[:-1], arc[1:]):
        segments += [a, b]
    radar = body_to_world((0.0, params.rear_axle_m, 0.0), pose.heading_deg)[0]
    world = body_to_world(np.array(segments), pose.heading_deg + pose.boresight_deg)
    return (world + radar + (pose.e, pose.n, 0.0)).astype(np.float32)


class _Chunk(NamedTuple):
    item: Any
    f0: int  # 帧 [f0, f1)
    f1: int
    t1: float  # 最后一帧的时刻


class WorldScene:
    """世界坐标下的 GL 图元。只管画,不持有推算与地图。"""

    def __init__(self, gl_view) -> None:
        import pyqtgraph.opengl as gl

        self._gl = gl
        self._view = gl_view
        self._visible = False
        self._chunks: list[_Chunk] = []
        self._shown: list[bool] = []
        self._key: tuple | None = None  # (地图 version, 只要静止点, 色带范围)
        self.height_range = HEIGHT_RANGE_M
        self.partial = self._scatter(MAP_POINT_SIZE)
        self.current = self._scatter(CURRENT_POINT_SIZE)
        self.trajectory = gl.GLLinePlotItem(
            pos=EMPTY3, color=TRAJECTORY_COLOR, width=2.0, mode="line_strip"
        )
        self.vehicle = gl.GLLinePlotItem(
            pos=EMPTY3, color=VEHICLE_COLOR, width=1.5, mode="lines"
        )
        self.fov = gl.GLLinePlotItem(pos=EMPTY3, color=FOV_COLOR, width=1, mode="lines")
        # 不做深度测试:地面下方的点(俯仰偏低时很多)不被网格线挡住
        self.grid = gl.GLGridItem(color=GRID_COLOR, glOptions="additive")
        self.grid.setSize(GRID_SIZE_M, GRID_SIZE_M, 1.0)
        self.grid.setSpacing(GRID_SPACING_M, GRID_SPACING_M, 1.0)
        self._grid_cell: tuple[float, float] | None = None
        self._fixed = [
            self.grid,
            self.trajectory,
            self.partial,
            self.fov,
            self.vehicle,
            self.current,
        ]
        for item in self._fixed:
            self._add(item)

    def _scatter(self, size: float):
        return self._gl.GLScatterPlotItem(
            pos=EMPTY3, size=size, pxMode=True, glOptions="translucent"
        )

    def _add(self, item) -> None:
        item.setVisible(self._visible)
        self._view.addItem(item)

    @property
    def chunks(self) -> list[_Chunk]:
        return list(self._chunks)

    def set_visible(self, on: bool) -> None:
        self._visible = bool(on)
        for item in self._fixed:
            item.setVisible(self._visible)
        for chunk, shown in zip(self._chunks, self._shown):
            chunk.item.setVisible(self._visible and shown)

    def _drop_chunks(self) -> None:
        for chunk in self._chunks:
            self._view.removeItem(chunk.item)
        self._chunks, self._shown = [], []

    def clear(self) -> None:
        self._drop_chunks()
        self._key = None
        self._grid_cell = None
        for item in (self.partial, self.current):
            item.setData(pos=EMPTY3, color=np.zeros((0, 4), np.float32))
        for item in (self.trajectory, self.vehicle, self.fov):
            item.setData(pos=EMPTY3)

    # ---------------- 地图 ----------------
    def update_map(
        self,
        world: WorldMap,
        *,
        static_only: bool,
        recent_s: float | None,
        now_t: float,
    ) -> None:
        key = (world.version, bool(static_only), self.height_range)
        if (
            key != self._key
        ):  # 已有的世界点变了(改参重算 / 丢最早的帧)、换了筛选或色带范围
            self._drop_chunks()
            self._key = key
        self._seal(world, static_only)
        t_lo = -math.inf if recent_s is None else now_t - recent_s
        f_now = world.frame_at(now_t)
        covered = 0  # 帧 [0, covered) 已由「截至当前帧」的封块画出
        for k, chunk in enumerate(self._chunks):
            past = chunk.f1 - 1 <= f_now
            if past:
                covered = chunk.f1
            shown = past and chunk.t1 >= t_lo
            if shown != self._shown[k]:
                self._shown[k] = shown
                chunk.item.setVisible(self._visible and shown)
        f_lo = max(covered, world.first_frame_from(t_lo))
        points = world.points(f_lo, f_now + 1, static_only=static_only)
        self._set_points(self.partial, points, world.flat)

    def _seal(self, world: WorldMap, static_only: bool) -> None:
        """把时间跨度已满 CHUNK_SECONDS 的帧段封成一块(之后不再改动)。"""
        times = world.times
        start = self._chunks[-1].f1 if self._chunks else 0
        while start < len(times):
            end = int(np.searchsorted(times, times[start] + CHUNK_SECONDS, "left"))
            if end >= len(times):
                return  # 这一段还没满:留给 partial
            item = self._scatter(MAP_POINT_SIZE)
            points = world.points(start, end, static_only=static_only)
            self._set_points(item, points, world.flat)
            item.setVisible(False)
            self._view.addItem(item)
            self._chunks.append(_Chunk(item, start, end, float(times[end - 1])))
            self._shown.append(False)
            start = end

    def _set_points(self, item, points: np.ndarray, flat: bool) -> None:
        """颜色按实测高度;压平时画在雷达高度的水平面上。"""
        item.setData(
            pos=flatten(points) if flat else points,
            color=height_colors(points[:, 2], span=self.height_range),
        )

    # ---------------- 本帧、车辆、轨迹 ----------------
    def update_current(self, world_xyz: np.ndarray, flags: np.ndarray) -> None:
        pos = np.asarray(world_xyz, np.float32).reshape((-1, 3))
        colors = np.empty((len(pos), 4), np.float32)
        colors[:] = STATIC_COLOR
        colors[np.asarray(flags).reshape(-1) == MOVING] = MOVING_COLOR
        self.current.setData(pos=pos, color=colors)

    def update_vehicle(
        self, pose: Pose | None, params: OdometryParams, fov_deg: tuple[float, float]
    ) -> None:
        if pose is None:
            self.vehicle.setData(pos=EMPTY3)
            self.fov.setData(pos=EMPTY3)
            return
        self.vehicle.setData(pos=vehicle_lines(pose, params))
        self.fov.setData(pos=fov_lines(pose, params, fov_deg))
        cell = (
            round(pose.e / GRID_SPACING_M) * GRID_SPACING_M,
            round(pose.n / GRID_SPACING_M) * GRID_SPACING_M,
        )
        if cell != self._grid_cell:
            self._grid_cell = cell
            self.grid.resetTransform()
            self.grid.translate(cell[0], cell[1], -GROUND_BELOW_RADAR_M)

    def update_trajectory(self, positions: np.ndarray) -> None:
        pos = np.asarray(positions, np.float32).reshape((-1, 2))
        if len(pos) < 2:
            self.trajectory.setData(pos=EMPTY3)
            return
        xyz = np.c_[pos, np.full(len(pos), -GROUND_BELOW_RADAR_M, np.float32)]
        self.trajectory.setData(pos=xyz)

    # ---------------- 相机 ----------------
    def chase(self) -> None:
        """透视相机摆到车后的默认距离与仰角(中心与方位角由 follow 每帧给)。"""
        self._view.setCameraParams(fov=CAMERA_FOV_DEG)
        self._view.setCameraPosition(
            distance=FOLLOW_DISTANCE_M, elevation=FOLLOW_ELEVATION_DEG
        )

    def follow(self, pose: Pose | None) -> None:
        """镜头中心在车前 15 m,方位角在车后;距离与仰角不动(用户可调)。"""
        if pose is None:
            return
        h = math.radians(pose.heading_deg)
        self._view.opts["center"] = pg.Vector(
            pose.e + FOLLOW_LOOKAHEAD_M * math.sin(h),
            pose.n + FOLLOW_LOOKAHEAD_M * math.cos(h),
            0.0,
        )
        self._view.opts["azimuth"] = math.degrees(
            math.atan2(-math.cos(h), -math.sin(h))
        )
        self._view.update()

    def fit_all(self, positions: np.ndarray, points: np.ndarray = EMPTY3) -> None:
        """仰角 60° 框住整段轨迹(外加 FIT_MARGIN_M)与地图点的主体(各轴 5%–95% 分位,
        不让零星远点把画面拉远)。pyqtgraph 的 fov 是水平视场角,宽而矮的窗口要按 y 方向
        的跨度定距离。"""
        pos = np.asarray(positions, np.float64).reshape((-1, 2))
        if len(pos) == 0:
            return
        lo, hi = pos.min(axis=0) - FIT_MARGIN_M, pos.max(axis=0) + FIT_MARGIN_M
        if len(points):
            body = np.percentile(np.asarray(points)[:, :2], [5, 95], axis=0)
            lo, hi = np.minimum(lo, body[0]), np.maximum(hi, body[1])
        center, span = (lo + hi) / 2, hi - lo
        half_tan = math.tan(math.radians(self._view.opts.get("fov", 60.0)) / 2)
        aspect = max(self._view.width(), 1) / max(self._view.height(), 1)
        need_x = span[0] / (2 * half_tan)
        need_y = span[1] * math.sin(math.radians(FIT_ELEVATION_DEG)) * aspect
        need_y /= 2 * half_tan
        self._view.setCameraPosition(
            pos=QVector3D(float(center[0]), float(center[1]), 0.0),
            distance=max(need_x, need_y),
            elevation=FIT_ELEVATION_DEG,
            azimuth=-90.0,
        )


class WorldLayer(QObject):
    """世界坐标的会话与控件。`changed`:参数或显示选项变了,点云视图据此重画当前帧。"""

    changed = Signal()

    def __init__(self, gl_view) -> None:
        super().__init__()
        self.scene = WorldScene(gl_view)
        self.params = OdometryParams()
        self.odometry = RadarOdometry(self.params)
        self.map = WorldMap(flat=True)  # 与 height_seg 的默认项一致
        self.period_s = 0.05
        self.fov_deg = (-70.0, 70.0)
        self._last: int | None = None  # 最近收下的帧序号
        self._epoch: int | None = None
        self._fit_pending = True
        self.height_range = HEIGHT_RANGE_M
        self._height_locked = False
        #: 本帧画出的位置(世界坐标,压平时高度为 0),与检出点同序;不在地图里时为空
        self.current_xyz = EMPTY3

        self.camera_seg = Segmented(
            ["跟随", "全图"], tip="跟随:镜头在车后跟着走;全图:框住截至当前帧的整段轨迹"
        )
        self.span_seg = Segmented(
            [f"最近 {RECENT_S:g} s", "全部"],
            tip=f"地图累积范围(按 {CHUNK_SECONDS:g} s 分块取整);"
            "推算的航向误差随里程增长,长距离开不回原点",
        )
        self.points_seg = Segmented(
            ["静止点", "全部点"],
            tip="静止点:只累积多普勒自运动拟合判为静止的点(运动目标、速度错的点不进地图)",
        )
        self.height_seg = Segmented(
            ["压平", "实测"],
            tip="压平:不用实测俯仰,点按斜距放到雷达高度的水平面上(颜色仍是实测高度);"
            "实测:按测角给出的 xyz 放。1113 实采的俯仰偏低约 12°,实测时远处的点沉到地下",
        )
        self.axle_spin = self._spin(
            "L ", " m", 0.5, 10.0, self.params.rear_axle_m,
            "雷达到后轴中心的水平距离(卷尺量)。量错 10%,每个 90° 弯约多差 9°",
        )  # fmt: skip
        self.align_seg = Segmented(
            ["配准", "纯推算"],
            tip="配准:每帧把本帧静止点对到前几帧的局部地图上修正航向,视轴角随之在线自适应"
            "(读数栏显示当前值);纯推算:只用多普勒,视轴角固定为右边的初值,"
            "差 0.5° 每 100 m 航向约漂 8°",
        )
        self.yaw_spin = self._spin(
            "视轴 ", "°", -30.0, 30.0, self.params.boresight_deg,
            "雷达视轴相对车身轴线的初值,向右为正。配准时只是起点(约 3.5 s 收敛),纯推算时固定用它",
        )  # fmt: skip
        self.controls = [
            self.camera_seg,
            self.span_seg,
            self.points_seg,
            self.height_seg,
            self.align_seg,
            self.axle_spin,
            self.yaw_spin,
        ]
        self.camera_seg.currentIndexChanged.connect(self._on_camera)
        self.span_seg.currentIndexChanged.connect(lambda _i: self.changed.emit())
        self.points_seg.currentIndexChanged.connect(lambda _i: self.changed.emit())
        self.height_seg.currentIndexChanged.connect(lambda _i: self._on_mount())
        self.align_seg.currentIndexChanged.connect(lambda _i: self._on_mount())
        self.axle_spin.valueChanged.connect(lambda _v: self._on_mount())
        self.yaw_spin.valueChanged.connect(lambda _v: self._on_mount())

    @staticmethod
    def _spin(prefix, suffix, lo, hi, value, tip) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(lo, hi)
        spin.setDecimals(2)
        spin.setSingleStep(0.05)
        spin.setPrefix(prefix)
        spin.setSuffix(suffix)
        spin.setValue(value)
        spin.setKeyboardTracking(False)  # 回车或失焦才生效:每次改都要整段重算
        spin.setMinimumHeight(theme.MIN_HIT)
        spin.setToolTip(tip)
        return spin

    # ---------------- 会话 ----------------
    def reset(self, period_s: float, fov_deg: tuple[float, float]) -> None:
        """换会话:推算与地图清零;安装参数(L、视轴角)是车上的,保留。"""
        self.period_s = float(period_s)
        self.fov_deg = (float(fov_deg[0]), float(fov_deg[1]))
        self.restart()

    def restart(self) -> None:
        self.odometry.clear()
        self.map.clear()
        self.scene.clear()
        self._last = self._epoch = None
        self._fit_pending = True
        self._set_height_range(HEIGHT_RANGE_M)
        self._height_locked = False
        self.current_xyz = EMPTY3

    def feed(self, frames) -> None:
        """一批按顺序新处理出的帧;新一轮处理(从航迹起点开始)先清零。"""
        if not frames:
            return
        if frames[0].index == frames[0].track_epoch:
            self.restart()
        for frame in frames:
            self._add(frame)

    def add_shown(self, frame) -> None:
        """显示的帧:只收紧接着上一帧的那一帧(没经过 feed 的顺序显示);回看不重复累加。"""
        if self._last is None or frame.track_epoch != self._epoch:
            if frame.index == frame.track_epoch:
                self.restart()
                self._add(frame)
            return
        if frame.index == self._last + 1:
            self._add(frame)

    def _add(self, frame) -> None:
        t = frame.index * self.period_s
        det = frame.detections
        points = np.c_[
            np.asarray(det.xyz, np.float64).reshape(-1, 3),
            np.asarray(det.velocity_mps, np.float64),
        ]
        ego = estimate_ego_motion(points)
        if self.odometry.add_frame(t, points, ego):
            pose = self.odometry.pose_at(t)
            inliers = None if ego is None else ego.inliers
            self.map.add_frame(t, points, inliers, pose, self.params)
        self._last, self._epoch = frame.index, frame.track_epoch

    # ---------------- 参数 ----------------
    def _on_camera(self, index: int) -> None:
        if index == 0:
            self.scene.chase()
        self._fit_pending = index == 1
        self.changed.emit()

    def _on_mount(self) -> None:
        """L、视轴角初值、配准或压平改了:整段重算推算(含配准),地图重映射。"""
        self.params = OdometryParams(
            self.axle_spin.value(),
            self.yaw_spin.value(),
            align=self.align_seg.currentIndex() == 0,
        )
        self.odometry.recompute(self.params)
        self.map.rebuild(
            self.odometry, self.params, flat=self.height_seg.currentIndex() == 0
        )
        if self.camera_seg.currentIndex() == 1:
            self._fit_pending = True
        self.changed.emit()

    # ---------------- 画 ----------------
    def _set_height_range(self, span: tuple[float, float]) -> None:
        self.height_range = span
        self.scene.height_range = span

    def _lock_height_range(self) -> None:
        """地图攒够点后按实测高度定一次色带范围,之后不动(已封的块颜色一次性上传)。"""
        world = self.map
        if self._height_locked or world.point_count < HEIGHT_FIT_POINTS:
            return
        z = world.points(0, world.frame_count, static_only=False)[:, 2]
        if len(z) >= HEIGHT_FIT_POINTS:
            self._height_locked = True
            self._set_height_range(fit_height_range(z))

    def render(self, frame) -> None:
        now_t = frame.index * self.period_s
        scene, world = self.scene, self.map
        self._lock_height_range()
        static_only = self.points_seg.currentIndex() == 0
        recent_s = RECENT_S if self.span_seg.currentIndex() == 0 else None
        scene.update_map(world, static_only=static_only, recent_s=recent_s, now_t=now_t)
        f = world.frame_at(now_t)
        if f >= 0 and world.times[f] == now_t:
            xyz, flags = world.frame_points(f)
            self.current_xyz = flatten(xyz) if world.flat else xyz
            scene.update_current(self.current_xyz, flags)
        else:
            self.current_xyz = EMPTY3
            scene.update_current(EMPTY3, np.zeros(0, np.uint8))
        pose = self.odometry.pose_at(now_t)
        scene.update_vehicle(pose, self.params, self.fov_deg)
        positions = self.odometry.positions(now_t)
        scene.update_trajectory(positions)
        if self._fit_pending and len(positions):
            self._fit_pending = False
            f_lo = 0 if recent_s is None else world.first_frame_from(now_t - recent_s)
            scene.fit_all(positions, world.points(f_lo, f + 1, static_only=static_only))
        elif self.camera_seg.currentIndex() == 0:
            scene.follow(pose)

    def enter(self) -> None:
        """切到世界坐标:透视相机;跟随时摆到车后,全图时重新框住轨迹。"""
        self.scene.chase()
        self._fit_pending = self.camera_seg.currentIndex() == 1
        self.scene.set_visible(True)

    def reset_camera(self) -> None:
        """双击:跟随 → 回到车后默认距离与仰角;全图 → 重新框住轨迹。"""
        if self.camera_seg.currentIndex() == 0:
            self.scene.chase()
        else:
            self._fit_pending = True
        self.changed.emit()

    # ---------------- 图例与读数 ----------------
    def legend_html(self) -> str:
        def swatch(rgb) -> str:
            r, g, b = (int(round(c * 255)) for c in rgb[:3])
            return f"<span style='color:rgb({r},{g},{b})'>■</span>"

        lo, hi = self.height_range
        ramp = " ".join(swatch(c) for c in _RAMP_RGB)
        return (
            f"实测高度 {lo:+g} {ramp} {hi:+g} m(相对雷达"
            f"{',压平显示' if self.map.flat else ''})"
            f"　<span style='color:{theme.TEXT}'>● 本帧静止</span>"
            f"　<span style='color:{theme.WARN}'>● 本帧运动</span>"
            f"　<span style='color:{theme.SIGNAL}'>━ 推算轨迹</span>"
            f"　<span style='color:{theme.ACCENT}'>◎ 选中</span>"
        )

    def readout_html(self, frame) -> str:
        now_t = frame.index * self.period_s
        summary = self.odometry.summary_at(now_t)
        if summary is None:
            return "推算 --"
        if self.params.align:
            yaw = (
                f"视轴 {summary.boresight_deg:+.2f}°(自适应)· "
                f"配准 {summary.aligned_ratio * 100:.0f}% 帧"
            )
        else:  # 纯推算:航向会随视轴角误差漂,标黄提醒
            yaw = (
                f"<span style='color:{theme.WARN}'>视轴 {summary.boresight_deg:+.2f}°"
                "(固定,纯推算)</span>"
            )
        fit_color = (
            theme.BAD
            if summary.measured_ratio < 0.5
            else theme.WARN if summary.measured_ratio < 0.8 else theme.DIM
        )
        f_now = self.map.frame_at(now_t)
        f_lo = (
            self.map.first_frame_from(now_t - RECENT_S)
            if self.span_seg.currentIndex() == 0
            else 0
        )
        stats = self.map.stats(f_lo, f_now + 1)
        ratio = stats.static_ratio
        static = "--" if ratio is None else f"{ratio * 100:.0f}%"
        drawn = stats.points - (
            stats.moving_points if self.points_seg.currentIndex() == 0 else 0
        )
        return (
            f"推算 {summary.speed_mps:.1f} m/s · 转向 {summary.yaw_rate_dps:+.1f}°/s · "
            f"里程 {summary.distance_m:.0f} m · "
            f"<span style='color:{fit_color}'>拟合 {summary.measured_ratio * 100:.0f}%</span>"
            f" · 地图 {drawn:,} 点(静止 {static})· {yaw}"
        )
