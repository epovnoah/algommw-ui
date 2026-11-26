"""世界地图:逐帧原始点按推算位姿放进同一张平面地图。纯 numpy,不碰 Qt。

地图保存每个原始点(雷达坐标 x, y, z、径向速度)和它的帧时刻,世界坐标随时可以重算:
改 L 或 b 时整段按新位姿重新映射(`rebuild`)。

单点变换(h = 车身航向,从初始方向顺时针;b = 该帧的视轴角,随位姿给出(在线自适应);L = 雷达到后轴):

    雷达位置 = 后轴 + Rz(h)·(0, L)
    [E N]    = 雷达位置 + Rz(h + b)·(x, y),   U = z(相对雷达高度)
    Rz(h)·(x, y) = (x·cos h + y·sin h,  −x·sin h + y·cos h)

压平(`flat`):水平位置按斜距放、不用实测俯仰(实测俯仰不可信时);U 仍存实测高度,界面用它着色。

点标记:有自运动拟合的帧用 RANSAC 内点(静止)/ 外点(运动);没有拟合的帧按推算车速预测
静止点该有的多普勒,容差 STATIC_TOL_MPS + STATIC_TOL_REL·v。
"""

import math
from dataclasses import dataclass

import numpy as np

from .ego import horizontal_directions
from .odometry import OdometryParams, Pose, RadarOdometry

STATIC, MOVING = 1, 2  # 点标记
STATIC_TOL_MPS = 0.5
STATIC_TOL_REL = 0.03


@dataclass(frozen=True)
class MapStats:
    frames: int
    points: int
    static_points: int
    moving_points: int

    @property
    def static_ratio(self) -> float | None:
        judged = self.static_points + self.moving_points
        return self.static_points / judged if judged else None


def radar_to_world(
    points_xyz: np.ndarray, pose: Pose, params: OdometryParams, flat: bool = False
) -> np.ndarray:
    """(N, ≥3) 雷达坐标点 → (N, 3) 世界坐标 (E, N, U),U 恒为实测高度 z。

    flat:水平位置按斜距 |p| 沿方位放(当俯仰为 0),不按 √(x² + y²) —— 实测俯仰不可信时
    (1113 实采中位 −11° ~ −13°)后者会把远处的点往雷达拉近约 2%。"""
    h = math.radians(pose.heading_deg)
    hb = math.radians(pose.heading_deg + pose.boresight_deg)
    c, s = math.cos(hb), math.sin(hb)
    radar_e = pose.e + params.rear_axle_m * math.sin(h)
    radar_n = pose.n + params.rear_axle_m * math.cos(h)
    p = np.asarray(points_xyz, dtype=np.float64)
    x, y = p[:, 0], p[:, 1]
    if flat:
        slant = np.linalg.norm(p[:, :3], axis=1)
        x, y = (horizontal_directions(p) * slant[:, None]).T
    out = np.empty((len(p), 3), dtype=np.float64)
    out[:, 0] = radar_e + x * c + y * s
    out[:, 1] = radar_n - x * s + y * c
    out[:, 2] = p[:, 2]
    return out


def classify_by_speed(
    points_xyzv: np.ndarray, speed_mps: float, boresight_deg: float
) -> np.ndarray:
    """没有自运动拟合的帧:静止点的径向速度 = −(v_雷达 · p̂),雷达速度取车速沿车身轴线,
    p̂ 取水平方向(同拟合,不用实测俯仰)。"""
    b = math.radians(boresight_deg)
    vx, vy = -speed_mps * math.sin(b), speed_mps * math.cos(b)  # 车身前向在雷达坐标里
    direction = horizontal_directions(points_xyzv)
    expected = -(vx * direction[:, 0] + vy * direction[:, 1])
    tolerance = STATIC_TOL_MPS + STATIC_TOL_REL * abs(speed_mps)
    static = np.abs(points_xyzv[:, 3] - expected) < tolerance
    return np.where(static, STATIC, MOVING).astype(np.uint8)


class WorldMap:
    """帧按时间顺序进来,当场映射。点数超过 max_points 时整帧丢掉最早的;
    `version` 在已有的世界点变化或消失时加 1(视图据此整体重画)。"""

    def __init__(self, max_points: int = 2_000_000, flat: bool = False):
        self.max_points = int(max_points)
        self.flat = bool(flat)  # 见 radar_to_world;改它要 rebuild
        self.version = 0
        self.clear()

    def clear(self) -> None:
        cap, fcap = 4096, 1024
        self._raw = np.empty((cap, 4), dtype=np.float32)
        self._world = np.empty((cap, 3), dtype=np.float32)
        self._flags = np.zeros(cap, dtype=np.uint8)
        self._frame_t = np.empty(fcap, dtype=np.float64)
        self._frame_start = np.zeros(fcap + 1, dtype=np.int64)
        self._frame_fitted = np.zeros(fcap, dtype=bool)  # 标记来自自运动拟合
        self._np = 0
        self._nf = 0
        self.version += 1

    @property
    def frame_count(self) -> int:
        return self._nf

    @property
    def point_count(self) -> int:
        return self._np

    @property
    def times(self) -> np.ndarray:
        return self._frame_t[: self._nf]

    # ---------------- 输入 ----------------
    def add_frame(
        self,
        t: float,
        points_xyzv: np.ndarray,
        inliers: np.ndarray | None,
        pose: Pose,
        params: OdometryParams,
    ) -> bool:
        """inliers:自运动拟合的静止点掩码;None = 该帧没有拟合。返回是否收下。"""
        if self._nf and t <= self._frame_t[self._nf - 1]:
            return False
        points = np.asarray(points_xyzv, dtype=np.float32).reshape((-1, 4))
        self._reserve(len(points))
        a, b = self._np, self._np + len(points)
        self._raw[a:b] = points
        f = self._nf
        self._frame_t[f] = t
        self._frame_fitted[f] = inliers is not None
        self._frame_start[f + 1] = b
        self._np, self._nf = b, f + 1
        if inliers is not None:
            self._flags[a:b] = np.where(inliers, STATIC, MOVING)
        self._map_frame(f, pose, params)
        return True

    def rebuild(
        self,
        odometry: RadarOdometry,
        params: OdometryParams,
        flat: bool | None = None,
    ) -> None:
        """按新的推算轨迹与 L(及压平与否)重映射每一帧。"""
        if flat is not None:
            self.flat = bool(flat)
        for f in range(self._nf):
            pose = odometry.pose_at(float(self._frame_t[f]))
            if pose is None:  # 推算里没有这一帧(不该发生):不画
                a, b = self._frame_start[f], self._frame_start[f + 1]
                self._world[a:b] = np.nan
            else:
                self._map_frame(f, pose, params)
        self.version += 1

    def _map_frame(self, f: int, pose: Pose, params: OdometryParams) -> None:
        a, b = self._frame_start[f], self._frame_start[f + 1]
        raw = self._raw[a:b]
        self._world[a:b] = radar_to_world(raw, pose, params, self.flat)
        if not self._frame_fitted[f]:
            self._flags[a:b] = classify_by_speed(
                raw, pose.speed_mps, pose.boresight_deg
            )

    def _reserve(self, n: int) -> None:
        if self._np + n > self.max_points and self._nf:
            self._drop_oldest(self._np + n - int(self.max_points * 0.9))
        while self._np + n > len(self._raw):
            grow = max(len(self._raw), n)
            self._raw = np.concatenate([self._raw, np.empty((grow, 4), np.float32)])
            self._world = np.concatenate([self._world, np.empty((grow, 3), np.float32)])
            self._flags = np.concatenate([self._flags, np.zeros(grow, np.uint8)])
        if self._nf + 1 > len(self._frame_t):
            grow = len(self._frame_t)
            self._frame_t = np.concatenate([self._frame_t, np.empty(grow)])
            self._frame_start = np.concatenate(
                [self._frame_start, np.zeros(grow, np.int64)]
            )
            self._frame_fitted = np.concatenate(
                [self._frame_fitted, np.zeros(grow, bool)]
            )

    def _drop_oldest(self, min_points: int) -> None:
        """从头整帧丢掉,至少腾出 min_points 个点。"""
        starts = self._frame_start[: self._nf + 1]
        drop_f = min(max(int(np.searchsorted(starts, min_points)), 1), self._nf)
        drop_p = int(starts[drop_f])
        keep_p, keep_f = self._np - drop_p, self._nf - drop_f
        for arr in (self._raw, self._world, self._flags):
            arr[:keep_p] = arr[drop_p : self._np]
        for arr in (self._frame_t, self._frame_fitted):
            arr[:keep_f] = arr[drop_f : self._nf]
        self._frame_start[: keep_f + 1] = (
            self._frame_start[drop_f : self._nf + 1] - drop_p
        )
        self._np, self._nf = keep_p, keep_f
        self.version += 1

    # ---------------- 输出 ----------------
    def frame_at(self, t: float) -> int:
        """时刻不晚于 t 的最后一帧;没有时为 -1。"""
        return int(np.searchsorted(self.times, t, side="right")) - 1

    def first_frame_from(self, t: float) -> int:
        """时刻不早于 t 的第一帧(可能等于 frame_count)。"""
        return int(np.searchsorted(self.times, t, side="left"))

    def points(self, f0: int, f1: int, *, static_only: bool) -> np.ndarray:
        """帧 [f0, f1) 的世界点,(K, 3) float32。"""
        f0, f1 = max(0, f0), min(self._nf, f1)
        if f1 <= f0:
            return np.zeros((0, 3), np.float32)
        a, b = self._frame_start[f0], self._frame_start[f1]
        world = self._world[a:b]
        keep = np.isfinite(world[:, 0])
        if static_only:
            keep &= self._flags[a:b] != MOVING
        return world[keep]

    def frame_points(self, f: int) -> tuple[np.ndarray, np.ndarray]:
        """第 f 帧的世界点与标记(与该帧检出点同序)。"""
        a, b = self._frame_start[f], self._frame_start[f + 1]
        return self._world[a:b], self._flags[a:b]

    def stats(self, f0: int, f1: int) -> MapStats:
        """帧 [f0, f1) 的点数与静止 / 运动占比。"""
        f0, f1 = max(0, f0), min(self._nf, f1)
        a = self._frame_start[f0] if f1 > f0 else 0
        b = self._frame_start[f1] if f1 > f0 else 0
        flags = self._flags[a:b]
        return MapStats(
            frames=max(f1 - f0, 0),
            points=int(b - a),
            static_points=int((flags == STATIC).sum()),
            moving_points=int((flags == MOVING).sum()),
        )
