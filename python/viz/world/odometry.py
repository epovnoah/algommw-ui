"""纯雷达推算:把逐帧的多普勒自运动拟合积分成车辆轨迹,并用场景配准修正航向。

运动学(后轴不侧滑):雷达装在后轴前方 L 米的车身中线上,车速 v、横摆角速度 ω(顺时针为正)时,
雷达在车身坐标(x 右、y 前)下的速度是 (ω·L, v)。把雷达测得的速度按视轴角 b 转回车身坐标,
每帧就量到

    v_x' = ω·L,   v_y' = v

两状态卡尔曼(v, ω;随机游走)按每帧拟合的协方差加权,弱帧不丢只降权;位姿用中点航向积分。
位姿原点 = 第一帧时的后轴中心,航向 0 = 初始行驶方向(画成北 / +y),没有高度信息。

只靠多普勒时对视轴角极敏感:b 的误差被当成横向速度,即 ω 的误差 = v·sin(Δb)/L,航向漂移
≈ Δb × 里程 / L(b 差 0.5°、L = 3.5 m 时每 100 m 约 8°;b 取 0 时 1113 直路上推出 −7°/s 的假转向,
地图拖成扇形)。b 还不是常数(1113 同一份采集里各 10 s 窗口的最优 b 差到 ±0.8°),标一次不够。

所以默认(`align`)每帧积分之后做航向配准(`align.py`):本帧静止点对前几帧的局部地图搜 ±2° 的
航向修正 d,航向加上 d;d 的平均值就是 b 的偏差造成的假转向,据此在线调 b:

    b ← b + ADAPT_GAIN · d · L / (v · Δt)        (ω 的误差 = v·Δb/L,一帧累积成 d)

收敛前那几秒的航向误差留在轨迹里(合成数据:b 初值差 2°,320 m 后航向差约 3.5°;
初值对时差 0.1°)。起步加大增益(按样本平均)试过,第一帧的修正量含配准前的漂移,会过冲,不更好。

1113 实采 b 初值 0 / 2.6° / 4.0° 收敛到同一值(+2.3° ~ +3.8°,随采集不同),轨迹基本重合。
车速 ≤ ALIGN_MIN_SPEED_MPS 时不配(静止时没有假转向,配准只会加噪声)。
L 量错 10%,每个 90° 弯的推算转角约多差 9°,配准能拉回大部分。长距离仍开不回原点(没有回环),
界面默认只画最近一段。
"""

import bisect
import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from .align import (
    ALIGN_FRAMES,
    ALIGN_MAX_RANGE_M,
    ALIGN_MIN_POINTS,
    ALIGN_MIN_REFERENCE,
    heading_correction,
    radar_xy_to_world,
)
from .ego import EgoMotion, horizontal_directions

DOPPLER_SIGMA_MPS = 0.1  # 拟合协方差用的单点多普勒噪声下限
SPEED_PROCESS = 2.0  # m/s²:帧间车速可变多快
YAW_RATE_PROCESS = 0.5  # rad/s²:帧间横摆角速度可变多快
ALIGN_MIN_SPEED_MPS = 3.0
#: 每帧按配准修正量调 b 的比例:时间常数约 1/ADAPT_GAIN 帧(70 ms 帧约 3.5 s)
ADAPT_GAIN = 0.02


@dataclass(frozen=True)
class OdometryParams:
    rear_axle_m: float = 3.5  # L:雷达到后轴中心的水平距离,用卷尺量
    #: b:雷达视轴相对车身轴线,顺时针(向右)为正。align 开着时是初值,之后在线自适应
    boresight_deg: float = 0.0
    align: bool = True  # 航向配准 + b 在线自适应;关 = 纯多普勒推算,b 固定


@dataclass(frozen=True)
class Pose:
    """后轴中心的位姿。e / n 是世界平面坐标(初始行驶方向 = n),米。"""

    t: float
    e: float
    n: float
    heading_deg: float  # 从初始行驶方向顺时针
    speed_mps: float
    yaw_rate_dps: float
    boresight_deg: float  # 这一帧用的视轴角(自适应后)


@dataclass(frozen=True)
class OdometrySummary:
    speed_mps: float
    yaw_rate_dps: float
    distance_m: float
    measured_ratio: float  # 有拟合的帧占比(截至该时刻)
    boresight_deg: float
    aligned_ratio: float  # 做了航向配准的帧占比(截至该时刻)


def align_points(points_xyzv: np.ndarray, ego: EgoMotion | None) -> np.ndarray:
    """配准用的点:静止(自运动拟合的内点)、斜距 ≤ ALIGN_MAX_RANGE_M,压平成水平 (x, y) ——
    斜距沿水平方向放,不用实测俯仰(见 ego.py)。没有拟合的帧没有可信的静止点,返回空。"""
    if ego is None:
        return np.zeros((0, 2))
    pts = np.asarray(points_xyzv, dtype=np.float64).reshape((-1, 4))
    slant = np.linalg.norm(pts[:, :3], axis=1)
    keep = ego.inliers & (slant <= ALIGN_MAX_RANGE_M) & np.isfinite(slant)
    return horizontal_directions(pts[keep]) * slant[keep, None]


def fit_covariance(points_xyzv: np.ndarray, ego: EgoMotion) -> np.ndarray:
    """(v_x, v_y) 的 2×2 协方差:σ²·(AᵀA)⁻¹,A 取内点的水平几何(同拟合)。"""
    pts = np.asarray(points_xyzv, dtype=np.float64).reshape((-1, 4))[ego.inliers]
    a = -horizontal_directions(pts)
    sigma = max(DOPPLER_SIGMA_MPS, ego.rms_mps)
    try:
        return np.linalg.inv(a.T @ a) * sigma**2
    except np.linalg.LinAlgError:
        return np.diag([1e6, 1e6])


class RadarOdometry:
    """帧按时间顺序进来。原始拟合与配准用的点全部留着:参数改了就 `recompute` 整段重算
    (含配准,每帧约 1.2 ms)。"""

    def __init__(self, params: OdometryParams = OdometryParams()):
        self.params = params
        #: (t, (v_x, v_y) 或 None, 协方差或 None, 配准用的水平点 (K, 2))
        self._fits: list[tuple] = []
        self._reset_state()

    def _reset_state(self) -> None:
        self._t: list[float] = []
        self._e: list[float] = []
        self._n: list[float] = []
        self._h: list[float] = []  # rad,从初始方向顺时针,不缠绕
        self._v: list[float] = []
        self._w: list[float] = []  # 横摆角速度,rad/s
        self._measured: list[bool] = []
        self._b: list[float] = []  # 每帧用的视轴角,度
        self._aligned: list[bool] = []
        self._x = np.zeros(2)  # 状态:车速、横摆角速度
        self._p = np.diag([25.0, 1.0])
        self._boresight = self.params.boresight_deg  # 当前 b(自适应)
        self._reference: deque[np.ndarray] = deque(maxlen=ALIGN_FRAMES)

    @property
    def frame_count(self) -> int:
        return len(self._t)

    @property
    def last_t(self) -> float | None:
        return self._t[-1] if self._t else None

    def add_frame(
        self, t: float, points_xyzv: np.ndarray, ego: EgoMotion | None
    ) -> bool:
        """返回是否收下(时间不晚于上一帧的丢掉)。"""
        if self._fits and t <= self._fits[-1][0]:
            return False
        xy = align_points(points_xyzv, ego)
        if ego is None:
            self._fits.append((float(t), None, None, xy))
        else:
            z = np.array([ego.vx, ego.vy])
            self._fits.append((float(t), z, fit_covariance(points_xyzv, ego), xy))
        self._step(*self._fits[-1])
        return True

    def recompute(self, params: OdometryParams) -> None:
        self.params = params
        self._reset_state()
        for fit in self._fits:
            self._step(*fit)

    def clear(self) -> None:
        self._fits.clear()
        self._reset_state()

    def _step(
        self, t: float, z: np.ndarray | None, cov: np.ndarray | None, xy: np.ndarray
    ) -> None:
        dt = t - self._t[-1] if self._t else 0.0
        if self._t:
            v, w = self._x
            h_mid = self._h[-1] + w * dt / 2
            e = self._e[-1] + v * math.sin(h_mid) * dt
            n = self._n[-1] + v * math.cos(h_mid) * dt
            h = self._h[-1] + w * dt
            self._p = self._p + np.diag(
                [SPEED_PROCESS**2 * dt, YAW_RATE_PROCESS**2 * dt]
            )
        else:
            e = n = h = 0.0
        if z is not None and cov is not None:
            b = math.radians(self._boresight)
            rot = np.array([[math.cos(b), math.sin(b)], [-math.sin(b), math.cos(b)]])
            z_vehicle = rot @ z  # 雷达坐标 → 车身坐标
            r = rot @ cov @ rot.T
            hm = np.array([[0.0, self.params.rear_axle_m], [1.0, 0.0]])  # (v, ω) → 量测
            s = hm @ self._p @ hm.T + r
            gain = self._p @ hm.T @ np.linalg.inv(s)
            self._x = self._x + gain @ (z_vehicle - hm @ self._x)
            self._p = (np.eye(2) - gain @ hm) @ self._p
        v = float(self._x[0])
        length = self.params.rear_axle_m
        aligned = (
            self.params.align
            and dt > 0
            and v > ALIGN_MIN_SPEED_MPS
            and len(xy) >= ALIGN_MIN_POINTS
            and len(self._reference) >= ALIGN_MIN_REFERENCE
        )
        if aligned:
            d = heading_correction(
                xy,
                np.concatenate(self._reference),
                e,
                n,
                h,
                length,
                math.radians(self._boresight),
            )
            h += d
            self._boresight += ADAPT_GAIN * math.degrees(d) * length / (v * dt)
        if len(xy):
            world = radar_xy_to_world(
                xy, e, n, np.array([h]), length, math.radians(self._boresight)
            )
            self._reference.append(world[0])
        self._t.append(t)
        self._e.append(e)
        self._n.append(n)
        self._h.append(h)
        self._v.append(float(self._x[0]))
        self._w.append(float(self._x[1]))
        self._measured.append(z is not None)
        self._b.append(self._boresight)
        self._aligned.append(aligned)

    # ---------------- 查询 ----------------
    def index_at(self, t: float) -> int:
        """时刻不晚于 t 的最后一帧;t 早于第一帧时为 -1。"""
        return bisect.bisect_right(self._t, t) - 1

    def pose_at(self, t: float) -> Pose | None:
        """t 时刻后轴的位姿,帧间线性插值;不外推(t 在已有帧之外时为 None)。"""
        ts = self._t
        if not ts or t < ts[0] or t > ts[-1]:
            return None
        i = min(bisect.bisect_right(ts, t), len(ts) - 1)
        i0 = max(i - 1, 0)
        frac = 0.0 if ts[i] == ts[i0] else (t - ts[i0]) / (ts[i] - ts[i0])

        def lerp(values: list[float]) -> float:
            return values[i0] + (values[i] - values[i0]) * frac

        return Pose(
            t=float(t),
            e=lerp(self._e),
            n=lerp(self._n),
            heading_deg=math.degrees(lerp(self._h)),
            speed_mps=lerp(self._v),
            yaw_rate_dps=math.degrees(lerp(self._w)),
            boresight_deg=lerp(self._b),
        )

    def positions(self, t_max: float | None = None) -> np.ndarray:
        """(K, 2) 后轴轨迹 (e, n),截至 t_max(含)。"""
        k = len(self._t) if t_max is None else self.index_at(t_max) + 1
        return (
            np.stack([self._e[:k], self._n[:k]], axis=1) if k > 0 else np.zeros((0, 2))
        )

    def summary_at(self, t: float) -> OdometrySummary | None:
        k = self.index_at(t)
        if k < 0:
            return None
        path = self.positions(t)
        distance = float(np.hypot(*np.diff(path, axis=0).T).sum()) if k else 0.0
        return OdometrySummary(
            speed_mps=self._v[k],
            yaw_rate_dps=math.degrees(self._w[k]),
            distance_m=distance,
            measured_ratio=sum(self._measured[: k + 1]) / (k + 1),
            boresight_deg=self._b[k],
            aligned_ratio=sum(self._aligned[: k + 1]) / (k + 1),
        )
