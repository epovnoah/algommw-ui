"""多普勒自运动拟合:由一帧检出点估计雷达自身的水平速度,并分出静止点。

静止点的径向速度满足(core 的 xDoppler 远离为正)

    v_r = −(v_x·x + v_y·y) / √(x² + y²)      (假设 v_z = 0、静止点在雷达高度附近)

(v_x, v_y) 是雷达在雷达坐标下的速度(x 右、y 前)。**只用水平几何,不用实测俯仰**:1113 实采
20 m 外的点俯仰中位数是 −11° ~ −13°,与装车上仰约 14° 一致(compass 开放项第 2 条;profile 的
`[mount] elev_tilt_deg` 还没补偿)。带上俯仰(分母取 |p|)时内点残差中位数 0.12 m/s、车速偏高
2.5–3.5%;去掉俯仰后残差降到 0.08–0.10 m/s(2026-11-20)。
近处高大的静止物(俯仰真不小)会因此略偏,但它们在一帧里占少数。

RANSAC:每次取两个点解出一个候选速度,
数残差小于 INLIER_TOL_MPS 的点,取最多的候选,再对这些内点做两轮最小二乘。内点就是静止点,
外点是运动目标或速度错的点(如双 PRF 的单帧折叠点)。点对数超过 MAX_PAIRS 时按固定种子抽样,
结果可复现。
"""

import math
from dataclasses import dataclass

import numpy as np

INLIER_TOL_MPS = 0.25
MIN_INLIERS = 4
#: 内点的方位跨度至少这么大,否则速度方向解不出来(所有点挤在一个方向上)
MIN_SPREAD_DEG = 10.0
MAX_PAIRS = 400


@dataclass(frozen=True, eq=False)
class EgoMotion:
    vx: float  # 雷达速度,雷达坐标,m/s(x 右、y 前)
    vy: float
    inliers: np.ndarray  # (N,) bool,静止点
    rms_mps: float  # 内点残差的均方根

    @property
    def speed_mps(self) -> float:
        return math.hypot(self.vx, self.vy)


def horizontal_directions(points: np.ndarray) -> np.ndarray:
    """(N, ≥2) 点 → (N, 2) 水平方向的单位向量 (x, y)/√(x² + y²)。"""
    xy = np.asarray(points, dtype=np.float64)[:, :2]
    return xy / np.maximum(np.hypot(xy[:, 0], xy[:, 1]), 1e-6)[:, None]


def _pairs(n: int) -> tuple[np.ndarray, np.ndarray]:
    if n * (n - 1) // 2 <= MAX_PAIRS:
        return np.triu_indices(n, k=1)
    rng = np.random.default_rng(0)
    i = rng.integers(0, n, MAX_PAIRS)
    return i, (i + rng.integers(1, n, MAX_PAIRS)) % n  # j ≠ i


def estimate_ego_motion(points_xyzv: np.ndarray) -> EgoMotion | None:
    """points_xyzv:(N, 4) 的 x, y, z, 径向速度(远离为正)。点太少或几何退化时返回 None。"""
    raw = np.asarray(points_xyzv, dtype=np.float64).reshape((-1, 4))
    finite = np.isfinite(raw).all(axis=1)
    pts = raw[finite]
    n = len(pts)
    if n < MIN_INLIERS:
        return None
    a = -horizontal_directions(pts)  # 线性模型 a·(v_x, v_y) = v_r 的各行
    v_r = pts[:, 3]

    i, j = _pairs(n)
    det = a[i, 0] * a[j, 1] - a[i, 1] * a[j, 0]
    ok = np.abs(det) > math.sin(math.radians(MIN_SPREAD_DEG)) * 0.5
    if not ok.any():
        return None
    i, j, det = i[ok], j[ok], det[ok]
    vx = (v_r[i] * a[j, 1] - v_r[j] * a[i, 1]) / det  # 每个点对的克拉默法则解
    vy = (a[i, 0] * v_r[j] - a[j, 0] * v_r[i]) / det
    residual = np.abs(a[None, :, 0] * vx[:, None] + a[None, :, 1] * vy[:, None] - v_r)
    near = residual < INLIER_TOL_MPS
    # 内点数相同时取残差小的:残差和按 tol·n 归一,不会压过一个内点
    score = near.sum(axis=1) - np.where(near, residual, 0).sum(axis=1) / (
        INLIER_TOL_MPS * n
    )
    inliers = near[int(np.argmax(score))]

    sol, res = np.zeros(2), np.zeros(n)
    for _ in range(2):  # 内点最小二乘,再按新解重选一次内点
        if inliers.sum() < MIN_INLIERS:
            return None
        sol, *_ = np.linalg.lstsq(a[inliers], v_r[inliers], rcond=None)
        res = np.abs(a @ sol - v_r)
        inliers = res < INLIER_TOL_MPS
    if inliers.sum() < MIN_INLIERS:
        return None
    azimuth = np.degrees(np.arctan2(pts[inliers, 0], pts[inliers, 1]))
    if np.ptp(azimuth) < MIN_SPREAD_DEG:
        return None
    rms = float(np.sqrt(np.mean(res[inliers] ** 2)))
    full = np.zeros(len(raw), bool)  # 与输入逐行对齐(非有限的行不是内点)
    full[finite] = inliers
    return EgoMotion(float(sol[0]), float(sol[1]), full, rms)
