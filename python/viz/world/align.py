"""航向配准:把本帧静止点对到前几帧拼出的局部地图上,找让它们重合得最好的航向修正。

为什么要它:纯多普勒推算看不见转向,只能从雷达横向速度反推(ω = v_x'/L),视轴角 b 差 Δb
就凭空多出 v·Δb/L 的转向,13 m/s 时 2.5° 约 9°/s,地图拖成扇形。b 还不是常数:1113 实采逐个
10 s 窗口求「让地图最锐的 b」在同一份采集里差到 ±0.8°,弯道上的 β 也分不开弯与偏差。场景能分开:
航向对了,同一个静止物在相邻帧里落在同一处。

做法(相关性扫描匹配,只搜航向):参考 = 前 ALIGN_FRAMES 帧的静止点(世界坐标,压平),
落进 ALIGN_CELL_M 栅格计数后做 3×3 平滑;本帧静止点按候选航向 h + d(d ∈ ±ALIGN_SEARCH_DEG,
步长 ALIGN_STEP_DEG,绕后轴转)变到世界坐标,取栅格值之和最大的 d。前后方向的平移不搜:车速由
多普勒给得很准,沿路方向的平移在平行路边线上本来也定不住。

1113 实采(2026-11-20):10 s 窗口地图锐度(占据格数 / 点数)由最好的常数 b 的 0.44–0.51
降到 0.34–0.36,每帧约 1.2 ms;b 初值 0 / 2.6 / 4.0 都收敛到同一 b(在线自适应,见 odometry)。
"""

import numpy as np

ALIGN_CELL_M = 0.3
ALIGN_FRAMES = 40  # 参考局部地图的帧数(70 ms 帧约 2.8 s)
ALIGN_SEARCH_DEG = 2.0
ALIGN_STEP_DEG = 0.1
ALIGN_MAX_RANGE_M = 60.0  # 远处的点方位误差的横向量大
ALIGN_MIN_POINTS = 10
ALIGN_MIN_REFERENCE = 5  # 参考至少这么多帧才配
_MARGIN = 5  # 栅格四周留的格数(候选转角把点挪出参考范围时不越界)

_OFFSETS = np.radians(
    np.arange(-ALIGN_SEARCH_DEG, ALIGN_SEARCH_DEG + 1e-9, ALIGN_STEP_DEG)
)


def radar_xy_to_world(
    xy: np.ndarray,
    e: float,
    n: float,
    heading_rad: np.ndarray,
    rear_axle_m: float,
    boresight_rad: float,
) -> np.ndarray:
    """(K, 2) 雷达坐标水平点 → 世界坐标;heading_rad 可以是 (C,) 一组候选,结果 (C, K, 2)。"""
    h = np.atleast_1d(heading_rad)[:, None]
    hb = h + boresight_rad
    c, s = np.cos(hb), np.sin(hb)
    radar_e = e + rear_axle_m * np.sin(h)
    radar_n = n + rear_axle_m * np.cos(h)
    x, y = xy[None, :, 0], xy[None, :, 1]
    return np.stack([radar_e + x * c + y * s, radar_n - x * s + y * c], axis=-1)


def heading_correction(
    xy: np.ndarray,
    reference: np.ndarray,
    e: float,
    n: float,
    heading_rad: float,
    rear_axle_m: float,
    boresight_rad: float,
) -> float:
    """本帧水平点 xy(雷达坐标)对参考点 reference(世界坐标)的最佳航向修正,弧度。"""
    lo = reference.min(axis=0) - _MARGIN * ALIGN_CELL_M
    ij = np.floor((reference - lo) / ALIGN_CELL_M).astype(np.int64)
    shape = ij.max(axis=0) + 1 + _MARGIN
    grid = np.zeros(shape)
    np.add.at(grid, (ij[:, 0], ij[:, 1]), 1.0)
    smooth = grid.copy()
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx or dy:
                smooth += 0.5 * np.roll(np.roll(grid, dx, 0), dy, 1)
    world = radar_xy_to_world(
        xy, e, n, heading_rad + _OFFSETS, rear_axle_m, boresight_rad
    )
    q = np.floor((world - lo) / ALIGN_CELL_M).astype(np.int64)
    inside = (q >= 0).all(axis=-1) & (q < shape).all(axis=-1)
    q = np.where(inside[..., None], q, 0)
    score = np.where(inside, smooth[q[..., 0], q[..., 1]], 0.0).sum(axis=1)
    if score.max() <= 0:
        return 0.0  # 与参考毫无重合:不修
    # 分数相同时取离 0 最近的修正(平局不该把航向推向搜索边界)
    return float(_OFFSETS[int(np.argmax(score - 1e-9 * np.abs(_OFFSETS)))])
