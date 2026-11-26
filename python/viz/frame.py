"""一帧的数据模型 —— 处理线程产出、显示与落盘线程消费的共同口径。

只放数据与纯换算,不碰线程/队列/Qt:`pipeline`(生产)、`saver`(落盘)与 `ui/`(显示)
都依赖它,它不依赖它们。
"""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

EMPTY3 = np.zeros((0, 3), np.float32)

#: 配对类别 → 标签(core PAIR_PAIRED / SINGLE / OVERLAP / CARRIED;待确认点不发布)。
#: 延续 = 没有相邻异类帧伙伴,沿用 k−2 同类帧已确认的速度假设。
#: 单帧点的速度只是单帧折叠值,目标超过单帧不模糊速度时它是错的;-1 = 没有配对级。
PAIR_LABELS = {0: "已配对", 1: "单帧", 2: "救回", 4: "延续", -1: "—"}


def signed_velocity(dop_bins, n_dop: int, step: float):
    """无符号多普勒 bin → **零频居中的轴坐标**(step=Δv 时读作 m/s,step=1 时即居中 bin)。

    RD 图/速度轴是 fftshift 过的零频居中轴,而 core 的 usDopplerBin 是无符号 fft bin
    (上半轴 [N/2, N) 是负频率),所以**在轴上定位**必须在 UI 侧折号。⚠ 这一步算的是
    **坐标**,不是速度真值 —— 速度一律取 core 的 xDoppler(它已折号、且带亚 bin 项,
    见 pipeline._extract 的 vel)。

    折号口径与 core 的 lWaveformSignedBin 同:平局 bin == N/2 折负。

    DDM 下 RD 图画的是**折叠基带图**(rd_scale=1、横轴即 bin),标记同样按本节口径折号
    定位 —— 它的 xDoppler 是 fold 展开后的真值(±28.8),与折叠图不在同一空间,标上去
    会整片跑出图外。
    """
    half = n_dop // 2
    shifted = (np.asarray(dop_bins, int) + half) % n_dop
    return (shifted - half) * step


class FrameResult(NamedTuple):
    """一帧的全部产物。det/vel/tracks 既服务当前帧面板,也**原样留在缓存里喂 3D
    累积层**(截至视图光标帧的全部历史帧都由视图按需取用,见 CloudView 的 hist
    参数);其余字段只服务当前帧,消费后即弃。"""

    fi: int
    proc_ms: float  # 本帧 5 级链耗时,用于显示实时余量
    det: np.ndarray  # (N,3) float32,点云 xyz
    vel: (
        np.ndarray
    )  # (N,) float32,与 det 行对齐的有符号径向速度(m/s),产帧时换算好(口径见 _extract)
    tracks: list  # [(track_id, x, y, z, v_r, |v|), ...]
    fields: dict  # 检出点全字段(每项 (N,) 数组),供检测表/联动
    rd_db: np.ndarray  # (n_range, n_doppler) float32,归一化 dB
    spectra_db: np.ndarray  # (N, bins) float32,每检出点方位谱(归一化 dB)
    spectra_axis: np.ndarray  # (bins,) float32,方位角轴(度)
    #: 本帧会话的谱导出状态("ok"/"notimpl"/"off"):空谱是"后端不导出"而不是
    #: "用户没选点"时要能说清楚。随帧携带(不是管道上的活属性)—— 回看旧帧时显示的
    #: 是那一帧当时的后端。
    spectra_state: str = "off"
    reprocessed: bool = False  # True = 改参后对同一帧的重算,消费者应替换而非追加
    #: 双 PRF:每个输出点的配对类别(PAIR_LABELS 的键)与展开周期,与 det 同序;其余板为 None。
    #: 不放进 fields:fields 是 Point_t 的原样镜像。
    pair_kind: np.ndarray | None = None
    pair_unfold: np.ndarray | None = None


@dataclass(frozen=True)
class RadarInfo:
    """一份会话(板 × 采集)的雷达几何与标度,界面的信息栏与坐标轴用它。

    数值一律取自已载入的 ChainCfg(`xWave` / `xArray`),不在这里从物理量另推一份 ——
    标度量是波形的函数,只认 profile 载入时那一个来源(见 .claude/CLAUDE.md 硬约定)。
    """

    profile: str  # 板名(profile 目录名,如 "sr61")
    modulation: str  # "tdm" / "ddm"
    n_tx: int
    n_rx: int
    n_virtual: int
    n_range: int  # RD 图距离维 = 每 chirp ADC 采样数
    n_doppler: int  # RD 图多普勒维
    range_step_m: float
    doppler_step_mps: float
    frame_period_s: float
    n_frames: int  # 采集总帧数
    capture: str  # 采集目录名
    #: DOA 方位视场 (min, max),度;取自 ChainCfg 的 xDoa.xFov,未启用(bEnable=0)时为 None
    fov_azimuth_deg: tuple[float, float] | None = None
    #: 有效距离门数:实采样(采集格式 afm761)的距离 FFT 后半是频谱镜像,只有前半有效
    #: (= num_adc_samples // 2);复采样 = num_adc_samples。0 = 未指定,按 n_range 算
    n_range_valid: int = 0
    #: 步进频的细距离步长 S_f(m / 全局慢时间 bin,带符号);0 = 波形无载频步进
    fine_range_step_m: float = 0.0
    #: 全帧 chirp 数 N(步进频的慢时间窗长);无步进时 0
    n_slow: int = 0

    @property
    def gates(self) -> int:
        """显示用的有效距离门数(所有以距离为轴的图只画前 gates 个门)。"""
        return self.n_range_valid or self.n_range

    @property
    def stepped(self) -> bool:
        """波形带载频步进(走解耦链,有「解耦」一级)。"""
        return self.fine_range_step_m != 0.0


# 下面几个类持有 numpy 数组:eq=False 让相等/哈希退回按对象身份 —— dataclass 自动生成的
# __eq__ 会拿数组逐元素比较,结果不是 bool,一比就抛 ValueError。


@dataclass(frozen=True, eq=False)
class Detections:
    """一帧检出点,列存放;第 i 行对应 `FrameResult.fields` 各列的第 i 个元素。

    速度只认 core 的 xDoppler(两板都已是有符号真值,口径见 pipeline.extract_result),
    这里不再折号。
    """

    range_bin: np.ndarray  # (N,) int32
    doppler_bin: np.ndarray  # (N,) int32,core 的无符号 fft bin
    range_m: np.ndarray  # (N,) float32
    velocity_mps: np.ndarray  # (N,) float32,= fields["xDoppler"]
    snr_db: np.ndarray  # (N,) float32
    noise_db: np.ndarray  # (N,) float32
    azimuth_deg: np.ndarray  # (N,) float32
    elevation_deg: np.ndarray  # (N,) float32
    xyz: np.ndarray  # (N,3) float32,世界系,= FrameResult.det
    pair_kind: np.ndarray  # (N,) int8,PAIR_LABELS 的键;非双 PRF 全为 -1
    pair_unfold: np.ndarray  # (N,) int8,展开周期数;非双 PRF 全为 0

    def __len__(self) -> int:
        return len(self.range_bin)

    def pair_label(self, k: int) -> str:
        label = PAIR_LABELS[int(self.pair_kind[k])]
        unfold = int(self.pair_unfold[k])
        return f"{label} {unfold:+d}" if unfold else label

    @classmethod
    def from_result(cls, result: FrameResult) -> "Detections":
        f = result.fields
        return cls(
            range_bin=f["usRangeBin"],
            doppler_bin=f["usDopplerBin"],
            range_m=f["xRange"],
            velocity_mps=f["xDoppler"],
            snr_db=f["xSnr"],
            noise_db=f["xNoise"],
            azimuth_deg=f["xAzimuth"],
            elevation_deg=f["xElevation"],
            xyz=result.det,
            pair_kind=(
                result.pair_kind
                if result.pair_kind is not None
                else np.full(len(result.det), -1, np.int8)
            ),
            pair_unfold=(
                result.pair_unfold
                if result.pair_unfold is not None
                else np.zeros(len(result.det), np.int8)
            ),
        )


@dataclass(frozen=True, eq=False)
class TrackSet:
    """一帧航迹,列存放;取自 `FrameResult.tracks` 的 (id, x, y, z, v_r, |v|)。"""

    track_id: np.ndarray  # (N,) int32
    xyz: np.ndarray  # (N,3) float32,世界系
    v_r: np.ndarray  # (N,) float32,径向速度(远离为正,与检出点同口径)
    speed: np.ndarray  # (N,) float32,速度矢量模

    def __len__(self) -> int:
        return len(self.track_id)

    @classmethod
    def from_result(cls, result: FrameResult) -> "TrackSet":
        t = result.tracks
        return cls(
            track_id=np.array([r[0] for r in t], np.int32),
            xyz=np.array([r[1:4] for r in t], np.float32).reshape(-1, 3),
            v_r=np.array([r[4] for r in t], np.float32),
            speed=np.array([r[5] for r in t], np.float32),
        )


@dataclass(frozen=True, eq=False)
class ProcessedFrame:
    """`pipeline.FrameProcessor` 处理出的一帧:原有 FrameResult 加上列存放的点 / 航迹、
    逐级耗时与跟踪起点,以及(诊断开时)各级之间的中间量 `products`。

    `products`:`FrameProcessor(diagnose=True)` 时由 `ChainSession.process_frame(capture=...)`
    当帧拷出的 dict(全部是拷贝,不引用 C 缓冲);`diagnose=False`(`--check`)或已出播放器
    中间量预算时为 None。记号:n_chirps / n_rx / n_samples(= n_range)/ n_doppler /
    n_virtual 取自已载入的 WaveCfg / Array(同 RadarInfo);N = DOA 之后点数,
    N_final = 最终检出点数,M = 本帧非 Free 航迹数。

    ========================  ==============================================================
    键                         内容
    ========================  ==============================================================
    adc_raw                    (n_chirps, n_rx, n_samples, 2) int16,AdcFrame 原样;
                               复数定序只由 binding.native.adc_complex 解释
    range_power                (n_chirps, n_range) float32,距离级之后 Σ_rx |X|²(RTI)
    range_profile              (n_rx, n_range) float32,各 RX 对全部 chirp 的 |X|² 平均
    dcsig                      dict:sig (16, 32) complex64(RangeDcSigCtx_t 原布局
                               [分组×numRx+rx][槽位])、ready、enabled、n_groups、n_rx、
                               n_pos_bins、n_neg_bins、accum_count / avg_chirps(校准进度)
    rd_log2                    (n_range, n_doppler) float32,多普勒级之后的 RdMap 原值
                               (log2 幅度,不减峰值、不移位)
    fold / fold_metric /       仅 DDM 且本帧诊断有效:(n_range, n_doppler) uint8 /
    band_pow                   (n_range, n_doppler, n_sub) float32 ×2;TDM 无这三个键
    cfar                       dict:noise_range / noise_doppler (n_range, n_doppler) float32
                               (log2 域)、flags uint8(binding.dtypes.DIAG_CFAR_* 位)、
                               thr_range / thr_doppler(log2)、range_on / doppler_on /
                               group_on、scheme
    decouple                   仅步进频波形(xFineRangeStep != 0)有:period (N,) int32 展开周期、
                               range_bin / global_bin / coarse_range (N,) float32(与 CFAR 之后
                               的 Cloud 同序);无步进无此键
    doa_cloud                  dict:range_bin / doppler_bin (N,) int32、azimuth_deg /
                               elevation_deg (N,) float32(DOA 之后、聚类之前的点)
    doa                        dict:state、snap (N, n_virtual) complex64(送进估计器的快照)、
                               peak_pow (N,) float32、second (N,) bool、dropped:
                               range_bin / doppler_bin int32、reason uint8 三列
    track                      dict:point_track_id (N_final,) int32(-1 = 未关联)、
                               point_score float32、point_new bool;tracks:id / state /
                               age / det2act / det2free / active2free (M,)、s / p_diag
                               (M, 9)、hs / innov / limits (M, 4)、nis / n_assoc (M,);
                               ended_ids
    ========================  ==============================================================
    """

    index: int  # 采集内全局帧序号(0 起);采集自带的帧号见 fi
    fi: int  # = result.fi(usFrameIdx,会重复 / 回绕,不能当唯一键)
    radar: RadarInfo
    result: FrameResult
    detections: Detections
    tracks: TrackSet
    #: 毫秒;键为 "read"、ChainSession.process_frame 的 7 级与 "extract";
    #: 诊断开时另有 "capture"(中间量拷贝 / 归约)与 "diagnostics"
    timings_ms: dict
    #: 本次连续处理的起始帧序号 —— 跟踪状态从这一帧起递推(之前的帧没进过跟踪)
    track_epoch: int
    #: 各级之间的中间量(键见类文档);诊断关或已出播放器中间量预算时为 None
    products: dict | None = None
    diagnostics: object | None = None
