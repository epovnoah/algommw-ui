"""UI 配置和数据字段。ctypes 仅作为标准库数据容器使用,不建立 C 库绑定。"""

import ctypes

from .sensor import (
    MAX_ADC_SAMPLES,
    MAX_CHIRPS_PER_FRAME,
    MAX_RX,
    MAX_VIRTUAL_ANT,
)

MAX_RANGE_BINS = MAX_ADC_SAMPLES  # = core radar.h(同源派生,不再手抄字面量)
MAX_DOPPLER_BINS = 128
MAX_DETECTIONS = 1024
MAX_TRACKS = 64
DOA_SPECTRUM_BINS = 64  # = core/types/spectra.h,C 侧有编译期断言钉死

# ⚠ 以上容量镜像(Python 侧为 ctypes 数组定尺寸)手抄 C 宏很容易漂;native.py 启动时
# 用 C 侧 ulSizeOf* 逐个自检(SIZE_CHECK_TABLE),并用 ulCap* 直接核对宏值
# (CAPACITY_CHECK_TABLE),对不上直接 RuntimeError——fail-closed,不靠肉眼看。

# ---- 枚举(C 侧默认 enum underlying type = int,4 字节) -----------------------

E_OK = 0
E_ERR_NOT_IMPL = 3
E_MOD_TDM = 0  # eModulationScheme:时分复用
E_MOD_DDM = 1  # eModulationScheme:Doppler 分复用(F3 起真实解调语义)

E_WINDOW_NONE = 0
E_WINDOW_HANN = 1
E_WINDOW_BLACKMAN = 2
E_WINDOW_HAMMING = 3

#: eCfarNoiseMode —— 前三个与 TI averageMode 取值一一对应
E_CFAR_CA = 0
E_CFAR_CAGO = 1
E_CFAR_CASO = 2
E_CFAR_OS = 3
E_CFAR_CACC = 4

#: eCfarPeakGroupScheme —— 邻居从哪来(检测矩阵 / CFAR 检出列表),对应 TI 的 1/2
E_CFAR_PEAK_GROUP_DET_MATRIX = 0
E_CFAR_PEAK_GROUP_CFAR_PEAK = 1

#: 门限单位换算:RdMap 是 log2 幅度,门限按功率域 dB 表达。20·log10(2)。
CFAR_DB_PER_LOG2 = 6.020599913279624

#: eDoaVariant —— 2026-09-07 C 侧融合 fft(2D)/fft1d(1D) 为 beam(1D/2D 由 init 自动分流)
#: F4:尾加 eDoaVariantDbf2d=3(不动 0-2 既有编号);ADR 0012:尾加 eDoaVariantCorr=4。
E_DOA_VARIANT_BEAM = 0
E_DOA_VARIANT_MUSIC = 1
E_DOA_VARIANT_DML = 2
E_DOA_VARIANT_DBF2D = 3
E_DOA_VARIANT_CORR = 4
E_DOA_VARIANT_PHASEFIT = 5  # 阶梯解模糊 + 相位平面最小二乘(core/dpu/doa/phasefit.h)


class Complex16(ctypes.Structure):
    _fields_ = [("sI", ctypes.c_int16), ("sQ", ctypes.c_int16)]


class ComplexF(ctypes.Structure):
    _fields_ = [("xRe", ctypes.c_float), ("xIm", ctypes.c_float)]


class AdcFrame(ctypes.Structure):
    _fields_ = [
        ("usFrameIdx", ctypes.c_uint16),
        (
            "xCube",
            (Complex16 * MAX_ADC_SAMPLES) * MAX_RX * MAX_CHIRPS_PER_FRAME,
        ),
    ]


class Cube(ctypes.Structure):
    _fields_ = [
        (
            "xData",
            (ComplexF * MAX_RANGE_BINS) * MAX_RX * MAX_CHIRPS_PER_FRAME,
        ),
    ]


class RdMap(ctypes.Structure):
    _fields_ = [("xData", (ctypes.c_float * MAX_DOPPLER_BINS) * MAX_RANGE_BINS)]


class Point(ctypes.Structure):
    _fields_ = [
        ("usRangeBin", ctypes.c_uint16),
        ("usDopplerBin", ctypes.c_uint16),
        ("xRange", ctypes.c_float),
        ("xDoppler", ctypes.c_float),
        ("xSnr", ctypes.c_float),
        ("xNoise", ctypes.c_float),
        ("xAzimuth", ctypes.c_float),
        ("xElevation", ctypes.c_float),
        ("xPosX", ctypes.c_float),
        ("xPosY", ctypes.c_float),
        ("xPosZ", ctypes.c_float),
    ]


class Cloud(ctypes.Structure):
    _fields_ = [
        ("xPoints", Point * MAX_DETECTIONS),
        ("ulNumPoints", ctypes.c_uint32),
    ]


class Track(ctypes.Structure):
    _fields_ = [
        ("ulTrackId", ctypes.c_uint32),
        ("xState", ctypes.c_float * 9),
        ("eState", ctypes.c_int32),
        ("xMaxZ", ctypes.c_float),
        ("xMinZ", ctypes.c_float),
    ]


class Tracks(ctypes.Structure):
    _fields_ = [
        ("xTracks", Track * MAX_TRACKS),
        ("ulNumTracks", ctypes.c_uint32),
    ]


class WaveCfg(ctypes.Structure):
    _fields_ = [
        ("eModulation", ctypes.c_int32),
        ("ulNumTx", ctypes.c_uint32),
        ("ulNumRx", ctypes.c_uint32),
        ("ulNumAdcSamples", ctypes.c_uint32),
        ("ulNumDopplerChirps", ctypes.c_uint32),
        ("ulNumChirpsPerFrame", ctypes.c_uint32),
        ("ulNumSubbands", ctypes.c_uint32),
        ("xRangeStep", ctypes.c_float),
        ("xDopplerStep", ctypes.c_float),
        ("xFrameTime", ctypes.c_float),
        # 步进频派生量(core/types/waveform.h):0 = 无载频步进
        ("xFineRangeStep", ctypes.c_float),
        ("xStepSlopeRatio", ctypes.c_float),
        ("xChirpInterval", ctypes.c_float),
    ]


class Array(ctypes.Structure):
    _fields_ = [
        ("ulNumVirtual", ctypes.c_uint32),
        ("xPosX", ctypes.c_float * MAX_VIRTUAL_ANT),
        ("xPosZ", ctypes.c_float * MAX_VIRTUAL_ANT),
        ("xCalib", ComplexF * MAX_VIRTUAL_ANT),
        ("ulGridRows", ctypes.c_uint32),
        ("ulGridCols", ctypes.c_uint32),
    ]


class CfarDimCfg(ctypes.Structure):
    """单维 CFAR 配置。见 core/dpu/cfar/cfar.h —— 门限是**功率域 dB**(TI CLI 同口径)。"""

    _fields_ = [
        ("eNoiseMode", ctypes.c_int32),
        ("xThrDb", ctypes.c_float),
        ("ulGuardLen", ctypes.c_uint32),
        ("ulWinLen", ctypes.c_uint32),
        ("ulOsRank", ctypes.c_uint32),
        ("bPeakGrouping", ctypes.c_uint8),
        ("bPowerMean", ctypes.c_uint8),
        ("bOsEdgeScale", ctypes.c_uint8),
    ]


class CfarCfg(ctypes.Structure):
    _fields_ = [
        ("xRange", CfarDimCfg),
        ("xDoppler", CfarDimCfg),
        ("ulMinRangeBin", ctypes.c_uint32),
        ("ulMaxRangeBin", ctypes.c_uint32),
        ("ePeakGroupScheme", ctypes.c_int32),
        ("ulRangeSkipLow", ctypes.c_uint32),
        ("ulRangeSkipHigh", ctypes.c_uint32),
        ("bSubBinInterp", ctypes.c_uint8),
        ("bCrossPeakGrouping", ctypes.c_uint8),
        ("bSegmented", ctypes.c_uint8),
        ("ulSegmentEnd", ctypes.c_uint32 * 2),
        ("xRangeThrDb", ctypes.c_float * 3),
        ("xDopplerThrDb", ctypes.c_float * 3),
        ("bDominantSupplement", ctypes.c_uint8),
        ("ulDominantStart", ctypes.c_uint32),
        ("ulDominantEnd", ctypes.c_uint32),
        ("xDominanceThrDb", ctypes.c_float),
        ("xSupplementThrDb", ctypes.c_float),
        ("ulSupplementGuard", ctypes.c_uint32),
        ("ulSupplementWin", ctypes.c_uint32),
        ("ulSupplementLimit", ctypes.c_uint32),
        ("ulDecouplePeakClimb", ctypes.c_uint32),
    ]


class CalibCfg(ctypes.Structure):
    """实测标定量。见 core/types/calib.h —— 与推导量 WaveCfg 分家是刻意的。"""

    _fields_ = [("xRangeBias", ctypes.c_float)]


class RangeDcSigCfg(ctypes.Structure):
    """DC 天线耦合签名去除。见 core/dpu/range/dcsig.h。"""

    _fields_ = [
        ("bEnable", ctypes.c_uint8),
        ("ulNumPosBins", ctypes.c_uint32),
        ("ulNumNegBins", ctypes.c_uint32),
        ("ulNumAvgChirps", ctypes.c_uint32),
    ]


E_CLUSTER_OFF = 0
E_CLUSTER_FILTER = 1
E_CLUSTER_CENTROID = 2

# eStaticPolicy(track.h)
E_STATIC_SUPPRESS = 0
E_STATIC_TRACK = 1


class ClusterCfg(ctypes.Structure):
    """DBSCAN 聚类。见 core/dpu/cluster/dbscan.h —— ε 是**半径**不是平方(TI 的形参是
    平方阈值,手册把它列为易错点);ulMinPoints **不含被测点自身**(TI 口径)。"""

    _fields_ = [
        ("eMode", ctypes.c_int32),
        ("xEpsilon", ctypes.c_float),
        ("xWeightZ", ctypes.c_float),
        ("xWeightVel", ctypes.c_float),
        ("ulMinPoints", ctypes.c_uint32),
    ]


class MountCfg(ctypes.Structure):
    """传感器安装姿态。见 core/types/mount.h —— 只有俯仰倾角,因为 gTrack 的 azimTilt
    在它自己的 3D 源码里零使用。"""

    _fields_ = [
        ("xElevTiltDeg", ctypes.c_float),
        ("xHeightM", ctypes.c_float),
    ]


class FovCfg(ctypes.Structure):
    """角度视场。见 core/dpu/doa/cfg.h —— bEnable 是必要的:四个角度全 0 的字面含义是
    "只接受正前方 0°",不是"不设限"。"""

    _fields_ = [
        ("bEnable", ctypes.c_uint8),
        ("xAzMinDeg", ctypes.c_float),
        ("xAzMaxDeg", ctypes.c_float),
        ("xElMinDeg", ctypes.c_float),
        ("xElMaxDeg", ctypes.c_float),
    ]


class MultiPeakCfg(ctypes.Structure):
    """多目标波束形成(第二峰)。见 core/dpu/doa/cfg.h。"""

    _fields_ = [
        ("bEnable", ctypes.c_uint8),
        ("xThrScale", ctypes.c_float),
    ]


class DoaScanCfg(ctypes.Structure):
    """角度扫描网格(单位度)。见 core/dpu/doa/cfg.h 的 DoaScanCfg_t。

    仅网格后端(eDoaBeamInit 的 bDirect 分支,由 bDoaScanConfigured 判定"配没配")
    消费并校验;未配网格(零值)时 FFT 快路径不碰它(= 未配 = 正常)。
    """

    _fields_ = [
        ("xAzStartDeg", ctypes.c_float),
        ("xAzStopDeg", ctypes.c_float),
        ("xAzStepDeg", ctypes.c_float),
        ("xElStartDeg", ctypes.c_float),
        ("xElStopDeg", ctypes.c_float),
        ("xElStepDeg", ctypes.c_float),
    ]


class DoaCfg(ctypes.Structure):
    """DOA 族共享配置。见 core/dpu/doa/cfg.h。"""

    _fields_ = [
        ("xFov", FovCfg),
        ("xMultiPeak", MultiPeakCfg),
        ("xScan", DoaScanCfg),
        ("bExtMaxVel", ctypes.c_uint8),
        ("bAngleInterp", ctypes.c_uint8),
        ("bNearElevationSkip", ctypes.c_uint8),
        ("ulNearElevationMaxBin", ctypes.c_uint32),
        ("xMinCoherence", ctypes.c_float),  # 0 = 关;DDM 逐点路径的测角相干度门限
        ("bElevationRepair", ctypes.c_uint8),  # 0 = 关;corr/DDM 近处俯仰混叠帧内修复
    ]


class GateCfg(ctypes.Structure):
    _fields_ = [
        ("xGain", ctypes.c_float),
        ("xDepth", ctypes.c_float),
        ("xWidth", ctypes.c_float),
        ("xHeight", ctypes.c_float),
        ("xVel", ctypes.c_float),
    ]


class TrackCfg(ctypes.Structure):
    """EKF 群跟踪参数。见 core/dpu/track/track.h。

    P7 起收进 ChainCfg —— 此前 eChainInit 直接调 vTrackCfgDefaults(),跟踪参数完全不在
    配置面上,想做敏感性扫描或 A/B 都无从下手。默认值仍是 core 单源(经
    ``ChainSession.track_defaults()`` 取),Python 不抄一份。
    """

    _fields_ = [
        ("xDeltaT", ctypes.c_float),
        ("xMaxAccel", ctypes.c_float * 3),
        ("xGate", GateCfg),
        ("xMeasNoise", ctypes.c_float * 4),
        ("ulAllocPointsThr", ctypes.c_uint32),
        ("xAllocSnrThr", ctypes.c_float),
        ("xAllocMaxDist", ctypes.c_float),
        ("xAllocMaxVel", ctypes.c_float),
        ("ulHitPointsThr", ctypes.c_uint32),
        ("ulDet2ActThr", ctypes.c_uint32),
        ("ulDet2FreeThr", ctypes.c_uint32),
        ("ulActive2FreeThr", ctypes.c_uint32),
        ("xSceneXMin", ctypes.c_float),
        ("xSceneXMax", ctypes.c_float),
        ("xSceneYMin", ctypes.c_float),
        ("xSceneYMax", ctypes.c_float),
        ("bGroupDispersion", ctypes.c_uint8),
        ("bVelUnroll", ctypes.c_uint8),
        ("xMaxRadialVel", ctypes.c_float),
        ("xUnrollAlpha", ctypes.c_float),
        ("xUnrollConfidence", ctypes.c_float),
        ("bJoseph", ctypes.c_uint8),
        ("bHeightDetect", ctypes.c_uint8),
        ("eStatic", ctypes.c_int32),
        ("ulMaxTracks", ctypes.c_uint32),
        ("bMergeEnable", ctypes.c_uint8),
        ("xMergeMdGate", ctypes.c_float),
        ("ulMergeMinLife", ctypes.c_uint32),
        ("bAvoidanceEnable", ctypes.c_uint8),
        ("xAvoidanceDist", ctypes.c_float),
        ("bEgoMotion", ctypes.c_uint8),
        ("xEgoStaticVel", ctypes.c_float),
        ("bCoast", ctypes.c_uint8),
        ("bGateNoise", ctypes.c_uint8),
        ("xInitTanVel", ctypes.c_float),
    ]


class PairCfg(ctypes.Structure):
    _fields_ = [
        ("bEnable", ctypes.c_uint8),
        ("ucMaxUnfold", ctypes.c_uint8),
        ("bShareTaken", ctypes.c_uint8),
        ("bCarryConfirmed", ctypes.c_uint8),
        ("bRescueUnfolded", ctypes.c_uint8),
    ] + [
        (name, ctypes.c_float)
        for name in (
            "xRangeGate",
            "xVelocityGate",
            "xMaxSpeed",
            "xAngleGate",
            "xMinCoherence",
            "xAmbiguityMargin",
            "xMinOverlapSpeed",
            "xOverlapRangeBins",
            "xOverlapDopplerBins",
            "xOccupantRangeBins",
            "xMaskMarginDb",
        )
    ] + [
        ("bZeroIdentity", ctypes.c_uint8),
        ("xAzimuthSigma", ctypes.c_float),
        ("xLateralExtent", ctypes.c_float),
    ]


class PairInfo(ctypes.Structure):
    _fields_ = [
        ("ulNumPoints", ctypes.c_uint32),
        ("ulNumTentative", ctypes.c_uint32),
        ("sCurrent", ctypes.c_int16 * MAX_DETECTIONS),
        ("sPrevious", ctypes.c_int16 * MAX_DETECTIONS),
        ("cUnfold", ctypes.c_int8 * MAX_DETECTIONS),
        ("ucKind", ctypes.c_uint8 * MAX_DETECTIONS),
    ]


class ChainCfg(ctypes.Structure):
    _fields_ = [
        ("xWave", WaveCfg),
        ("xArray", Array),
        ("xCalib", CalibCfg),
        ("eRangeWindow", ctypes.c_int32),
        ("xDcSig", RangeDcSigCfg),
        ("eDopplerWindow", ctypes.c_int32),
        ("xCfar", CfarCfg),
        ("eDoa", ctypes.c_int32),
        ("xMount", MountCfg),
        ("xDoa", DoaCfg),
        ("xCluster", ClusterCfg),
        ("xTrack", TrackCfg),
        ("bClutter", ctypes.c_uint8),
        ("bMti", ctypes.c_uint8),
        ("bMtd", ctypes.c_uint8),
        ("xAlternateWave", WaveCfg),
        ("xPair", PairCfg),
    ]


class Spectra(ctypes.Structure):
    """每检出点的方位谱(旁路诊断产物,不参与算法)。见 core/types/spectra.h。

    xAxisDeg 已 fftshift(下标 0 = 最负角),xPower 每行同序。行下标与 Cloud
    的点下标对齐(DOA 级压缩后)。~64KB,由调用方持有并 attach 给 C 侧。
    """

    _fields_ = [
        ("ulNumPoints", ctypes.c_uint32),
        ("ulNumBins", ctypes.c_uint32),
        ("xAxisDeg", ctypes.c_float * DOA_SPECTRUM_BINS),
        ("xPower", (ctypes.c_float * DOA_SPECTRUM_BINS) * MAX_DETECTIONS),
    ]


class SnapCube(ctypes.Structure):
    """DDM 解调产物(F3):per (R,D) fold + 16 通道解调后复快照。见 core/types/radar.h。

    xSnap 与 RdMap 同源(同一次窗后 FFT 的输出取值,无第二条数值路径);未施 xCalib
    (标定在 DOA 消费侧)。DDM 模式必 attach(eChainDoppler 拒绝 NULL)。
    ~6.3MB,由调用方持有并 attach 给 C 侧。
    """

    _fields_ = [
        ("xSnap", ((ComplexF * MAX_VIRTUAL_ANT) * MAX_DOPPLER_BINS) * MAX_RANGE_BINS),
        ("ucFold", (ctypes.c_uint8 * MAX_DOPPLER_BINS) * MAX_RANGE_BINS),
    ]


# ---- 诊断出口(uiplan.md §5 E1–E3,已删,见 git 历史 fa7d3d4;C 侧 core/types/diag.h 与 chain/chain.h) -----

RANGE_DCSIG_MAX_BINS = 32  # = core/dpu/range/dcsig.h
DIAG_MAX_SUBBANDS = 6  # = core/types/diag.h
DOA_PROBE_MAX_CELLS = 32768  # = core/types/diag.h
#: 格数上限的镜像。判据在 core/dpu/doa/cfg.h;参数编辑器提前预警用(C 侧 eChainInit 才是权威)。
DOA_SCAN_MAX_CELLS = 32768  # = core/dpu/doa/cfg.h

#: CfarMap_t.ucFlags 位
DIAG_CFAR_SKIP = 0x01  # 不参与:range skip 区或 range FOV 外
DIAG_CFAR_DOPPLER_PASS = 0x02  # 多普勒向过门限(该维关闭时视为过)
DIAG_CFAR_SUPPLEMENT = 0x10  # 主导 Doppler 补点候选,不等价于两向通过
DIAG_CFAR_RANGE_PASS = 0x04  # 距离向过门限(同上)
DIAG_CFAR_PEAK_KEEP = 0x08  # 峰值分组存活(仅 det_matrix 方案定义)

#: 测角剔除原因(DoaDiag_t.ucDropReason / DoaProbe_t.ucReason)
DIAG_DOA_DROP_NONE = 0
DIAG_DOA_DROP_GEOM = 1  # 几何不可解:cosφ 过小或 |Wx/cosφ| > 1
DIAG_DOA_DROP_FOV = 2  # 角度出视场
DIAG_DOA_DROP_COHERENCE = 3  # 相干度低于 xDoa.xMinCoherence(非单源)

#: DoaDiag_t.ucState
DIAG_DOA_STATE_OK = 0
DIAG_DOA_STATE_UNSUPPORTED = 1  # music / dml 不出测角诊断

#: DoaProbe_t.ucBackend
DIAG_PROBE_FFT2D = 0  # 行 = Wz、列 = Wx(归一化空间频率,fftshift 后升序)
DIAG_PROBE_FFT1D = 1  # 单行,列 = Wx
DIAG_PROBE_CORR = 3  # 单行相关谱,列 = u = sin az·cos el;俯仰来自基线
DIAG_PROBE_DIRECT = 2  # 直接加权:行 = 俯仰格(度)、列 = 方位格(度)


class RangeDcSigCtx(ctypes.Structure):
    """DC 签名与校准进度。见 core/dpu/range/dcsig.h —— 这里只为 ChainDiag.xDcSig 镜像
    (处理链自己的那份在 ChainCtx_t 里,不透明)。xSig 下标 [分组×numRx+rx][槽位]。"""

    _fields_ = [
        ("xCfg", RangeDcSigCfg),
        ("ulNumGroups", ctypes.c_uint32),
        ("ulNumRx", ctypes.c_uint32),
        ("ulNumRangeBins", ctypes.c_uint32),
        ("ulNumChirpsPerFrame", ctypes.c_uint32),
        ("ulChirpsPerGroup", ctypes.c_uint32),
        ("ulNumBins", ctypes.c_uint32),
        ("ulAccumCount", ctypes.c_uint32),
        ("bReady", ctypes.c_uint8),
        ("xSig", (ComplexF * RANGE_DCSIG_MAX_BINS) * MAX_VIRTUAL_ANT),
    ]


class DopplerDiag(ctypes.Structure):
    """E1 ② DDM 折叠搜索过程量,下标 [距离 bin][子带内 bin][fold 或子带]。见 core/types/diag.h。

    TDM 链 bValid = 0;只写 [0, 活动距离 bin) × [0, 每子带 bin) × [0, ulNumSubbands)。
    """

    _fields_ = [
        (
            "xFoldMetric",
            ((ctypes.c_float * DIAG_MAX_SUBBANDS) * MAX_DOPPLER_BINS) * MAX_RANGE_BINS,
        ),
        (
            "xBandPow",
            ((ctypes.c_float * DIAG_MAX_SUBBANDS) * MAX_DOPPLER_BINS) * MAX_RANGE_BINS,
        ),
        ("ulNumSubbands", ctypes.c_uint32),
        ("bValid", ctypes.c_uint8),
    ]


class DecoupleDiag(ctypes.Structure):
    """步进频解耦的逐点过程量,与 eChainCfar 之后的 Cloud 同序。见 core/types/diag.h。
    无步进时 ulNumPoints = 0。"""

    _fields_ = [
        ("ulNumPoints", ctypes.c_uint32),
        ("lPeriod", ctypes.c_int32 * MAX_DETECTIONS),
        ("xRangeBin", ctypes.c_float * MAX_DETECTIONS),
        ("xGlobalBin", ctypes.c_float * MAX_DETECTIONS),
        ("xCoarseRange", ctypes.c_float * MAX_DETECTIONS),
    ]


class DoaDiag(ctypes.Structure):
    """E1 ③ 测角逐点记录:保留点与 eChainDoa 之后的 Cloud 同序同数;剔除记录按处理次序。
    见 core/types/diag.h。"""

    _fields_ = [
        ("ulNumPoints", ctypes.c_uint32),
        ("xSnap", (ComplexF * MAX_VIRTUAL_ANT) * MAX_DETECTIONS),
        ("xPeakPow", ctypes.c_float * MAX_DETECTIONS),
        ("ucSecondPeak", ctypes.c_uint8 * MAX_DETECTIONS),
        ("ulNumDropped", ctypes.c_uint32),
        ("usDropRangeBin", ctypes.c_uint16 * MAX_DETECTIONS),
        ("usDropDopplerBin", ctypes.c_uint16 * MAX_DETECTIONS),
        ("ucDropReason", ctypes.c_uint8 * MAX_DETECTIONS),
        ("ucState", ctypes.c_uint8),
    ]


class TrackDiagItem(ctypes.Structure):
    """一条非 Free 航迹在本帧结束时的快照。见 core/types/diag.h —— 本帧新建的航迹
    xHs / xInnov / xLimits / xNis 全为 NaN。"""

    _fields_ = [
        ("ulId", ctypes.c_uint32),
        ("eState", ctypes.c_int32),
        ("ulAge", ctypes.c_uint32),
        ("ulDet2Act", ctypes.c_uint32),
        ("ulDet2Free", ctypes.c_uint32),
        ("ulActive2Free", ctypes.c_uint32),
        ("xS", ctypes.c_float * 9),
        ("xPDiag", ctypes.c_float * 9),
        ("xHs", ctypes.c_float * 4),
        ("xInnov", ctypes.c_float * 4),
        ("xNis", ctypes.c_float),
        ("ulNumAssoc", ctypes.c_uint32),
        ("xLimits", ctypes.c_float * 4),
    ]


class TrackDiag(ctypes.Structure):
    """E1 ④ 跟踪诊断:逐点数组与 eChainTrack 收到的 Cloud 同序。见 core/types/diag.h。"""

    _fields_ = [
        ("ulNumPoints", ctypes.c_uint32),
        ("lPointTrackId", ctypes.c_int32 * MAX_DETECTIONS),
        ("xPointScore", ctypes.c_float * MAX_DETECTIONS),
        ("ucPointNew", ctypes.c_uint8 * MAX_DETECTIONS),
        ("ulNumTracks", ctypes.c_uint32),
        ("xTracks", TrackDiagItem * MAX_TRACKS),
        ("ulNumEnded", ctypes.c_uint32),
        ("ulEndedId", ctypes.c_uint32 * MAX_TRACKS),
    ]


class ChainDiag(ctypes.Structure):
    """E1 处理过程捕获的汇总缓冲(eChainDiagAttach 挂到 ChainCtx_t)。见 core/chain/chain.h。"""

    _fields_ = [
        ("xDcSig", RangeDcSigCtx),
        ("xDoppler", DopplerDiag),
        ("xDoa", DoaDiag),
        ("xTrack", TrackDiag),
        ("xDecouple", DecoupleDiag),
    ]


class CfarMap(ctypes.Structure):
    """E2 CFAR 全图(eChainCfarDiag)。噪声与门限为 log2 幅度域。见 core/types/diag.h。"""

    _fields_ = [
        ("xRangeThrLog2", ctypes.c_float),
        ("xDopplerThrLog2", ctypes.c_float),
        ("bRangeOn", ctypes.c_uint8),
        ("bDopplerOn", ctypes.c_uint8),
        ("bGroupOn", ctypes.c_uint8),
        ("ucScheme", ctypes.c_uint8),
        ("ulNumRange", ctypes.c_uint32),
        ("ulNumDoppler", ctypes.c_uint32),
        ("xNoiseRange", (ctypes.c_float * MAX_DOPPLER_BINS) * MAX_RANGE_BINS),
        ("xNoiseDoppler", (ctypes.c_float * MAX_DOPPLER_BINS) * MAX_RANGE_BINS),
        ("ucFlags", (ctypes.c_uint8 * MAX_DOPPLER_BINS) * MAX_RANGE_BINS),
        ("xRangeThreshold", ctypes.c_float * MAX_RANGE_BINS),
        ("xDopplerThreshold", ctypes.c_float * MAX_RANGE_BINS),
    ]


class DoaProbe(ctypes.Structure):
    """E3 测角探针对一个快照的整张谱(eDoaBeamProbe)。xPower 行优先(下标 = 行·ulCols + 列)。
    见 core/types/diag.h。"""

    _fields_ = [
        ("ucBackend", ctypes.c_uint8),
        ("ucReason", ctypes.c_uint8),
        ("ulRows", ctypes.c_uint32),
        ("ulCols", ctypes.c_uint32),
        ("ulPeakRow", ctypes.c_uint32),
        ("ulPeakCol", ctypes.c_uint32),
        ("xAzimuth", ctypes.c_float),
        ("xElevation", ctypes.c_float),
        ("xPeakPow", ctypes.c_float),
        ("xSnapPow", ctypes.c_float),
        ("xContrib", ComplexF * MAX_VIRTUAL_ANT),
        ("xAxisRow", ctypes.c_float * DOA_PROBE_MAX_CELLS),
        ("xAxisCol", ctypes.c_float * DOA_PROBE_MAX_CELLS),
        ("xPower", ctypes.c_float * DOA_PROBE_MAX_CELLS),
    ]


# (Python 类型, C 侧 sizeof 查询函数名) —— native.py 用这张表在加载时自检对齐。
SIZE_CHECK_TABLE = [
    (PairCfg, "ulSizeOfPairCfg"),
    (PairInfo, "ulSizeOfPairInfo"),
    (Complex16, "ulSizeOfComplex16"),
    (ComplexF, "ulSizeOfComplexF"),
    (AdcFrame, "ulSizeOfAdcFrame"),
    (Cube, "ulSizeOfCube"),
    (RdMap, "ulSizeOfRdMap"),
    (Point, "ulSizeOfPoint"),
    (Cloud, "ulSizeOfCloud"),
    (Track, "ulSizeOfTrack"),
    (Tracks, "ulSizeOfTracks"),
    (WaveCfg, "ulSizeOfWaveCfg"),
    (Array, "ulSizeOfArray"),
    (CfarDimCfg, "ulSizeOfCfarDimCfg"),
    (CfarCfg, "ulSizeOfCfarCfg"),
    (MountCfg, "ulSizeOfMountCfg"),
    (ClusterCfg, "ulSizeOfClusterCfg"),
    (GateCfg, "ulSizeOfGateCfg"),
    (TrackCfg, "ulSizeOfTrackCfg"),
    (FovCfg, "ulSizeOfFovCfg"),
    (MultiPeakCfg, "ulSizeOfMultiPeakCfg"),
    (DoaScanCfg, "ulSizeOfDoaScanCfg"),
    (DoaCfg, "ulSizeOfDoaCfg"),
    (CalibCfg, "ulSizeOfCalibCfg"),
    (RangeDcSigCfg, "ulSizeOfRangeDcSigCfg"),
    (ChainCfg, "ulSizeOfChainCfg"),
    (Spectra, "ulSizeOfSpectra"),
    (SnapCube, "ulSizeOfSnapCube"),
    (RangeDcSigCtx, "ulSizeOfRangeDcSigCtx"),
    (DopplerDiag, "ulSizeOfDopplerDiag"),
    (DecoupleDiag, "ulSizeOfDecoupleDiag"),
    (DoaDiag, "ulSizeOfDoaDiag"),
    (TrackDiagItem, "ulSizeOfTrackDiagItem"),
    (TrackDiag, "ulSizeOfTrackDiag"),
    (ChainDiag, "ulSizeOfChainDiag"),
    (CfarMap, "ulSizeOfCfarMap"),
    (DoaProbe, "ulSizeOfDoaProbe"),
]

#: Python 侧镜像的 C 容量宏 → 导出函数(bind/reflect.c 末尾)。native.py 导入时逐项自检,
#: 对不上直接 RuntimeError;sizeof 自检只能间接发现容量漂移,这里直接点名宏。
CAPACITY_CHECK_TABLE = [
    ("ulCapMaxAdcSamples", MAX_ADC_SAMPLES),
    ("ulCapMaxRx", MAX_RX),
    ("ulCapMaxChirpsPerFrame", MAX_CHIRPS_PER_FRAME),
    ("ulCapMaxVirtualAnt", MAX_VIRTUAL_ANT),
    ("ulCapMaxRangeBins", MAX_RANGE_BINS),
    ("ulCapMaxDopplerBins", MAX_DOPPLER_BINS),
    ("ulCapMaxDetections", MAX_DETECTIONS),
    ("ulCapMaxTracks", MAX_TRACKS),
    ("ulCapDoaSpectrumBins", DOA_SPECTRUM_BINS),
    ("ulCapRangeDcsigMaxBins", RANGE_DCSIG_MAX_BINS),
    ("ulCapDiagMaxSubbands", DIAG_MAX_SUBBANDS),
    ("ulCapDoaProbeMaxCells", DOA_PROBE_MAX_CELLS),
    ("ulCapDoaScanMaxCells", DOA_SCAN_MAX_CELLS),
]
