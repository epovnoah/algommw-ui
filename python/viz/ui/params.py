"""界面参数的声明与校验。参数副本仅用于预览,不创建算法处理器。"""

import ctypes
import math

from binding import dtypes
from binding.config import (
    AVG_MODES,
    CLUSTER_MODES,
    DOA_VARIANT_NAMES,
    PEAK_GROUP_SCHEMES,
    WINDOW_NAMES,
    doa_route_recommend,
)
from binding.dtypes import DOA_SCAN_MAX_CELLS, MAX_RANGE_BINS

__all__ = [
    "clone_config",
    "group_tip",
    "scan_grid_cells",
    "scan_ready",
    "scan_tag",
    "variant_name",
]


def _choices(by_name: dict) -> list:
    """{名: 编号} → 下拉用的 [(名, 编号)],按编号升序。"""
    return sorted(by_name.items(), key=lambda kv: kv[1])


# 下拉选项全部从 binding.config 那几张表派生(profile 契约的唯一权威),不在这里
# 手抄第二份 —— 抄了就会出现“下拉里能选 dml、profile 里写 dml 却报 unknown”这类
# 漂移。编号顺序 = 枚举顺序,与手写时的意图一致。
_WINDOWS = _choices(WINDOW_NAMES)
_DOA = sorted(
    ((name, var) for var, name in DOA_VARIANT_NAMES.items()), key=lambda kv: kv[1]
)
_AVG = _choices(AVG_MODES)
_PKGRP = _choices(PEAK_GROUP_SCHEMES)
_CLUSTER = _choices(CLUSTER_MODES)

#: 扫描网格组的标题(活标签要找它,写死两处会漂)。
_SCAN_GROUP = "DOA · 扫描网格"

#: 从 corr/music/dml 切到 dbf2d 时自动补的默认网格(AFM761 参考值,141×121)。
#: 用户可在「DOA · 扫描网格」里改;切走再切回会恢复最后一次合法网格。
_DEFAULT_SCAN = (-70.0, 70.0, 1.0, -30.0, 30.0, 0.5)

#: (ChainCfg 上的属性路径, 显示名, 编辑器类型, 选项)
_SPEC = [
    (
        "剪枝级",
        "默认全关",
        [
            ("bClutter", "静杂波滤除", "bool", {}),
            ("bMti", "MTI(分组内慢时间两脉冲对消)", "bool", {}),
            ("bMtd", "MTD(多普勒域)", "bool", {}),
        ],
    ),
    (
        "Range FFT",
        "",
        [("eRangeWindow", "窗函数", "list", {"limits": _WINDOWS})],
    ),
    (
        "Doppler FFT",
        "",
        [("eDopplerWindow", "窗函数", "list", {"limits": _WINDOWS})],
    ),
    (
        "DC 耦合签名",
        "默认关 · 先校准后补偿",
        [
            ("xDcSig.bEnable", "启用", "bool", {}),
            ("xDcSig.ulNumPosBins", "正侧 bin 数", "int", {"limits": (0, 32)}),
            ("xDcSig.ulNumNegBins", "负侧 bin 数", "int", {"limits": (0, 32)}),
            ("xDcSig.ulNumAvgChirps", "校准 chirp 数", "int", {"limits": (1, 128)}),
        ],
    ),
    (
        "CFAR · Range",
        "非循环窗 · 门限 dB",
        [
            ("xCfar.xRange.enabled", "启用距离 CFAR", "bool", {}),
            ("xCfar.xRange.eNoiseMode", "噪声估计", "list", {"limits": _AVG}),
            ("xCfar.xRange.bPowerMean", "功率域算术平均", "bool", {}),
            ("xCfar.xRange.bOsEdgeScale", "OS 边缘秩缩放", "bool", {}),
            (
                "xCfar.xRange.xThrDb",
                "门限 (dB)",
                "float",
                {"limits": (0.0, 72.0), "step": 0.5},
            ),
            ("xCfar.xRange.ulGuardLen", "保护单元", "int", {"limits": (0, 16)}),
            ("xCfar.xRange.ulWinLen", "参考窗", "int", {"limits": (1, 32)}),
            ("xCfar.xRange.ulOsRank", "OS 秩 k", "int", {"limits": (1, 64)}),
            ("xCfar.xRange.bPeakGrouping", "峰值分组", "bool", {}),
        ],
    ),
    (
        "CFAR · Doppler",
        "循环窗 · 门限 dB",
        [
            ("xCfar.xDoppler.enabled", "启用速度 CFAR", "bool", {}),
            ("xCfar.xDoppler.eNoiseMode", "噪声估计", "list", {"limits": _AVG}),
            ("xCfar.xDoppler.bPowerMean", "功率域算术平均", "bool", {}),
            ("xCfar.xDoppler.bOsEdgeScale", "OS 边缘秩缩放", "bool", {}),
            (
                "xCfar.xDoppler.xThrDb",
                "门限 (dB)",
                "float",
                {"limits": (0.0, 72.0), "step": 0.5},
            ),
            ("xCfar.xDoppler.ulGuardLen", "保护单元", "int", {"limits": (0, 16)}),
            ("xCfar.xDoppler.ulWinLen", "参考窗", "int", {"limits": (1, 15)}),
            ("xCfar.xDoppler.ulOsRank", "OS 秩 k", "int", {"limits": (1, 30)}),
            ("xCfar.xDoppler.bPeakGrouping", "峰值分组", "bool", {}),
        ],
    ),
    (
        "CFAR · 输出裁剪",
        "FOV 只裁输出 · skip 连训练都不当",
        [
            ("xCfar.ePeakGroupScheme", "分组方案", "list", {"limits": _PKGRP}),
            ("xCfar.ulMinRangeBin", "起始距离 bin", "int", {"limits": (0, 255)}),
            ("xCfar.ulMaxRangeBin", "终止距离 bin", "int", {"limits": (0, 255)}),
            ("xCfar.ulRangeSkipLow", "低端 skip", "int", {"limits": (0, 64)}),
            # 上限取距离门总数:实采样板的镜像区 skip 是 192 / 256(旧上限 64 会在建参数树时把它
            # 夹到 64,树发出一次「改参」,处理器带着被夹过的配置重建)
            (
                "xCfar.ulRangeSkipHigh",
                "高端 skip",
                "int",
                {"limits": (0, MAX_RANGE_BINS)},
            ),
            ("xCfar.bSubBinInterp", "亚 bin 插值", "bool", {}),
            ("xCfar.bCrossPeakGrouping", "十字形峰值分组核", "bool", {}),
            # 仅步进频:解耦沿同列爬到局部峰解速度(0 = 关),见 core/dpu/cfar/cfar.h
            (
                "xCfar.ulDecouplePeakClimb",
                "解耦同列爬坡格数",
                "int",
                {"limits": (0, 8)},
            ),
        ],
    ),
    (
        "DOA",
        "",
        [
            ("eDoa", "估角变体", "list", {"limits": _DOA}),
            ("xDoa.bExtMaxVel", "扩展最大速度", "bool", {}),
            ("xDoa.bAngleInterp", "角度亚 bin 插值", "bool", {}),
            (
                "xDoa.xMinCoherence",
                "最小相干度 (0=关,仅 DDM)",
                "float",
                {"limits": (0.0, 1.0), "step": 0.05},
            ),
            ("xDoa.bElevationRepair", "俯仰混叠修复 (corr/DDM)", "bool", {}),
        ],
    ),
    (
        _SCAN_GROUP,
        "",  # 标签是活的(后端/格数随输入变),见 _refresh_scan_tag
        [
            ("xDoa.xScan.enabled", "启用网格后端", "bool", {}),
            (
                "xDoa.xScan.xAzStartDeg",
                "方位起 (°)",
                "float",
                {"limits": (-90.0, 90.0), "step": 1.0},
            ),
            (
                "xDoa.xScan.xAzStopDeg",
                "方位止 (°)",
                "float",
                {"limits": (-90.0, 90.0), "step": 1.0},
            ),
            (
                "xDoa.xScan.xAzStepDeg",
                "方位步 (°)",
                "float",
                {"limits": (0.0, 90.0), "step": 0.5},
            ),
            (
                "xDoa.xScan.xElStartDeg",
                "俯仰起 (°)",
                "float",
                {"limits": (-90.0, 90.0), "step": 1.0},
            ),
            (
                "xDoa.xScan.xElStopDeg",
                "俯仰止 (°)",
                "float",
                {"limits": (-90.0, 90.0), "step": 1.0},
            ),
            (
                "xDoa.xScan.xElStepDeg",
                "俯仰步 (°)",
                "float",
                {"limits": (0.0, 90.0), "step": 0.5},
            ),
        ],
    ),
    (
        "DOA · 角度 FOV",
        "默认关 · 逐点拒绝",
        [
            ("xDoa.xFov.bEnable", "启用", "bool", {}),
            ("xDoa.bNearElevationSkip", "近距俯仰豁免 (corr/DDM)", "bool", {}),
            (
                "xDoa.ulNearElevationMaxBin",
                "豁免最大距离 bin",
                "int",
                {"limits": (0, MAX_RANGE_BINS - 1)},
            ),
            (
                "xDoa.xFov.xAzMinDeg",
                "方位下限 (°)",
                "float",
                {"limits": (-90.0, 90.0), "step": 1.0},
            ),
            (
                "xDoa.xFov.xAzMaxDeg",
                "方位上限 (°)",
                "float",
                {"limits": (-90.0, 90.0), "step": 1.0},
            ),
            (
                "xDoa.xFov.xElMinDeg",
                "俯仰下限 (°)",
                "float",
                {"limits": (-90.0, 90.0), "step": 1.0},
            ),
            (
                "xDoa.xFov.xElMaxDeg",
                "俯仰上限 (°)",
                "float",
                {"limits": (-90.0, 90.0), "step": 1.0},
            ),
        ],
    ),
    (
        "跟踪",
        "门限与起航迹条件",
        [
            (
                "xTrack.xGate.xGain",
                "门限增益 (χ²₃ 分位)",
                "float",
                {"limits": (1.0, 20.0), "step": 0.5},
            ),
            ("xTrack.bGroupDispersion", "组弥散 gD", "bool", {}),
            ("xTrack.bVelUnroll", "径向速度解缠", "bool", {}),
            (
                "xTrack.ulAllocPointsThr",
                "起航迹最少点数",
                "int",
                {"limits": (1, dtypes.MAX_DETECTIONS)},
            ),
            (
                "xTrack.xAllocSnrThr",
                "起航迹 SNR 门限",
                "float",
                {"limits": (0.0, 1000.0), "step": 1.0},
            ),
            (
                "xTrack.xAllocMaxDist",
                "起航迹聚类距离 (m)",
                "float",
                {"limits": (0.01, 100.0), "step": 0.1},
            ),
            (
                "xTrack.xAllocMaxVel",
                "起航迹速度差 (m/s)",
                "float",
                {"limits": (0.01, 100.0), "step": 0.1},
            ),
        ],
    ),
    (
        "聚类 (DBSCAN)",
        "默认关 · A/B 见 bench/cluster",
        [
            ("xCluster.eMode", "模式", "list", {"limits": _CLUSTER}),
            (
                "xCluster.xEpsilon",
                "邻域半径 (m)",
                "float",
                {"limits": (0.1, 10.0), "step": 0.1},
            ),
            (
                "xCluster.xWeightZ",
                "z 维权重",
                "float",
                {"limits": (0.0, 10.0), "step": 0.1},
            ),
            (
                "xCluster.xWeightVel",
                "速度维权重",
                "float",
                {"limits": (0.0, 100.0), "step": 0.5},
            ),
            ("xCluster.ulMinPoints", "minPts (不含自身)", "int", {"limits": (1, 32)}),
        ],
    ),
    (
        "DOA · 第二峰",
        "同 range/doppler 多目标",
        [
            ("xDoa.xMultiPeak.bEnable", "启用", "bool", {}),
            (
                "xDoa.xMultiPeak.xThrScale",
                "次峰/主峰 功率比",
                "float",
                {"limits": (0.01, 1.0), "step": 0.05},
            ),
        ],
    ),
]

# 与 core 的字段逐项绑定,分段模式下实际使用的是这六个门限,普通门限只保留开关符号。
_SPEC += [
    (
        "CFAR · 分段门限",
        "启用时覆盖普通门限",
        [("xCfar.bSegmented", "启用", "bool", {})]
        + [
            (
                f"xCfar.ulSegmentEnd.{i}",
                f"第 {i + 1} 段终止 bin",
                "int",
                {"limits": (0, MAX_RANGE_BINS - 1)},
            )
            for i in range(2)
        ]
        + [
            (
                f"xCfar.{field}.{i}",
                f"{label}第 {i + 1} 段 (dB)",
                "float",
                {"limits": (0.1, 72.0), "step": 0.5},
            )
            for field, label in (("xRangeThrDb", "距离"), ("xDopplerThrDb", "速度"))
            for i in range(3)
        ],
    ),
    (
        "CFAR · 主导列补点",
        "受距离 CFAR 总开关控制",
        [
            ("xCfar.bDominantSupplement", "启用", "bool", {}),
            (
                "xCfar.ulDominantStart",
                "起始距离 bin",
                "int",
                {"limits": (0, MAX_RANGE_BINS - 1)},
            ),
            (
                "xCfar.ulDominantEnd",
                "终止距离 bin",
                "int",
                {"limits": (0, MAX_RANGE_BINS - 1)},
            ),
            (
                "xCfar.xDominanceThrDb",
                "主导强度门限 (dB)",
                "float",
                {"limits": (0.1, 72.0), "step": 0.1},
            ),
            (
                "xCfar.xSupplementThrDb",
                "补点门限 (dB)",
                "float",
                {"limits": (0.1, 72.0), "step": 0.1},
            ),
            (
                "xCfar.ulSupplementGuard",
                "保护单元",
                "int",
                {"limits": (0, MAX_RANGE_BINS - 1)},
            ),
            (
                "xCfar.ulSupplementWin",
                "参考窗",
                "int",
                {"limits": (1, MAX_RANGE_BINS - 1)},
            ),
            (
                "xCfar.ulSupplementLimit",
                "每帧补点上限 (0=不限)",
                "int",
                {"limits": (0, dtypes.MAX_DETECTIONS)},
            ),
        ],
    ),
    (
        "双 PRF · 配对",
        "需要双波形与 CP_ADC 采集",
        [
            ("xPair.bEnable", "启用", "bool", {}),
            ("xPair.ucMaxUnfold", "最大展开周期数", "int", {"limits": (0, 8)}),
            ("xPair.bShareTaken", "扩展目标共享伙伴", "bool", {}),
            ("xPair.bCarryConfirmed", "同类帧延续", "bool", {}),
            ("xPair.bRescueUnfolded", "高速点救回", "bool", {}),
        ]
        + [
            (f"xPair.{field}", label, "float", {"limits": limits, "step": step})
            for field, label, limits, step in (
                ("xRangeGate", "距离关联门 (m)", (0.01, 100.0), 0.1),
                ("xVelocityGate", "速度关联门 (m/s)", (0.01, 100.0), 0.1),
                ("xMaxSpeed", "最大速度 (m/s)", (0.01, 300.0), 1.0),
                ("xAngleGate", "视线夹角门 (°)", (0.01, 180.0), 1.0),
                ("xMinCoherence", "最小阵列相干度", (0.0, 1.0), 0.05),
                ("xAmbiguityMargin", "候选唯一性裕度", (0.0, 10.0), 0.05),
                ("xMinOverlapSpeed", "重叠救回最小速度 (m/s)", (0.0, 100.0), 0.1),
                ("xOverlapRangeBins", "重叠距离门 (bin)", (0.0, 100.0), 0.1),
                ("xOverlapDopplerBins", "重叠速度门 (bin)", (0.0, 100.0), 0.1),
                ("xOccupantRangeBins", "占用距离门 (bin)", (0.0, 100.0), 0.1),
                ("xMaskMarginDb", "遮挡功率裕度 (dB)", (0.0, 72.0), 0.1),
            )
        ],
    ),
]

#: CFAR 门限的单位口径 —— 这是 P3 把 log2 换成 dB 的全部理由。
_CFAR_TIP = (
    "门限是**功率域 dB**,与 TI CLI 的 cfarCfg 同口径:判据 cell − 噪声估计 > 门限,"
    "在 log2 幅度域是加性的。换算 dB = log2 × 6.0206(幅度 log2 +1 ⇒ 功率 ×4)。\n\n"
    "⚠ 别拿教科书公式 α = N(Pfa^(-1/N)−1) 直接换算:log2 域的算术平均在功率域是"
    "**几何平均**,同一 α 下门限期望低约 2.4 dB(N=16),实际 Pfa 会高于公式预测。"
    "门限只能实测标定 —— 实测表见 docs/config_schema.md。\n\n"
    "距离和速度检测可分别开关;关闭时保留门限,再次开启恢复。两维都关闭不输出检测点。"
    "关闭距离检测也会停止主导 Doppler 列的距离补点。峰值分组是独立设置。\n\n"
    "⚠ cacc(功率域求和)的门限**不可与 ca 直接比**:等值参考窗下它的噪声估计比 ca 高 "
    "10·log10(N) dB(N=16 时 12.04 dB),照抄 ca 的门限会几乎什么都检不出来。\n\n"
    "分组方案决定邻居从哪来:det_matrix 读检测矩阵(没过门限的大邻居也能淘汰当前峰),"
    "cfar_peak 只读 CFAR 检出列表(未检出的邻居按 0 计,保留得更多)。"
)

#: DC 签名去除的两条必须说清楚的行为(否则用户会以为开关没生效)。
_DCSIG_TIP = (
    "收发天线直接耦合在 DC 附近 bin 产生固定强杂波。开启后**当前这一帧只做校准、"
    "不做补偿**(签名就是从它算出来的),要到下一帧才看得到效果 —— 在回看模式下尤其"
    "容易被误认为开关没生效。签名按 (发射通道, RX) 分别求:TDM 下三个 TX 的耦合路径"
    "不同,混在一起求一个均值是去不干净的。\n\n"
    "距离偏置(range_bias)不在这里 —— 它是把 bin 翻译成米的平移量,只走 profile.toml,"
    "理由同下方只读组。"
)

#: 扫描网格 = 估计器的**求值后端**开关(2026-09-12 beam/dbf2d 兼并后)。
_SCAN_TIP = (
    "角度扫描网格是「逐格复加权取最大」这个**求值后端**的开关,不是某个变体的私有配置:"
    "六项全零 ⇒ 走角度 FFT 快路径(阵元须落在规范正弦格上,规则阵才成立);任意一项"
    "非零 ⇒ 改走直接加权扫描(逐格算 Σw·snap 的功率,对阵列几何无要求 —— 稀疏阵、"
    "非均匀阵都归它,afm761 的 16 元 3-z 阵就靠它)。\n\n"
    "两个后端是**同一个估计器**(波束形成)的两种求值方式,argmax 口径完全一致;换后端"
    "只改候选集与代价。同口径实测(afm761 网格 141×121 = 17061 格):直接加权 "
    "28 ms/帧 vs FFT 快路径 4 ms/帧 —— 这是它这一帧里最贵的一级。\n\n"
    "⚠ variant=dbf2d 是「必须配网格」的显式声明:eChainInit 见它没配网格直接报错,"
    "不回落快路径。music/dml 不消费网格(它们是各自的估角器),配了同样在 init 处"
    "显式拒 —— 悄悄收下的后果是「配了却没生效」。\n\n"
    "⚠ 直接加权后端没有第二峰/扩展最大速度/角度亚 bin 插值,任何一个开着都会让会话"
    "重建失败(同上,显式拒)。\n\n"
    "⚠ 网格盖不住的目标会被拉向网格边缘 —— 部署方负责让网格盖住视场"
    "(参考 mmwlab afm761:方位 −70..70° 步 1°、俯仰 −30..30° 步 0.5°)。\n\n"
    "⚠ 全 z=0 的直线阵(1D)会让直接加权塌缩成**方位-only**:俯仰只留 0° 一格,填的"
    "俯仰配置被忽略、俯仰一律报 0 —— 线阵只有锥角,az/el 本来就不可分;不塌缩的 2D "
    "扫描会**伪造俯仰**(实测真值 (10.3°, 0°) 被报成 (11.0°, −20.5°))。\n\n"
    f"格数上限 {DOA_SCAN_MAX_CELLS}(≈181×181,含端点),超了 eChainInit 直接拒。\n\n"
    "在「DOA · 估角变体」里切到 dbf2d 会自动补一套默认网格;切到 corr/music/dml 会"
    "自动清零并记住上次合法网格 —— 不再需要手动制造 corr+网格 / dbf2d+无网格的非法中间态。"
)

#: DOA 的三条易误解行为。
_DOA_TIP = (
    "角度 FOV 的 enabled 不可省:四个角度全 0 的字面含义是「只接受正前方 0 度」,"
    "不是「不设限」。判据与 TI 的 Wx 落在 limitScale×[sinMin,sinMax] 内逐字等价"
    "(limitScale = cosφ,两边同除即得 sin(方位角) 落在 [sinMin,sinMax] 内)。\n\n"
    "扩展最大速度只对 TDM 有意义,且判别裕度随 TX 数缩小 —— 错误假设的方位峰,2TX 时是"
    "正确假设的 0.65(−3.7dB),3TX 时 0.85(−1.4dB)。SR61 是 3TX,裕度不宽裕;同一 "
    "range/doppler 有多目标时该算法失效(TI 自己也声明)。\n\n"
    "角度亚 bin 插值 TI 没有这一步(它靠补零做插值),属本项目扩展。"
)

#: 聚类的两条必须说清楚的事(默认关是有实测依据的,不是保守)。
_CLUSTER_TIP = (
    "SDK 3.6 的检测链**没有**聚类环节(clusproc 是 2.x 的),而本仓库的跟踪器自带 "
    "allocation 期聚类 —— 两者职责重叠。全采集 A/B 实测:**试过的每一档都让跟踪变差**"
    "(航迹寿命中位 644 → 最好 321、最差 91;ID 数从 4 涨到 5~7,说明航迹在碎裂)。"
    "机制是 DBSCAN 的噪声过滤与跟踪器的关联在抢同一件事:被判为孤立噪声的点恰是远距/"
    "低 SNR 上稀疏但真实的回波。\n\n"
    "centroid 模式一簇一点,会减少跟踪输入点数;若起航迹最少点数仍大于可用点数,"
    "会看不到航迹。可在「跟踪」组调整起航迹条件,需结合目标密度重新标定。\n\n"
    "minPts **不含被测点自身**(TI 口径);邻域半径是半径不是平方(TI 形参是平方阈值)。"
)

#: 跟踪的两条口径。
_TRACK_TIP = (
    "门限增益是**偏马氏距离**(range/az/el 三维,去 doppler)的 χ²(3) 分位点:"
    "6.251→门概率 90%、7.815→95%、**9.0→97.1%**、11.345→99%。\n\n"
    "对照 gTrack:它默认 2.0、demo 取 3.0,按全维 χ²(4) 反查只有 26.4%/44.2% —— "
    "**不是概率门**,是「小 G 少进杂波、漏检由状态机兜底」的经验增益。两种取法都成立,"
    "但要知道自己用的是哪一种。\n\n"
    "组弥散 gD 与径向速度解缠**实测都不改变本采集的跟踪结果**:前者会改状态估计"
    "(位置差几厘米)但离散指标一致、拿 20km/h 的外部真值比也没更准;后者因为 "
    "v_max=±13.7 m/s 远大于车的 5.5 m/s 而从不触发。默认关,细节见 bench/track/p7_ab.py。"
)

#: 只读:波形/几何。见模块文档串第 2 条 —— 这些不是可调项。
_READONLY_TIP = (
    "波形的函数,不是可调参数。真值由波形第一性原理推导"
    "(~/Downloads/mmwave_Calculator.m)。2026-07-17 把它当可调项从参考 profile 抄了"
    "错值,20km/h 的车被报成 4.3km/h,且三层验证同时失守。要改请改 profile.toml 并"
    "重新推导。"
)


def _get(cfg, path):
    if path == "xDoa.xScan.enabled":
        return any(_scan_fields(cfg.xDoa.xScan))
    # UI 开关复用 core 的门限符号约定;负号保存关闭状态,绝对值保留用户门限。
    for dim in ("xRange", "xDoppler"):
        if path == f"xCfar.{dim}.enabled":
            return getattr(cfg.xCfar, dim).xThrDb > 0
        if path == f"xCfar.{dim}.xThrDb":
            return abs(getattr(cfg.xCfar, dim).xThrDb)
    obj = cfg
    for part in path.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


def _set(cfg, path, value):
    if path == "xDoa.xScan.enabled":
        for (field, _), v in zip(
            cfg.xDoa.xScan._fields_, _DEFAULT_SCAN if value else (0.0,) * 6
        ):
            setattr(cfg.xDoa.xScan, field, v)
        return
    for dim in ("xRange", "xDoppler"):
        field = getattr(cfg.xCfar, dim)
        if path == f"xCfar.{dim}.enabled":
            # 旧 profile 的 0 表示关闭且没有可恢复值;首次开启采用编辑器的最小正步长。
            threshold = abs(field.xThrDb) or 0.5
            field.xThrDb = threshold if value else -threshold
            return
        if path == f"xCfar.{dim}.xThrDb":
            field.xThrDb = float(value) if field.xThrDb > 0 else -float(value)
            return
    parts = path.split(".")
    obj = cfg
    for part in parts[:-1]:
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    if parts[-1].isdigit():
        obj[int(parts[-1])] = value
    else:
        setattr(obj, parts[-1], value)


def clone_config(cfg):
    """ChainCfg 的深拷贝(ctypes 结构体没有 __deepcopy__,按字节复制)。

    改参发的是副本:处理线程正拿着旧配置在跑,原地改是数据竞争。参数抽屉
    (`drawer.ParameterPanel`)用它,外部要拿新配置请从抽屉的 `config_changed` 信号取。
    """
    dup = type(cfg)()
    ctypes.memmove(ctypes.byref(dup), ctypes.byref(cfg), ctypes.sizeof(cfg))
    return dup


def fft_options_supported(cfg) -> bool:
    """第二峰/插值只在二维 FFT 后端实现,不是所有 beam 后端都消费这些字段。"""
    a = cfg.xArray
    z = list(a.xPosZ)[: a.ulNumVirtual]
    return (
        variant_name(cfg.eDoa) == "beam"
        and not any(_scan_fields(cfg.xDoa.xScan))
        and bool(z)
        and max(z) - min(z) >= 0.25
    )


def parameter_error(cfg) -> str:
    """可编辑字段的组合预检。只读几何仍由 core 校验;这里不调用有全局副作用的 eChainInit。"""
    w, c, d = cfg.xWave, cfg.xCfar, cfg.xDoa
    for _title, _tag, rows in _SPEC:
        for path, label, kind, _opts in rows:
            if kind == "float" and not math.isfinite(_get(cfg, path)):
                return f"{label}必须是有限数值"
    n_range = int(w.ulNumAdcSamples)
    active = n_range - c.ulRangeSkipLow - c.ulRangeSkipHigh
    if active <= 0:
        return "CFAR:两端 skip 总和必须小于距离门数"
    for label, dim, length in (
        ("距离", c.xRange, active),
        ("速度", c.xDoppler, w.ulNumDopplerChirps),
    ):
        if dim.xThrDb <= 0:
            continue
        if not dim.ulWinLen or 2 * (dim.ulWinLen + dim.ulGuardLen) >= length:
            return f"{label} CFAR:2×(参考窗+保护单元)必须小于有效门数 {length}"
        if (
            dim.eNoiseMode == dtypes.E_CFAR_OS
            and not 1 <= dim.ulOsRank <= 2 * dim.ulWinLen
        ):
            return f"{label} CFAR:OS 秩必须在 1..{2 * dim.ulWinLen} 内"
    if not 0 <= c.ulMinRangeBin <= c.ulMaxRangeBin < n_range:
        return f"CFAR 输出裁剪:需要 0 ≤ 起始 bin ≤ 终止 bin < {n_range}"
    if c.bSegmented:
        if not 0 <= c.ulSegmentEnd[0] < c.ulSegmentEnd[1] < n_range:
            return "分段 CFAR:两个终止 bin 必须递增且在距离门范围内"
        if any(v <= 0 for v in (*c.xRangeThrDb, *c.xDopplerThrDb)):
            return "分段 CFAR:六个门限必须为正"
    if c.bDominantSupplement:
        if not 0 <= c.ulDominantStart <= c.ulDominantEnd < n_range // 2:
            return "主导列补点:距离区间必须递增且位于距离 FFT 前半区"
        if (
            c.ulSupplementWin == 0
            or c.ulSupplementGuard + c.ulSupplementWin >= n_range // 2
        ):
            return "主导列补点:参考窗必须为正,且保护单元+参考窗小于距离门数的一半"
        if min(c.xDominanceThrDb, c.xSupplementThrDb) <= 0:
            return "主导列补点:两个门限必须为正"
    dc = cfg.xDcSig
    if dc.bEnable and (
        not 0
        < dc.ulNumPosBins + dc.ulNumNegBins
        <= min(dtypes.RANGE_DCSIG_MAX_BINS, n_range)
        or dc.ulNumAvgChirps == 0
    ):
        return f"DC 耦合签名:正负 bin 数之和须在 1..{min(dtypes.RANGE_DCSIG_MAX_BINS, n_range)} 内,校准 chirp 数须为正"
    name = variant_name(cfg.eDoa)
    if w.eModulation == dtypes.E_MOD_DDM and name in ("music", "dml"):
        return "DDM 单快照路径支持 beam/dbf2d/corr/phasefit,不支持 MUSIC/DML"
    scan = any(_scan_fields(d.xScan))
    if name in ("corr", "phasefit", "music", "dml") and scan:
        return f"{name} 不使用扫描网格,请切到 beam/dbf2d"
    if name == "dbf2d" and not scan:
        return "dbf2d 必须启用扫描网格"
    if scan and not scan_ready(d.xScan):
        return f"DOA 扫描网格:两轴步长须为正、起止递增且各至少 2 格,总格数不超过 {DOA_SCAN_MAX_CELLS}"
    if (
        name == "beam"
        and not scan
        and doa_route_recommend(cfg.xArray)[0] == dtypes.E_DOA_VARIANT_DBF2D
    ):
        return "当前阵列无法使用 FFT 后端,请启用扫描网格"
    if (
        d.bAngleInterp or d.bExtMaxVel or d.xMultiPeak.bEnable
    ) and not fft_options_supported(cfg):
        return "第二峰、角度插值和扩展最大速度仅支持二维 beam FFT 后端"
    if d.bExtMaxVel and (w.eModulation != dtypes.E_MOD_TDM or w.ulNumTx < 2):
        return "扩展最大速度仅支持至少两个 TX 的 TDM 波形"
    f = d.xFov
    if f.bEnable and not (
        -90 <= f.xAzMinDeg <= f.xAzMaxDeg <= 90
        and -90 <= f.xElMinDeg <= f.xElMaxDeg <= 90
    ):
        return "DOA FOV:角度下限不能大于上限,且须在 ±90° 内"
    if d.bNearElevationSkip and w.eModulation != dtypes.E_MOD_DDM:
        return "近距俯仰豁免仅支持 DDM"
    if cfg.xPair.bEnable:
        if not cfg.xAlternateWave.xChirpInterval or w.eModulation != dtypes.E_MOD_DDM:
            return "双 PRF 配对需要双波形 DDM 配置与 CP_ADC 采集"
        if cfg.xCluster.eMode != dtypes.E_CLUSTER_OFF or dc.bEnable:
            return "双 PRF 配对与 DBSCAN/DC 耦合签名互斥;请先关闭配对再启用"
        p = cfg.xPair
        if (
            min(p.xRangeGate, p.xVelocityGate, p.xMaxSpeed, p.xAngleGate) <= 0
            or p.xAngleGate > 180
            or not 0 <= p.xMinCoherence <= 1
        ):
            return "双 PRF 配对:距离/速度/夹角门须为正,相干度须在 0..1 内"
    return ""


def _scan_fields(scan):
    """xScan 的六个字段,顺序 = (起,止,步)×(方位,俯仰)。"""
    return (
        scan.xAzStartDeg,
        scan.xAzStopDeg,
        scan.xAzStepDeg,
        scan.xElStartDeg,
        scan.xElStopDeg,
        scan.xElStepDeg,
    )


def variant_name(e_doa):
    """eDoaVariant 编号 → profile 里的变体名;未知编号返回 "?"。

    与下拉框、profile 解析共用 binding.config 那张表(见文件头 _DOA 注释):
    雷达信息页拿它当名字用,不自己抄一份编号↔名字。
    """
    return DOA_VARIANT_NAMES.get(int(e_doa), "?")


def scan_grid_cells(scan):
    """扫描网格 → (方位格数, 俯仰格数);六项全零(未配)⇒ None。

    ⚠ 这是**显示**用的复算(参数抽屉里扫描网格组的活标签),判据永远是 C 侧的
    eDoaScanValidate/eChainInit,口径与 core 的 ulDoaScanAxisPoints(cfg.c,校验与
    beam 建表共用的那一份)一致(含端点、跨距/步长四舍五入)。算错只影响标签上的
    数字与配色,不改任何真值 —— 也正因为如此,这里**不作**任何否决性判断。
    非法轴(步长 ≤ 0 或止 < 起)计 0。
    """
    values = [float(v) for v in _scan_fields(scan)]
    if not any(values):
        return None

    def axis(start, stop, step):
        if step <= 0.0 or stop < start:
            return 0
        return int((stop - start) / step + 0.5) + 1

    return axis(values[0], values[1], values[2]), axis(values[3], values[4], values[5])


def scan_ready(scan) -> bool:
    """网格已配且能过 eDoaScanValidate 的显示侧预判(不作为 C 侧判据)。"""
    cells = scan_grid_cells(scan)
    if cells is None:
        return False
    az, el = cells
    return (
        az >= 2
        and el >= 2
        and az * el <= DOA_SCAN_MAX_CELLS
        and -90 <= scan.xAzStartDeg <= scan.xAzStopDeg <= 90
        and -90 <= scan.xElStartDeg <= scan.xElStopDeg <= 90
    )


def scan_tag(scan) -> str:
    """扫描网格组的活标签:后端与格数随当前值复算。

    兼并 dbf2d 后「配没配网格」决定的是**求值后端**(FFT 快路径 vs 逐格复加权),
    所以标签报的是后端而不只是数字 —— 改完就能看到后端起没起来。
    """
    cells = scan_grid_cells(scan)
    if cells is None:
        return "未配 ⇒ 角度 FFT 快路径"
    az, el = cells
    if az < 2 or el < 2:
        return "非法轴 ⇒ 会被拒"
    if az * el > DOA_SCAN_MAX_CELLS:
        return f"{az}×{el} 超上限"
    return f"≈{az}×{el} 格 ⇒ 逐格复加权"


def group_tip(title: str) -> str:
    """参数组标题 → 该组的说明(tooltip);没有说明的组返回空串。"""
    if title == _SCAN_GROUP:
        return _SCAN_TIP
    if title == "DC 耦合签名":
        return _DCSIG_TIP
    if title.startswith("CFAR"):
        if title == "CFAR · 分段门限":
            return "启用后按两个距离 bin 边界使用三段距离/速度门限,覆盖普通门限值;距离/速度启用开关仍有效。关闭后恢复普通门限。"
        if title == "CFAR · 主导列补点":
            return "对主导 Doppler 列在指定距离区间补充距离检测;参数位于距离 FFT 前半区,关闭距离 CFAR 会同时停止补点。"
        return _CFAR_TIP
    if title.startswith("DOA"):
        return _DOA_TIP
    if title.startswith("聚类"):
        return _CLUSTER_TIP
    if title == "跟踪":
        return _TRACK_TIP
    if title == "双 PRF · 配对":
        return "仅已声明备用波形与 CP_ADC 的 DDM 采集可启用。配对要求 DC 耦合签名与聚类关闭;关闭配对后可单独对照这些功能。修改会重建配对状态与航迹。"
    return ""
