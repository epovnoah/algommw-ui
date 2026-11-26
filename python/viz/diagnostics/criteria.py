"""判据:按阈值把每个检测每一级的诊断指标分成「正常 / 需要看 / 异常」,并给出一句结论。

接口照 ccm `diagnostics/criteria.py`:界面(KPI 条、列表、溯源检查器、全链状态条)只认
`Level` / `STAGES` / `Verdicts` 这一套。阈值是 `CriteriaConfig` 的字段,每个字段带界面元数据
(参数抽屉「判据」页按它生成),可按板存进 data/criteria.json(读失败回退默认)。

默认值的来由(uiplan.md §6「标定记录」,已删,见 git 历史 fa7d3d4):
- NIS 的 χ² 分位、贴边即需要看:物理含义直接给定;
- fold 余量、空子带抑制、CFAR 余量、相干度:两块板默认采集各 200 帧的分布标定
  (python/bench/diag/calib.py),「越大越好」的指标取 WARN = p10、BAD = p1,按板放在 `DEFAULTS`;
- 「未关联到航迹」默认不标黄(未关联是常态,见该字段注释)。原则:被标「需要看」的检出点应是少数,
  否则 KPI 与检查器失去筛选意义(ccm 的「注意力管理」)。
"""

import dataclasses
import json
import math
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path

import numpy as np

from binding.paths import PROJECT_ROOT

from .metrics import DetectionMetrics


class Level(IntEnum):
    NA = -1  # 不适用 / 未提供诊断
    OK = 0
    WARN = 1  # 需要看
    BAD = 2  # 异常


#: 逐检测的级:id → 显示名,顺序即处理顺序。ADC / 距离 FFT 是帧级检查,不在这里。
STAGES = {"doppler": "多普勒", "cfar": "CFAR", "doa": "测角", "track": "跟踪"}

#: 界面改过的判据按板存在这里(ALGOMMW_UI_NO_SETTINGS=1 时界面不读不写)
CRITERIA_PATH = PROJECT_ROOT / "data" / "criteria.json"
FORMAT = "algommw-criteria"


def _param(default, label: str, unit: str = "", lo=None, hi=None, step=None, tip=""):
    """带界面元数据的字段(「判据」页的 ParameterTree 按它生成;lo / hi 也是取值校验)。"""
    meta = {"label": label, "unit": unit, "lo": lo, "hi": hi, "step": step, "tip": tip}
    return field(default=default, metadata=meta)


@dataclass(frozen=True)
class CriteriaConfig:
    """判据阈值。按板的默认见 `DEFAULTS`;字段默认值用于未登记的板,取两块板标定值里
    较宽松(标得少)的一档 —— DDM 两项只有 AFM761 有数,即 AFM761 的值。"""

    # 多普勒(仅 DDM)。ccm 的对照值是余量 0.3 dB(其 1.25 dB 理论上限)/ 空带 10 dB,
    # 本仓库候选度量定义不同,按实采样标定(uiplan.md §9)
    fold_margin_warn_db: float = _param(
        0.45,
        "fold 判决余量下限",
        "dB",
        0.0,
        6.0,
        0.05,
        "胜出 fold 的候选度量领先次优多少 dB;低于它标「需要看」(只看 DDM)",
    )
    empty_suppression_warn_db: float = _param(
        13.0,
        "空子带抑制下限",
        "dB",
        0.0,
        60.0,
        0.5,
        "选中 fold 的占用子带均功率高出空子带多少 dB;低于它标「需要看」",
    )
    # CFAR:贴门限。原计划取物理阈值 3 dB,标定显示两块板检出点余量中位只有 3.7 / 4.3 dB,
    # 3 dB 会把 43% / 32% 的检出点标黄 —— 改与其它指标同口径取 p10(SR61 0.5、AFM761 1.2)
    cfar_margin_warn_db: float = _param(
        0.5,
        "CFAR 超门限余量下限",
        "dB",
        0.0,
        30.0,
        0.5,
        "两向余量的较小值低于它标「需要看」(贴门限)",
    )
    # 测角。ccm 的对照值是 0.8 / 0.5
    coherence_warn: float = _param(
        0.51, "相干度标黄阈值", "", 0.0, 1.0, 0.01, "相干度低于它标「需要看」"
    )
    coherence_bad: float = _param(
        0.44, "相干度标红阈值", "", 0.0, 1.0, 0.01, "相干度低于它标「异常」"
    )
    edge_is_warn: bool = _param(
        True, "贴扫描边界即需要看", tip="仅直接加权后端:角度落在扫描网格端点"
    )
    # 跟踪:χ²(0.99, 4) / χ²(0.999, 4),量测 4 维
    nis_warn: float = _param(
        13.28, "NIS 标黄阈值", "", 0.0, 100.0, 0.1, "χ²(0.99, 4) = 13.28"
    )
    nis_bad: float = _param(
        18.47, "NIS 标红阈值", "", 0.0, 100.0, 0.1, "χ²(0.999, 4) = 18.47"
    )
    # 默认关:标定显示未关联的检出点占 37%(SR61)/ 92%(AFM761)—— 静止杂波与偶发点本来就
    # 不进航迹,「未关联」是常态而不是异常;关着时它只作一句结论(等级 OK),不标黄
    unassociated_is_warn: bool = _param(
        False, "未关联到航迹即需要看", tip="点没有关联到任何航迹(也没有起新航迹)"
    )

    def with_values(self, **values) -> "CriteriaConfig":
        """按字段类型转换并校验范围后返回新判据;未知字段或越界 → ValueError。"""
        fields = {f.name: f for f in dataclasses.fields(self)}
        clean = {}
        for name, value in values.items():
            f = fields.get(name)
            if f is None:
                raise ValueError(f"未知判据字段:{name}")
            if isinstance(f.default, bool):
                if not isinstance(value, (bool, np.bool_)):
                    raise ValueError(f"{name} 应为布尔值,实际 {value!r}")
                clean[name] = bool(value)
                continue
            value = float(value)
            lo, hi = f.metadata.get("lo"), f.metadata.get("hi")
            if not math.isfinite(value) or (lo is not None and value < lo):
                raise ValueError(f"{name} = {value} 越界")
            if hi is not None and value > hi:
                raise ValueError(f"{name} = {value} 越界")
            clean[name] = value
        return dataclasses.replace(self, **clean)

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}


#: 按板的默认判据(profile 目录名 → 判据)。2026-10-27 用两块板默认采集各 200 帧标定
#: (python/bench/diag/calib.py;分位表与占比见 uiplan.md §6「标定记录」):WARN = p10、BAD = p1。
#: SR61 是 TDM,没有 DDM 两项。对照:ccm 的相干度阈值是 0.8 / 0.5 —— 本仓库两块板的
#: 相干度中位只有 0.79(SR61)/ 0.84(AFM761),照搬 0.8 会把 51% / 43% 的检出点标黄。
DEFAULTS = {
    "sr61": CriteriaConfig(
        cfar_margin_warn_db=0.5, coherence_warn=0.62, coherence_bad=0.52
    ),
    "afm761": CriteriaConfig(
        fold_margin_warn_db=0.45,
        empty_suppression_warn_db=13.0,
        cfar_margin_warn_db=1.2,
        coherence_warn=0.51,
        coherence_bad=0.44,
    ),
    # 步进频(CCM)板:未标定,沿用 afm761 的判据(同一块硬件、同一 DDM 折叠搜索,但 512 采样 +
    # 载频步进 + 另一组通道标定,分位值未必相同;标定要用 python/bench/diag/calib.py 在 afmccm 采集上重跑)
    "afmccm": CriteriaConfig(
        fold_margin_warn_db=0.45,
        empty_suppression_warn_db=13.0,
        cfar_margin_warn_db=1.2,
        coherence_warn=0.51,
        coherence_bad=0.44,
    ),
}


def criteria_for(profile: str) -> CriteriaConfig:
    """该板的默认判据;未登记的板用字段默认值。"""
    return DEFAULTS.get(profile, CriteriaConfig())


def load_criteria(profile: str, path: Path = CRITERIA_PATH) -> CriteriaConfig:
    """读 path 里该板存过的判据;文件缺失、格式不对、字段未知或越界时回退该板默认。"""
    default = criteria_for(profile)
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("format") != FORMAT:
            return default
        stored = data.get("boards", {}).get(profile)
        if not isinstance(stored, dict):
            return default
        return default.with_values(**stored)
    except (OSError, ValueError, TypeError, AttributeError):
        return default


def save_criteria(
    criteria: CriteriaConfig, profile: str, path: Path = CRITERIA_PATH
) -> None:
    """把该板的判据写进 path(保留其它板的条目;原文件读不出时整份重写)。"""
    path = Path(path)
    boards = {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("format") == FORMAT and isinstance(data.get("boards"), dict):
            boards = data["boards"]
    except (OSError, ValueError, AttributeError):
        pass
    boards[profile] = criteria.to_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        {"format": FORMAT, "boards": boards}, ensure_ascii=False, indent=2
    )
    path.write_text(text + "\n", encoding="utf-8")


@dataclass
class Verdicts:
    """levels[级] 为 (n,) 的 Level 整数数组;text(级, k) 生成第 k 个检测在该级的一句结论。"""

    levels: dict[str, np.ndarray]
    metrics: DetectionMetrics
    criteria: CriteriaConfig

    def level(self, stage: str, k: int) -> Level:
        return Level(int(self.levels[stage][k]))

    def flagged(self, stage: str) -> np.ndarray:
        """该级需要看或异常的检测(布尔数组)。"""
        return self.levels[stage] >= Level.WARN

    def flagged_stages(self, k: int) -> list[str]:
        return [s for s in STAGES if self.levels[s][k] >= Level.WARN]

    def worst(self, k: int) -> Level:
        return Level(max(int(self.levels[s][k]) for s in STAGES))

    def text(self, stage: str, k: int) -> str:
        m, c = self.metrics, self.criteria
        if self.levels[stage][k] == Level.NA:
            if not m.available.get(stage, False):
                return m.notes.get(stage, "不适用")
            if stage == "doa" and m.doa_index[k] < 0:
                return "无对应的测角记录(聚类合并后的点)"
            return "该检测无此级指标"
        if stage == "doppler":
            margin, supp = m.fold_margin_db[k], m.empty_suppression_db[k]
            text = f"fold {int(m.fold[k])}:领先次优 {margin:.2f} dB"
            if margin < c.fold_margin_warn_db:
                text += f"(低于 {c.fold_margin_warn_db:.2f})"
            if np.isfinite(supp):
                text += f",空子带低 {supp:.1f} dB"
                if supp < c.empty_suppression_warn_db:
                    text += f"(低于 {c.empty_suppression_warn_db:.1f})"
            return text
        if stage == "cfar":
            text = (
                f"超门限:距离向 {_margin(m.cfar_range_margin_db[k])},"
                f"多普勒向 {_margin(m.cfar_doppler_margin_db[k])}"
            )
            if m.cfar_margin_db[k] < c.cfar_margin_warn_db:
                text += f"(贴门限,< {c.cfar_margin_warn_db:g} dB)"
            return text
        if stage == "doa":
            coherence = m.coherence[k]
            text = f"相干度 {coherence:.2f}"
            if coherence < c.coherence_bad:
                text += f"(低于 {c.coherence_bad:.2f})"
            elif coherence < c.coherence_warn:
                text += f"(低于 {c.coherence_warn:.2f})"
            if m.at_scan_edge[k] == 1.0:
                text += ",贴扫描边界"
            if m.second_peak[k] == 1.0:
                text += ",第二峰点"
            return text
        tid = int(m.track_id[k])
        if tid < 0:
            return "未关联到航迹"
        if m.track_new[k]:
            return f"新建航迹 {tid}"
        if m.track_ended[k]:
            return f"关联航迹 {tid} 本帧已终止"
        nis = m.track_nis[k]
        if not np.isfinite(nis):
            return f"航迹 {tid}:本帧无 NIS"
        if nis > c.nis_bad:
            return f"航迹 {tid}:NIS {nis:.1f} > {c.nis_bad:.2f}"
        if nis > c.nis_warn:
            return f"航迹 {tid}:NIS {nis:.1f} > {c.nis_warn:.2f}"
        return f"航迹 {tid}:NIS {nis:.2f}"


def judge(metrics: DetectionMetrics, criteria: CriteriaConfig) -> Verdicts:
    """逐级逐检测分级。指标为 NaN(该检测无此级指标)的为 NA;整级不可用的全为 NA。"""
    m, c, n = metrics, criteria, len(metrics)
    av = m.available

    def grade(stage, valid, warn, bad=None) -> np.ndarray:
        if not av.get(stage, False):
            return np.full(n, int(Level.NA), int)
        out = np.where(warn, int(Level.WARN), int(Level.OK))
        if bad is not None:
            out = np.where(bad, int(Level.BAD), out)
        return np.where(valid, out, int(Level.NA)).astype(int)

    with np.errstate(invalid="ignore"):
        fold_ok = np.isfinite(m.fold_margin_db)
        low_fold = (m.fold_margin_db < c.fold_margin_warn_db) | (
            m.empty_suppression_db < c.empty_suppression_warn_db
        )
        cfar = m.cfar_margin_db
        coherence = m.coherence
        doa_warn = coherence < c.coherence_warn
        if c.edge_is_warn:
            doa_warn = doa_warn | (m.at_scan_edge == 1.0)
        tracked = m.track_id >= 0
        nis = np.where(tracked & ~m.track_new & ~m.track_ended, m.track_nis, np.nan)
        track_warn = nis > c.nis_warn
        if c.unassociated_is_warn:
            track_warn = track_warn | ~tracked
        levels = {
            "doppler": grade("doppler", fold_ok, low_fold),
            "cfar": grade("cfar", np.isfinite(cfar), cfar < c.cfar_margin_warn_db),
            "doa": grade(
                "doa", np.isfinite(coherence), doa_warn, coherence < c.coherence_bad
            ),
            "track": grade("track", np.ones(n, bool), track_warn, nis > c.nis_bad),
        }
    return Verdicts(levels, metrics, criteria)


def _margin(value: float) -> str:
    """余量文字:带符号一位小数;该向关闭(NaN)写「关」。"""
    return "关" if not np.isfinite(value) else f"{value:+.1f} dB"
