"""处理链一行摘要:每级一个状态等级 + 一个有代表性的实时数(界面的处理链行逐帧取用)。

纯数据,不碰 Qt。取数只读帧本身与 `frame.diagnostics`(帧级汇总 / 逐检测判据),
某级没有指标时数值为「—」。每级选哪个数:

- ADC       饱和率(饱和 int16 / 全部 int16);
- 距离      最强距离门的距离;
- 多普勒    fold 余量中位(DDM;TDM 没有 fold,为「—」);
- CFAR      检测数;
- 解耦      细距离分辨率 |S_f|·N/(N−1)(仅步进频);
- 测角      相干度中位;
- 跟踪      航迹数;
- 场景      点数。
"""

import numpy as np

from .criteria import Level
from .slices import fine_resolution_m

#: 处理链各级的 key,顺序即处理顺序(与 ui.screens.SCREEN_TABS 一致)
STAGE_KEYS = (
    "adc",
    "rangefft",
    "doppler",
    "cfar",
    "decouple",
    "doa",
    "track",
    "results",
)
#: 屏 key → 诊断级 id(逐检测判据);adc / 距离是帧级,其余没有判据
_VERDICT_STAGE = {"doppler": "doppler", "cfar": "cfar", "doa": "doa", "track": "track"}
NA = int(Level.NA)
DASH = "—"


def _verdict_level(diag, stage: str) -> int:
    """本级需要看的检测:没有诊断为 NA;有「异常」级的检测为 BAD;有「需要看」为 WARN;否则 OK。"""
    levels = diag.verdicts.levels[stage]
    if not (levels != Level.NA).any():
        return NA
    if (levels >= Level.BAD).any():
        return int(Level.BAD)
    return int(Level.WARN) if diag.summary.flagged.get(stage, 0) else int(Level.OK)


def _finite(value, fmt: str) -> str:
    return fmt.format(value) if value is not None and np.isfinite(value) else DASH


def chain_cells(frame) -> dict[str, tuple[int, str]]:
    """{级 key: (等级 Level 的整数值, 实时数文字)}。"""
    diag = frame.diagnostics
    s = diag.summary if diag is not None else None
    radar = frame.radar
    cells = {key: (NA, DASH) for key in STAGE_KEYS}
    cells["cfar"] = (NA, f"{len(frame.detections)} 检测")
    cells["track"] = (NA, f"{len(frame.tracks)} 航迹")
    cells["results"] = (NA, f"{len(frame.detections)} 点")
    if radar.stepped:
        cells["decouple"] = (
            int(Level.OK) if diag is not None else NA,
            f"细分辨 {fine_resolution_m(radar):.2f} m",
        )
    if s is None:
        return cells
    if s.adc is not None:
        rate = s.adc.saturation_rate
        text = (
            DASH
            if not np.isfinite(rate)
            else ("饱和 0%" if rate == 0 else f"饱和 {rate * 100:.1f}%")
        )
        cells["adc"] = (s.adc_level, text)
    if s.range_peak_bin >= 0:
        cells["rangefft"] = (
            s.range_level,
            f"峰 {s.range_peak_bin * radar.range_step_m:.1f} m",
        )
    cells["doppler"] = (
        _verdict_level(diag, "doppler"),
        _finite(s.fold_margin_median_db, "余量 {:.1f} dB"),
    )
    cells["cfar"] = (_verdict_level(diag, "cfar"), cells["cfar"][1])
    cells["doa"] = (
        _verdict_level(diag, "doa"),
        _finite(s.coherence_median, "相干 {:.2f}"),
    )
    cells["track"] = (_verdict_level(diag, "track"), cells["track"][1])
    return cells


def trace_facts(frame, k: int) -> dict[str, tuple[int, str]]:
    """选中检测 k 在每一级留下的痕迹:{级 key: (等级, 一行事实)},溯源页的竖直时间线用。

    ADC / 距离是帧级事实(没有逐检测判据);多普勒 / CFAR / 测角 / 跟踪取判据层的一句结论;
    解耦(仅步进频)是该检测的展开周期与解出的距离、速度。取不到的级不在结果里。"""
    from .slices import decouple_table

    diag = frame.diagnostics
    det = frame.detections
    out: dict[str, tuple[int, str]] = {}
    s = diag.summary if diag is not None else None
    if s is not None and s.adc is not None:
        spread = s.adc.rms_spread_db
        out["adc"] = (
            s.adc_level,
            f"饱和 {s.adc.total_saturated} 个样本 · RX 电平极差 "
            + (f"{spread:.1f} dB" if np.isfinite(spread) else DASH),
        )
    gate = int(det.range_bin[k])
    out["rangefft"] = (
        s.range_level if s is not None else NA,
        f"第 {gate} 格 · {float(det.range_m[k]):.2f} m",
    )
    if frame.radar.stepped:
        table = decouple_table(frame)
        if table is not None and table["index"][k] >= 0:
            out["decouple"] = (
                int(Level.OK),
                f"第 {int(table['period'][k])} 周期 → {float(det.range_m[k]):.2f} m · "
                f"{float(det.velocity_mps[k]):+.2f} m/s",
            )
    if diag is not None:
        for key, stage in _VERDICT_STAGE.items():
            out[key] = (
                int(diag.verdicts.level(stage, k)),
                diag.verdicts.text(stage, k),
            )
    return out
