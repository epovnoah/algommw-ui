"""ADC 屏:采集本身正常吗(削波、通道电平、直流)?

主图是小倍数网格:每路 RX 一张 chirp × 采样时域热图,同时可见(views/adc.py 的 AdcGrid,格数随
板的真实 RX 数走);列表是各 RX 的帧级统计(RMS / 峰值 / 饱和数 / DC),点一行把对应那一格的
标签标成强调色。ADC 没有逐检测等级,这一屏的问题都是「整帧」层面的;选中 chirp 的各 RX 波形
在溯源页的 ADC 一级。
"""

import numpy as np

from ...diagnostics.summary import ADC_SPREAD_WARN_DB
from ..plots.mini import AdcWave
from ..state import ViewState
from ..views.adc import AdcGrid
from ..views.panels import flatten_card
from ..widgets.conclusion import Conclusion
from ..widgets.grid import GridTable
from ..widgets.stagelist import Column
from .base import Screen
from .common import summary

DC_WARN_DB = (
    -20.0
)  # 复均值模 / RMS 超过它 → DC 偏大(与 summary.ADC_SPREAD_WARN_DB 同门)


def _adc(frame):
    s = summary(frame)
    return None if s is None else s.adc


COL_RX = Column("通道", lambda f, r: f"RX{r + 1}", "{:s}")
COL_RMS = Column("RMS dBFS", lambda f, r: float(_adc(f).rms_dbfs[r]), "{:.2f}")
COL_PEAK = Column("峰值 LSB", lambda f, r: int(_adc(f).peak[r]), "{:d}")
COL_SAT = Column("饱和样本", lambda f, r: int(_adc(f).saturated[r]), "{:d}")
COL_DC = Column("DC dB", lambda f, r: float(_adc(f).dc_db[r]), "{:.1f}")


def clauses(frame):
    """削波、各 RX 电平极差、直流(有问题的排前面由 Conclusion 挑)。"""
    adc = _adc(frame)
    if adc is None:
        return [("该帧没有 ADC 中间量(诊断未开)", 0)]
    out = []
    n = adc.total_saturated
    out.append((f"削波 {n} 个样本" if n else "无削波", 1 if n else 0))
    spread = adc.rms_spread_db
    if np.isfinite(spread):
        out.append(
            (f"各 RX 电平极差 {spread:.1f} dB", 1 if spread > ADC_SPREAD_WARN_DB else 0)
        )
    if np.isfinite(adc.dc_db).any():
        worst = float(np.nanmax(adc.dc_db))
        out.append(
            (
                f"直流最大 {worst:.1f} dB(RX{int(np.nanargmax(adc.dc_db)) + 1})",
                1 if worst > DC_WARN_DB else 0,
            )
        )
    return out


class AdcScreen(Screen):
    key = "adc"
    title = "ADC"
    list_label = "各 RX 统计"

    def __init__(self, state: ViewState) -> None:
        self.view = AdcGrid(state)
        self.list = GridTable(
            state,
            "各 RX 帧级统计",
            [COL_RX, COL_RMS, COL_PEAK, COL_SAT, COL_DC],
            rows_for=lambda frame: _rx_rows(frame),
            on_select=lambda r: self.view.set_rx(r),
        )
        super().__init__(
            state,
            self.view,
            self.list,
            Conclusion(state, clauses),
            trace_plots=[AdcWave(self.view, state)],
        )
        flatten_card(self)


def _rx_rows(frame):
    adc = _adc(frame)
    return np.arange(0 if adc is None else len(adc.rms_dbfs), dtype=int)
