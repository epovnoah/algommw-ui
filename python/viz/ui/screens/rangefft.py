"""距离像屏:噪声底稳不稳、目标在哪个距离门、直流去干净了没有?

主图是共轴上下堆叠:上 RTI、下选中 chirp 的距离像(对照全帧平均、DC 签名去除前后),共用距离轴
与一条游标(views/rangefft.py);分段控件只剩通道。列表是本帧检测按格电平排序;选中检测所在距离门附近的距离像切片在溯源页的距离一级。距离级没有
逐检测判据。
"""

import numpy as np

from ..plots.mini import RangeLocal
from ..state import ViewState
from ..views.panels import flatten_card
from ..views.rangefft import RangeView
from ..widgets.conclusion import Conclusion
from ..widgets.stagelist import StageList
from .base import Screen
from .common import (
    COL_INDEX,
    COL_POWER,
    COL_RANGE,
    COL_SNR,
    COL_VELOCITY,
    order_by,
    summary,
)


def clauses(frame):
    s = summary(frame)
    if s is None:
        return [(f"{len(frame.detections)} 个检测", 0)]
    out = []
    if s.dcsig is not None and s.dcsig.startswith("校准中"):
        out.append((f"DC 签名{s.dcsig}(本帧距离像没有补偿)", 1))
    if s.range_peak_bin >= 0:
        out.append(
            (f"最强距离门 {s.range_peak_bin * frame.radar.range_step_m:.1f} m", 0)
        )
    if np.isfinite(s.range_noise_db):
        out.append((f"噪底中位 {s.range_noise_db:+.1f} dB", 0))
    return out


class RangeScreen(Screen):
    key = "rangefft"
    title = "距离 FFT"

    def __init__(self, state: ViewState) -> None:
        self.view = RangeView(state)
        self.list = StageList(
            state,
            "检测(按格电平)",
            [COL_INDEX, COL_RANGE, COL_VELOCITY, COL_SNR, COL_POWER],
            order=order_by("power_db"),
        )
        super().__init__(
            state,
            self.view,
            self.list,
            Conclusion(state, clauses),
            trace_plots=[RangeLocal()],
        )
        flatten_card(self)
