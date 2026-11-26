"""视图共享状态:当前会话的雷达信息、当前帧、选中的检测 / 单元、选中的航迹、列表筛选,
以及跨屏跳转请求(照 ccm `gui/state.py`)。

所有视图只通过它取数据、发选择:任一屏点选,各屏同步;换屏不丢选中。
换帧时保持选中的 (距离 bin, 多普勒 bin) 不变;第一次显示时自动选 SNR 最高的检测。
例外是跟随运动目标:选中的是检测、而新一帧该单元没有检测时,按上一帧的距离与速度外推到
新帧时刻(R + v·dt;检出点速度远离为正,2026-10-26 用 SR61 采集核对:正速度点的距离逐帧
增大),跟到门限内最近的检测。dt = 帧序号差 × 帧周期,须在 (0, 1] s(回退、跳转、改参重算
同一帧都不跟)。本仓库没有真值,所以没有 ccm 的「按真值跟随」分支。
"""

from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QObject, Signal

#: 滑出栏的页:溯源页的序号(屏底部「看它的来路」按钮请求打开它)
PANEL_TRACE = 0

# 跟随运动目标的门限:外推位置与检测的距离差、速度差
FOLLOW_RANGE_M = 1.5
FOLLOW_VELOCITY_MPS = 1.0


@dataclass(frozen=True)
class Selection:
    range_bin: int
    doppler_bin: int  # core 的无符号多普勒 bin
    detection: int | None  # 该单元若是检测,则为检测在本帧中的行号


@dataclass(frozen=True)
class HistoryPick:
    """点云累积层里点中的一个历史检测:所在帧与该帧的检测字段(值,不是行号引用)。"""

    index: int  # 处理序号(时间线位置)
    frame_id: int  # 采集帧号(usFrameIdx)
    row: int  # 该帧检测行号
    range_bin: int
    doppler_bin: int
    range_m: float
    velocity_mps: float
    snr_db: float
    noise_db: float
    azimuth_deg: float
    elevation_deg: float
    xyz: tuple[float, float, float]
    pair_label: str  # 双 PRF 配对标签(Detections.pair_label),其余板为 —


class ViewState(QObject):
    source_changed = Signal(object)  # RadarInfo
    frame_changed = Signal(object)  # ProcessedFrame
    frames_processed = Signal(object)  # 本批全部新处理帧,供累积层接收跳帧时未显示的帧
    selection_changed = Signal(object)  # Selection
    track_changed = Signal(object)  # 航迹 id(int)或 None
    filter_changed = Signal(object)  # diagnostics.STAGES 的键或 None
    screen_requested = Signal(str)  # 请求切换到某个屏(溯源页、耗时页里的跳转)
    panel_requested = Signal(
        int
    )  # 请求打开滑出栏的某一页(0 溯源 1 参数 2 判据 3 雷达 4 耗时)
    config_changed = Signal(object)  # ChainCfg:换会话或改处理参数(测角屏据此重建探针)
    velocity_limit_changed = Signal(float)  # 点云速度色标量程(m/s),三维与二维投影共用
    #: 点云余辉 / 全量累积里点中的历史检测(HistoryPick)或 None。它不是当前帧的选中:
    #: 不换帧、不改 selection,只供点云高亮与信息页显示。
    history_pick_changed = Signal(object)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.source = None  # RadarInfo
        self.frame = None  # ProcessedFrame
        self.selection: Selection | None = None
        self.track: int | None = None
        self.stage_filter: str | None = None
        self.config = None  # 当前处理器用的 ChainCfg(只读;视图需要配置时取它)
        self.velocity_limit_mps = 5.0
        self.history_pick: HistoryPick | None = None

    def set_history_pick(self, pick: "HistoryPick | None") -> None:
        if pick != self.history_pick:
            self.history_pick = pick
            self.history_pick_changed.emit(pick)

    def set_config(self, config) -> None:
        self.config = config
        self.config_changed.emit(config)

    def set_velocity_limit(self, limit_mps: float) -> None:
        if limit_mps != self.velocity_limit_mps:
            self.velocity_limit_mps = limit_mps
            self.velocity_limit_changed.emit(limit_mps)

    def set_filter(self, stage: str | None) -> None:
        self.stage_filter = stage
        self.filter_changed.emit(stage)

    def request_screen(self, key: str) -> None:
        self.screen_requested.emit(key)

    def request_panel(self, page: int) -> None:
        self.panel_requested.emit(page)

    def set_source(self, source) -> None:
        self.source = source
        self.frame = None
        self.selection = None
        self.track = None
        self.stage_filter = None
        self.set_history_pick(None)
        self.source_changed.emit(source)

    def set_frame(self, frame) -> None:
        previous, self.frame = self.frame, frame
        if self.track is not None and self.track not in frame.tracks.track_id:
            self.track = None  # 航迹已终止(或改参后航迹清零):不留悬空的 id
            self.track_changed.emit(None)
        self.frame_changed.emit(frame)
        if self.selection is None:
            det = frame.detections
            if len(det):
                self.select_detection(int(np.argmax(det.snr_db)))
            else:
                radar = frame.radar
                self.select_cell(radar.n_range // 2, radar.n_doppler // 2)
            return
        followed = self._follow(previous, frame)
        if followed is not None:
            self.select_detection(followed)
        else:
            self.select_cell(self.selection.range_bin, self.selection.doppler_bin)

    def _follow(self, previous, frame) -> int | None:
        """上一帧选中的检测在新帧里对应的检测行号;原单元仍有检测或跟不上时为 None。"""
        sel = self.selection
        if (
            previous is None
            or sel.detection is None
            or sel.detection >= len(previous.detections)
        ):
            return None
        det = frame.detections
        same_cell = (det.range_bin == sel.range_bin) & (
            det.doppler_bin == sel.doppler_bin
        )
        if np.any(same_cell):
            return None
        dt = (frame.index - previous.index) * frame.radar.frame_period_s
        if not 0 < dt <= 1.0 or not len(det):
            return None
        old = previous.detections
        k = sel.detection
        range_m = old.range_m[k] + old.velocity_mps[k] * dt
        cost = (
            np.abs(det.range_m - range_m) / FOLLOW_RANGE_M
            + np.abs(det.velocity_mps - old.velocity_mps[k]) / FOLLOW_VELOCITY_MPS
        )
        best = int(np.argmin(cost))
        return best if cost[best] <= 1.0 else None

    def select_cell(self, range_bin: int, doppler_bin: int) -> None:
        if self.frame is None:
            return
        radar = self.frame.radar
        range_bin = int(np.clip(range_bin, 0, radar.n_range - 1))
        doppler_bin = int(doppler_bin) % radar.n_doppler
        det = self.frame.detections
        rows = np.flatnonzero(
            (det.range_bin == range_bin) & (det.doppler_bin == doppler_bin)
        )
        detection = int(rows[0]) if len(rows) else None
        self.selection = Selection(range_bin, doppler_bin, detection)
        self.selection_changed.emit(self.selection)

    def select_detection(self, row: int) -> None:
        """选中第 row 个检测。直接记行号(同一单元可能有多个检测,如测角第二峰)。"""
        if self.frame is None:
            return
        det = self.frame.detections
        if 0 <= row < len(det):
            self.selection = Selection(
                int(det.range_bin[row]), int(det.doppler_bin[row]), int(row)
            )
            self.selection_changed.emit(self.selection)

    def select_track(self, track_id: int | None) -> None:
        self.track = None if track_id is None else int(track_id)
        self.track_changed.emit(self.track)
