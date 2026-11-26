"""航迹生灭带:每条航迹一个水平条,横轴是近 60 帧,回答「航迹何时新建、何时终止」。

横轴是相对当前帧的帧数(−59 … 0,右端「当前」);条从航迹出现到消失(在当前帧仍在的画到右端),
条上标 `#id`,颜色与俯视图同 id;新建处画绿竖线、终止处画红竖线。新建 / 终止从逐帧记下的航迹 id
集合推出:某帧有而前一帧(已记录)没有 = 新建,前一帧有而这帧没有 = 终止。窗口左缘之外的历史、
还没记录过的帧都不下结论(条在左缘被截断,不画绿线)。

航迹 id 取跟踪中间量里的全部航迹(含待确认 = 刚新建的),与航迹栏同一口径;没有中间量时退回已确认输出。
历史由本视图自己累积(与是否可见无关):`accumulate(frame)` 记显示过的帧,`frames_processed`
补齐跳帧时没显示的中间帧;按帧序号覆盖,回看 / 重显不重复;每条记录带 `track_epoch`,只画与当前
帧同一跟踪起点的记录(改参重算后旧 id 不混进来);换源清空。
"""

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QVBoxLayout, QWidget

from ...diagnostics.slices import track_table
from .. import theme
from ..state import ViewState
from ..views.base import ResultView, set_title
from ..views.common import ClipLabel
from ..views.tracks import track_color

#: 窗口帧数、最多画的行数
WINDOW_FRAMES = 60
MAX_ROWS = 8
BAR_HEIGHT = 0.62
#: 纵轴上缘留出的空带(行高的约一格),标题 / 图例小标签压在这里,不盖住第一条航迹
TOP = -1.1


class _FrameAxis(pg.AxisItem):
    """横轴刻度:相对当前帧的偏移,0 写成「当前」。"""

    def tickStrings(self, values, scale, spacing):
        return ["当前" if round(v) == 0 else f"{round(v):d}" for v in values]


class LifetimeBand(ResultView):
    title = "航迹生灭"
    KEEP_AUTORANGE = True  # 范围每次渲染都手动设,不要「首次有数据后锁」

    def __init__(self, state: ViewState, parent: QWidget | None = None) -> None:
        super().__init__(state, parent)
        self.widget = pg.PlotWidget(axisItems={"bottom": _FrameAxis("bottom")})
        self.plot = theme.style_plot(self.widget.getPlotItem())
        self.plot.hideAxis("left")
        self.plot.hideButtons()
        self.plot.setMenuEnabled(False)
        self.plot.setMouseEnabled(False, False)
        self.plot.getViewBox().setDefaultPadding(0.0)
        self.plot.getViewBox().invertY(True)  # 第一条航迹在最上
        self.bars = pg.BarGraphItem(x0=[], x1=[], y=[], height=BAR_HEIGHT)
        self.births = pg.PlotCurveItem(
            pen=theme.pen(theme.OK, 1.2), connect="pairs", skipFiniteCheck=True
        )
        self.deaths = pg.PlotCurveItem(
            pen=theme.pen(theme.BAD, 1.2), connect="pairs", skipFiniteCheck=True
        )
        self.births.setZValue(8)
        self.deaths.setZValue(8)
        for item in (self.bars, self.births, self.deaths):
            self.plot.addItem(item)
        #: `#id` 文字:按需增补,每次渲染只改文字与位置,多余的隐藏
        self.labels: list[pg.TextItem] = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.widget)
        self.key = ClipLabel(
            "<span style='color:%s'>▍</span>新建　"
            "<span style='color:%s'>▍</span>终止" % (theme.OK, theme.BAD),
            self,
        )
        self.key.setObjectName("LifetimeKey")
        self.key.setStyleSheet(
            f"#LifetimeKey {{ background: rgba(12, 14, 18, 190); color: {theme.TEXT}; "
            f"border: 1px solid {theme.LINE}; border-radius: 5px; padding: 1px 7px; "
            f"font-size: {theme.FONT_NOTE}px; }}"
        )
        self.key.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        #: 帧序号 → (跟踪起点, 该帧输出的航迹 id 集合)
        self._seen: dict[int, tuple[int, frozenset]] = {}
        state.frames_processed.connect(self._on_processed)
        self.setMinimumHeight(0)
        self.set_note("")

    # ---- 累积 ----
    def _on_processed(self, frames) -> None:
        for frame in frames:
            self.accumulate(frame)

    def accumulate(self, frame) -> None:
        """记下本帧的航迹 id 集合(与是否可见无关);新一轮处理(从跟踪起点开始)清掉旧记录。"""
        if frame.index == frame.track_epoch:
            self._seen.clear()
        table = track_table(frame)  # 含待确认(新建)的航迹;没有中间量时退回已确认输出
        raw = frame.tracks.track_id if table is None else table["id"]
        ids = frozenset(int(t) for t in raw)
        self._seen[frame.index] = (frame.track_epoch, ids)

    def reset(self, source) -> None:
        self._seen.clear()
        self._draw([], [], [], 0)

    # ---- 渲染 ----
    def render_frame(self, frame) -> None:
        # 窗口 [index-59, index]:只取同一跟踪起点的记录(改参重算后旧 id 不混进来)
        now = frame.index
        lo = now - WINDOW_FRAMES + 1
        window = {
            i: ids
            for i, (epoch, ids) in self._seen.items()
            if lo <= i <= now and epoch == frame.track_epoch
        }
        runs, births, deaths = _lifetimes(window, lo, now)
        self._draw(runs, births, deaths, now)

    def _draw(self, runs, births, deaths, now: int) -> None:
        """runs = [(id, 起帧, 终帧)](按起帧排序);births / deaths = [帧序号]。"""
        # 行数有限:优先留当前在场的、再留最近才终止的,画出来仍按起帧排
        shown = sorted(runs, key=lambda r: (r[2], r[1], r[0]))[-MAX_ROWS:]
        shown.sort(key=lambda r: (r[1], r[0]))
        rows = {track_id: row for row, track_id in enumerate(_ordered_ids(shown))}
        x0 = np.array([a - now - 0.5 for _, a, _ in shown], float)
        x1 = np.array([b - now + 0.5 for _, _, b in shown], float)
        y = np.array([rows[t] for t, _, _ in shown], float)
        self.bars.setOpts(
            x0=x0,
            x1=x1,
            y=y,
            height=BAR_HEIGHT,
            brushes=[track_color(t) for t, _, _ in shown],
            pen=None,
        )
        n_rows = max(len(rows), 3)
        for curve, frames in ((self.births, births), (self.deaths, deaths)):
            x = np.repeat(np.array(frames, float) - now - 0.5, 2)
            if len(x):
                y_line = np.tile([TOP, n_rows - 0.5], len(frames))
                curve.setData(x=x, y=y_line)
            else:
                curve.setData(x=[], y=[])
        self.plot.setXRange(-WINDOW_FRAMES + 0.5, 0.5, padding=0)
        self.plot.setYRange(TOP, n_rows - 0.5, padding=0)
        while len(self.labels) < len(shown):
            label = pg.TextItem(color=theme.GROUND, anchor=(0.0, 0.5))
            label.setZValue(9)
            self.plot.addItem(label, ignoreBounds=True)
            self.labels.append(label)
        for k, label in enumerate(self.labels):
            if k < len(shown):
                track_id, start, _end = shown[k]
                label.setText(f"#{track_id}")
                label.setPos(
                    max(start - now - 0.5, -WINDOW_FRAMES + 0.5) + 0.3, rows[track_id]
                )
            label.setVisible(k < len(shown))
        self.set_note(
            "窗口内没有航迹"
            if not shown
            else f"仅显示 {MAX_ROWS} 条" if len(runs) > MAX_ROWS else ""
        )

    def set_note(self, note: str) -> None:
        set_title(
            self.plot,
            f"航迹生灭 · 近 {WINDOW_FRAMES} 帧" + (f"({note})" if note else ""),
        )

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.key.adjustSize()
        self.key.move(self.width() - self.key.width() - 8, 6)
        self.key.raise_()


def _ordered_ids(runs) -> list[int]:
    """行序:按各航迹在窗口里的首次出现(runs 已按起帧排好)。"""
    seen: list[int] = []
    for track_id, _a, _b in runs:
        if track_id not in seen:
            seen.append(track_id)
    return seen


def _lifetimes(window: dict[int, frozenset], lo: int, now: int):
    """逐帧 id 集合 → (连续在场的区间, 新建帧, 终止帧)。

    区间 = (id, 起帧, 终帧),按起帧排序;新建帧 = 该 id 本帧在、前一帧(有记录)不在;
    终止帧 = 该 id 本帧不在、前一帧在(记在「不在」的那一帧)。前一帧没有记录(窗口左缘、跳过的帧)
    不下结论。"""
    runs: list[tuple[int, int, int]] = []
    births: list[int] = []
    deaths: list[int] = []
    open_runs: dict[int, int] = {}
    previous: frozenset | None = None
    for i in range(lo, now + 1):
        ids = window.get(i)
        if ids is None:
            for track_id, start in open_runs.items():  # 记录断了:区间到此为止,不下结论
                runs.append((track_id, start, i - 1))
            open_runs, previous = {}, None
            continue
        for track_id in ids:
            if track_id not in open_runs:
                open_runs[track_id] = i
                if previous is not None:
                    births.append(i)
        for track_id in [t for t in open_runs if t not in ids]:
            runs.append((track_id, open_runs.pop(track_id), i - 1))
            deaths.append(i)
        previous = ids
    for track_id, start in open_runs.items():
        runs.append((track_id, start, now))
    runs.sort(key=lambda r: (r[1], r[0]))
    return runs, sorted(set(births)), sorted(set(deaths))
