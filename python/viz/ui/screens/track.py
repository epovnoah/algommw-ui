"""跟踪屏:点关联到哪条航迹、航迹的生灭与门限门是否对?

版式是「航迹栏 + 地图 + 生灭带」(设计稿 track):
- 左侧 260 px 常驻航迹栏:每条航迹一行 —— id(与地图方块同色)、状态徽标(新建 / 确认)、龄期、
  NIS 小条,按 NIS 从大到小(关联最差的在最上面),点一行即选中那条航迹;它就是这一屏的列表,
  所以上下文条里没有「列表」按钮、也没有滑出的列表抽屉;
- 中间是世界系俯视地图(检测点按所属航迹着色,航迹为方块 + id,近 30 帧轨迹线);
- 底部 150 px 是「航迹生灭」时间带(近 60 帧,每条航迹一个水平条,绿线新建、红线终止)。
各航迹 NIS 柱与量测空间(方位 × 距离)里的门限矩形在检查器溯源页的跟踪一级。
"""

from PySide6.QtWidgets import QFrame, QGridLayout, QVBoxLayout, QWidget

from .. import theme
from ..plots.lifetime import LifetimeBand
from ..plots.mini import GateMap, NisBars
from ..state import ViewState
from ..views.tracks import TrackLanes, TrackView
from ..widgets.conclusion import Conclusion
from .base import Screen
from .common import summary

#: 航迹栏宽度、生灭带高度(设计稿)
LANES_WIDTH = 260
BAND_HEIGHT = 150


def _events(frame) -> str:
    s = summary(frame)
    if s is None or frame.products is None:
        return ""
    parts = [f"新建 +{len(s.new_track_ids)}"] if s.new_track_ids else []
    if s.ended_track_ids:
        parts.append(f"终止 −{len(s.ended_track_ids)}")
    return " / ".join(parts)


def clauses(frame):
    n = len(frame.tracks)
    out = [(f"{n} 条航迹", 0)]
    s = summary(frame)
    if s is None or frame.products is None:
        return out
    flagged = s.flagged.get("track", 0)
    if flagged:
        out.insert(0, (f"{flagged} 个检测的航迹关联需要看", 1))
    events = _events(frame)
    if events:
        out.append((events, 1 if s.ended_track_ids else 0))
    if s.n_unassociated:
        out.append((f"{s.n_unassociated} 个点未关联", 0))
    return out


def _panel(child: QWidget) -> QFrame:
    """8 圆角、PLOT 底、LINE_SOFT 1px 描边的面板,里面放一个视图。"""
    panel = QFrame()
    panel.setObjectName("Panel")
    panel.setStyleSheet(
        f"#Panel {{ background: {theme.PLOT}; border: 1px solid {theme.LINE_SOFT}; "
        f"border-radius: {theme.RADIUS_CARD}px; }}"
    )
    column = QVBoxLayout(panel)
    column.setContentsMargins(1, 1, 1, 1)
    column.addWidget(child)
    return panel


class TrackBoard(QWidget):
    """跟踪屏的主视图:航迹栏 | 地图 / 底部生灭带。三块各是一个面板,间距 GAP。
    这是个复合部件,没有分段控件、图例与读数(`controls` 等仍声明为空,屏骨架按属性取)。"""

    def __init__(self, lanes: TrackLanes, view: TrackView, band: LifetimeBand) -> None:
        super().__init__()
        self.controls, self.legends, self.readout = [], [], None
        self.lanes, self.view, self.band = lanes, view, band
        self.lanes_panel, self.map_panel, self.band_panel = (
            _panel(lanes),
            _panel(view),
            _panel(band),
        )
        self.lanes_panel.setFixedWidth(LANES_WIDTH)
        self.band_panel.setFixedHeight(BAND_HEIGHT)
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(theme.GAP)
        grid.addWidget(self.lanes_panel, 0, 0)
        grid.addWidget(self.map_panel, 0, 1)
        grid.addWidget(self.band_panel, 1, 0, 1, 2)
        grid.setColumnStretch(1, 1)
        grid.setRowStretch(0, 1)

    def dock(self, lanes: TrackLanes) -> None:
        """屏骨架把「列表」收进了滑出抽屉;这一屏的列表是常驻航迹栏,取回来放回面板里。"""
        self.lanes_panel.layout().addWidget(lanes)
        lanes.show()


class TrackScreen(Screen):
    key = "track"
    title = "跟踪"
    #: 航迹栏常驻在主区左侧,没有列表抽屉(探针按此跳过抽屉断言)
    list_inline = True

    def __init__(self, state: ViewState) -> None:
        self.state = state
        self.view = TrackView(state)
        self.lanes = TrackLanes(state)
        self.band = LifetimeBand(state)
        self.board = TrackBoard(self.lanes, self.view, self.band)
        super().__init__(
            state,
            self.board,
            self.lanes,
            Conclusion(state, clauses),
            trace_plots=[NisBars(), GateMap()],
        )
        # 骨架把 lanes 当列表收进了抽屉:取回常驻面板,隐藏抽屉与「列表」按钮;卡片只做透明容器
        self.board.dock(self.lanes)
        self.list_button.hide()
        self.drawer.hide()
        self.card.setStyleSheet("#Card { background: transparent; border: 0; }")

    def set_list_open(self, open_: bool) -> None:
        """没有列表抽屉:航迹栏常驻。"""
