"""视图基类:订阅 ViewState,换源 / 换帧 / 换选中时重画(照 ccm `gui/views/base.py`)。"""

import pyqtgraph as pg
from PySide6.QtWidgets import QGraphicsItem, QWidget

from .. import theme
from ..state import Selection, ViewState


class ResultView(QWidget):
    """结果视图基类。回放流畅的三条规则都在这里统一处理,子类只管 reset / render_frame /
    render_selection:

    - 只在可见时渲染:不可见的视图(别的屏、收起的列表抽屉与滑出栏)只记「脏」,重新显示时补画;
    - 坐标范围在换源后的第一帧自动缩放,之后锁定(KEEP_AUTORANGE=True 的视图除外)。
      每帧自动缩放会让坐标轴每帧重新生成并重绘全部刻度文字,是回放卡顿的主因;
      用户仍可点图左下角的「A」或右键菜单恢复自动缩放;
    - 坐标轴用 DeviceCoordinateCache 缓存成位图,范围不变时直接贴图
      (需配合 theme.apply 调大位图缓存)。
    """

    title = "未命名视图"
    KEEP_AUTORANGE = False  # 数据范围随帧增长或随选中变化的视图设为 True

    def __init__(self, state: ViewState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.state = state
        #: 主视图可声明三样东西,由屏摆到标题行 / 底部一行(见 screens/base.py);默认都没有:
        #: controls = 标题行右侧的分段控件;legends = 底部的图例标签;readout = 悬停读数标签
        self.controls: list[QWidget] = []
        self.legends: list[QWidget] = []
        self.readout = None
        self._dirty = False
        #: 上次画的内容键(帧号 + 选中):切回已画过且没变化的屏时不再重画
        self._shown = None
        self._ranges_locked = False
        #: 由 _lock_ranges 锁上的视图框;子类手动设定范围的不在其中
        self._locked_boxes: list[pg.ViewBox] = []
        self._axes_cached = False
        state.source_changed.connect(self._on_source)
        state.frame_changed.connect(self._on_frame)
        state.selection_changed.connect(self._on_selection)

    # ---- 子类实现 ----
    def reset(self, source) -> None:
        """换会话(source 是 RadarInfo):清空画面与累积的历史,按新雷达参数设坐标轴。"""

    def render_frame(self, frame) -> None:
        """画与选中无关的部分(整帧的点、图像、曲线)。"""

    def render_selection(self, frame, selection: Selection) -> None:
        """画与选中有关的部分(选中圈、光标、切片)。"""

    def accumulate(self, frame) -> None:
        """每帧都会调用(不论是否可见),需要跨帧累积数据的视图在这里记录。"""

    # ---- 分发 ----
    def _on_source(self, source) -> None:
        self._dirty = False
        self._shown = None
        self._ranges_locked = False
        for view_box in self._locked_boxes:  # 只解锁自己锁上的,子类手设的固定范围不动
            view_box.enableAutoRange()
        self._locked_boxes = []
        self.reset(source)

    def _on_frame(self, frame) -> None:
        self.accumulate(frame)
        if self.isVisible():
            self._render(frame, None)
            # 画的内容键按「本帧 + 当前选中/筛选」记,否则切回来时会被判成「内容变了」而重画一遍
            self._shown = self._content_key(frame, self.state.selection)
        else:
            self._dirty = True

    def _on_selection(self, selection: Selection) -> None:
        if self.isVisible() and self.state.frame is not None:
            self.render_selection(self.state.frame, selection)
            self._shown = self._content_key(self.state.frame, selection)
            self._lock_ranges()
        else:
            self._dirty = True

    def _content_key(self, frame, selection: Selection | None) -> tuple:
        """「画的是哪一帧的什么状态」:帧号 + 选中单元 + 筛选级 + 跟的航迹。内容没变就不必重画。"""
        if frame is None:
            return (None, None, None, None)
        cell = (
            None if selection is None else (selection.range_bin, selection.doppler_bin)
        )
        return (frame.index, cell, self.state.stage_filter, self.state.track)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        frame = self.state.frame
        if frame is None:
            return
        # 切回一屏时只在「没画过 / 画过之后帧或选中变了」时补画:重画会重排图层、重设视图范围,
        # 无谓的重画正是换面时画面抖动的来源之一
        if self._dirty or self._shown != self._content_key(frame, self.state.selection):
            self._dirty = False
            self._render(frame, self.state.selection)

    def _render(self, frame, selection: Selection | None) -> None:
        self._cache_axes()
        self.render_frame(frame)
        if selection is not None:
            self.render_selection(frame, selection)
        self._shown = self._content_key(frame, selection)
        self._lock_ranges()

    # ---- 性能 ----
    def _plot_items(self) -> list[pg.PlotItem]:
        """本视图里**显示着**的图(分段控件切走的那张还没有数据,按它现在空的范围锁会锁死在默认值)。"""
        items = []
        for view in self.findChildren(pg.GraphicsView):
            if not view.isVisibleTo(self):
                continue
            items += [it for it in view.scene().items() if isinstance(it, pg.PlotItem)]
        return items

    def relock_ranges(self) -> None:
        """切到另一张图(分段控件)后:让下一次渲染按新图的数据重新锁一次范围。"""
        self._ranges_locked = False

    def _view_boxes(self) -> list[pg.ViewBox]:
        return [item.vb for item in self._plot_items()]

    def _cache_axes(self) -> None:
        if self._axes_cached:
            return
        for view in self.findChildren(pg.GraphicsView):
            for item in view.scene().items():
                if isinstance(item, pg.AxisItem):
                    item.setCacheMode(QGraphicsItem.CacheMode.DeviceCoordinateCache)
                elif isinstance(item, pg.ViewBox) and any(
                    isinstance(child, pg.ImageItem) for child in item.addedItems
                ):
                    # 热图贴边:默认留白按视图尺寸给到每边约 5%,主图上是一圈约 20 px 的空带
                    item.setDefaultPadding(0.0)
        self._axes_cached = True

    def _lock_ranges(self) -> None:
        """首次有数据后按数据定一次坐标范围并锁定;还没有数据的图下次再试。"""
        if self._ranges_locked or self.KEEP_AUTORANGE:
            return
        locked = True
        for view_box in self._view_boxes():
            auto_x, auto_y = view_box.state["autoRange"][:2]
            if not auto_x and not auto_y:
                continue  # 子类已手动设定范围,或已锁定
            bounds = view_box.childrenBounds()
            # 仍在自动缩放的方向都要有数据范围才锁,否则会锁在尚无数据时的默认 0~1
            if (auto_x and bounds[0] is None) or (auto_y and bounds[1] is None):
                locked = False
                continue
            view_box.updateAutoRange()
            view_box.disableAutoRange()
            self._locked_boxes.append(view_box)
        self._ranges_locked = locked


def set_title(plot: pg.PlotItem, text: str) -> None:
    """标题没变就不重设:每帧重设会触发文字重排与布局更新。

    `plot._algommw_quiet` 为真时(溯源页里 64 px 的小图)不画标题,只记下文字。"""
    if getattr(plot, "_algommw_title", None) != text:
        if not getattr(plot, "_algommw_quiet", False):
            theme.set_plot_title(plot, text)
        plot._algommw_title = text
