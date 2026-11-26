"""主窗口外壳:顶栏(含处理链)· 屏 | 检查器 · 传输条(布局见 docs/uidesign.html,决定见 ADR 0011)。

从上到下三层(默认窗口 1440×920,外壳合计约 122 px):
- 顶栏(40 px,一行):品牌、板与采集胶囊(点开「打开…」)、处理链(ADC | 距离 | 多普勒 | CFAR |
  解耦(仅步进频)| 测角 | 跟踪 | 场景,每级形状 + 级名 + 一个实时数,既是导航也是全链摘要,
  互斥,Ctrl+1…9)、每帧「处理耗时 / 帧周期」、参数、检查器、搜索 ⌘K;
- 主区:各屏叠放(QStackedWidget),同一时刻只显示一个,只有可见的部件渲染;每屏自带 30 px 上下文条
  (屏名 · 结论 · 分段控件 · 列表开关)。右侧是检查器(溯源 / 参数 / 判据 / 雷达 / 耗时,300 px),
  **默认展开**(场景屏自动收起、切走恢复),开合状态记进 data/ui_state.json;收起 / 再滑出时主区随之变宽 / 变窄;
- 传输条(34 px):播放 / 单步、帧计数、瀑布时间轴(拖它就是在拖数据)、倍速;
  快捷键:空格 播放暂停、← → 单步、[ ] 异常帧、Ctrl+K 命令面板、Ctrl+反斜杠 开合检查器、Esc 关检查器;
- ViewState 负责联动:任一部件里点选检测 / 航迹,各屏同步;溯源页、耗时页请求跳屏、请求开
  检查器,都由这里处理;
- 一个进程只有一个窗口:换板 / 换采集不重建窗口,只换处理器(`set_session`);改处理参数从当前帧
  起建新处理器并重算当前帧(航迹在此帧清零,**立即生效**);改判据只重新分级(不重算);
  判据按板记进 data/criteria.json;
- 状态栏已取消:瞬时消息(落盘、改参、失败)走顶栏下沿居中的提示条,常驻提示(载入失败、
  换板说明)在它上面,二者都不在布局里,显隐不会让画布重排。`statusBar()` 返回提示条,
  调用方(session.py)不用改;
- 关闭时把窗口几何、当前屏、检查器开合记进 data/ui_state.json(经 boards 的读写函数;旧版记过的
  面板开合与分隔条不再读;无记录时检查器按展开);环境变量 ALGOMMW_UI_NO_SETTINGS=1 时不读不写(探针用;data/criteria.json 同此)。
会话的建 / 换 / 收(预检、落盘线程、换板纪律)归 `session.SessionController`,窗口只发
`reload_requested` / `save_toggled`。
"""

import os

from PySide6.QtCore import QByteArray, QEvent, QPoint, Qt, Signal
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QHBoxLayout,
    QMainWindow,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .. import boards, diagnostics
from . import theme
from .drawer import Drawer
from .params import parameter_error
from .player import FramePlayer
from .popup import ConfigPopup
from .screens import SCREEN_TABS, discover_screens
from .state import ViewState
from .widgets import icons
from .widgets.chain import ChainRow
from .widgets.palette import Command, CommandPalette, filter_commands
from .widgets.slide import SlideOut
from .widgets.toast import Toast
from .widgets.topbar import TopBar
from .widgets.trace import TracePage
from .widgets.transport import TransportBar, play_icons

TITLE = "algommw 处理链工作台"
SPEEDS = {
    "最快(逐帧)": None,
    "0.25×": 0.25,
    "0.5×": 0.5,
    "1× 实时": 1.0,
    "2×": 2.0,
    "4×": 4.0,
}
DEFAULT_SPEED = "1× 实时"
DEFAULT_SCREEN = "results"
#: 处理耗时页里点某一级 → 跳到的屏(未实施的屏退回「结果」)
STAGE_TARGETS = {
    "read": "adc",
    "range": "rangefft",
    "doppler": "doppler",
    "cfar": "cfar",
    "doa": "doa",
    "pair": "decouple",
    "cluster": "track",
    "track": "track",
    "world": "results",
    "extract": "results",
}
NO_SETTINGS_ENV = "ALGOMMW_UI_NO_SETTINGS"
_NOTICE_TONES = {"ok": 0, "warn": 1, "bad": 2}
#: 提示条离顶栏下沿的距离
_TOAST_GAP = 4
#: 屏切换的快捷键只编前 9 级
MAX_SHORTCUTS = 9


def settings_enabled() -> bool:
    return os.environ.get(NO_SETTINGS_ENV) != "1"


def make_application(argv: list[str]) -> QApplication:
    """建 QApplication 并套主题。放在 ui 层:main.py 不直接 import Qt(testarch 检查)。"""
    app = QApplication.instance() or QApplication(list(argv))
    app.setApplicationName(TITLE)
    theme.apply(app)
    return app


def _b64(data: QByteArray) -> str:
    return bytes(data.toBase64()).decode()


def _unb64(text) -> QByteArray | None:
    return QByteArray.fromBase64(text.encode()) if isinstance(text, str) else None


class MainWindow(QMainWindow):
    reload_requested = Signal(str, str)  # (板目录, 采集目录)
    save_toggled = Signal(bool)

    def __init__(self, *, catalog=(), capture_for=None, recents=None) -> None:
        super().__init__()
        self.setWindowTitle(TITLE)
        self.resize(1440, 920)
        #: 「打开…」弹层用:板目录(boards.scan_boards)、板 → 采集目录、最近用过的目录
        self.catalog = tuple(catalog)
        self.capture_for = capture_for or (lambda _name: "")
        self.recents = recents or (lambda: [])
        self.config = None  # 当前处理器用的 ChainCfg
        self._popup: ConfigPopup | None = None
        self._palette: CommandPalette | None = None
        self._controls_state: tuple | None = None
        self._fitted = False  # 首次显示时已把窗口夹进屏幕
        #: 检查器是被场景屏自动收起的(切走时要恢复);只在窗口里记,不写进 ui_state.json
        self._scene_closed_inspector = False
        self._auto_inspector = False  # 正在由 _sync_inspector 开合(区分用户手动)

        self.state = ViewState(self)
        self.player = FramePlayer(self)
        self.player.frame_shown.connect(self._on_frame_shown)
        self.player.frames_processed.connect(self.state.frames_processed)
        self.player.failed.connect(self._on_failed)
        self.player.state_changed.connect(self._sync_controls)
        self.state.screen_requested.connect(self.show_screen)
        self.state.panel_requested.connect(self.show_panel)

        # ---- 屏 ----
        self.stack = QStackedWidget()
        self.screens: dict[str, QWidget] = {}
        # 屏按 SCREEN_TABS 的 key 自动发现:screens/<key>.py 里有 key 相同的 Screen 子类就启用该标签
        classes = discover_screens()
        available = set(classes)
        for key, _label in SCREEN_TABS:
            if key in classes:
                screen = classes[key](self.state)
                self.screens[screen.key] = screen
                self.stack.addWidget(screen)

        # ---- 检查器:溯源页由各屏声明的次要视图(trace_plots)拼成 ----
        self.trace = TracePage(
            self.state,
            SCREEN_TABS,
            {key: screen.trace_plots for key, screen in self.screens.items()},
        )
        self.trace.screen_chosen.connect(self.show_screen)
        self.drawer = Drawer(self.state, self.trace)
        self.drawer.close_requested.connect(self.close_panel)
        self.slide = SlideOut(self.drawer, open_=self._saved_inspector_open())
        self.drawer.params.config_changed.connect(self._on_config_changed)
        self.drawer.params.motion_changed.connect(self._on_motion_changed)
        self.drawer.criteria.criteria_changed.connect(self._on_criteria_changed)
        self.drawer.timings.stage_activated.connect(self._on_stage_activated)

        # ---- 动作(快捷键挂在窗口上;按钮只是它们的另一种摆法) ----
        self._build_actions()
        self.chain = ChainRow(self.state, SCREEN_TABS, available)
        self.chain.screen_chosen.connect(self.show_screen)
        self.topbar = TopBar(
            self.open_action,
            self.palette_action.shortcut().toString(
                QKeySequence.SequenceFormat.NativeText
            ),
            self.chain,
        )
        self.topbar.palette_requested.connect(self.open_palette)
        self.topbar.params_requested.connect(lambda: self.toggle_panel(Drawer.PARAMS))
        self.topbar.inspector_requested.connect(lambda: self.toggle_panel())
        self.slide.opened_changed.connect(self.topbar.set_inspector_open)
        self.slide.opened_changed.connect(self._on_inspector_toggled)
        self.topbar.set_inspector_open(self.slide.is_open())
        self.transport = TransportBar(
            {
                "step_back": self.step_back_action,
                "play": self.play_action,
                "step": self.step_action,
            },
            list(SPEEDS),
            DEFAULT_SPEED,
        )
        self.transport.speed_chosen.connect(self.set_speed)
        # 回放控件的老名字:探针与会话控制器按这些名字取
        self.slider = self.transport.timeline
        self.frame_label = self.transport.frame_label
        self.slider.valueChanged.connect(self._on_slider)
        self.player.frames_processed.connect(self.slider.add_frames)
        # 落盘开关低频,不占顶栏 / 传输条:放在检查器「参数」页的顶部(命令面板里也有)
        self.save_box = QCheckBox("落盘")
        self.save_box.setToolTip(
            "开 = 从这一刻起累积新处理出的帧,读完 / 换板 / 关窗时写 "
            "data/results/<采集名>.npz(同名覆盖);关 = 不写文件"
        )
        self.save_box.toggled.connect(self.save_toggled.emit)
        self.drawer.add_param_tool(self.save_box)

        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self.topbar)
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        body.addWidget(self.stack, 1)
        body.addWidget(self.slide)
        outer.addLayout(body, 1)
        outer.addWidget(self.transport)
        self.setCentralWidget(central)
        self._central = central
        central.installEventFilter(self)
        # 提示条在顶栏正下方居中:常驻提示在上,瞬时提示叠在它下面
        self.notice = Toast(central)
        self.toast = Toast(central, follow=self.notice)

        self.set_speed(DEFAULT_SPEED)
        self._restore_settings()
        self._sync_controls()
        self.toast.showMessage("载入采集后按「播放」开始")

    def statusBar(self):  # type: ignore[override]
        """状态栏已取消:返回提示条(同名的 showMessage / currentMessage 方法)。"""
        return self.toast

    @property
    def current_screen(self) -> str:
        return self.stack.currentWidget().key

    def current_screen_widget(self) -> QWidget:
        return self.stack.currentWidget()

    @staticmethod
    def _saved_inspector_open() -> bool:
        """检查器上次的开合(无记录 / 不读设置时按展开)。"""
        if not settings_enabled():
            return True
        return bool(boards.load_ui_state().get(boards.UI_KEY_INSPECTOR, True))

    # ---- 构建 ----
    def _action(self, text, tip, shortcut=None, icon=None, slot=None) -> QAction:
        action = QAction(text, self)
        action.setToolTip(f"{tip}  {shortcut}" if shortcut else tip)
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
        if icon is not None:
            action.setIcon(icon)
        if slot is not None:
            action.triggered.connect(slot)
        self.addAction(action)
        return action

    def _build_actions(self) -> None:
        dim = theme.DIM
        self._play_icon, self._pause_icon = play_icons()
        self.open_action = self._action(
            "打开…",
            "换板 / 换采集目录(新会话:缓存与航迹从零开始)",
            "Ctrl+O",
            slot=self.open_popup,
        )
        self.play_action = self._action(
            "播放", "播放 / 暂停", "Space", self._play_icon, self._toggle_play
        )
        self.step_back_action = self._action(
            "上一帧",
            "上一帧",
            "Left",
            icons.step_back(dim),
            lambda: self.player.step(-1),
        )
        self.step_action = self._action(
            "下一帧",
            "下一帧",
            "Right",
            icons.step_forward(dim),
            lambda: self.player.step(1),
        )
        # 异常帧跳转不占底栏:快捷键与命令面板项
        self.prev_anomaly_action = self._action(
            "上一个异常帧",
            "上一个异常帧(只含已处理的帧)",
            "[",
            slot=lambda: self._jump_anomaly(-1),
        )
        self.next_anomaly_action = self._action(
            "下一个异常帧",
            "下一个异常帧(只含已处理的帧)",
            "]",
            slot=lambda: self._jump_anomaly(1),
        )
        self.palette_action = self._action(
            "命令面板", "命令面板", "Ctrl+K", slot=self.open_palette
        )
        self.panel_action = self._action(
            "检查器",
            "显示 / 隐藏右侧检查器",
            "Ctrl+\\",
            slot=lambda: self.toggle_panel(),
        )
        self.close_panel_action = self._action(
            "关闭检查器", "关闭检查器", "Esc", slot=self.close_panel
        )
        self.screen_actions: dict[str, QAction] = {}
        for key, label in SCREEN_TABS:
            action = QAction(label, self)
            action.triggered.connect(lambda _=False, k=key: self.show_screen(k))
            if key not in self.screens:
                action.setEnabled(False)
                action.setToolTip(f"{label}:该屏不可用(模块缺失)")
            self.addAction(action)
            self.screen_actions[key] = action
        self._assign_shortcuts()

    def _visible_screens(self) -> list[str]:
        """处理链上现在显示的级(没有步进的波形没有「解耦」)。"""
        return [k for k in self.chain.visible_keys() if k in self.screens]

    def _assign_shortcuts(self) -> None:
        """Ctrl+1…9 按**显示着的级**依次编号(隐藏的级不占号)。"""
        order = (
            self._visible_screens()
            if hasattr(self, "chain")
            else [k for k, _ in SCREEN_TABS if k != "decouple"]
        )
        for action in self.screen_actions.values():
            action.setShortcut(QKeySequence())
        for i, key in enumerate(order[:MAX_SHORTCUTS], start=1):
            self.screen_actions[key].setShortcut(QKeySequence(f"Ctrl+{i}"))

    def show_screen(self, key: str) -> None:
        """切屏:整次切换（藏旧页 + 显示新页 + 新页 showEvent 里的补画）压在**同一次重绘**里。

        不这么做的实测后果（sr61,1500×950,离屏）:切换要 14~20 ms,其中至少两次重绘 —— Qt 先画
        一次刚显示的页(此时补画排的 update 还没合并),再画一次补画后的内容,画面上就是切过去时
        抖一下 / 闪一下。`setUpdatesEnabled(False)` 把这两次合并成一次,同一实测降到 10~14 ms,
        且只有一帧可见变化。
        """
        screen = self.screens.get(key)
        if screen is None or (
            hasattr(self, "chain") and key not in self._visible_screens()
        ):
            return  # 未实施的屏,或当前波形没有这一级
        if screen is self.stack.currentWidget():
            self.chain.set_current(key)
            self.trace.set_current(key)
            return
        self.setUpdatesEnabled(False)
        try:
            self.stack.setCurrentWidget(screen)  # showEvent 里的补画也在这段里完成
        finally:
            self.setUpdatesEnabled(True)
        self.chain.set_current(key)
        self.trace.set_current(key)
        self._sync_inspector(key)

    def _sync_inspector(self, key: str) -> None:
        """场景屏让出 300 px 给点云:切进去时收起检查器,切走时恢复原来的开合。
        在场景屏里用户手动开合过(含点选中卡的「看它的来路」),就以他最后的选择为准、不再恢复。"""
        if key == "results":
            if not self._scene_closed_inspector and self.slide.is_open():
                self._scene_closed_inspector = True
                self._auto_inspector = True
                self.slide.set_open(False, animate=False)
                self._auto_inspector = False
        elif self._scene_closed_inspector:
            self._scene_closed_inspector = False
            self._auto_inspector = True
            self.slide.set_open(True, animate=False)
            self._auto_inspector = False

    def _on_inspector_toggled(self, _open: bool) -> None:
        if not self._auto_inspector:
            self._scene_closed_inspector = False  # 用户手动开合:不再自动恢复

    # ---- 检查器(右栏) ----
    def show_panel(self, page: int) -> None:
        """检查器打开并切到第 page 页(Drawer.TRACE …)。"""
        self.drawer.setCurrentIndex(page)
        self.slide.set_open(True)

    def close_panel(self) -> None:
        self.slide.set_open(False)

    def toggle_panel(self, page: int | None = None) -> None:
        """page 给定:开着且正在那一页就收起,否则打开到那一页;不给:开 / 关。"""
        if page is None:
            self.slide.set_open(not self.slide.is_open())
        elif self.slide.is_open() and self.drawer.currentIndex() == page:
            self.close_panel()
        else:
            self.show_panel(page)

    def _on_stage_activated(self, stage: str) -> None:
        screen = STAGE_TARGETS.get(stage, DEFAULT_SCREEN)
        self.show_screen(screen if screen in self.screens else DEFAULT_SCREEN)

    # ---- 布局记忆 ----
    def _restore_settings(self) -> None:
        self.show_screen(DEFAULT_SCREEN)
        if settings_enabled():
            state = boards.load_ui_state()
            try:
                geometry = _unb64(state.get(boards.UI_KEY_GEOMETRY))
                if geometry is not None:
                    self.restoreGeometry(geometry)
                self.show_screen(str(state.get(boards.UI_KEY_SCREEN, DEFAULT_SCREEN)))
            except Exception:  # noqa: BLE001 -- 旧版记忆与当前界面不匹配时用默认值
                self.show_screen(DEFAULT_SCREEN)

    def _save_settings(self) -> None:
        if not settings_enabled():
            return
        state = boards.load_ui_state()
        state[boards.UI_KEY_GEOMETRY] = _b64(self.saveGeometry())
        state[boards.UI_KEY_SCREEN] = self.current_screen
        state[boards.UI_KEY_INSPECTOR] = (
            self.slide.is_open() or self._scene_closed_inspector
        )
        for legacy in ("panel", "splitters"):  # 旧版记过的面板开合与分隔条,新版不用
            state.pop(legacy, None)
        boards.save_ui_state(state)

    # ---- 会话与参数 ----
    def set_session(self, processor, config) -> None:
        """换会话(板 × 采集):换处理器并让各视图按新雷达参数复位,从帧 0 开始。"""
        self.config = config
        self.state.set_config(config)
        radar = processor.radar
        if settings_enabled():
            # 处理器还没交给播放器(未开始处理),此时换它的判据是安全的
            processor.criteria = diagnostics.load_criteria(radar.profile)
        self.drawer.set_config(config)
        self.drawer.set_criteria(
            processor.criteria, diagnostics.criteria_for(radar.profile)
        )
        self.state.set_source(radar)  # 处理链据此显示 / 隐藏「解耦」一级
        self.trace.set_stepped(radar.stepped)
        self._assign_shortcuts()
        if self.current_screen not in self._visible_screens():
            self.show_screen(DEFAULT_SCREEN)  # 上一块板停在「解耦」,新的波形没有这一级
        n_frames = len(processor)
        self.slider.reset(n_frames)
        mode = "步进频" if radar.stepped else radar.modulation.upper()
        self.topbar.set_source(radar.profile, mode, radar.capture, n_frames)
        self.topbar.set_budget(None, None)
        self.setWindowTitle(f"{TITLE} — {radar.profile} · {radar.capture}")
        self.player.set_processor(processor)
        self._sync_controls()

    def _on_config_changed(self, config) -> None:
        """改处理参数(立即生效):从当前帧起建新处理器(= eChainInit,航迹清零)并重算当前帧。"""
        old = self.player.processor
        if old is None:
            self.drawer.params.set_status("请先载入采集,再修改处理参数", error=True)
            return
        start = self._restart(config, old.criteria, "改参")
        if start is not None:
            self.toast.showMessage(
                f"参数已改 · 改参数会清零航迹 · 正在用新参数重算帧 {start + 1}"
            )

    def _on_motion_changed(self, enabled: bool) -> None:
        """走动补偿研究开关:不改 ChainCfg,只让新处理器带/不带补偿重建并重算当前帧。"""
        old = self.player.processor
        if old is None:
            return
        start = self._restart(self.config, old.criteria, "走动补偿", motion=enabled)
        if start is not None:
            self.toast.showMessage(
                f"走动补偿已{'打开' if enabled else '关闭'} · 航迹清零 · 正在重算帧 {start + 1}"
            )

    def _on_criteria_changed(self, criteria) -> None:
        """改判据(同 ccm):只重新分级,不重算处理链;按板记进 data/criteria.json
        (ALGOMMW_UI_NO_SETTINGS=1 时不存)。"""
        old = self.player.processor
        if old is None:
            return
        if settings_enabled():
            try:
                diagnostics.save_criteria(criteria, old.radar.profile)
            except OSError as exc:
                self.show_notice(f"判据未能保存:{exc}", "warn")
        # 只重新分级(同 ccm 的 rejudge):处理器之后算的帧用新判据,缓存帧由播放器补分级;
        # 不重建处理链,航迹不受影响。属性赋值是原子的,后台线程下一帧即读到新判据。
        old.criteria = criteria
        self.player.set_criteria(criteria, diagnostics.rejudge)
        self.toast.showMessage("判据已改 · 只重新分级,不重算处理链(航迹不受影响)")

    def _restart(self, config, criteria, what, motion=None):
        """独立 UI 只编辑预览参数,没有算法处理器可重启。"""
        self.toast.showMessage("无数据界面预览 · 不执行信号处理")
        return None

    def open_popup(self) -> None:
        """「打开…」:换板 / 换采集目录的弹层,位置在来源芯片正下方,贴屏幕右缘时向内收。"""
        if self._popup is None:
            self._popup = ConfigPopup(self)
            self._popup.loadRequested.connect(self.reload_requested.emit)
        popup = self._popup
        radar = self.state.source
        popup.set_catalog(
            self.catalog,
            self.capture_for,
            self.recents(),
            current_profile=radar.profile if radar is not None else "",
        )
        popup.adjustSize()
        anchor = self.topbar.source_button
        pos = anchor.mapToGlobal(QPoint(0, anchor.height() + 4))
        screen = QApplication.primaryScreen()
        if screen is not None:
            right = screen.availableGeometry().right()
            pos.setX(max(8, min(pos.x(), right - popup.width() - 8)))
        popup.move(pos)
        popup.show()

    def set_saving(self, on: bool, locked: bool = False) -> None:
        """同步「落盘」开关(会话控制器调;locked = 命令行 --no-save)。只改控件,不回发信号。
        开关在检查器「参数」页顶部。"""
        self.save_box.blockSignals(True)
        self.save_box.setChecked(bool(on) and not locked)
        self.save_box.blockSignals(False)
        self.save_box.setEnabled(not locked)
        if locked:
            self.save_box.setToolTip("命令行 --no-save:本次运行不落盘。")

    def show_notice(self, text: str, tone: str = "warn") -> None:
        """右下角的常驻提示(载入失败、落盘失败、换板说明);空串清除。"""
        self.notice.set_color(theme.LEVEL_COLORS[_NOTICE_TONES.get(tone, 1)])
        self.notice.showMessage(text, 0)

    # ---- 命令面板 ----
    def open_palette(self) -> None:
        if self._palette is None:
            self._palette = CommandPalette(self)
        self._palette.open_palette(self._commands)

    def _commands(self, query: str) -> list[Command]:
        commands = []
        labels = dict(SCREEN_TABS)
        for i, key in enumerate(self._visible_screens(), start=1):
            commands.append(
                Command(
                    f"切换到「{labels[key]}」屏",
                    lambda k=key: self.show_screen(k),
                    f"Ctrl+{i}" if i <= MAX_SHORTCUTS else "",
                    f"screen {key}",
                )
            )
        commands += [
            Command("播放 / 暂停", self._toggle_play, "Space", "play pause"),
            Command("上一帧", self.step_back_action.trigger, "←", "step prev"),
            Command("下一帧", self.step_action.trigger, "→", "step next"),
            Command(
                "上一个异常帧", lambda: self._jump_anomaly(-1), "[", "anomaly 异常"
            ),
            Command("下一个异常帧", lambda: self._jump_anomaly(1), "]", "anomaly 异常"),
            Command("打开板卡与采集…", self.open_popup, "Ctrl+O", "open load 换板"),
            Command("落盘 开 / 关", self.save_box.toggle, "", "save 保存"),
            Command(
                "显示 / 隐藏检查器",
                self.toggle_panel,
                "Ctrl+\\",
                "panel 面板 inspector",
            ),
        ]
        commands += [
            Command(
                f"检查器 · {name}", lambda i=i: self.show_panel(i), "", "panel 面板"
            )
            for i, name in enumerate(Drawer.NAMES)
        ]
        commands += [
            Command(f"播放速度 · {text}", lambda t=text: self.set_speed(t), "", "speed")
            for text in SPEEDS
        ]
        commands = filter_commands(commands, query)
        digits = query.strip()
        if digits.isdigit() and 1 <= int(digits) <= self.player.n_frames:
            k = int(digits) - 1
            commands.insert(
                0, Command(f"跳到帧 {k + 1}", lambda: self._goto(k), "Enter")
            )
        return commands

    # ---- 回放 ----
    def _toggle_play(self) -> None:
        if self.player.playing:
            self.player.pause()
        else:
            self.player.play()

    def _goto(self, index: int) -> None:
        self.player.pause()
        self.player.seek(index)

    def _on_slider(self, value: int) -> None:
        self._goto(value)

    def _jump_anomaly(self, direction: int) -> None:
        target = self.slider.step_anomaly(direction)
        if target is None:
            side = "之后" if direction > 0 else "之前"
            self.toast.showMessage(
                f"当前帧{side}没有已处理的异常帧(时间轴只含已处理过的帧;"
                "播放或跳转后会长出更多刻度)"
            )
            return
        self._goto(target)

    def set_speed(self, key: str) -> None:
        self.player.set_speed(SPEEDS[key])
        self.transport.set_speed(key)

    def _on_frame_shown(self, frame) -> None:
        self.state.set_frame(frame)
        if self.drawer.params.status.text().startswith("参数已提交"):
            self.drawer.params.set_status(
                f"已生效 · 当前帧 {frame.index + 1} 已按新配置重算"
            )
        self.slider.blockSignals(True)
        self.slider.setValue(frame.index)
        self.slider.blockSignals(False)
        self.slider.refresh_head()
        period_ms = frame.radar.frame_period_s * 1e3
        self.topbar.set_budget(diagnostics.processing_ms(frame.timings_ms), period_ms)
        if frame.track_epoch > 0 and frame.index == frame.track_epoch:
            self.toast.showMessage("航迹自本帧重新起始")
        self._sync_controls()

    def _on_failed(self, index: int, message: str) -> None:
        if self.drawer.params.status.text().startswith("参数已提交"):
            self.drawer.params.set_status(
                f"重算失败:{message}。请修正参数后重试。", error=True
            )
        self.toast.showMessage(f"帧 {index + 1} 处理失败 — {message}")

    def _sync_controls(self) -> None:
        n = self.player.n_frames
        has_source = n > 0
        playing = self.player.playing
        busy = self.player.busy
        # 动作图标 / 可用性只在状态翻转时改:逐帧调用 setIcon 会触发按钮重画,
        # 实测让结果屏回放帧率掉约一成
        state = (has_source, playing, busy)
        if state != self._controls_state:
            self._controls_state = state
            self.play_action.setText("暂停" if playing else "播放")
            self.play_action.setIcon(self._pause_icon if playing else self._play_icon)
            self.play_action.setEnabled(has_source)
            for action in (self.step_back_action, self.step_action):
                action.setEnabled(has_source and not playing)
            for action in (self.prev_anomaly_action, self.next_anomaly_action):
                action.setEnabled(has_source)
            self.slider.setEnabled(has_source)
        current = self.player.current
        self.frame_label.setText(f"{current + 1 if current >= 0 else 0} / {n}")
        frame = self.state.frame
        if frame is not None:
            seconds = max(current, 0) * frame.radar.frame_period_s
            tip = f"采集帧号 #{frame.fi} · {seconds:.2f} s"
            if tip != self.frame_label.toolTip():
                self.frame_label.setToolTip(tip)

    # ---- 窗口尺寸 ----
    def eventFilter(self, obj, event) -> bool:
        if (
            obj is getattr(self, "_central", None)
            and event.type() == QEvent.Type.Resize
            and hasattr(self, "toast")
        ):
            self._place_toasts()
        return super().eventFilter(obj, event)

    def _place_toasts(self) -> None:
        """提示条贴在顶栏的下沿(顶栏高度随字体而变,所以现量)。"""
        below_bar = self.topbar.mapTo(
            self._central, QPoint(0, self.topbar.height())
        ).y()
        for toast in (self.notice, self.toast):
            toast.top = below_bar + _TOAST_GAP
        self.notice.reposition()
        self.toast.reposition()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not self._fitted:
            self._fitted = True
            self._fit_to_screen()

    def _fit_to_screen(self) -> None:
        """窗口大小与位置夹进所在屏幕的可用区域:默认 1440×920 或记住的几何在小屏 / 换了显示器后
        会越界。离屏测试平台的虚拟屏幕只有 800×600,不是真显示器,不夹。"""
        screen = self.screen() or QApplication.primaryScreen()
        if screen is None or QApplication.platformName() == "offscreen":
            return
        avail = screen.availableGeometry()
        frame = self.frameGeometry()
        width = min(self.width(), avail.width() - (frame.width() - self.width()))
        height = min(self.height(), avail.height() - (frame.height() - self.height()))
        if (width, height) != (self.width(), self.height()):
            self.resize(width, height)
            frame = self.frameGeometry()
        if not avail.contains(frame):
            self.move(
                min(max(frame.x(), avail.left()), avail.right() + 1 - frame.width()),
                min(max(frame.y(), avail.top()), avail.bottom() + 1 - frame.height()),
            )

    def closeEvent(self, event) -> None:
        self.player.shutdown()
        self._save_settings()
        super().closeEvent(event)
