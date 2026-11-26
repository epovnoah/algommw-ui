"""「打开…」弹层 —— 选板 + 选采集目录 + 载入(预检就地做,失败就地报错)。"""

from pathlib import Path

from PySide6 import QtCore, QtWidgets

from .. import boards
from . import theme

#: 预检状态 → 文字颜色
_TONES = {"ok": theme.LEVEL_COLORS[0], "error": theme.LEVEL_COLORS[2]}


def _config_qss() -> str:
    """弹层的 QSS。单独一份:`Qt.Popup` 是顶层窗口,主窗的样式表管不到它。"""
    return f"""
    QFrame#radarConfig {{
        background: {theme.SURFACE}; border: 1px solid {theme.LINE};
        border-radius: {theme.RADIUS_CARD}px;
    }}
    QLabel#cfgTitle {{ color: {theme.TEXT}; font-size: {theme.FONT_BODY}px; font-weight: 600; }}
    QLabel#cfgHint {{ color: {theme.DIM}; font-size: {theme.FONT_NOTE}px; }}
    QLabel#cfgState {{ font-family: {theme.MONO}; font-size: {theme.FONT_TICK}px; }}
    QComboBox, QLineEdit {{
        background: {theme.PLOT}; color: {theme.TEXT};
        border: 1px solid {theme.LINE}; border-radius: {theme.RADIUS_CONTROL}px; padding: 3px 8px;
        font-family: {theme.MONO}; font-size: {theme.FONT_TICK}px;
    }}
    QComboBox QAbstractItemView {{
        background: {theme.SURFACE}; color: {theme.TEXT};
        selection-background-color: {theme.ACCENT_BG};
    }}
    QPushButton#cfgLoad {{
        background: {theme.RAISED}; color: {theme.TEXT}; border: 1px solid {theme.LINE_HOVER};
        border-radius: {theme.RADIUS_CONTROL}px; padding: 3px 14px; min-height: 28px;
    }}
    QPushButton#cfgLoad:disabled {{
        background: transparent; color: {theme.DIM}; border: 1px solid {theme.LINE};
    }}
    """


class ConfigPopup(QtWidgets.QFrame):
    """工具栏「打开…」的弹层:板(profile)+ 采集目录 + 载入。

    预检就地做(`boards.preflight`:profile 能否载、采集能数出几帧),失败就地报错且
    **不关窗** —— 否则就是“点了载入没反应,原因在别处闪一下”。

    ⚠ 预检只证**格式自洽**:文件大小能按该 profile 切出帧,不证这份采集是这块板录的
    (拿 afm761 的 profile 去数 SR61 的目录也能数出帧,只是数量对不上)。所以这里报的是
    「可载入 · N 帧」而不是“校验通过”;物理对不对得看点云/RD 图,不在这层假装能判。
    """

    #: (板目录, 采集目录)
    loadRequested = QtCore.Signal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent, QtCore.Qt.Popup)
        self.setObjectName("radarConfig")
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.setStyleSheet(_config_qss())
        self.setMinimumWidth(560)
        self._catalog = ()
        self._capture_for = lambda name: ""  # noqa: E731 -- 默认空表,set_catalog 会换掉

        grid = QtWidgets.QGridLayout(self)
        grid.setContentsMargins(14, 12, 14, 12)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(7)

        title = QtWidgets.QLabel("雷达配置")
        title.setObjectName("cfgTitle")
        grid.addWidget(title, 0, 0, 1, 4)

        hint = QtWidgets.QLabel(
            "切换即新会话:缓存与航迹从零开始(EKF 跟踪状态不能跨会话迁移)。"
        )
        hint.setObjectName("cfgHint")
        hint.setWordWrap(True)
        grid.addWidget(hint, 1, 0, 1, 4)

        self.combo_board = QtWidgets.QComboBox()
        self.combo_board.currentIndexChanged.connect(self._on_board_changed)
        grid.addWidget(QtWidgets.QLabel("雷达"), 2, 0)
        grid.addWidget(self.combo_board, 2, 1, 1, 3)

        self.edit_dir = QtWidgets.QLineEdit()
        self.edit_dir.setPlaceholderText(".bin 所在的采集目录")
        self.edit_dir.editingFinished.connect(self._refresh)
        self.btn_browse = QtWidgets.QPushButton("浏览…")
        self.btn_browse.clicked.connect(self._on_browse)
        grid.addWidget(QtWidgets.QLabel("采集"), 3, 0)
        grid.addWidget(self.edit_dir, 3, 1, 1, 2)
        grid.addWidget(self.btn_browse, 3, 3)

        self.combo_recent = QtWidgets.QComboBox()
        self.combo_recent.currentIndexChanged.connect(self._on_recent)
        grid.addWidget(QtWidgets.QLabel("最近"), 4, 0)
        grid.addWidget(self.combo_recent, 4, 1, 1, 3)

        self.lbl_state = QtWidgets.QLabel("—")
        self.lbl_state.setObjectName("cfgState")
        self.lbl_state.setWordWrap(True)
        grid.addWidget(self.lbl_state, 5, 0, 1, 4)

        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        self.btn_cancel = QtWidgets.QPushButton("取消")
        self.btn_cancel.clicked.connect(self.close)
        row.addWidget(self.btn_cancel)
        self.btn_load = QtWidgets.QPushButton("载入")
        self.btn_load.setObjectName("cfgLoad")
        self.btn_load.setDefault(True)
        self.btn_load.clicked.connect(self._on_load)
        row.addWidget(self.btn_load)
        grid.addLayout(row, 6, 0, 1, 4)

    # -- 内容 ---------------------------------------------------------------
    def set_catalog(
        self, catalog, capture_for, recents, current_profile: str = ""
    ) -> None:
        """刷新下拉。capture_for(板名) → 该板记住/内置的采集目录。"""
        self._catalog = tuple(catalog)
        self._capture_for = capture_for
        self.combo_board.blockSignals(True)
        self.combo_board.clear()
        for board in self._catalog:
            if board.error:
                label = f"⚠ {board.name} · 载入失败"
            else:
                label = f"{board.name} · {board.summary}"
            self.combo_board.addItem(label, board.name)
            item = self.combo_board.count() - 1
            tip = board.error or f"{board.summary}\n{board.profile_dir}"
            self.combo_board.setItemData(item, tip, QtCore.Qt.ToolTipRole)
            if board.name == current_profile:
                self.combo_board.setCurrentIndex(item)
        self.combo_board.blockSignals(False)

        self.combo_recent.blockSignals(True)
        self.combo_recent.clear()
        self.combo_recent.addItem("—")
        for path in recents:
            self.combo_recent.addItem(path)
        self.combo_recent.blockSignals(False)

        self.edit_dir.setText(self._capture_for(self._board_name()))
        self._refresh()

    def set_error(self, message: str) -> None:
        """载入失败的原因写回弹层(下次打开能看到)。"""
        self._set_state(message, "error")

    # -- 事件 ---------------------------------------------------------------
    def _board_name(self) -> str:
        return self.combo_board.currentData() or ""

    def _board(self):
        name = self._board_name()
        return next((b for b in self._catalog if b.name == name), None)

    def _on_board_changed(self, _index: int) -> None:
        # 换板顺带把采集目录换成该板记住/内置的那份:两块板的采集格式与目录都不同,
        # 留着上一块的路径几乎必然是错的。
        self.edit_dir.setText(self._capture_for(self._board_name()))
        self._refresh()

    def _on_recent(self, index: int) -> None:
        if index > 0:
            self.edit_dir.setText(self.combo_recent.itemText(index))
            self._refresh()

    def _on_browse(self) -> None:
        # 先收起弹层再开文件对话框:Popup 一失焦就自动关,直接从弹层里开对话框会把它
        # 关掉(用户看到的是“点浏览就没了”)。选完按原位置弹回来。
        pos = self.pos()
        self.hide()
        start = self.edit_dir.text().strip() or str(Path.home())
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self.parentWidget(), "选择采集目录", start
        )
        if path:
            self.edit_dir.setText(path)
            self._refresh()
        self.move(pos)
        self.show()

    def _on_load(self) -> None:
        pf = self._preflight()
        if pf is None or not pf.ok:
            self._set_state(pf.error if pf is not None else "没有可载入的板", "error")
            return
        self.loadRequested.emit(self._profile_dir(), self.edit_dir.text().strip())
        self.close()

    # -- 预检与状态 ---------------------------------------------------------
    def _profile_dir(self) -> str:
        board = self._board()
        return str(board.profile_dir) if board is not None else ""

    @staticmethod
    def _bad(message: str):
        return boards.Preflight(False, message, 0, 0.0, None, None)

    def _preflight(self):
        board = self._board()
        if board is None:
            return None
        if board.error:
            return self._bad(f"这块板载入失败:{board.error}")
        raw = self.edit_dir.text().strip()
        if not raw:
            return self._bad("采集目录为空:点『浏览…』选 .bin 所在目录")
        return boards.preflight(board.profile_dir, Path(raw))

    def _refresh(self) -> None:
        pf = self._preflight()
        if pf is None:
            self._set_state(
                "profiles/ 下没有可选的板(profile.toml 一个都没找到)", "error"
            )
            return
        if not pf.ok:
            self._set_state(pf.error, "error")
            return
        self._set_state(
            f"可载入 · 采集 {pf.total} 帧 @ {1.0 / pf.period_s:.1f}Hz —— "
            "只证格式能切出帧,不证这份采集是这块板录的",
            "ok",
        )

    def _set_state(self, text: str, tone: str) -> None:
        self.lbl_state.setText(text)
        self.lbl_state.setStyleSheet(f"color: {_TONES[tone]};")
        self.btn_load.setEnabled(tone != "error")
