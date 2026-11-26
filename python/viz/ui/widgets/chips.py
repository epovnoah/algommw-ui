"""各级「需要看」的检测计数芯片:点一下只看该级需要看的检测(ViewState.stage_filter),再点取消。

住在结果屏列表抽屉的头部(列表是被它筛选的)。本帧没有任何一级给出诊断时整排隐藏,
否则会显示一句不成立的「各级正常」。形状 + 颜色双编码;芯片各占定宽槽位,出现 / 消失
不会让后面的芯片左右跳。
"""

from collections.abc import Sequence

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget

from ...diagnostics import STAGES, Level
from .. import theme
from ..state import ViewState
from ..views.base import ResultView


def diagnosed(diag, stages: Sequence[str]) -> bool:
    """本帧是否有任何一级给出了诊断(不全是「不适用」)。"""
    return diag is not None and any(
        bool(np.any(diag.verdicts.levels[s] != Level.NA)) for s in stages
    )


class StageChips(ResultView):
    title = "各级需要看的检测"

    def __init__(self, state: ViewState, parent: QWidget | None = None) -> None:
        super().__init__(state, parent)
        self.stages = list(STAGES)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        self.chips: dict[str, QPushButton] = {}
        for stage in self.stages:
            button = QPushButton()
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(lambda _=False, s=stage: self._on_chip(s))
            button.setText(
                f"{theme.LEVEL_SHAPES[1]} {STAGES[stage]} 888"
            )  # 按三位数预留宽度
            slot = QWidget()
            slot.setFixedWidth(button.sizeHint().width())
            slot_row = QHBoxLayout(slot)
            slot_row.setContentsMargins(0, 0, 0, 0)
            slot_row.addWidget(button)
            button.hide()
            row.addWidget(slot)
            self.chips[stage] = button
        self.all_ok = QLabel(
            f"<span style='color:{theme.OK}'>{theme.LEVEL_SHAPES[0]} 各级正常</span>"
        )
        self.all_ok.hide()
        row.addWidget(self.all_ok)
        state.filter_changed.connect(lambda _stage: self._sync_checks())

    def reset(self, source) -> None:
        for button in self.chips.values():
            button.hide()
        self.all_ok.hide()

    def render_frame(self, frame) -> None:
        diag = frame.diagnostics
        if not diagnosed(diag, self.stages):
            for button in self.chips.values():
                button.hide()
            self.all_ok.hide()
            return
        any_flagged = False
        for stage, button in self.chips.items():
            count = diag.summary.flagged.get(stage, 0)
            button.setVisible(count > 0)
            if count == 0:
                continue
            any_flagged = True
            bad = bool((diag.verdicts.levels[stage] >= Level.BAD).any())
            level = 2 if bad else 1
            color = theme.LEVEL_COLORS[level]
            text = f"{theme.LEVEL_SHAPES[level]} {STAGES[stage]} {count}"
            if button.text() != text:
                button.setText(text)
                button.setStyleSheet(
                    f"QPushButton {{ color: {color}; background: {theme.SURFACE}; "
                    f"border: 1px solid {theme.LINE}; border-radius: 10px; padding: 3px 10px; "
                    f"font-size: {theme.FONT_TICK}px; }}"
                    f"QPushButton:hover {{ background: {theme.RAISED}; }}"
                    f"QPushButton:checked {{ border: 1px solid {color}; }}"
                )
        self.all_ok.setVisible(not any_flagged)
        self._sync_checks()

    def _sync_checks(self) -> None:
        for stage, button in self.chips.items():
            button.blockSignals(True)
            button.setChecked(self.state.stage_filter == stage)
            button.blockSignals(False)

    def _on_chip(self, stage: str) -> None:
        self.state.set_filter(None if self.state.stage_filter == stage else stage)
