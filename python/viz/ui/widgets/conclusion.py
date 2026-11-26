"""一句话结论:屏标题旁的一行字,取代原来的 KPI 条。

每屏声明一个取值函数 `clauses(frame) -> [(文字, 等级)]`(等级 0 正常 / 1 需要看 / 2 异常,
按重要性排序);这里只挑 1~2 句拼成一行:有 ▲ / ■ 的句子优先,并用对应颜色。函数返回空列表
(没有可说的)时显示空串。数据只读帧与诊断,不做计算。
"""

from collections.abc import Callable

from PySide6.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QWidget

from .. import theme
from ..state import ViewState
from ..views.base import ResultView

Clause = tuple[str, int]
MAX_CLAUSES = 2


def compose(clauses: list[Clause]) -> tuple[str, str]:
    """(富文本, 纯文本):有等级的句子排前面,最多 MAX_CLAUSES 句。"""
    flagged = [c for c in clauses if c[1] >= 1]
    calm = [c for c in clauses if c[1] < 1]
    chosen = (flagged + calm)[:MAX_CLAUSES]
    html, plain = [], []
    for text, level in chosen:
        plain.append(f"{theme.LEVEL_SHAPES[level]} {text}" if level >= 1 else text)
        if level >= 1:
            color = theme.LEVEL_COLORS[level]
            html.append(f"<span style='color:{color}'>{plain[-1]}</span>")
        else:
            html.append(text)
    return " · ".join(html), " · ".join(plain)


class Conclusion(ResultView):
    title = "一句话结论"

    def __init__(
        self,
        state: ViewState,
        clauses: Callable[[object], list[Clause]],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(state, parent)
        self.clauses = clauses
        self.text = ""
        self.label = QLabel()
        self.label.setStyleSheet(f"color: {theme.DIM}; font-size: {theme.FONT_BODY}px;")
        # 句子长时宁可裁掉,也不要把窗口撑宽
        self.label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.label)
        self._html = ""

    def reset(self, source) -> None:
        self._html, self.text = "", ""
        self.label.clear()

    def render_frame(self, frame) -> None:
        html, self.text = compose(self.clauses(frame))
        if html != self._html:  # 没变的不重设:逐帧 setText 会触发重排
            self._html = html
            self.label.setText(html)
