"""内含标签的胶囊按钮(QPushButton 里放 QLabel)的尺寸:QPushButton 的 sizeHint 只看按钮自己的文字,
内含布局撑不开它 —— 文字一变就得按布局的 sizeHint 重设宽度,否则内容被裁。"""

from PySide6.QtWidgets import QPushButton


def fit_pill(button: QPushButton) -> None:
    layout = button.layout()
    layout.activate()
    button.setFixedWidth(layout.sizeHint().width())
