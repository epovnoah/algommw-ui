"""工作台 UI 层(照 ccm 的界面架构):外壳 window、播放器 player、共享状态 state、主题
theme、参数抽屉 drawer、配置弹层 popup、参数声明 params,以及 screens/ views/ widgets/ plots/。

只有这一层可以依赖 Qt / pyqtgraph(tests/py/testarch.py 检查)。必须先固定 Qt 绑定再导入
pyqtgraph,并在创建任何图元之前设好全局选项,所以放在包初始化里(照 ccm `gui/__init__`)。
"""

import os

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

from . import theme  # noqa: E402

theme.configure_pyqtgraph()
