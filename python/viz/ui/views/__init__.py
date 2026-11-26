"""结果视图。每个视图继承 ResultView,订阅 ViewState 自动随帧与选中刷新。

各阶段屏的主视图:ADC 时域(adc)、距离像与 RTI(rangefft)、RD 图 / DDM 矩阵(rdmap、ddm)、
测角空间谱(angle)、跟踪俯视(tracks);检测明细表(table)供需要全字段的地方使用。
"""

from .base import ResultView
from .table import DetectionTableView

__all__ = ["DetectionTableView", "ResultView"]
