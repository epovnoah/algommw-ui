"""CFAR 屏:这些检测是不是贴着门限?哪一向在约束它?

版式「主图 + 常驻表」:上半是 RD 图(分段控件选叠加:无叠加 / CFAR 逐格判决 / fold 分区),
下半是常驻检测表(`list_docked`,默认展开、固定 250 px,「列表 N」按钮仍可收起);表按两向余量的
较小值升序(最贴门限的在最上面),两向余量分开列,能看出是哪一向没余量;选中单元的距离向与多普勒向
切片(功率 / 噪声 / 门限)与全帧余量散点在溯源页的 CFAR 一级。
"""

from .. import theme
from ..plots.mini import CfarSlices, DetectionScatter
from ..state import ViewState
from ..views.rdmap import OVERLAY_MODES, RdMapView
from ..widgets.conclusion import Conclusion
from ..widgets.stagelist import StageList
from .base import Screen
from .common import (
    COL_CFAR_DOPPLER,
    COL_CFAR_MARGIN,
    COL_CFAR_RANGE,
    COL_INDEX,
    COL_POWER,
    COL_RANGE,
    COL_SNR,
    COL_VELOCITY,
    order_by,
)

MARGIN_RANGE_DB = (-8.0, 24.0)


def _criteria(frame):
    return frame.diagnostics.verdicts.criteria


def cfar_scatter() -> DetectionScatter:
    """全帧余量散点(模块级不能建 QWidget,所以做成工厂,屏构造时再建)。"""
    return DetectionScatter(
        "cfar",
        lambda f: f.diagnostics.metrics.cfar_margin_db,
        "CFAR 余量 dB · 本帧全部检测",
        MARGIN_RANGE_DB,
        lines=[(lambda f: _criteria(f).cfar_margin_warn_db, theme.LEVEL_COLORS[1])],
    )


def clauses(frame):
    """「23 个检测,其中 3 个贴近门限」:有贴门限的检测时整句标「需要看」。"""
    n = len(frame.detections)
    s = frame.diagnostics.summary if frame.diagnostics is not None else None
    if (
        s is not None
        and n
        and s.n_near_threshold
        and frame.diagnostics.metrics.available.get("cfar") is not False
    ):
        return [(f"{n} 个检测,其中 {s.n_near_threshold} 个贴近门限", 1)]
    return [(f"{n} 个检测", 0)]


class CfarScreen(Screen):
    key = "cfar"
    title = "CFAR"
    list_docked = True

    def __init__(self, state: ViewState) -> None:
        self.view = RdMapView(state, stage="cfar")
        self.view.mode.setCurrentIndex(OVERLAY_MODES.index("CFAR 判决"))
        self.list = StageList(
            state,
            "检测(按 CFAR 余量)",
            [
                COL_INDEX,
                COL_RANGE,
                COL_VELOCITY,
                COL_SNR,
                COL_POWER,
                COL_CFAR_RANGE,
                COL_CFAR_DOPPLER,
                COL_CFAR_MARGIN,
            ],
            order=order_by("cfar_margin_db"),
        )
        scatter = cfar_scatter()
        scatter.on_select = state.select_detection
        super().__init__(
            state,
            self.view,
            self.list,
            Conclusion(state, clauses),
            trace_plots=[CfarSlices(), scatter],
        )
        # 常驻表不再多留一行抽屉头:表头行已是列名,排序口径放进按钮提示
        self.caption.hide()
        self.list_button.setToolTip("检测表,按 CFAR 余量升序(最贴门限的在最上面)")
