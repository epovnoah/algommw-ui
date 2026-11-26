"""检查器(右栏,300 px)的五页:溯源 / 参数 / 判据 / 雷达 / 耗时(后四页照 ccm `gui/panels.py`;
「溯源」页是选中检测的竖直时间线,见 `widgets/trace.py`)。开合由 `widgets/slide.py` 管。

- 处理参数:pyqtgraph ParameterTree,节点由 `params._SPEC` 生成;改值 → `clone_config` 出新
  ChainCfg → 发 `config_changed(cfg)`,窗口据此建新 FrameProcessor(从当前帧起)并重算当前帧。
  改参 = eChainInit = 航迹清零(跟踪状态是 file-static),页顶如实写明。波形 / 标度量只读。
- 判据:ParameterTree,节点由 `diagnostics.CriteriaConfig` 字段的元数据生成;改值 → 发
  `criteria_changed(criteria)`,窗口据此用新判据建处理器并重算当前帧,并按板存进
  data/criteria.json。行内 ↺ 复位到该板的标定默认值。
- 雷达信息:RadarInfo 与 ChainCfg 里的波形参数和推导量,加当前帧、选中检测与选中航迹的详情。
  数值只读 ChainCfg 已有字段(推导量只是这些字段的乘积),不从物理量另推一份标度。
- 处理耗时:各级耗时与占帧周期比例(块字符文字条),合计与余量;点某一级发 `stage_activated`。
  合计取 `diagnostics.processing_ms`(不含诊断开销),诊断开销(取中间量、诊断)单列。
"""

import dataclasses

import numpy as np
from pyqtgraph.parametertree import Parameter, ParameterTree
from PySide6.QtCore import QSignalBlocker, Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QLabel,
    QPushButton,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from binding import dtypes
from binding.config import doa_route_recommend

from .. import diagnostics
from . import theme
from .params import (
    _DEFAULT_SCAN,
    _READONLY_TIP,
    _SCAN_GROUP,
    _SPEC,
    _get,
    _scan_fields,
    _set,
    clone_config,
    fft_options_supported,
    group_tip,
    parameter_error,
    scan_ready,
    scan_tag,
    variant_name,
)
from .state import Selection, ViewState
from .views.common import TIMING_STAGES, rd_column

_BAR_CELLS = 12
_PARTIAL = " ▏▎▍▌▋▊▉"
_STAGE_ROLE = 256  # Qt.ItemDataRole.UserRole

#: 扫描网格六项的 ChainCfg 路径,顺序与 params._scan_fields 一致。
_SCAN_PATHS = tuple(
    path
    for title, _tag, rows in _SPEC
    if title == _SCAN_GROUP
    for path, _label, kind, _opts in rows
    if kind != "bool"
)


def _text_bar(fraction: float) -> str:
    """用块字符画占比条(每帧只改文字;QProgressBar 每次 setValue 都要重绘控件,在 Retina
    屏上明显拖慢回放)。"""
    eighths = int(round(min(max(fraction, 0.0), 1.0) * _BAR_CELLS * 8))
    full, rest = divmod(eighths, 8)
    return "█" * full + (_PARTIAL[rest] if rest else "")


def _hint(text: str) -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    label.setStyleSheet(
        f"color: {theme.DIM}; padding: 2px 4px; font-size: {theme.FONT_NOTE}px;"
    )
    return label


# ---------------------------------------------------------------- 处理参数
class ParameterPanel(QWidget):
    """处理参数:每组一个树节点,改动立即生成新的 ChainCfg 副本并发出。"""

    config_changed = Signal(object)  # 新的 ChainCfg 副本
    motion_changed = Signal(bool)  # 走动补偿研究开关(不在 ChainCfg 里,由驱动层执行)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.cfg = None
        self.motion = QCheckBox("走动补偿(研究):运动点用补偿后的快照测角")
        self.motion.setToolTip(
            "仅双 PRF 步进频。打开后,|v| ≥ 2 m/s 的检测点在测角前换成按其 R/v 做多普勒域\n"
            "走动补偿的局部匹配快照,距离/速度/检测不变,只影响角度及其后的配对与跟踪。\n"
            "逐点在 Python 里计算,每帧约慢 0.3~0.5 s;开关即重建处理链并重算当前帧。\n"
            "实采上没有观察到俯仰改善,见 docs/archive/dopplercompensation.md。"
        )
        self.motion.setEnabled(False)
        self.motion.toggled.connect(self.motion_changed)
        self.params: Parameter | None = None
        self._kinds: dict[str, str] = {}
        self._saved_scan: tuple[float, ...] | None = None
        self._applied_cfg = None
        self.status = _hint("尚未载入处理配置")
        self.revert = QPushButton("撤销未生效修改")
        self.revert.setEnabled(False)
        self.revert.clicked.connect(self._revert)
        self.tree = ParameterTree(showHeader=False)
        self.tree.setIndentation(
            8
        )  # 检查器只有 300 px 宽:默认缩进 20 会让值列出水平滚动
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.addWidget(
            _hint(
                "改参数即重建处理链(eChainInit),航迹在当前帧清零、id 重新开始,当前帧按"
                "新参数重算;之后的帧从这一帧顺序往前算。波形与标度量只读(改 profile.toml)。"
            )
        )
        layout.addWidget(self.motion)
        layout.addWidget(self.tree, 1)
        layout.addWidget(self.status)
        layout.addWidget(self.revert)

    def set_config(self, config) -> None:
        """换会话:按新配置重建参数树(基线 = profile 载入值,行内 ↺ 复位到它)。"""
        # 换会话后研究开关回到关:新处理器是不带补偿建的
        with QSignalBlocker(self.motion):
            self.motion.setChecked(False)
        self.motion.setEnabled(bool(config.xPair.bEnable))
        self.cfg = clone_config(config)
        self._kinds = {}
        self._saved_scan = None
        children = []
        for title, tag, rows in _SPEC:
            items = [
                self._item(path, label, kind, opts) for path, label, kind, opts in rows
            ]
            children.append(
                {
                    "name": title,
                    "title": self._group_title(title, tag),
                    "type": "group",
                    "children": items,
                    "tip": group_tip(title),
                    "expanded": title.startswith("CFAR"),
                }
            )
        children.append(self._readonly_group(self.cfg))
        self.params = Parameter.create(
            name="processing", type="group", children=children
        )
        self.params.sigTreeStateChanged.connect(self._on_change)
        self.tree.setParameters(self.params, showTop=False)
        self._applied_cfg = clone_config(self.cfg)
        self.revert.setEnabled(False)
        self._refresh_controls()
        self.set_status("已载入配置;有效修改会立即重算当前帧")

    def _group_title(self, title: str, tag: str) -> str:
        if title == _SCAN_GROUP:
            name = variant_name(self.cfg.eDoa)
            tag = (
                scan_tag(self.cfg.xDoa.xScan)
                if name in ("beam", "dbf2d")
                else f"{name} 不使用网格"
            )
        return f"{title} · {tag}" if tag else title

    def _scan_values(self) -> tuple[float, ...]:
        return tuple(float(v) for v in _scan_fields(self.cfg.xDoa.xScan))

    def _write_scan(self, values) -> None:
        for path, value in zip(_SCAN_PATHS, values):
            _set(self.cfg, path, float(value))

    def _sync_scan_widgets(self) -> None:
        values = self._scan_values()
        with QSignalBlocker(self.params):
            for path, value in zip(_SCAN_PATHS, values):
                child = self.params.child(_SCAN_GROUP, path)
                if child is not None:
                    child.setValue(float(value))
            self.params.child(_SCAN_GROUP).setOpts(
                title=self._group_title(_SCAN_GROUP, "")
            )

    def _sync_scan_for_variant(self) -> None:
        """切变体时同步扫描网格,避免 UI 必然经过一个非法中间态。

        dbf2d 必须有网格,corr/music/dml 不能有网格;beam 两种都接受。
        切走时记下最后一次合法网格,切回 dbf2d 时恢复,便于 A/B 对比同一套格子。
        """
        name = variant_name(self.cfg.eDoa)
        values = self._scan_values()
        if name == "dbf2d":
            if not any(values):
                saved = self._saved_scan
                if saved is None:
                    saved = _DEFAULT_SCAN
                self._write_scan(saved)
                self._sync_scan_widgets()
        elif (
            name == "beam"
            and not any(values)
            and doa_route_recommend(self.cfg.xArray)[0] == dtypes.E_DOA_VARIANT_DBF2D
        ):
            self._write_scan(self._saved_scan or _DEFAULT_SCAN)
            self._sync_scan_widgets()
        if name in ("corr", "phasefit", "music", "dml") and any(values):
            if scan_ready(self.cfg.xDoa.xScan):
                self._saved_scan = values
            self._write_scan((0.0,) * len(_SCAN_PATHS))
            self._sync_scan_widgets()
        if not fft_options_supported(self.cfg):
            # 切后端时一次性清除目标后端不消费的开关,避免必然经过加载失败的中间态。
            for path in (
                "xDoa.bExtMaxVel",
                "xDoa.bAngleInterp",
                "xDoa.xMultiPeak.bEnable",
            ):
                _set(self.cfg, path, 0)
        elif self.cfg.xWave.eModulation == dtypes.E_MOD_DDM:
            self.cfg.xDoa.bExtMaxVel = 0

    def _item(self, path: str, label: str, kind: str, opts: dict) -> dict:
        self._kinds[path] = kind
        value = _get(self.cfg, path)
        item = {"name": path, "title": label}
        if kind == "list":
            item.update(type="list", limits=dict(opts["limits"]), value=int(value))
        elif kind == "bool":
            item.update(type="bool", value=bool(value))
        elif kind == "int":
            item.update(type="int", value=int(value))
        else:
            item.update(type="float", value=float(value))
        if kind in ("int", "float"):
            if opts.get("limits") is not None:
                lo, hi = opts["limits"]
                n = int(self.cfg.xWave.ulNumAdcSamples)
                if path in (
                    "xCfar.ulMinRangeBin",
                    "xCfar.ulMaxRangeBin",
                    "xCfar.ulRangeSkipLow",
                    "xCfar.ulRangeSkipHigh",
                ) or path.startswith("xCfar.ulSegmentEnd."):
                    hi = n - 1
                if path.endswith(("ulWinLen", "ulGuardLen", "ulOsRank")):
                    length = (
                        self.cfg.xWave.ulNumDopplerChirps if ".xDoppler." in path else n
                    )
                    hi = length - 1 if path.endswith("ulOsRank") else (length - 1) // 2
                # 关闭功能的零值及合法 profile 的大尺寸不能在创建控件时被静默夹断。
                item["limits"] = (min(lo, value), max(hi, value))
            if opts.get("step") is not None:
                item["step"] = opts["step"]
        # 只给 default(= profile 载入值)不给 value:pyqtgraph 0.14 把显式给的 value 记成
        # 「改过」,行内 ↺ 复位按钮会在每一行都亮着
        item["default"] = item.pop("value")
        return item

    def set_status(self, message: str, error: bool = False) -> None:
        self.status.setText(message)
        color = theme.LEVEL_COLORS[2] if error else theme.DIM
        self.status.setStyleSheet(f"color: {color}; padding: 2px 4px;")

    def reject_config(self, config, message: str) -> None:
        self._applied_cfg = clone_config(config)
        self.revert.setEnabled(True)
        self.set_status(f"未生效:{message}。请修正或撤销。", error=True)

    def _revert(self) -> None:
        self.cfg = clone_config(self._applied_cfg)
        self._refresh_controls()
        self.revert.setEnabled(False)
        self.set_status("已撤销未生效修改,继续使用上次有效配置")

    def _refresh_controls(self) -> None:
        """值、可用性与提示一起刷新;只阻断根节点通知,不阻断控件自己的更新。"""
        cfg = self.cfg
        fft = fft_options_supported(cfg)
        pair = bool(cfg.xPair.bEnable)
        with QSignalBlocker(self.params):
            for title, _tag, rows in _SPEC:
                for path, _label, _kind, _opts in rows:
                    child = self.params.child(title, path)
                    enabled, tip = True, ""
                    if path == "eDoa":
                        choices = dict(_opts["limits"])
                        if cfg.xWave.eModulation == dtypes.E_MOD_DDM:
                            choices = {
                                k: v
                                for k, v in choices.items()
                                if k not in ("music", "dml")
                            }
                        child.setOpts(limits=choices)
                    if path in (
                        "xDoa.bAngleInterp",
                        "xDoa.xMultiPeak.bEnable",
                        "xDoa.xMultiPeak.xThrScale",
                        "xDoa.bExtMaxVel",
                    ):
                        enabled = fft
                        tip = "仅二维 beam 的 FFT 后端支持;网格后端与 corr/music/dml 不支持"
                        if path == "xDoa.bExtMaxVel":
                            enabled &= (
                                cfg.xWave.eModulation == dtypes.E_MOD_TDM
                                and cfg.xWave.ulNumTx > 1
                            )
                            tip += ";扩展速度还要求多 TX 的 TDM 波形"
                        if path.endswith("xThrScale"):
                            enabled &= bool(cfg.xDoa.xMultiPeak.bEnable)
                    if path.startswith("xDoa.xScan."):
                        enabled = variant_name(cfg.eDoa) in ("beam", "dbf2d")
                        tip = "扫描网格仅供 beam/dbf2d 使用"
                        if path != "xDoa.xScan.enabled":
                            enabled &= bool(_get(cfg, "xDoa.xScan.enabled"))
                    if path.startswith("xDoa.xFov.") and not path.endswith("bEnable"):
                        enabled = bool(cfg.xDoa.xFov.bEnable)
                        tip = "先启用角度 FOV"
                    if path in (
                        "xDoa.bNearElevationSkip",
                        "xDoa.ulNearElevationMaxBin",
                    ):
                        enabled = (
                            cfg.xWave.eModulation == dtypes.E_MOD_DDM
                            and variant_name(cfg.eDoa) == "corr"
                        )
                        tip = "近距俯仰豁免仅由 corr/DDM 后端消费"
                        if path.endswith("ulNearElevationMaxBin"):
                            enabled &= bool(cfg.xDoa.bNearElevationSkip)
                    if path.startswith("xDcSig."):
                        enabled = not pair and (
                            path.endswith("bEnable") or bool(cfg.xDcSig.bEnable)
                        )
                        tip = (
                            "双 PRF 配对与 DC 耦合签名互斥"
                            if pair
                            else "启用后先校准,后续帧才补偿"
                        )
                    if path.startswith("xCluster."):
                        enabled = not pair and (
                            path.endswith("eMode")
                            or cfg.xCluster.eMode != dtypes.E_CLUSTER_OFF
                        )
                        tip = (
                            "双 PRF 配对要求聚类关闭;可先关闭配对"
                            if pair
                            else group_tip("聚类")
                        )
                    if path.startswith("xPair."):
                        enabled = bool(cfg.xAlternateWave.xChirpInterval) and (
                            path.endswith("bEnable") or pair
                        )
                        tip = "需要 profile 声明双波形与 CP_ADC;DC 耦合签名和聚类须关闭"
                    if (
                        path in ("xCfar.xRange.xThrDb", "xCfar.xDoppler.xThrDb")
                        and cfg.xCfar.bSegmented
                    ):
                        enabled, tip = (
                            False,
                            "分段模式已覆盖普通门限,请修改「CFAR · 分段门限」中的三段门限",
                        )
                    if path.endswith(("ulOsRank", "bOsEdgeScale")):
                        dim = (
                            cfg.xCfar.xRange
                            if ".xRange." in path
                            else cfg.xCfar.xDoppler
                        )
                        enabled = dim.eNoiseMode == dtypes.E_CFAR_OS
                        tip = "只在 OS 噪声估计模式下使用;OS 秩须 ≤ 2×参考窗"
                        if path == "xCfar.xDoppler.bOsEdgeScale":
                            enabled = False
                            tip = "Doppler 使用循环参考窗,边缘不缩短,无需边缘秩缩放"
                    if path.endswith("bPowerMean"):
                        dim = (
                            cfg.xCfar.xRange
                            if ".xRange." in path
                            else cfg.xCfar.xDoppler
                        )
                        enabled = dim.eNoiseMode not in (
                            dtypes.E_CFAR_OS,
                            dtypes.E_CFAR_CACC,
                        )
                        tip = "仅 CA/CAGO/CASO 使用平均域;OS 排序与域无关,CACC 固定为功率求和"
                    if path == "xCfar.bCrossPeakGrouping":
                        enabled = bool(
                            cfg.xCfar.xRange.bPeakGrouping
                            and cfg.xCfar.xDoppler.bPeakGrouping
                        )
                        tip = "两维峰值分组都开启时选择十字核,不比较对角邻居"
                    if title == "CFAR · 分段门限" and path != "xCfar.bSegmented":
                        enabled = bool(cfg.xCfar.bSegmented)
                        tip = "先启用分段门限;两边界递增,分别控制三个距离区间"
                    if (
                        title == "CFAR · 主导列补点"
                        and path != "xCfar.bDominantSupplement"
                    ):
                        enabled = bool(cfg.xCfar.bDominantSupplement)
                        tip = "先启用主导列补点;距离 CFAR 关闭时不补点"
                    child.setOpts(enabled=bool(enabled), tip=tip)
                    child.setValue(_get(cfg, path))
            self.params.child(_SCAN_GROUP).setOpts(
                title=self._group_title(_SCAN_GROUP, "")
            )

    def _enable_defaults(self, path) -> None:
        """首次启用未配置的可选级时补合法值,已有 profile 值完整保留。"""
        c = self.cfg.xCfar
        for dim in ("xRange", "xDoppler"):
            field = getattr(c, dim)
            if (
                path in (f"xCfar.{dim}.eNoiseMode", f"xCfar.{dim}.ulWinLen")
                and field.eNoiseMode == dtypes.E_CFAR_OS
            ):
                field.ulOsRank = min(
                    field.ulOsRank or field.ulWinLen, 2 * field.ulWinLen
                )
        if path == "xCfar.bSegmented" and c.bSegmented and not any(c.xRangeThrDb):
            n = self.cfg.xWave.ulNumAdcSamples - c.ulRangeSkipLow - c.ulRangeSkipHigh
            c.ulSegmentEnd[:] = (max(1, n // 3), max(2, 2 * n // 3))
            c.xRangeThrDb[:] = (abs(c.xRange.xThrDb) or 0.5,) * 3
            c.xDopplerThrDb[:] = (abs(c.xDoppler.xThrDb) or 0.5,) * 3
        if (
            path == "xCfar.bDominantSupplement"
            and c.bDominantSupplement
            and c.ulSupplementWin == 0
        ):
            c.ulDominantEnd = self.cfg.xWave.ulNumAdcSamples // 2 - 1
            c.xDominanceThrDb, c.xSupplementThrDb = 3.0, 3.0
            c.ulSupplementGuard, c.ulSupplementWin = 2, 8
            c.ulSupplementLimit = 80

    @staticmethod
    def _readonly_group(cfg) -> dict:
        wave, arr = cfg.xWave, cfg.xArray
        rows = [
            ("距离分辨 ΔR", f"{wave.xRangeStep:.4f} m"),
            ("速度分辨 ΔV", f"{wave.xDopplerStep:.4f} m/s"),
            ("帧周期", f"{wave.xFrameTime * 1000:.0f} ms"),
            ("发/收/虚拟", f"{wave.ulNumTx}·{wave.ulNumRx}·{arr.ulNumVirtual}"),
            (
                "ADC 采样 / doppler",
                f"{wave.ulNumAdcSamples} / {wave.ulNumDopplerChirps}",
            ),
            ("阵列网格", f"{arr.ulGridRows}×{arr.ulGridCols}"),
        ]
        items = [
            {
                "name": f"ro{i}",
                "title": label,
                "type": "str",
                "value": value,
                "readonly": True,
                "tip": _READONLY_TIP,
            }
            for i, (label, value) in enumerate(rows)
        ]
        return {
            "name": "波形 · 几何",
            "title": "波形 · 几何 · 只读",
            "type": "group",
            "children": items,
            "tip": _READONLY_TIP,
            "expanded": False,
        }

    def _on_change(self, _root, changes) -> None:
        changed = False
        variant_changed = False
        for param, change, value in changes:
            if change != "value":
                continue
            path = param.name()
            kind = self._kinds.get(path)
            if kind is None:
                continue
            if kind == "float":
                value = float(value)
            else:  # list(枚举编号)/ bool(uint8)/ int
                value = int(value)
            if (
                path == "xDoa.xScan.enabled"
                and not value
                and scan_ready(self.cfg.xDoa.xScan)
            ):
                self._saved_scan = self._scan_values()
            _set(self.cfg, path, value)
            self._enable_defaults(path)
            if path == "xDoa.xScan.enabled":
                if value and self._saved_scan:
                    self._write_scan(self._saved_scan)
                if value:
                    self._sync_scan_for_variant()
            if path == "eDoa":
                variant_changed = True
            if path.startswith(("xCfar.xRange.", "xCfar.xDoppler.")):
                # 门限为零或从旧的零门限配置开启时,同步两行显示,不重复发起链重建。
                with QSignalBlocker(self.params):
                    for dim, title in (
                        ("xRange", "CFAR · Range"),
                        ("xDoppler", "CFAR · Doppler"),
                    ):
                        for suffix in ("enabled", "xThrDb"):
                            key = f"xCfar.{dim}.{suffix}"
                            self.params.child(title, key).setValue(_get(self.cfg, key))
            if path.startswith("xDoa.xScan."):
                group = self.params.child(_SCAN_GROUP)
                group.setOpts(title=self._group_title(_SCAN_GROUP, ""))
            changed = True
        if changed:
            if variant_changed:
                self._sync_scan_for_variant()
            self._refresh_controls()
            error = parameter_error(self.cfg)
            self.revert.setEnabled(bool(error))
            if error:
                self.set_status(
                    f"未生效: {error}。请修正或撤销;播放仍使用上次有效配置。",
                    error=True,
                )
                return
            self._applied_cfg = clone_config(self.cfg)
            self.set_status("参数已提交,正在按新配置重算当前帧")
            # 发副本:处理线程正拿着旧配置在跑,原地改是数据竞争
            self.config_changed.emit(clone_config(self.cfg))


# ---------------------------------------------------------------- 判据
class CriteriaPanel(QWidget):
    """判据阈值(由 CriteriaConfig 的字段元数据生成)。改动发 `criteria_changed`。"""

    criteria_changed = Signal(object)  # diagnostics.CriteriaConfig

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.criteria = None
        self.params: Parameter | None = None
        self.tree = ParameterTree(showHeader=False)
        self.tree.setIndentation(
            8
        )  # 检查器只有 300 px 宽:默认缩进 20 会让值列出水平滚动
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.addWidget(
            _hint(
                "判据决定处理链行的状态形状、一句话结论、溯源页与列表里的标色。默认值按两块板各 200 帧的分布"
                "标定(python python/bench diag calib)。改判据只重新分级,不重算处理链、"
                "航迹不受影响;改过的值按板记进 data/criteria.json。"
            )
        )
        layout.addWidget(self.tree, 1)

    def set_criteria(self, criteria, default) -> None:
        """换会话:按当前判据重建树;行内 ↺ 复位到 default(该板的标定默认值)。"""
        self.criteria = criteria
        children = []
        for f in dataclasses.fields(criteria):
            meta = f.metadata
            value, base = getattr(criteria, f.name), getattr(default, f.name)
            item = {"name": f.name, "title": meta["label"], "tip": meta.get("tip", "")}
            if isinstance(base, bool):
                item["type"] = "bool"
            else:
                item["type"] = "float"
                if meta.get("lo") is not None and meta.get("hi") is not None:
                    item["limits"] = (meta["lo"], meta["hi"])
                if meta.get("step") is not None:
                    item["step"] = meta["step"]
                if meta.get("unit"):
                    item["suffix"] = f" {meta['unit']}"
            # 同处理参数页:只给 default 时 ↺ 不亮;与默认不同才另给 value
            item["default"] = base
            if value != base:
                item["value"] = value
            children.append(item)
        self.params = Parameter.create(name="criteria", type="group", children=children)
        self.params.sigTreeStateChanged.connect(self._on_change)
        self.tree.setParameters(self.params, showTop=False)

    def _on_change(self, _root, changes) -> None:
        values = {p.name(): v for p, change, v in changes if change == "value"}
        if not values or self.criteria is None:
            return
        try:
            self.criteria = self.criteria.with_values(**values)
        except ValueError:
            return  # 越界值由控件的范围限制挡住,这里只是兜底
        self.criteria_changed.emit(self.criteria)


# ---------------------------------------------------------------- 雷达信息
class InfoPanel(QWidget):
    """雷达信息:波形参数、推导量、当前帧、选中检测、选中航迹与点云里点中的历史点的详情。"""

    def __init__(self, state: ViewState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.state = state
        self.cfg = None
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["项目", "值"])
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.addWidget(self.tree)
        self.groups: dict[str, QTreeWidgetItem] = {}
        for name in (
            "波形与阵列",
            "推导量",
            "当前帧",
            "选中检测",
            "选中航迹",
            "选中累积点",
        ):
            group = QTreeWidgetItem([name, ""])
            self.tree.addTopLevelItem(group)
            group.setExpanded(True)
            self.groups[name] = group
        state.source_changed.connect(self.show_source)
        state.frame_changed.connect(self.show_frame)
        state.selection_changed.connect(self.show_selection)
        state.track_changed.connect(self.show_track)
        state.history_pick_changed.connect(self.show_history_pick)

    def set_config(self, config) -> None:
        self.cfg = config

    def _fill(self, group: str, rows: list[tuple[str, str]]) -> None:
        item = self.groups[group]
        item.takeChildren()
        for key, value in rows:
            item.addChild(QTreeWidgetItem([key, value]))

    def show_source(self, radar) -> None:
        cfg = self.cfg
        wave, arr = cfg.xWave, cfg.xArray
        ddm = radar.modulation == "ddm"
        subbands = max(int(wave.ulNumSubbands), 1)  # TDM 子带未用,按 1
        rows = [
            ("板", radar.profile or "—"),
            ("采集", f"{radar.capture}({radar.n_frames} 帧)"),
            ("调制", "DDM(多普勒分复用)" if ddm else "TDM(时分复用)"),
            ("TX × RX", f"{radar.n_tx} × {radar.n_rx}"),
            ("虚拟通道", f"{radar.n_virtual}(网格 {arr.ulGridRows}×{arr.ulGridCols})"),
            ("每 chirp 采样", str(radar.n_range)),
            (
                "有效距离门",
                (
                    f"{radar.gates}(实采样只有前半有效)"
                    if radar.gates != radar.n_range
                    else str(radar.gates)
                ),
            ),
            ("每帧 chirp", str(int(wave.ulNumChirpsPerFrame))),
            ("多普勒 chirp", str(radar.n_doppler)),
        ]
        if ddm:
            rows.append(("子带", str(int(wave.ulNumSubbands))))
        rows += [
            ("帧周期", f"{radar.frame_period_s * 1e3:g} ms"),
            ("测角变体", variant_name(cfg.eDoa)),
        ]
        self._fill("波形与阵列", rows)
        v_max = subbands * radar.n_doppler / 2 * radar.doppler_step_mps
        self._fill(
            "推导量",
            [
                ("距离分辨率 ΔR", f"{radar.range_step_m:.4f} m"),
                (
                    "最大距离(有效门数 × ΔR)",
                    f"{radar.gates * radar.range_step_m:.1f} m",
                ),
                ("速度分辨率 ΔV", f"{radar.doppler_step_mps:.4f} m/s"),
                ("最大不模糊速度", f"±{v_max:.2f} m/s(子带 × N_dop / 2 × ΔV)"),
            ],
        )
        if radar.stepped:
            self._fill_stepped(radar)
        for name in ("当前帧", "选中检测", "选中航迹"):
            self._fill(name, [])

    def _fill_stepped(self, radar) -> None:
        """步进频波形的细距离量(只读 RadarInfo 里取自 profile 的 S_f 与 N)。"""
        from ..diagnostics.slices import fine_resolution_m

        item = self.groups["推导量"]
        for key, value in (
            (
                "细距离步长 |S_f|",
                f"{abs(radar.fine_range_step_m):.4f} m / 全局慢时间 bin",
            ),
            ("细距离分辨率", f"{fine_resolution_m(radar):.3f} m"),
            (
                "细尺模糊周期 |S_f|·N",
                f"{abs(radar.fine_range_step_m) * radar.n_slow:.1f} m",
            ),
        ):
            item.addChild(QTreeWidgetItem([key, value]))

    def show_frame(self, frame) -> None:
        rows = [
            ("帧序号", f"{frame.index + 1} / {frame.radar.n_frames}"),
            ("帧号 usFrameIdx", str(frame.fi)),
            ("检测数", str(len(frame.detections))),
            ("航迹数", str(len(frame.tracks))),
            ("航迹起点", f"帧 {frame.track_epoch + 1}"),
        ]
        self._fill("当前帧", rows)

    def show_selection(self, selection: Selection) -> None:
        frame = self.state.frame
        if frame is None:
            return
        radar = frame.radar
        r, d = selection.range_bin, selection.doppler_bin
        rd = frame.result.rd_db
        rows = [("距离 bin", str(r)), ("多普勒 bin", str(d))]
        if rd.shape == (radar.n_range, radar.n_doppler):
            rows.append(
                ("RD 功率", f"{rd[r, rd_column(d, radar.n_doppler)]:.1f} dB(相对峰值)")
            )
        k = selection.detection
        if k is not None:
            det = frame.detections
            x, y, z = (float(v) for v in det.xyz[k])
            rows += [
                ("检测序号", f"#{k}"),
                ("距离", f"{det.range_m[k]:.3f} m"),
                ("径向速度(远离为正)", f"{det.velocity_mps[k]:+.3f} m/s"),
                ("配对", det.pair_label(k)),
                ("SNR / 噪声", f"{det.snr_db[k]:.1f} / {det.noise_db[k]:.1f} dB"),
                (
                    "方位 / 俯仰",
                    f"{det.azimuth_deg[k]:+.2f}° / {det.elevation_deg[k]:+.2f}°",
                ),
                ("位置 x / y / z", f"{x:+.2f} / {y:.2f} / {z:+.2f} m"),
            ]
        self._fill("选中检测", rows)

    def show_history_pick(self, pick) -> None:
        """三维点云余辉 / 全量累积里点中的历史检测(不是当前帧,取值时已拷出)。"""
        rows = []
        if pick is not None:
            x, y, z = pick.xyz
            frame = self.state.frame
            ago = (
                f"(当前帧之前 {frame.index - pick.index} 帧)"
                if frame is not None
                else ""
            )
            rows = [
                ("采集帧号", f"{pick.frame_id}{ago}"),
                ("检测序号", f"#{pick.row}"),
                ("距离 / 多普勒 bin", f"{pick.range_bin} / {pick.doppler_bin}"),
                ("距离", f"{pick.range_m:.3f} m"),
                ("径向速度(远离为正)", f"{pick.velocity_mps:+.3f} m/s"),
                ("配对", pick.pair_label),
                ("SNR / 噪声", f"{pick.snr_db:.1f} / {pick.noise_db:.1f} dB"),
                (
                    "方位 / 俯仰",
                    f"{pick.azimuth_deg:+.2f}° / {pick.elevation_deg:+.2f}°",
                ),
                ("位置 x / y / z", f"{x:+.2f} / {y:.2f} / {z:+.2f} m"),
            ]
        self._fill("选中累积点", rows)

    def show_track(self, track_id) -> None:
        frame = self.state.frame
        rows = []
        if frame is not None and track_id is not None:
            hit = np.flatnonzero(frame.tracks.track_id == track_id)
            if len(hit):
                i = int(hit[0])
                t = frame.tracks
                x, y, z = (float(v) for v in t.xyz[i])
                rows = [
                    ("航迹 id", str(track_id)),
                    ("位置 x / y / z", f"{x:+.2f} / {y:.2f} / {z:+.2f} m"),
                    ("径向速度(远离为正)", f"{t.v_r[i]:+.3f} m/s"),
                    ("速度模 |v|", f"{t.speed[i]:.3f} m/s"),
                ]
        self._fill("选中航迹", rows)


# ---------------------------------------------------------------- 处理耗时
class PipelinePanel(QWidget):
    """处理耗时:各级耗时条(占帧周期的比例)、合计与余量;点某一级跳到对应的屏。"""

    stage_activated = Signal(str)  # 级 id(TIMING_STAGES 的键)

    def __init__(self, state: ViewState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["处理级", "耗时 ms", "占帧周期"])
        self.tree.setRootIsDecorated(False)
        for stage, name in TIMING_STAGES.items():
            item = QTreeWidgetItem([name, "—", ""])
            item.setData(0, _STAGE_ROLE, stage)
            item.setForeground(2, QColor(theme.SIGNAL))
            self.tree.addTopLevelItem(item)
        self.tree.itemClicked.connect(
            lambda item, _col: self.stage_activated.emit(item.data(0, _STAGE_ROLE))
        )
        self.summary = QLabel("—")
        self.summary.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.addWidget(self.tree, 1)
        layout.addWidget(self.summary)
        state.frame_changed.connect(self.show_frame)

    def show_frame(self, frame) -> None:
        period_ms = frame.radar.frame_period_s * 1e3
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            ms = frame.timings_ms.get(item.data(0, _STAGE_ROLE))
            item.setText(1, "—" if ms is None else f"{ms:.1f}")
            item.setText(2, "" if ms is None else _text_bar(ms / period_ms))
        total = diagnostics.processing_ms(frame.timings_ms)
        overhead = [
            f"{name} {frame.timings_ms[key]:.1f} ms"
            for key, name in (("capture", "取中间量"), ("diagnostics", "诊断"))
            if key in frame.timings_ms
        ]
        extra = f"<br>诊断开销(不计入合计):{' · '.join(overhead)}" if overhead else ""
        self.summary.setText(
            f"合计 {total:.1f} ms / 帧周期 {period_ms:.0f} ms,"
            f"余量 {period_ms - total:.1f} ms{extra}<br>"
            f"检测 {len(frame.detections)} 个 · 航迹 {len(frame.tracks)} 条"
        )


# ---------------------------------------------------------------- 滑出栏的页签
class Drawer(QTabWidget):
    """检查器的五页:溯源 / 参数 / 判据 / 雷达 / 耗时(类名沿用「抽屉」)。

    「溯源」页由窗口建好传进来(它要各屏的小图);其余四页是原参数抽屉的四页。窗口直接连
    params.config_changed、criteria.criteria_changed 与 timings.stage_activated。
    页签右端的「×」发 `close_requested`(Esc 同效,在窗口里)。页签在 300 px 宽下不换行。"""

    TRACE, PARAMS, CRITERIA, INFO, TIMINGS = range(5)
    NAMES = ("溯源", "参数", "判据", "雷达", "耗时")
    close_requested = Signal()

    def __init__(self, state: ViewState, trace: QWidget, parent: QWidget | None = None):
        super().__init__(parent)
        self.trace = trace
        self.params = ParameterPanel()
        self.criteria = CriteriaPanel()
        self.info = InfoPanel(state)
        self.timings = PipelinePanel(state)
        for page, name in zip(
            (trace, self.params, self.criteria, self.info, self.timings), self.NAMES
        ):
            self.addTab(page, name)
        close = QPushButton("×")
        close.setToolTip("关闭  Esc")
        close.setCursor(Qt.CursorShape.PointingHandCursor)
        close.setStyleSheet(
            f"QPushButton {{ border: 0; background: transparent; color: {theme.DIM}; "
            f"font-size: 18px; min-width: {theme.MIN_HIT}px; min-height: {theme.MIN_HIT}px; }}"
            f"QPushButton:hover {{ color: {theme.TEXT}; }}"
        )
        close.clicked.connect(self.close_requested.emit)
        self.setCornerWidget(close, Qt.Corner.TopRightCorner)
        self.tabBar().setUsesScrollButtons(
            False
        )  # 300 px 里五个页签排得下,不要出滚动钮
        self.tabBar().setExpanding(False)
        self.setStyleSheet(
            f"QTabWidget::pane {{ border: 0; border-top: 1px solid {theme.LINE_SOFT}; }}"
            "QTabBar { border: 0; }"
            "QTabBar::tab { background: transparent; color: "
            f"{theme.DIM}; padding: 5px 9px; border: 0; min-height: {theme.MIN_HIT - 12}px; "
            f"font-size: {theme.FONT_BODY}px; border-bottom: 2px solid transparent; }}"
            f"QTabBar::tab:selected {{ color: {theme.TEXT}; "
            f"border-bottom: 2px solid {theme.ACCENT}; }}"
            f"QTabBar::tab:hover {{ color: {theme.TEXT}; }}"
        )

    def add_param_tool(self, widget: QWidget) -> None:
        """参数页顶部放一个低频开关(落盘),不占顶栏。"""
        self.params.layout().insertWidget(0, widget)

    def set_config(self, config) -> None:
        """换会话(先于 ViewState.set_source 调用:信息页的波形组要读它)。"""
        self.params.set_config(config)
        self.info.set_config(config)

    def set_criteria(self, criteria, default) -> None:
        """换会话:判据页显示该会话的判据,↺ 复位到 default(该板的标定默认值)。"""
        self.criteria.set_criteria(criteria, default)
