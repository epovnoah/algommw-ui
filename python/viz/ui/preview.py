"""无数据 UI 预览:复用完整工作台,只载入雷达配置,不创建采集处理器。"""

from pathlib import Path

from PySide6.QtWidgets import QLabel

from binding import config as bind_config
from binding import dtypes
from binding.config import valid_range_gates

from .. import boards, diagnostics
from ..frame import RadarInfo
from .params import parameter_error
from .popup import ConfigPopup
from .window import MainWindow


class PreviewPopup(ConfigPopup):
    """保留配置弹层的布局,用 profile 校验替代采集预检。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        for label in self.findChildren(QLabel):
            if label.objectName() == "cfgHint":
                label.setText("无数据预览:选择雷达即可查看对应页面,无需采集文件。")
        self.edit_dir.setReadOnly(True)
        self.edit_dir.setPlaceholderText("无需加载数据")
        self.btn_browse.setEnabled(False)
        self.combo_recent.setEnabled(False)
        self.btn_load.setText("预览界面")

    def _preflight(self):
        board = self._board()
        if board is None:
            return None
        if board.error:
            return self._bad(board.error)
        try:
            config = bind_config.load_profile(board.profile_dir)
        except Exception as exc:
            return self._bad(str(exc))
        return boards.Preflight(
            True, "", 0, float(config.xWave.xFrameTime), config, None
        )

    def _refresh(self):
        result = self._preflight()
        if result is None or not result.ok:
            self._set_state(result.error if result else "没有可预览的配置", "error")
        else:
            self._set_state("配置可用 · 无数据预览 · 不读取采集文件", "ok")

    def _on_load(self):
        result = self._preflight()
        if result is not None and result.ok:
            self.loadRequested.emit(self._profile_dir(), "")
            self.close()


class PreviewWindow(MainWindow):
    """不向播放器交付处理器:播放/单步自动禁用,各页面仍可交互。"""

    def __init__(self, profile_dir=None):
        super().__init__(catalog=boards.scan_boards(boards.BOARDS_ROOT))
        self.reload_requested.connect(self.set_profile)
        self.set_saving(False, locked=True)
        self.set_profile(profile_dir or boards.BOARDS_ROOT / "afmccmdual")
        self.toast.clearMessage()
        self.show_notice("无数据预览 · 可切换处理页面、雷达配置与检查器", "ok")

    def set_profile(self, profile_dir, _capture=""):
        profile_dir = Path(profile_dir)
        config = bind_config.load_profile(profile_dir)
        ingest_spec = bind_config.load_ingest_spec(profile_dir)
        wave, fov = config.xWave, config.xDoa.xFov
        radar = RadarInfo(
            profile=profile_dir.name,
            modulation="ddm" if wave.eModulation == dtypes.E_MOD_DDM else "tdm",
            n_tx=int(wave.ulNumTx),
            n_rx=int(wave.ulNumRx),
            n_virtual=int(config.xArray.ulNumVirtual),
            n_range=int(wave.ulNumAdcSamples),
            n_doppler=int(wave.ulNumDopplerChirps),
            range_step_m=float(wave.xRangeStep),
            doppler_step_mps=float(wave.xDopplerStep),
            frame_period_s=float(wave.xFrameTime),
            n_frames=0,
            capture="无数据预览",
            fov_azimuth_deg=(
                (float(fov.xAzMinDeg), float(fov.xAzMaxDeg)) if fov.bEnable else None
            ),
            n_range_valid=valid_range_gates(wave, ingest_spec),
            fine_range_step_m=float(wave.xFineRangeStep),
            n_slow=int(wave.ulNumChirpsPerFrame) if wave.xFineRangeStep else 0,
        )
        self.config = config
        self.criteria = diagnostics.criteria_for(radar.profile)
        self.state.set_config(config)
        self.drawer.set_config(config)
        self.drawer.set_criteria(self.criteria, self.criteria)
        self.state.set_source(radar)
        self.trace.set_stepped(radar.stepped)
        self._assign_shortcuts()
        if self.current_screen not in self._visible_screens():
            self.show_screen("results")
        self.slider.reset(0)
        self.topbar.set_source(
            radar.profile,
            "步进频" if radar.stepped else radar.modulation.upper(),
            radar.capture,
            0,
        )
        self.topbar.set_budget(None, None)
        self.setWindowTitle(f"algommw UI 无数据预览 — {radar.profile}")
        self.drawer.params.set_status("预览模式 · 参数可编辑,不执行数据处理")
        self._sync_controls()

    def open_popup(self):
        if self._popup is None:
            self._popup = PreviewPopup(self)
            self._popup.loadRequested.connect(self.reload_requested.emit)
        super().open_popup()

    def _on_config_changed(self, config):
        error = parameter_error(config)
        if error:
            self.drawer.params.set_status(error, error=True)
            return
        self.config = config
        self.state.set_config(config)
        self.drawer.info.set_config(config)
        self.drawer.params.set_status("预览参数已更新 · 未加载数据")

    def _on_criteria_changed(self, criteria):
        self.criteria = criteria
        self.toast.showMessage("预览判据已更新 · 未加载数据")

    def _on_motion_changed(self, enabled):
        self.toast.showMessage(f"走动补偿预览:{'开' if enabled else '关'} · 未加载数据")
