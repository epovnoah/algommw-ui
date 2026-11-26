"""双 PRF 单步进频的离线局部聚焦预览；仅供诊断视图读取。"""

import tomllib

import numpy as np

from . import config
from .paths import profile_dir


class MotionPreview:
    """复用距离 FFT，按给定距离/速度候选生成 16 通道相干快照。"""

    def __init__(self, profile, board="afmccmdual"):
        cfg = config.load_profile(profile_dir(board))
        wave = cfg.xAlternateWave if profile else cfg.xWave
        w = tomllib.loads((profile_dir(board) / "profile.toml").read_text())["waveform"]
        # B 帧按 [waveform.alternate] 逐键覆盖(起频、步进符号、idle),与 config 的口径相同。
        if profile:
            w = {**w, **w.get("alternate", {})}
        self.c = 299792458.0
        self.slope = w["freq_slope_mhz_per_us"] * 1e12
        self.step = w["chirp_step_khz"] * 1e3
        self.fs = w["sample_rate_msps"] * 1e6
        self.pri = wave.xChirpInterval
        self.n = np.arange(768)
        self.m = self.n - 767 / 2
        self.carrier = (
            w["start_freq_ghz"] * 1e9
            + self.slope * (w["adc_start_time_us"] * 1e-6 + 512 / self.fs / 2)
            + 767 * self.step / 2
        )
        self.codes = np.exp(-2j * np.pi * np.arange(4)[:, None] * self.n / 6)
        self.norm = np.sum(np.hanning(768)) * np.sum(np.hanning(512)[:-2])

    def snapshot(self, range_fft, distance, velocity, compensate):
        if range_fft.shape != (768, 4, 512):
            raise ValueError("局部补偿仅支持双 PRF 的 768×4×512 距离立方体")
        frequency = 2 / self.c * (self.slope * distance + self.carrier * velocity)
        if compensate:
            frequency += (
                2 / self.c * (self.slope * self.pri + self.step) * velocity * self.m
            )
        frequency = np.broadcast_to(frequency, (768,)) * 512 / self.fs
        bins = np.floor(frequency).astype(int)[:, None] + np.arange(-3, 5)
        delta = bins - frequency[:, None]
        weights = np.sinc(delta) / np.sinc(delta / 512)
        # 复谱先恢复居中 DFT 相位再插值，否则局部幅值看似正确而虚拟阵列相位会错。
        samples = range_fft[
            self.n[:, None, None], np.arange(4)[None, :, None], bins[:, None, :] % 512
        ]
        samples *= np.exp(2j * np.pi * bins[:, None, :] * 511 / 2 / 512)
        symbols = np.sum(samples * weights[:, None, :], axis=2)
        slow = (
            2
            / self.c
            * ((self.step * distance + self.carrier * velocity * self.pri) * self.m)
        )
        if compensate:
            slow += 2 / self.c * self.step * velocity * self.pri * self.m**2
        symbols *= (np.hanning(768) * np.exp(-2j * np.pi * slow))[:, None]
        return (self.codes @ symbols).ravel() / self.norm

    def gain_db(self, range_fft, distance, velocity):
        plain = self.snapshot(range_fft, distance, velocity, False)
        focused = self.snapshot(range_fft, distance, velocity, True)
        old = float(np.vdot(plain, plain).real)
        new = float(np.vdot(focused, focused).real)
        return 10 * np.log10(new / old) if old > 0 and new > 0 else float("nan")
