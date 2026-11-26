"""拉取式逐帧播放器:后台线程处理 → 主线程显示完 → 再请求下一帧(照 ccm `gui/player.py`)。

下一帧只在上一帧显示完之后才请求,处理慢时只会变慢,不会丢帧或堆积;单步、跳转、回退都只是
请求一个帧号。播放速度:speed=None 时尽快逐帧;否则按「帧周期 / speed」的节拍发出请求
(节拍对齐请求时刻,所以平均帧间隔等于该周期),处理跟不上节拍时立即请求下一帧。

与 ccm 的差别来自跟踪状态是 track.c 的 file-static、只能顺序往前算:
- `processor.process(k)` 返回 list[ProcessedFrame](顺序往前;回退从帧 0 重算),worker 发回
  整张列表,主线程把**每一帧**放进缓存、只显示最后一帧,并经 `frames_processed` 交给落盘;
- 缓存:主线程 OrderedDict[帧序号 → ProcessedFrame],上限 2000 帧(超出丢最早放进去的);
  跳转 / 播放下一帧先查缓存,命中就直接显示、不发请求(播放节拍照常);
- 缓存分两层:轻量部分(点、航迹、指标)按上面的 2000 帧;重的中间量 `products` 按内存
  预算 PRODUCTS_BUDGET_BYTES 只留最近的若干帧 —— 超预算时从最早放进去的帧起把 products
  置 None(轻量部分留着),最新放进去的那一帧不丢;
- 换处理器(换会话 / 改参)= 代号加 1 并清空缓存;换之前发出、换之后才回来的结果直接丢弃。
处理器随请求传入后台线程,只有那一个线程调它(单线程 worker 保证两个处理器不会交错处理)。
"""

import dataclasses
import threading
import time
from collections import OrderedDict

import numpy as np
from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal, Slot

CACHE_FRAMES = 2000
#: 缓存里全部帧 products 的字节上限(numpy nbytes 递归累计)
PRODUCTS_BUDGET_BYTES = 512 * 1024 * 1024


def products_nbytes(obj) -> int:
    """products(dict / list / tuple 嵌套)里全部 numpy 数组的 nbytes 之和。"""
    if isinstance(obj, np.ndarray):
        return obj.nbytes
    if isinstance(obj, dict):
        return sum(products_nbytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(products_nbytes(v) for v in obj)
    return 0


def trim_products(cache: "OrderedDict[int, object]", budget: int) -> None:
    """超预算时从最早放进缓存的帧起把 products 置 None(`dataclasses.replace`,帧本身与
    轻量部分留在缓存里、位置不变),直到总量不超过 budget。最新放进去的那一帧不丢。"""
    if not cache:
        return
    newest = next(reversed(cache))
    sizes = [
        (k, products_nbytes(f.products))
        for k, f in cache.items()
        if f.products is not None
    ]
    total = sum(n for _, n in sizes)
    for k, n in sizes:
        if total <= budget:
            break
        if k == newest:
            continue
        cache[k] = dataclasses.replace(cache[k], products=None)
        total -= n


class _Worker(QObject):
    """住在后台线程里,只负责处理被请求的那一帧。处理器随请求传入,不与主线程共享可变状态。"""

    done = Signal(int, object)  # 代号, list[ProcessedFrame]
    failed = Signal(int, int, str)  # 代号, 帧号, 错误信息

    def __init__(self, idle: threading.Event) -> None:
        super().__init__()
        self.idle = idle

    @Slot(object, int, int)
    def process(self, processor, generation: int, index: int) -> None:
        try:
            frames = processor.process(index)
        except Exception as exc:  # 线程里的异常不会自己冒泡到界面
            self.idle.set()
            self.failed.emit(generation, index, f"{type(exc).__name__}: {exc}")
            return
        self.idle.set()
        self.done.emit(generation, frames)


class FramePlayer(QObject):
    #: ProcessedFrame;连接的槽在主线程同步执行,返回即视为「已显示」
    frame_shown = Signal(object)
    #: list[ProcessedFrame]:每批新处理出的帧,按顺序(缓存命中的回看不发),供落盘
    frames_processed = Signal(object)
    failed = Signal(int, str)  # 帧号, 错误信息;出错时自动暂停
    state_changed = Signal()  # 播放/暂停、忙闲、当前帧变化

    _request = Signal(object, int, int)  # 处理器, 代号, 帧号

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.processor = None
        self.current = -1  # 最近一次显示的帧号
        self.playing = False
        self.busy = False  # 有一帧正在后台处理(结果尚未回到主线程)
        self.speed: float | None = None  # None:尽快逐帧;1.0:按帧周期实时
        self._pending: int | None = None  # 忙时收到的跳转请求,只保留最后一个
        self._generation = 0
        self._last_issue = 0.0
        self.cache: "OrderedDict[int, object]" = OrderedDict()
        self._criteria = None  # set_criteria 换过的判据(None = 用处理器自带的)
        self._rejudge = None

        self._pace = QTimer(self)
        self._pace.setSingleShot(True)
        # 默认的粗精度定时器会提前约 5% 触发
        self._pace.setTimerType(Qt.TimerType.PreciseTimer)
        self._pace.timeout.connect(self._paced_next)

        #: 后台线程没有在跑处理器时置位(wait_idle 用;主线程发请求前清掉)
        self._idle = threading.Event()
        self._idle.set()
        self._thread = QThread()
        self._worker = _Worker(self._idle)
        self._worker.moveToThread(self._thread)
        self._request.connect(self._worker.process)
        self._worker.done.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._thread.start()

    @property
    def n_frames(self) -> int:
        return len(self.processor) if self.processor is not None else 0

    def set_processor(self, processor, keep_frame: bool = False) -> None:
        """换会话(从第 0 帧开始)或换处理参数(keep_frame=True,重新处理当前帧)。"""
        resume = keep_frame and self.playing
        self.pause()
        self.processor = processor
        self._generation += 1
        self.cache.clear()
        self._criteria = self._rejudge = None  # 新处理器自带判据,旧的补分级不再适用
        target = self.current if keep_frame and self.current >= 0 else 0
        self.current = self.current if keep_frame else -1
        self.seek(target)
        if resume:
            self.play()

    def set_speed(self, speed: float | None) -> None:
        self.speed = speed

    def set_criteria(self, criteria, rejudge) -> None:
        """只换判据(同 ccm 的 rejudge):缓存里的帧重新分级,不重算处理链、不清零航迹。

        `rejudge(diagnostics, criteria)` 由调用方给(viz.diagnostics.rejudge)。调用方同时把
        处理器的判据换掉,之后算的帧直接用新判据;已在后台处理、带旧判据回来的帧在
        `_on_done` 里补分级。当前帧立即按新判据重显。"""
        self._criteria, self._rejudge = criteria, rejudge
        for k, f in list(self.cache.items()):
            self.cache[k] = self._judged(f)
        frame = self.cache.get(self.current)
        if frame is not None:
            self.frame_shown.emit(frame)

    def _judged(self, frame):
        d = frame.diagnostics
        if self._rejudge is None or d is None or d.verdicts.criteria == self._criteria:
            return frame
        return dataclasses.replace(frame, diagnostics=self._rejudge(d, self._criteria))

    def play(self) -> None:
        if self.n_frames == 0:
            return
        if self.current >= self.n_frames - 1:
            self.current = -1  # 播完后再按播放就从头开始
        self.playing = True
        self.state_changed.emit()
        if not self.busy and self._pending is None:
            self._issue(self.current + 1)

    def pause(self) -> None:
        self.playing = False
        self._pace.stop()
        self.state_changed.emit()

    def step(self, delta: int = 1) -> None:
        self.pause()
        self.seek(self.current + delta)

    def seek(self, index: int) -> None:
        if not 0 <= index < self.n_frames:
            return
        if self.busy:
            self._pending = index
        else:
            self._issue(index)

    def wait_idle(self, timeout: float | None = None) -> bool:
        """阻塞到后台线程不再跑处理器(换会话前用:跟踪是 file-static,新旧会话不能交错)。
        已发出的结果仍会回到主线程,由代号丢弃。"""
        return self._idle.wait(timeout)

    def shutdown(self) -> None:
        self.playing = False
        self._pace.stop()
        self._thread.quit()
        self._thread.wait()

    # ---- 内部 ----
    def _issue(self, index: int) -> None:
        self._last_issue = time.perf_counter()
        frame = self.cache.get(index)
        if frame is not None:  # 缓存命中:直接显示,不发请求
            self._show(frame)
            self._next()
            return
        self.busy = True
        self._idle.clear()
        self.state_changed.emit()
        self._request.emit(self.processor, self._generation, index)

    def _show(self, frame) -> None:
        self.current = frame.index
        self.frame_shown.emit(frame)

    def _on_done(self, generation: int, frames: list) -> None:
        self.busy = False
        if generation == self._generation and frames:
            # 这批是新开的一轮(处理器从航迹起点重建了会话):缓存里排在它后面的帧属于
            # 上一轮,航迹递推不连贯,作废 —— 否则往前走会命中旧一轮的帧。
            if frames[0].index == frames[0].track_epoch:
                for k in [k for k in self.cache if k > frames[-1].index]:
                    del self.cache[k]
            frames = [self._judged(f) for f in frames]
            for f in frames:
                self.cache[f.index] = f
                self.cache.move_to_end(f.index)
            while len(self.cache) > CACHE_FRAMES:
                self.cache.popitem(last=False)
            trim_products(self.cache, PRODUCTS_BUDGET_BYTES)
            self.frames_processed.emit(frames)
            self._show(frames[-1])
        self._next()

    def _on_failed(self, generation: int, index: int, message: str) -> None:
        self.busy = False
        if generation == self._generation:
            self.playing = False
            self._pace.stop()
            self.failed.emit(index, message)
        self._next()

    def _next(self) -> None:
        if self._pending is not None:
            index, self._pending = self._pending, None
            self._issue(index)
        elif self.playing and self.current + 1 < self.n_frames:
            if self.speed is None:
                if self.current + 1 in self.cache:
                    self._pace.start(0)  # 缓存命中走定时器,不在这里递归
                else:
                    self._issue(self.current + 1)
            else:
                period_s = self.processor.radar.frame_period_s / self.speed
                wait_s = period_s - (time.perf_counter() - self._last_issue)
                self._pace.start(max(0, int(wait_s * 1000)))
            self.state_changed.emit()
        else:
            self.playing = False
            self.state_changed.emit()

    def _paced_next(self) -> None:
        if self.playing and not self.busy and self.current + 1 < self.n_frames:
            self._issue(self.current + 1)
