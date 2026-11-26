"""板卡目录(catalog)、载入前预检、UI 记忆 —— 配置区与会话控制器共用的判据。

**为什么单独一个模块**:顶栏配置区要在**选完就先**知道"这块板 + 这个采集目录能不能载"
(失败就地报错,而不是点了载入没反应),会话控制器要真正建会话。两处若各写一份判定,
漂开就会出现"预检说能载、载入却炸"。本模块不依赖 Qt,可单独跑:

    python python/viz/boards.py                  # 列出 profiles/ 下各板与摘要
    python python/viz/boards.py <板名> <采集目录>   # 对该组合做一次预检

判据口径:**能不能载**只看 profile 自身是否合法(load_profile 的键校验、step 与波形
第一性原理的 1% 对照)+ 采集目录能否按该 profile 数出至少一帧。**不去猜"这个采集是不
是这块板录的"** —— 文件大小能数出帧只说明格式自洽,不说明物理对得上;真判据是跑起来
看点数/角度是否合理(点云面板),不在这里假装能判。
"""

import json
import sys
from pathlib import Path
from typing import NamedTuple, Optional

from binding import config as bind_config
from binding import dtypes
from binding.paths import BOARD_CAPTURES, BOARDS_ROOT, PROJECT_ROOT

#: 每块板的默认采集目录取自 binding.paths.BOARD_CAPTURES(本机路径只在那里写一遍)——
#: 换板时配置区自动填它(再退到界面里上次选的那份)。
DEFAULT_PROFILE_DIR = BOARDS_ROOT / "sr61"
DEFAULT_SAVE_DIR = PROJECT_ROOT / "data" / "results"
DEFAULT_CAPTURE_DIR = PROJECT_ROOT / "data"

#: UI 状态:上次用的板与采集目录、每块的采集目录、最近用过的目录。
#: 与 data/results/ 同一棵树 —— 都不入库、都归本机;放仓库里是为了好找好删。
UI_STATE_PATH = PROJECT_ROOT / "data" / "ui_state.json"

#: 「最近用过的采集目录」保留条数(去重后按最近使用排序)
_RECENT_MAX = 6

#: 工作台布局记忆的键(ui/window.py 经 load_ui_state / save_ui_state 读写,值的格式归
#: 窗口管):窗口几何(Qt saveGeometry 的 base64)、当前屏 key、检查器(右栏)是否展开
#: (无记录时按展开)。旧版记过的 "panel" / "splitters" 两个键现在不读,窗口存盘时顺手删掉。
UI_KEY_GEOMETRY = "window_geometry"
UI_KEY_SCREEN = "screen"
UI_KEY_INSPECTOR = "inspector_open"


class BoardEntry(NamedTuple):
    """一块板在配置区下拉里的一行。"""

    name: str  # profile 目录名(EXPECTED_BOARDS 的键,也是记忆键)
    profile_dir: Path
    summary: str  # 例 "TDM 3TX×4RX · 12V 4×4 · ΔR 0.1831 ΔV 0.8565 · 50ms"
    error: str  # 非空 = 这块板载入失败(仍列出,选中就地报错,不藏起来)


class Preflight(NamedTuple):
    """一次「板 + 采集目录」组合的预检结果。error 为空即 ok。"""

    ok: bool
    error: str
    total: int  # 采集总帧数(按文件大小算,不解析采样)
    period_s: float  # 传感器帧周期 = 1x 实时节拍
    config: Optional[object]  # ChainCfg(ok 时非 None)
    ingest_spec: Optional[dict]  # [ingest] 段;None = SR61 默认格式


def summarize(cfg) -> str:
    """一行摘要,全部取自实载的 profile —— 不手写第二份板卡表。"""
    w, a = cfg.xWave, cfg.xArray
    mod = "DDM" if w.eModulation == dtypes.E_MOD_DDM else "TDM"
    return (
        f"{mod} {w.ulNumTx}TX×{w.ulNumRx}RX · {a.ulNumVirtual}V "
        f"{a.ulGridRows}×{a.ulGridCols} · ΔR {w.xRangeStep:.4f} "
        f"ΔV {w.xDopplerStep:.4f} · {float(w.xFrameTime) * 1000:.0f}ms"
    )


def scan_boards(boards_root: Path) -> list[BoardEntry]:
    """扫 <boards_root>/*/profile.toml。**载入失败的板也列出来**(带原因)—— 从下拉里
    悄悄消失,会让人以为是路径写错而不是 profile 有毛病。"""
    root = Path(boards_root)
    entries: list[BoardEntry] = []
    if not root.is_dir():
        return entries
    for d in sorted(p for p in root.iterdir() if (p / "profile.toml").is_file()):
        try:
            summary = summarize(bind_config.load_profile(d))
            entries.append(BoardEntry(d.name, d, summary, ""))
        except Exception as exc:  # noqa: BLE001 -- 逐板隔离:一块坏不该拖垮整个下拉
            entries.append(
                BoardEntry(d.name, d, "载入失败", f"{type(exc).__name__}: {exc}")
            )
    return entries


def preflight(profile_dir, capture_dir):
    """独立 UI 仅预览配置,没有采集读取入口。"""
    return Preflight(False, "本仓库为无数据 UI 预览,不加载采集", 0, 0.0, None, None)


# ------------------------------------------------------------------ UI 记忆
def load_ui_state() -> dict:
    """读状态。文件缺失或损坏一律当空 —— 便利设施不该挡住启动。"""
    try:
        with open(UI_STATE_PATH, "r", encoding="utf-8") as f:
            state = json.load(f)
    except (OSError, ValueError):
        return {}
    return state if isinstance(state, dict) else {}


def save_ui_state(state: dict) -> None:
    try:
        UI_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(UI_STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
            f.write("\n")
    except OSError:
        pass  # 写不了不影响本次会话,下次退回默认


def remember(profile_name: str, capture_dir: Path) -> dict:
    """记下这次选择并落盘:上次的板、每块的采集目录、最近用过的目录。返回新状态。"""
    state = load_ui_state()
    state["profile"] = profile_name
    captures = state.get("captures")
    captures = dict(captures) if isinstance(captures, dict) else {}
    captures[profile_name] = str(capture_dir)
    state["captures"] = captures

    recent = [str(capture_dir)] + [
        p for p in state.get("recent", []) if str(p) != str(capture_dir)
    ]
    state["recent"] = recent[:_RECENT_MAX]
    save_ui_state(state)
    return state


def remembered_profile(state: dict) -> Optional[str]:
    name = state.get("profile")
    return name if isinstance(name, str) and name else None


def remembered_capture(state: dict, profile_name: str) -> Optional[str]:
    captures = state.get("captures")
    if isinstance(captures, dict):
        path = captures.get(profile_name)
        if isinstance(path, str) and path:
            return path
    return None


def recents(state: dict) -> list[str]:
    items = state.get("recent")
    return [p for p in items if isinstance(p, str)] if isinstance(items, list) else []


def default_capture_for(profile_dir: Path) -> Path:
    """这块板的内置采集目录;没登记就退回 SR61 那份(让用户自己去选)。"""
    return BOARD_CAPTURES.get(Path(profile_dir).name, DEFAULT_CAPTURE_DIR)


def resolve_start(
    state: dict,
    profile_dir: Optional[Path] = None,
    capture_dir: Optional[Path] = None,
) -> tuple[Path, Path]:
    """启动用的 (profile 目录, 采集目录):命令行 > 界面记忆 > 内置默认。

    记忆里的板目录不在了(板被删/改名)就当没记过;采集目录不在这里校验 —— 载入时
    的预检会报,启动那一侧据此退回内置默认。
    """
    if profile_dir is None:
        name = remembered_profile(state)
        candidate = BOARDS_ROOT / name if name else None
        if candidate is not None and (candidate / "profile.toml").is_file():
            profile_dir = candidate
        else:
            profile_dir = DEFAULT_PROFILE_DIR
    if capture_dir is None:
        remembered = remembered_capture(state, profile_dir.name)
        capture_dir = (
            Path(remembered) if remembered else default_capture_for(profile_dir)
        )
    return profile_dir, capture_dir


# ------------------------------------------------------------------ 直接跑
def _main(argv: list[str]) -> int:
    boards_root = BOARDS_ROOT
    if len(argv) >= 3:
        profile_dir = Path(argv[1])
        if not profile_dir.is_dir():
            profile_dir = boards_root / argv[1]  # 允许只给板名(profile 目录名)
        pf = preflight(profile_dir, Path(argv[2]))
        if pf.config is not None:
            print(f"profile {profile_dir.name}: {summarize(pf.config)}")
        verdict = pf.error or f"{pf.total} 帧 @ {1.0 / pf.period_s:.1f}Hz"
        print(f"预检 {'通过' if pf.ok else '失败'}: {verdict}")
        return 0 if pf.ok else 1
    for b in scan_boards(boards_root):
        flag = f"  ⚠ {b.error}" if b.error else ""
        print(f"{b.name:12s} {b.summary}{flag}")
    state = load_ui_state()
    print(f"\n上次: {remembered_profile(state) or '—'} · 状态文件 {UI_STATE_PATH}")
    for p in recents(state):
        print(f"  最近 {p}")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
