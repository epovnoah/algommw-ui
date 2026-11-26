"""独立 UI 的资源位置,均相对于当前源码目录。"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BOARDS_ROOT = PROJECT_ROOT / "profiles"
BOARD_CAPTURES = {}


def profile_dir(board):
    return BOARDS_ROOT / board
