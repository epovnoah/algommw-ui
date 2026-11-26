#!/usr/bin/env python3
"""UI 包入口:默认无数据启动;--smoke-test 逐配置/逐屏验证。"""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("QT_MAC_WANTS_LAYER", "1")
# 预览不读写用户的真实会话、判据与窗口记忆。
os.environ["ALGOMMW_UI_NO_SETTINGS"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent / "python"))


def audit(event, args):
    if event == "ctypes.dlopen" and args and "algommw" in str(args[0]).lower():
        raise RuntimeError("纯 Python UI 不应加载项目原生库")
    if event == "open" and args and str(args[0]).lower().endswith(".bin"):
        raise RuntimeError("无数据 UI 不应读取采集文件")


sys.addaudithook(audit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--screenshots", type=Path)
    args = parser.parse_args()

    from PySide6.QtCore import QTimer

    from viz.ui.preview import PreviewWindow
    from viz.ui.window import make_application

    errors = []

    def on_error(*exc):
        errors.append("".join(traceback.format_exception(*exc)))
        sys.__excepthook__(*exc)

    sys.excepthook = on_error
    app = make_application([sys.argv[0]])

    window = PreviewWindow()
    window.show()
    result = {"configurations": [], "screens": 0, "errors": errors}

    def verify():
        try:
            from viz.ui.drawer import Drawer
            from viz.ui.screens import SCREEN_TABS

            cloud = window.screens["results"].cloud
            result["scene_view"] = type(cloud).__name__
            if hasattr(cloud, "gl"):
                app.processEvents()
                assert cloud.gl.isValid(), "三维视图 OpenGL 上下文未就绪"
                assert not cloud.gl.grabFramebuffer().isNull(), "三维画布无法渲染"
                result["opengl"] = "通过"
            else:
                result["opengl"] = "离屏平台使用二维后备视图"
            for board in window.catalog:
                assert not board.error, board.error
                window.set_profile(board.profile_dir)
                app.processEvents()
                expected = {key for key, _ in SCREEN_TABS}
                if not window.state.source.stepped:
                    expected.remove("decouple")
                assert set(window._visible_screens()) == expected
                for key in window._visible_screens():
                    window.show_screen(key)
                    app.processEvents()
                    assert window.current_screen == key
                    if args.screenshots:
                        args.screenshots.mkdir(parents=True, exist_ok=True)
                        path = args.screenshots / f"{board.name}-{key}.png"
                        assert window.grab().save(str(path))
                    result["screens"] += 1
                for page in range(len(Drawer.NAMES)):
                    window.show_panel(page)
                    app.processEvents()
                    assert window.drawer.currentIndex() == page
                window.open_popup()
                app.processEvents()
                assert window._popup.btn_load.isEnabled()
                window._popup._on_load()
                app.processEvents()
                window.open_palette()
                app.processEvents()
                window._palette.close()
                # 按一次参数应用路径;不得创建处理器或访问数据。
                window._on_config_changed(window.config)
                app.processEvents()
                assert window.player.processor is None
                assert window.state.frame is None
                assert window.player.n_frames == 0
                assert not window.play_action.isEnabled()
                assert not window.save_box.isEnabled()
                result["configurations"].append(board.name)
            assert not errors, "\n".join(errors)
            print("UI_PREVIEW_OK " + json.dumps(result, ensure_ascii=False), flush=True)
            window.close()
            app.exit(0)
        except BaseException:
            traceback.print_exc()
            window.close()
            app.exit(1)

    if args.smoke_test:
        QTimer.singleShot(100, verify)
    rc = app.exec()
    window.player.shutdown()
    # PySide6 6.11 在解释器收官时可能访问已释放对象,遵循原项目的退出约定。
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc or bool(errors))


if __name__ == "__main__":
    main()
