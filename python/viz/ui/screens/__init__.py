"""屏:每级一屏,骨架见 base.Screen。处理链行上的屏顺序即处理链顺序。

新屏只要在 screens/<key>.py 里定义一个 key 相同的 Screen 子类,窗口就会自动发现并启用该标签
(`discover_screens`);没有对应文件的标签置灰。「解耦」只在步进频波形里出现(处理链行按
`RadarInfo.stepped` 显示 / 隐藏它)。
"""

import importlib
import inspect

#: (屏 key, 处理链行上的名字);顺序 = Ctrl+1…(隐藏的级不占序号)
SCREEN_TABS = (
    ("adc", "ADC"),
    ("rangefft", "距离"),
    ("doppler", "多普勒"),
    ("cfar", "CFAR"),
    ("decouple", "解耦"),
    ("doa", "测角"),
    ("track", "跟踪"),
    ("results", "场景"),
)


def discover_screens() -> dict:
    """{key: Screen 子类}:按 SCREEN_TABS 逐个 import screens.<key>,取其中 key 相同的 Screen 子类。
    模块不存在 = 该屏未实施(标签置灰);模块存在但导入出错照常抛出,不吞。"""
    from .base import Screen

    found = {}
    for key, _label in SCREEN_TABS:
        try:
            module = importlib.import_module(f"{__name__}.{key}")
        except ModuleNotFoundError as exc:
            if exc.name == f"{__name__}.{key}":
                continue
            raise
        for _name, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, Screen) and obj is not Screen and obj.key == key:
                found[key] = obj
                break
    return found
