# algommw Python UI

从 algommw 提取的完整 UI 源码。安装 Python 依赖后,直接运行,无需数据、C 编译器或项目动态库。不包含 `.app`、C/C++ 源码及预编译程序。

```bash
cd algommw-python-ui
python -m pip install -r requirements.txt
python main.py
```

需要独立环境时:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py
```

需要 Python 3.11 以上;本次实际验证环境为 Python 3.14.7、PySide6 6.11.0、pyqtgraph 0.14.0、NumPy 2.3.4、PyOpenGL 3.1.10。

## 界面

包含 ADC、距离、多普勒、CFAR、解耦、测角、跟踪、场景八个页面,以及参数、溯源、判据、雷达信息、耗时检查器,配置弹层和命令面板。

点击左上角雷达名称,选择配置,再点「预览界面」即可切换。支持全部六套配置;SR61 和 AFM761 按原设计隐藏解耦页面。

启动不读取采集数据,图表保持空态,可浏览页面、展开列表、切换三维视角、编辑预览参数。播放、单步和落盘禁用。该仓库只提供 UI,不包含原项目的信号处理后端。

三维显示使用 PyOpenGL;无法建立 OpenGL 窗口的平台会使用原有二维后备视图。

## 代码

- `main.py`:唯一启动入口;附带 `--smoke-test` 无数据自检。
- `python/viz/ui/`:全部界面、绘图与控件,`preview.py` 提供无数据窗口。
- `python/viz/frame.py`:视图数据类型。
- `python/viz/diagnostics/`、`python/viz/world/`:界面依赖的诊断与坐标显示逻辑。
- `python/binding/`:纯 Python 配置解析、枚举及数据结构;不加载任何项目动态库。`ctypes` 仅作为标准库数据容器保留。
- `python/binding/defaults.json`:UI 缺省参数快照,由 `profiles` 中的显式配置覆盖。
- `profiles/`:六套雷达 TOML 配置与阵列 CSV。

自检命令(正常桌面):

```bash
python main.py --smoke-test
```

无显示会话时可检查二维界面:

```bash
QT_QPA_PLATFORM=offscreen python main.py --smoke-test
```

本目录独立于原项目,不读写原项目设置和数据。原仓库未修改。
