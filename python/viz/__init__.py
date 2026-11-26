"""Offline 3D visualization toolkit for the algommw radar signal processing pipeline.

Drives the C core in-process via ctypes (python/binding, ADR 0007) and renders
interactive 3D point clouds and tracks (pyqtgraph/OpenGL). Static matplotlib
plotting and the 2D range-doppler viewer have been removed — 3D realtime only.
Deliberately decoupled from the C core: no pybind11 dependency, just ctypes.
"""

__version__ = "0.1.0"
