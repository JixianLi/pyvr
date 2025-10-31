"""PyVR: Python Volume Rendering Toolkit

PyVR provides GPU-accelerated 3D volume rendering using OpenGL/ModernGL
for real-time interactive visualization.

Version 0.3.0 adds interactive matplotlib-based interface with real-time
transfer function editing and camera controls.
"""

from . import camera, datasets, interface, lighting, moderngl_renderer, transferfunctions, volume
from .config import RenderConfig

__version__ = "0.3.0"
__all__ = [
    "camera",
    "datasets",
    "interface",
    "lighting",
    "moderngl_renderer",
    "transferfunctions",
    "volume",
    "RenderConfig",
]

