"""PyVR: Python Volume Rendering Toolkit

PyVR provides GPU-accelerated 3D volume rendering using OpenGL/ModernGL
for real-time interactive visualization.

Version 0.2.7 removes unused abstract base renderer for simpler architecture.
"""

from . import camera, datasets, lighting, moderngl_renderer, transferfunctions, volume
from .config import RenderConfig

__version__ = "0.2.7"
__all__ = [
    "camera",
    "datasets",
    "lighting",
    "moderngl_renderer",
    "transferfunctions",
    "volume",
    "RenderConfig",
]

