"""PyVR: Python Volume Rendering Toolkit

PyVR provides GPU-accelerated 3D volume rendering using OpenGL/ModernGL
for real-time interactive visualization.

Version 0.3.4 reworks examples with clean flat structure and comprehensive
inline documentation for improved usability.
"""

from . import camera, datasets, interface, lighting, moderngl_renderer, transferfunctions, volume
from .config import RenderConfig

__version__ = "0.3.4"
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

