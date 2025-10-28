"""PyVR: Python Volume Rendering Toolkit

PyVR provides GPU-accelerated 3D volume rendering using OpenGL/ModernGL
for real-time interactive visualization.

Version 0.2.5 introduces Volume class and backend-agnostic renderer architecture.
"""

from . import camera, datasets, lighting, moderngl_renderer, renderer, transferfunctions, volume

__version__ = "0.2.5"
__all__ = [
    "camera",
    "datasets",
    "lighting",
    "moderngl_renderer",
    "renderer",
    "transferfunctions",
    "volume",
]

