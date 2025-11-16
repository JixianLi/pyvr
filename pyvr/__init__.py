"""PyVR: Python Volume Rendering Toolkit

PyVR provides GPU-accelerated 3D volume rendering using OpenGL/ModernGL
for real-time interactive visualization.

Version 0.4.0 adds VTK data loader for loading scientific volume data
from .vti files with automatic normalization and aspect ratio preservation.
"""

from . import camera, dataloaders, datasets, interface, lighting, moderngl_renderer, transferfunctions, volume
from .config import RenderConfig

__version__ = "0.4.0"
__all__ = [
    "camera",
    "dataloaders",
    "datasets",
    "interface",
    "lighting",
    "moderngl_renderer",
    "transferfunctions",
    "volume",
    "RenderConfig",
]

