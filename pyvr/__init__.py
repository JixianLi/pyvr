"""PyVR: Python Volume Rendering Toolkit

PyVR provides GPU-accelerated 3D volume rendering using OpenGL/ModernGL
for real-time interactive visualization.

Version 0.2.4 introduces pipeline-aligned Light system with configurable lighting.
"""

from . import camera, datasets, lighting, moderngl_renderer, transferfunctions

__version__ = "0.2.4"
__all__ = ["camera", "datasets", "lighting", "moderngl_renderer", "transferfunctions"]

