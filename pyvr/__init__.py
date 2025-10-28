"""PyVR: Python Volume Rendering Toolkit

PyVR provides GPU-accelerated 3D volume rendering using OpenGL/ModernGL
for real-time interactive visualization.

Version 0.2.3 introduces pipeline-aligned Camera system with matrix generation.
"""

from . import camera, datasets, moderngl_renderer, transferfunctions

__version__ = "0.2.3"
__all__ = ["camera", "datasets", "moderngl_renderer", "transferfunctions"]
