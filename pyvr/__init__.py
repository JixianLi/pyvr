"""PyVR: Python Volume Rendering Toolkit

PyVR provides GPU-accelerated 3D volume rendering using OpenGL/ModernGL
for real-time interactive visualization.
"""

from . import datasets, moderngl_renderer

__version__ = "0.1.0"
__all__ = ["datasets", "moderngl_renderer"]
