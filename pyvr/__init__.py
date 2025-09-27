"""PyVR: Python Volume Rendering Toolkit

PyVR provides multiple backends for 3D volume rendering including 
ModernGL (OpenGL) for real-time visualization and PyTorch for 
differentiable and research-oriented workflows.
"""

from . import datasets
from . import moderngl_renderer
from . import torch_renderer

__version__ = "0.1.0"
__all__ = ["datasets", "moderngl_renderer", "torch_renderer"]