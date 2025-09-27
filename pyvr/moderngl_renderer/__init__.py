"""PyVR ModernGL Renderer

GPU-accelerated, real-time OpenGL volume rendering using ModernGL.
Provides interactive visualization with camera controls and transfer functions.
"""

# Re-export dataset functions for convenience
from ..datasets import compute_normal_volume, create_sample_volume
from ..camera.control import CameraController
from ..camera.parameters import CameraParameters
from ..transferfunctions.color import ColorTransferFunction
from ..transferfunctions.opacity import OpacityTransferFunction
from .manager import ModernGLManager
from .renderer import VolumeRenderer

__all__ = [
    "ColorTransferFunction",
    "OpacityTransferFunction", 
    "VolumeRenderer",
    "ModernGLManager",
    "CameraController",
    "CameraParameters",
    "create_sample_volume",
    "compute_normal_volume",
]
