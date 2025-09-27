"""PyVR ModernGL Renderer

GPU-accelerated, real-time OpenGL volume rendering using ModernGL.
Provides interactive visualization with camera controls and transfer functions.
"""

# Re-export dataset functions for convenience
from ..datasets import compute_normal_volume, create_sample_volume
from .camera_control import get_camera_pos
from .moderngl_manager import ModernGLManager
from .transfer_functions import ColorTransferFunction, OpacityTransferFunction
from .volume_renderer import VolumeRenderer

__all__ = [
    "ColorTransferFunction",
    "OpacityTransferFunction",
    "VolumeRenderer",
    "ModernGLManager",
    "get_camera_pos",
    "create_sample_volume",
    "compute_normal_volume",
]
