"""PyVR ModernGL Renderer

GPU-accelerated, real-time OpenGL volume rendering using ModernGL.
Provides interactive visualization with camera controls and transfer functions.
"""

from ..camera.control import CameraController
from ..camera.camera import Camera

# Re-export dataset functions for convenience
from ..datasets import compute_normal_volume, create_sample_volume
from ..transferfunctions.color import ColorTransferFunction
from ..transferfunctions.opacity import OpacityTransferFunction
from .manager import ModernGLManager
from .renderer import ModernGLVolumeRenderer, VolumeRenderer  # VolumeRenderer is alias

__all__ = [
    "ColorTransferFunction",
    "OpacityTransferFunction",
    "ModernGLVolumeRenderer",
    "VolumeRenderer",  # For backward compatibility
    "ModernGLManager",
    "CameraController",
    "Camera",
    "create_sample_volume",
    "compute_normal_volume",
]
