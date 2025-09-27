"""PyVR ModernGL Renderer

GPU-accelerated, real-time OpenGL volume rendering using ModernGL.
Provides interactive visualization with camera controls and transfer functions.
"""

from .transfer_functions import ColorTransferFunction, OpacityTransferFunction
from .volume_renderer import VolumeRenderer
from .camera_control import get_camera_pos

# For backward compatibility, re-export common dataset functions
from ..datasets import create_sample_volume, compute_normal_volume

__all__ = [
    'ColorTransferFunction',
    'OpacityTransferFunction', 
    'VolumeRenderer',
    'get_camera_pos',
    'create_sample_volume',
    'compute_normal_volume'
]