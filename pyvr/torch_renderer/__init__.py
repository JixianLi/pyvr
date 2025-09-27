"""PyVR PyTorch Renderer

Fully vectorized, differentiable volume rendering in PyTorch.
Ideal for research applications and machine learning workflows.
"""

from .camera import Camera
from .transfer_functions import ColorTransferFunction, OpacityTransferFunction
from .volume_renderer import VolumeRenderer

# For backward compatibility, re-export common dataset functions
from ..datasets import create_test_volume, create_medical_phantom

__all__ = [
    'Camera',
    'ColorTransferFunction',
    'OpacityTransferFunction',
    'VolumeRenderer',
    'create_test_volume',
    'create_medical_phantom'
]