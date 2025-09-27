"""
PyVR Transfer Functions Module

This module provides transfer function implementations for volume rendering.
Transfer functions define how scalar values in volume data are mapped to 
colors and opacity for visualization.

Classes:
    BaseTransferFunction: Abstract base class for all transfer functions
    ColorTransferFunction: Maps scalar values to RGB colors
    OpacityTransferFunction: Maps scalar values to opacity/alpha values

Exceptions:
    TransferFunctionError: Base exception for transfer function errors
    InvalidControlPointError: Raised when control points are invalid

Examples:
    # Create a color transfer function from colormap
    ctf = ColorTransferFunction.from_colormap('viridis', value_range=(0.2, 0.8))
    
    # Create an opacity transfer function with linear ramp
    otf = OpacityTransferFunction.linear(0.0, 0.8)
    
    # Create opacity peaks at specific densities
    otf = OpacityTransferFunction.with_peaks([0.3, 0.7], opacities=[0.9, 0.9])
"""

__version__ = "0.2.1"

from .base import BaseTransferFunction, TransferFunctionError, InvalidControlPointError
from .color import ColorTransferFunction
from .opacity import OpacityTransferFunction

__all__ = [
    'BaseTransferFunction',
    'ColorTransferFunction', 
    'OpacityTransferFunction',
    'TransferFunctionError',
    'InvalidControlPointError',
]