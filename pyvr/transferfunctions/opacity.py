"""
Opacity Transfer Function implementation for PyVR.

This module provides the OpacityTransferFunction class which maps scalar 
values (e.g., density) to opacity (alpha) values for volume rendering.
Supports piecewise linear transfer functions and various preset patterns.
"""

import numpy as np
from typing import List, Tuple, Optional, Any, Union

from .base import BaseTransferFunction, InvalidControlPointError, validate_control_points_format


class OpacityTransferFunction(BaseTransferFunction):
    """
    Maps scalar values (e.g., density) to opacity (alpha) values for volume rendering.
    
    Supports piecewise linear transfer functions and can output a 1D texture.
    The opacity values typically range from 0.0 (transparent) to 1.0 (opaque).
    
    Examples:
        # Linear ramp from transparent to opaque
        otf = OpacityTransferFunction.linear(0.0, 1.0)
        
        # Step function
        otf = OpacityTransferFunction.one_step(step=0.5, low=0.0, high=0.8)
        
        # Peaks at specific densities
        otf = OpacityTransferFunction.peaks([0.2, 0.7], opacity=0.9)
        
        # Custom control points
        otf = OpacityTransferFunction([(0.0, 0.0), (0.3, 0.1), (0.8, 0.9), (1.0, 0.5)])
    """
    
    def __init__(self, control_points: Optional[List[Tuple[float, float]]] = None, lut_size: int = 256):
        """
        Initialize opacity transfer function.
        
        Args:
            control_points: List of (scalar, opacity) tuples, sorted by scalar.
                          Example: [(0.0, 0.0), (0.2, 0.1), (0.5, 0.8), (1.0, 1.0)]
            lut_size: Size of the lookup table and texture.
        """
        super().__init__(control_points, lut_size)
    
    def _get_default_control_points(self) -> List[Tuple[float, float]]:
        """Return default control points: linear ramp from 0 to 1."""
        return [(0.0, 0.0), (1.0, 1.0)]
    
    def _validate_control_points(self) -> None:
        """Validate that control points are appropriate for opacity transfer function."""
        validate_control_points_format(self.control_points, expected_value_length=1)
        self._validate_scalar_range(0.0, 1.0)
        
        # Validate opacity values are in reasonable range
        for i, (scalar, opacity) in enumerate(self.control_points):
            if not (0.0 <= opacity <= 1.0):
                raise InvalidControlPointError(
                    f"Control point {i} opacity {opacity} outside valid range [0.0, 1.0]"
                )
    
    def _get_texture_channels(self) -> int:
        """Opacity textures have 1 channel."""
        return 1
    
    def _create_texture_legacy(self, ctx: Any, size: Optional[int]) -> Any:
        """Create texture using legacy ModernGL context method."""
        import moderngl
        
        effective_size = size or self.lut_size
        lut = self.to_lut(effective_size)
        data = lut.reshape((effective_size, 1)).astype(np.float32)
        tex = ctx.texture((effective_size, 1), 1, data.tobytes(), dtype="f4")
        tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        tex.repeat_x = False
        tex.repeat_y = False
        return tex
    
    @classmethod
    def linear(cls, low: float = 0.0, high: float = 1.0, lut_size: int = 256) -> 'OpacityTransferFunction':
        """
        Create a linear opacity ramp from low to high.
        
        Args:
            low: Opacity value at scalar = 0.0
            high: Opacity value at scalar = 1.0  
            lut_size: Size of the lookup table
            
        Returns:
            OpacityTransferFunction with linear mapping
        """
        return cls([(0.0, low), (1.0, high)], lut_size=lut_size)
    
    @classmethod
    def one_step(cls, step: float = 0.5, low: float = 0.0, high: float = 1.0, 
                lut_size: int = 256) -> 'OpacityTransferFunction':
        """
        Create a step function: low opacity up to 'step', then high opacity.
        
        Args:
            step: Scalar value where the step occurs (between 0 and 1)
            low: Opacity value before the step
            high: Opacity value after the step
            lut_size: Size of the lookup table
            
        Returns:
            OpacityTransferFunction with step mapping
            
        Example:
            step=0.5, low=0.0, high=1.0 gives:
            [(0.0, 0.0), (0.5, 0.0), (0.5+ε, 1.0), (1.0, 1.0)]
        """
        epsilon = 1e-12  # Small offset to create sharp transition
        control_points = [
            (0.0, low), 
            (step, low), 
            (step + epsilon, high), 
            (1.0, high)
        ]
        return cls(control_points, lut_size=lut_size)
    
    @classmethod
    def peaks(cls, peaks: List[float], opacity: float = 1.0, eps: float = 0.02, 
             lut_size: int = 256, base: float = 0.0) -> 'OpacityTransferFunction':
        """
        Create an opacity transfer function with one or more narrow peaks.
        
        Useful for highlighting specific density ranges in volume data.
        
        Args:
            peaks: List of scalar positions where peaks occur (between 0 and 1)
            opacity: Opacity value at the peak(s)
            eps: Half-width of each peak (controls sharpness)
            lut_size: Size of the lookup table
            base: Base opacity outside peaks
            
        Returns:
            OpacityTransferFunction with peak-based mapping
            
        Example:
            peaks=[0.3, 0.7], opacity=0.9, eps=0.05, base=0.1
            Creates peaks at densities 0.3 and 0.7 with opacity 0.9,
            base opacity of 0.1 elsewhere.
        """
        if not peaks:
            raise ValueError("At least one peak position must be specified")
        
        # Validate peak positions
        for peak in peaks:
            if not (0.0 <= peak <= 1.0):
                raise ValueError(f"Peak position {peak} must be between 0 and 1")
        
        control_points = [(0.0, base)]
        
        for peak in sorted(peaks):
            left = max(0.0, peak - eps)
            right = min(1.0, peak + eps)
            
            # Add left edge of peak if it doesn't overlap with previous points
            if left > control_points[-1][0]:
                control_points.append((left, base))
            
            # Add peak
            control_points.append((peak, opacity))
            
            # Add right edge of peak
            if right > peak:
                control_points.append((right, base))
        
        # Ensure we end at scalar = 1.0
        if control_points[-1][0] < 1.0:
            control_points.append((1.0, base))
        
        return cls(control_points, lut_size=lut_size)
    
    def to_lut(self, size: Optional[int] = None) -> np.ndarray:
        """
        Generate a lookup table (numpy array) for fast mapping.
        
        Uses linear interpolation between control points for smooth transitions.
        
        Args:
            size: Size of the LUT. If None, uses self.lut_size.
            
        Returns:
            1D numpy array of shape (size,) with opacity values
        """
        effective_size = size or self.lut_size
        x = np.linspace(0, 1, effective_size)
        
        # Extract scalar and opacity values for interpolation
        scalars, opacities = zip(*self.control_points)
        
        # Linear interpolation between control points
        lut = np.interp(x, scalars, opacities).astype(np.float32)
        return lut
    
    def apply_to_array(self, scalar_array: np.ndarray) -> np.ndarray:
        """
        Apply the opacity transfer function to an array of scalar values.
        
        Args:
            scalar_array: Input array with scalar values (typically 0-1 range)
            
        Returns:
            Array of same shape with opacity values
        """
        scalars, opacities = zip(*self.control_points)
        return np.interp(scalar_array.flatten(), scalars, opacities).reshape(scalar_array.shape).astype(np.float32)
    
    def __repr__(self) -> str:
        """String representation showing control points and LUT size."""
        return f"OpacityTransferFunction(control_points={self.control_points}, lut_size={self.lut_size})"