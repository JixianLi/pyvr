"""
Color Transfer Function implementation for PyVR.

This module provides the ColorTransferFunction class which maps scalar 
values (e.g., density) to RGB color values for volume rendering.
Supports piecewise linear color transfer functions and matplotlib colormap integration.
"""

import numpy as np
from typing import List, Tuple, Optional, Any, Union

from .base import BaseTransferFunction, InvalidControlPointError, validate_control_points_format


class ColorTransferFunction(BaseTransferFunction):
    """
    Maps scalar values (e.g., density) to RGB color values for volume rendering.
    
    Supports piecewise linear color transfer functions and can output a 1D LUT as a 2D texture.
    Color values typically range from 0.0 to 1.0 for each RGB component.
    
    Examples:
        # From colormap string
        ctf = ColorTransferFunction.from_colormap('viridis', value_range=(0.2, 0.8))
        
        # Custom control points
        ctf = ColorTransferFunction([
            (0.0, (0.0, 0.0, 0.0)),  # Black
            (0.5, (1.0, 0.0, 0.0)),  # Red  
            (1.0, (1.0, 1.0, 1.0))   # White
        ])
        
        # Grayscale
        ctf = ColorTransferFunction.grayscale()
    """
    
    def __init__(self, control_points: Optional[List[Tuple[float, Tuple[float, float, float]]]] = None, 
                 lut_size: int = 256):
        """
        Initialize color transfer function.
        
        Args:
            control_points: List of (scalar, (r, g, b)) tuples, sorted by scalar.
                          Example: [(0.0, (0,0,0)), (0.5, (1,0,0)), (1.0, (1,1,1))]
            lut_size: Size of the lookup table and texture.
        """
        super().__init__(control_points, lut_size)
    
    def _get_default_control_points(self) -> List[Tuple[float, Tuple[float, float, float]]]:
        """Return default control points: grayscale ramp from black to white."""
        return [(0.0, (0.0, 0.0, 0.0)), (1.0, (1.0, 1.0, 1.0))]
    
    def _validate_control_points(self) -> None:
        """Validate that control points are appropriate for color transfer function."""
        validate_control_points_format(self.control_points, expected_value_length=3)
        self._validate_scalar_range(0.0, 1.0)
        
        # Validate RGB values are in reasonable range
        for i, (scalar, rgb) in enumerate(self.control_points):
            for j, component in enumerate(rgb):
                if not (0.0 <= component <= 1.0):
                    raise InvalidControlPointError(
                        f"Control point {i} RGB component {j} value {component} outside valid range [0.0, 1.0]"
                    )
    
    def _get_texture_channels(self) -> int:
        """Color textures have 3 channels (RGB)."""
        return 3
    
    @classmethod
    def grayscale(cls, lut_size: int = 256) -> 'ColorTransferFunction':
        """
        Create a grayscale color transfer function from black to white.
        
        Args:
            lut_size: Size of the lookup table
            
        Returns:
            ColorTransferFunction with grayscale mapping
        """
        return cls([(0.0, (0.0, 0.0, 0.0)), (1.0, (1.0, 1.0, 1.0))], lut_size=lut_size)
    
    @classmethod
    def single_color(cls, color: Tuple[float, float, float], lut_size: int = 256) -> 'ColorTransferFunction':
        """
        Create a single-color transfer function that maps all values to one color.
        
        Args:
            color: RGB color tuple (r, g, b) with values between 0 and 1
            lut_size: Size of the lookup table
            
        Returns:
            ColorTransferFunction that maps all scalars to the given color
        """
        return cls([(0.0, color), (1.0, color)], lut_size=lut_size)
    
    @classmethod
    def from_colormap(cls, colormap_name: str, value_range: Tuple[float, float] = (0.0, 1.0), 
                     lut_size: int = 256) -> 'ColorTransferFunction':
        """
        Create a ColorTransferFunction from a matplotlib colormap name.
        
        Args:
            colormap_name: Name of matplotlib colormap (e.g., 'viridis', 'plasma', 'jet')
            value_range: Tuple of (min_value, max_value) to map the colormap to
            lut_size: Size of the lookup table
            
        Returns:
            ColorTransferFunction with colors sampled from the colormap
            
        Example:
            ctf = ColorTransferFunction.from_colormap('viridis', value_range=(0.2, 0.8))
        """
        try:
            import matplotlib
            import matplotlib.pyplot as plt
            cmap = matplotlib.colormaps.get_cmap(colormap_name)
        except ImportError:
            raise ImportError("matplotlib is required for from_colormap method")
        except Exception as e:
            raise ValueError(f"Unknown colormap name '{colormap_name}': {e}")
        
        # Sample colormap
        x = np.linspace(0, 1, lut_size)
        colors = cmap(x)
        
        # Use only RGB, ignore alpha if present
        if colors.shape[1] == 4:
            colors = colors[:, :3]
        
        # Map to the specified value range
        min_val, max_val = value_range
        x_mapped = min_val + x * (max_val - min_val)
        
        control_points = [
            (float(xi), tuple(map(float, rgb))) 
            for xi, rgb in zip(x_mapped, colors)
        ]
        return cls(control_points, lut_size=lut_size)
    
    @classmethod
    def two_color_ramp(cls, color1: Tuple[float, float, float], 
                      color2: Tuple[float, float, float], 
                      lut_size: int = 256) -> 'ColorTransferFunction':
        """
        Create a linear ramp between two colors.
        
        Args:
            color1: Starting RGB color (r, g, b) at scalar = 0
            color2: Ending RGB color (r, g, b) at scalar = 1
            lut_size: Size of the lookup table
            
        Returns:
            ColorTransferFunction with linear color interpolation
        """
        return cls([(0.0, color1), (1.0, color2)], lut_size=lut_size)
    
    def to_lut(self, size: Optional[int] = None) -> np.ndarray:
        """
        Generate a lookup table (numpy array) for fast mapping.
        
        Uses linear interpolation between control points for smooth color transitions.
        
        Args:
            size: Size of the LUT. If None, uses self.lut_size.
            
        Returns:
            2D numpy array of shape (size, 3) with RGB color values
        """
        effective_size = size or self.lut_size
        x = np.linspace(0, 1, effective_size)
        
        # Extract scalar values and RGB colors for interpolation
        scalars, colors = zip(*self.control_points)
        colors = np.array(colors)  # Shape: (n_points, 3)
        
        # Create LUT with linear interpolation for each RGB channel
        lut = np.empty((effective_size, 3), dtype=np.float32)
        for c in range(3):  # For each RGB channel
            lut[:, c] = np.interp(x, scalars, colors[:, c])
        
        return lut
    
    def apply_to_array(self, scalar_array: np.ndarray) -> np.ndarray:
        """
        Apply the color transfer function to an array of scalar values.
        
        Args:
            scalar_array: Input array with scalar values (typically 0-1 range)
            
        Returns:
            Array of shape (*scalar_array.shape, 3) with RGB color values
        """
        original_shape = scalar_array.shape
        flat_scalars = scalar_array.flatten()
        
        # Extract scalar values and RGB colors for interpolation
        scalars, colors = zip(*self.control_points)
        colors = np.array(colors)  # Shape: (n_points, 3)
        
        # Apply interpolation for each RGB channel
        result = np.empty((flat_scalars.size, 3), dtype=np.float32)
        for c in range(3):  # For each RGB channel
            result[:, c] = np.interp(flat_scalars, scalars, colors[:, c])
        
        # Reshape to match input with additional RGB dimension
        return result.reshape(original_shape + (3,))
    
    def get_color_at(self, scalar: float) -> Tuple[float, float, float]:
        """
        Get the RGB color at a specific scalar value.
        
        Args:
            scalar: Input scalar value (typically 0-1 range)
            
        Returns:
            RGB color tuple (r, g, b)
        """
        scalars, colors = zip(*self.control_points)
        colors = np.array(colors)
        
        rgb = np.empty(3, dtype=np.float32)
        for c in range(3):
            rgb[c] = np.interp(scalar, scalars, colors[:, c])
        
        return tuple(rgb.tolist())
    
    def __repr__(self) -> str:
        """String representation showing number of control points and LUT size."""
        return (f"ColorTransferFunction({len(self.control_points)} control points, "
                f"lut_size={self.lut_size})")