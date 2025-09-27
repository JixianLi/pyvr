"""
Base transfer function classes and utilities for PyVR.

This module provides the foundational classes and common functionality
for all transfer functions in PyVR, including base classes, validation
utilities, and shared rendering interfaces.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple, Union, Optional, Any


class BaseTransferFunction(ABC):
    """
    Abstract base class for all transfer functions in PyVR.
    
    Transfer functions map scalar values (typically density or intensity)
    to visual properties like color or opacity for volume rendering.
    
    This base class provides common functionality for control point
    management, LUT generation, and texture creation while allowing
    concrete implementations to define their specific mapping behavior.
    """
    
    def __init__(self, control_points: Optional[List[Tuple]] = None, lut_size: int = 256):
        """
        Initialize base transfer function.
        
        Args:
            control_points: List of (scalar, value) tuples defining the mapping.
                          Subclasses define the specific format of 'value'.
            lut_size: Size of the lookup table for texture generation.
        """
        if control_points is None:
            control_points = self._get_default_control_points()
        
        self.control_points = sorted(control_points, key=lambda x: x[0])
        self.lut_size = lut_size
        
        # Validate control points
        self._validate_control_points()
    
    @abstractmethod
    def _get_default_control_points(self) -> List[Tuple]:
        """Return default control points for this transfer function type."""
        pass
    
    @abstractmethod
    def _validate_control_points(self) -> None:
        """Validate that control points are appropriate for this transfer function."""
        pass
    
    @abstractmethod
    def to_lut(self, size: Optional[int] = None) -> np.ndarray:
        """
        Generate a lookup table (numpy array) for fast mapping.
        
        Args:
            size: Size of the LUT. If None, uses self.lut_size.
            
        Returns:
            numpy array representing the lookup table
        """
        pass
    
    def __call__(self, size: Optional[int] = None) -> np.ndarray:
        """Convenience method - alias for to_lut()."""
        return self.to_lut(size)
    
    def _validate_scalar_range(self, min_val: float = 0.0, max_val: float = 1.0) -> None:
        """
        Validate that control point scalars are within expected range.
        
        Args:
            min_val: Minimum allowed scalar value
            max_val: Maximum allowed scalar value
        """
        for scalar, _ in self.control_points:
            if not (min_val <= scalar <= max_val):
                raise ValueError(f"Control point scalar {scalar} outside valid range [{min_val}, {max_val}]")
    
    def _get_scalar_values(self) -> np.ndarray:
        """Extract scalar values from control points."""
        return np.array([point[0] for point in self.control_points])
    
    def _get_mapped_values(self) -> np.ndarray:
        """Extract mapped values from control points. Subclasses may override."""
        return np.array([point[1] for point in self.control_points])


class TransferFunctionError(Exception):
    """Base exception class for transfer function errors."""
    pass


class InvalidControlPointError(TransferFunctionError):
    """Raised when control points are invalid."""
    pass


def validate_control_points_format(control_points: List[Tuple], expected_value_length: int) -> None:
    """
    Validate that control points have the expected format.
    
    Args:
        control_points: List of (scalar, value) tuples to validate
        expected_value_length: Expected length of the value component
                              (1 for scalars, 3 for RGB, etc.)
    
    Raises:
        InvalidControlPointError: If format is invalid
    """
    if not control_points:
        raise InvalidControlPointError("Control points cannot be empty")
    
    for i, point in enumerate(control_points):
        if not isinstance(point, (tuple, list)) or len(point) != 2:
            raise InvalidControlPointError(
                f"Control point {i} must be a (scalar, value) tuple, got {point}"
            )
        
        scalar, value = point
        if not isinstance(scalar, (int, float)):
            raise InvalidControlPointError(
                f"Control point {i} scalar must be numeric, got {type(scalar)}"
            )
        
        # Check value format based on expected length
        if expected_value_length == 1:
            # Single value (opacity)
            if not isinstance(value, (int, float)):
                raise InvalidControlPointError(
                    f"Control point {i} value must be numeric, got {type(value)}"
                )
        else:
            # Multi-value (color)
            if not isinstance(value, (tuple, list, np.ndarray)) or len(value) != expected_value_length:
                raise InvalidControlPointError(
                    f"Control point {i} value must be a {expected_value_length}-element sequence, got {value}"
                )
            
            for j, component in enumerate(value):
                if not isinstance(component, (int, float)):
                    raise InvalidControlPointError(
                        f"Control point {i} value component {j} must be numeric, got {type(component)}"
                    )