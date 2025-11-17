"""
Tests for the base transfer function functionality.
"""

import pytest
import numpy as np
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from pyvr.transferfunctions.base import (
    BaseTransferFunction,
    TransferFunctionError,
    InvalidControlPointError,
    validate_control_points_format,
)


class ConcreteTransferFunction(BaseTransferFunction):
    """Concrete implementation for testing base class."""

    def _get_default_control_points(self):
        return [(0.0, 0.0), (1.0, 1.0)]

    def _validate_control_points(self):
        validate_control_points_format(self.control_points, expected_value_length=1)
        self._validate_scalar_range(0.0, 1.0)

    def to_lut(self, size=None):
        effective_size = size or self.lut_size
        x = np.linspace(0, 1, effective_size)
        scalars, values = zip(*self.control_points)
        return np.interp(x, scalars, values).astype(np.float32)

    def _get_texture_channels(self):
        return 1

    def _create_texture_legacy(self, ctx, size):
        # Mock implementation for testing
        return "mock_texture"


def test_base_transfer_function_initialization():
    """Test basic initialization of BaseTransferFunction."""
    # Default initialization
    tf = ConcreteTransferFunction()
    assert len(tf.control_points) == 2
    assert tf.lut_size == 256

    # Custom initialization
    control_points = [(0.0, 0.2), (0.5, 0.8), (1.0, 0.1)]
    tf = ConcreteTransferFunction(control_points, lut_size=512)
    assert tf.control_points == control_points
    assert tf.lut_size == 512


def test_control_point_sorting():
    """Test that control points are sorted by scalar value."""
    control_points = [(1.0, 0.1), (0.0, 0.2), (0.5, 0.8)]
    tf = ConcreteTransferFunction(control_points)

    expected_sorted = [(0.0, 0.2), (0.5, 0.8), (1.0, 0.1)]
    assert tf.control_points == expected_sorted


def test_call_method():
    """Test that __call__ method works as alias for to_lut."""
    tf = ConcreteTransferFunction()
    lut1 = tf.to_lut(100)
    lut2 = tf(100)
    np.testing.assert_array_equal(lut1, lut2)


def test_validate_control_points_format():
    """Test control points validation function."""
    # Valid single-value control points
    valid_points = [(0.0, 0.5), (1.0, 0.8)]
    validate_control_points_format(valid_points, 1)  # Should not raise

    # Valid multi-value control points
    valid_color_points = [(0.0, (0.1, 0.2, 0.3)), (1.0, (0.8, 0.9, 1.0))]
    validate_control_points_format(valid_color_points, 3)  # Should not raise

    # Empty control points
    with pytest.raises(InvalidControlPointError, match="cannot be empty"):
        validate_control_points_format([], 1)

    # Wrong tuple length
    with pytest.raises(
        InvalidControlPointError, match="must be a \\(scalar, value\\) tuple"
    ):
        validate_control_points_format([(0.0,)], 1)

    # Non-numeric scalar
    with pytest.raises(InvalidControlPointError, match="scalar must be numeric"):
        validate_control_points_format([("a", 0.5)], 1)

    # Wrong value length for multi-value
    with pytest.raises(InvalidControlPointError, match="must be a 3-element sequence"):
        validate_control_points_format([(0.0, (0.1, 0.2))], 3)

    # Non-numeric value component
    with pytest.raises(InvalidControlPointError, match="must be numeric"):
        validate_control_points_format([(0.0, (0.1, "a", 0.3))], 3)


def test_abstract_method_coverage():
    """Test that abstract methods are properly defined."""
    from pyvr.transferfunctions.base import BaseTransferFunction

    # Verify abstract methods exist and are abstract
    assert hasattr(BaseTransferFunction, "_get_default_control_points")
    assert hasattr(BaseTransferFunction, "_validate_control_points")
    assert hasattr(BaseTransferFunction, "to_lut")

    # Cannot instantiate abstract class
    with pytest.raises(TypeError):
        BaseTransferFunction()


def test_control_point_validation_edge_cases():
    """Test edge cases in control point validation."""
    from pyvr.transferfunctions.color import ColorTransferFunction

    # Test empty control points (should use defaults)
    ctf = ColorTransferFunction()
    assert len(ctf.control_points) > 0

    # Test single control point (edge case)
    single_point_ctf = ColorTransferFunction([(0.5, [0.5, 0.5, 0.5])])
    assert len(single_point_ctf.control_points) == 1

    # Test control points out of order (should be sorted)
    unordered_ctf = ColorTransferFunction(
        [(1.0, [1.0, 1.0, 1.0]), (0.0, [0.0, 0.0, 0.0]), (0.5, [0.5, 0.5, 0.5])]
    )
    # Should be sorted by first element (position)
    positions = [point[0] for point in unordered_ctf.control_points]
    assert positions == sorted(positions)


def test_lut_size_parameter():
    """Test LUT size parameter handling."""
    from pyvr.transferfunctions.color import ColorTransferFunction

    ctf = ColorTransferFunction(lut_size=512)
    assert ctf.lut_size == 512

    # Test default size
    lut_default = ctf.to_lut()
    assert lut_default.shape[0] == 512

    # Test custom size
    lut_custom = ctf.to_lut(1024)
    assert lut_custom.shape[0] == 1024

    # Test call method with size
    lut_call = ctf(256)
    assert lut_call.shape[0] == 256


def test_call_method_alias():
    """Test that __call__ is proper alias for to_lut."""
    from pyvr.transferfunctions.color import ColorTransferFunction

    ctf = ColorTransferFunction()

    # Test without parameters
    lut1 = ctf.to_lut()
    lut2 = ctf()
    assert np.array_equal(lut1, lut2)

    # Test with size parameter
    lut3 = ctf.to_lut(128)
    lut4 = ctf(128)
    assert np.array_equal(lut3, lut4)


if __name__ == "__main__":
    pytest.main([__file__])
