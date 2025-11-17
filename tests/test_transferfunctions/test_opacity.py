"""
Tests for OpacityTransferFunction.
"""

import pytest
import numpy as np
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from pyvr.transferfunctions import OpacityTransferFunction, InvalidControlPointError


def test_opacity_transfer_function_initialization():
    """Test basic OpacityTransferFunction initialization."""
    # Default initialization
    otf = OpacityTransferFunction()
    assert len(otf.control_points) == 2
    assert otf.control_points == [(0.0, 0.0), (1.0, 1.0)]
    assert otf.lut_size == 256

    # Custom initialization
    control_points = [(0.0, 0.1), (0.5, 0.8), (1.0, 0.3)]
    otf = OpacityTransferFunction(control_points, lut_size=512)
    assert otf.control_points == control_points
    assert otf.lut_size == 512


def test_linear_opacity_transfer_function():
    """Test linear opacity transfer function creation."""
    otf = OpacityTransferFunction.linear(0.2, 0.8, lut_size=128)

    assert len(otf.control_points) == 2
    assert otf.control_points[0] == (0.0, 0.2)
    assert otf.control_points[1] == (1.0, 0.8)
    assert otf.lut_size == 128

    # Test LUT generation
    lut = otf.to_lut()
    assert lut.shape == (128,)
    assert lut[0] == pytest.approx(0.2, abs=1e-6)
    assert lut[-1] == pytest.approx(0.8, abs=1e-6)


def test_one_step_opacity_transfer_function():
    """Test step opacity transfer function creation."""
    otf = OpacityTransferFunction.one_step(step=0.5, low=0.1, high=0.9)

    # Should have 4 control points for step function
    assert len(otf.control_points) == 4
    assert otf.control_points[0] == (0.0, 0.1)
    assert otf.control_points[1] == (0.5, 0.1)
    assert otf.control_points[2][0] == pytest.approx(0.5, abs=1e-10)  # 0.5 + epsilon
    assert otf.control_points[2][1] == 0.9
    assert otf.control_points[3] == (1.0, 0.9)

    # Test LUT generation
    lut = otf.to_lut(1000)  # High resolution for step function
    # Values before step should be low
    assert np.all(lut[:500] <= 0.1 + 1e-6)
    # Values after step should be high
    assert np.all(lut[500:] >= 0.9 - 1e-6)


def test_peaks_opacity_transfer_function():
    """Test peaks opacity transfer function creation."""
    # Single peak
    otf = OpacityTransferFunction.peaks([0.5], opacity=0.9, eps=0.1, base=0.1)

    lut = otf.to_lut(1000)
    peak_idx = 500  # Middle of LUT for peak at 0.5

    # Peak value should be near maximum
    assert lut[peak_idx] >= 0.8  # Allow some tolerance for interpolation

    # Base values should be low
    assert lut[0] == pytest.approx(0.1, abs=1e-6)
    assert lut[-1] == pytest.approx(0.1, abs=1e-6)

    # Multiple peaks
    otf_multi = OpacityTransferFunction.peaks([0.2, 0.8], opacity=0.9, eps=0.05)
    lut_multi = otf_multi.to_lut(1000)

    # Check peak locations
    peak1_idx, peak2_idx = 200, 800
    assert lut_multi[peak1_idx] >= 0.8
    assert lut_multi[peak2_idx] >= 0.8


def test_peaks_validation():
    """Test peaks method input validation."""
    # Empty peaks list
    with pytest.raises(
        ValueError, match="At least one peak position must be specified"
    ):
        OpacityTransferFunction.peaks([])

    # Peak outside valid range
    with pytest.raises(ValueError, match="Peak position .* must be between 0 and 1"):
        OpacityTransferFunction.peaks([1.5])

    with pytest.raises(ValueError, match="Peak position .* must be between 0 and 1"):
        OpacityTransferFunction.peaks([-0.1])


def test_lut_generation():
    """Test LUT generation with different sizes."""
    otf = OpacityTransferFunction.linear(0.0, 1.0)

    # Test different sizes
    lut_256 = otf.to_lut(256)
    lut_512 = otf.to_lut(512)
    lut_100 = otf.to_lut(100)

    assert lut_256.shape == (256,)
    assert lut_512.shape == (512,)
    assert lut_100.shape == (100,)

    # Test default size
    lut_default = otf.to_lut()
    assert lut_default.shape == (256,)  # Default lut_size

    # All should be float32
    assert lut_256.dtype == np.float32
    assert lut_512.dtype == np.float32


def test_apply_to_array():
    """Test applying opacity transfer function to arrays."""
    otf = OpacityTransferFunction.linear(0.0, 1.0)

    # 1D array
    test_array_1d = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    result_1d = otf.apply_to_array(test_array_1d)

    assert result_1d.shape == test_array_1d.shape
    np.testing.assert_array_almost_equal(result_1d, test_array_1d, decimal=6)

    # 2D array
    test_array_2d = np.array([[0.0, 0.5], [0.25, 1.0]])
    result_2d = otf.apply_to_array(test_array_2d)

    assert result_2d.shape == test_array_2d.shape
    np.testing.assert_array_almost_equal(result_2d, test_array_2d, decimal=6)


def test_validation():
    """Test validation of control points."""
    # Valid control points
    OpacityTransferFunction([(0.0, 0.0), (1.0, 1.0)])  # Should not raise

    # Invalid opacity values
    with pytest.raises(
        InvalidControlPointError, match="opacity .* outside valid range"
    ):
        OpacityTransferFunction([(0.0, -0.1), (1.0, 1.0)])

    with pytest.raises(
        InvalidControlPointError, match="opacity .* outside valid range"
    ):
        OpacityTransferFunction([(0.0, 0.0), (1.0, 1.5)])

    # Invalid scalar values
    with pytest.raises(ValueError, match="scalar .* outside valid range"):
        OpacityTransferFunction([(-0.1, 0.0), (1.0, 1.0)])


def test_repr():
    """Test string representation."""
    otf = OpacityTransferFunction([(0.0, 0.1), (1.0, 0.9)], lut_size=512)
    repr_str = repr(otf)

    assert "OpacityTransferFunction" in repr_str
    assert "control_points" in repr_str
    assert "lut_size=512" in repr_str


if __name__ == "__main__":
    pytest.main([__file__])
