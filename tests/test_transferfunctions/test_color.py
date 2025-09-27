"""
Tests for ColorTransferFunction.
"""

import pytest
import numpy as np
import sys
import os
from unittest.mock import patch, Mock

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from pyvr.transferfunctions import ColorTransferFunction, InvalidControlPointError


def test_color_transfer_function_initialization():
    """Test basic ColorTransferFunction initialization."""
    # Default initialization
    ctf = ColorTransferFunction()
    assert len(ctf.control_points) == 2
    assert ctf.control_points == [(0.0, (0.0, 0.0, 0.0)), (1.0, (1.0, 1.0, 1.0))]
    assert ctf.lut_size == 256
    
    # Custom initialization
    control_points = [(0.0, (1.0, 0.0, 0.0)), (0.5, (0.0, 1.0, 0.0)), (1.0, (0.0, 0.0, 1.0))]
    ctf = ColorTransferFunction(control_points, lut_size=512)
    assert ctf.control_points == control_points
    assert ctf.lut_size == 512


def test_grayscale_color_transfer_function():
    """Test grayscale color transfer function creation."""
    ctf = ColorTransferFunction.grayscale(lut_size=128)
    
    assert len(ctf.control_points) == 2
    assert ctf.control_points[0] == (0.0, (0.0, 0.0, 0.0))
    assert ctf.control_points[1] == (1.0, (1.0, 1.0, 1.0))
    assert ctf.lut_size == 128
    
    # Test LUT generation
    lut = ctf.to_lut()
    assert lut.shape == (128, 3)
    
    # Check that it's actually grayscale (R == G == B)
    np.testing.assert_array_almost_equal(lut[:, 0], lut[:, 1], decimal=6)
    np.testing.assert_array_almost_equal(lut[:, 1], lut[:, 2], decimal=6)
    
    # Check endpoints
    np.testing.assert_array_almost_equal(lut[0], [0.0, 0.0, 0.0], decimal=6)
    np.testing.assert_array_almost_equal(lut[-1], [1.0, 1.0, 1.0], decimal=6)


def test_single_color_transfer_function():
    """Test single color transfer function creation."""
    red_color = (1.0, 0.0, 0.0)
    ctf = ColorTransferFunction.single_color(red_color, lut_size=100)
    
    assert len(ctf.control_points) == 2
    assert ctf.control_points[0] == (0.0, red_color)
    assert ctf.control_points[1] == (1.0, red_color)
    
    # Test LUT generation - all values should be the same color
    lut = ctf.to_lut()
    assert lut.shape == (100, 3)
    
    for row in lut:
        np.testing.assert_array_almost_equal(row, red_color, decimal=6)


def test_two_color_ramp():
    """Test two-color ramp transfer function."""
    blue = (0.0, 0.0, 1.0)
    yellow = (1.0, 1.0, 0.0)
    ctf = ColorTransferFunction.two_color_ramp(blue, yellow)
    
    assert len(ctf.control_points) == 2
    assert ctf.control_points[0] == (0.0, blue)
    assert ctf.control_points[1] == (1.0, yellow)
    
    # Test LUT generation
    lut = ctf.to_lut(256)
    assert lut.shape == (256, 3)
    
    # Check endpoints
    np.testing.assert_array_almost_equal(lut[0], blue, decimal=6)
    np.testing.assert_array_almost_equal(lut[-1], yellow, decimal=6)
    
    # Check middle should be interpolated
    middle = lut[128]  # Middle of 256-element array
    expected_middle = ((np.array(blue) + np.array(yellow)) / 2)
    np.testing.assert_array_almost_equal(middle, expected_middle, decimal=2)


def test_lut_generation():
    """Test LUT generation with different sizes."""
    ctf = ColorTransferFunction.grayscale()
    
    # Test different sizes
    lut_256 = ctf.to_lut(256)
    lut_512 = ctf.to_lut(512)
    lut_100 = ctf.to_lut(100)
    
    assert lut_256.shape == (256, 3)
    assert lut_512.shape == (512, 3)
    assert lut_100.shape == (100, 3)
    
    # Test default size
    lut_default = ctf.to_lut()
    assert lut_default.shape == (256, 3)  # Default lut_size
    
    # All should be float32
    assert lut_256.dtype == np.float32
    assert lut_512.dtype == np.float32


def test_get_color_at():
    """Test getting color at specific scalar value."""
    ctf = ColorTransferFunction.two_color_ramp((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    
    # Test endpoints
    color_0 = ctf.get_color_at(0.0)
    color_1 = ctf.get_color_at(1.0)
    
    np.testing.assert_array_almost_equal(color_0, (1.0, 0.0, 0.0), decimal=6)
    np.testing.assert_array_almost_equal(color_1, (0.0, 1.0, 0.0), decimal=6)
    
    # Test middle
    color_mid = ctf.get_color_at(0.5)
    np.testing.assert_array_almost_equal(color_mid, (0.5, 0.5, 0.0), decimal=6)


def test_apply_to_array():
    """Test applying color transfer function to arrays."""
    ctf = ColorTransferFunction.grayscale()
    
    # 1D array
    test_array_1d = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    result_1d = ctf.apply_to_array(test_array_1d)
    
    assert result_1d.shape == (5, 3)
    # Should be grayscale, so all components equal
    np.testing.assert_array_almost_equal(result_1d[:, 0], test_array_1d, decimal=6)
    np.testing.assert_array_almost_equal(result_1d[:, 1], test_array_1d, decimal=6)
    np.testing.assert_array_almost_equal(result_1d[:, 2], test_array_1d, decimal=6)
    
    # 2D array
    test_array_2d = np.array([[0.0, 0.5], [0.25, 1.0]])
    result_2d = ctf.apply_to_array(test_array_2d)
    
    assert result_2d.shape == (2, 2, 3)
    # Check specific values
    np.testing.assert_array_almost_equal(result_2d[0, 0], [0.0, 0.0, 0.0], decimal=6)
    np.testing.assert_array_almost_equal(result_2d[1, 1], [1.0, 1.0, 1.0], decimal=6)


def test_colormap_integration():
    """Test colormap integration with string colormap names."""
    # Test with different colormaps
    for cmap_name in ['viridis', 'plasma', 'jet', 'coolwarm']:
        ctf = ColorTransferFunction.from_colormap(cmap_name, lut_size=128)
        
        assert len(ctf.control_points) == 128
        assert ctf.lut_size == 128
        
        # Test LUT generation
        lut = ctf.to_lut()
        assert lut.shape == (128, 3)
        
        # Colors should be in valid range
        assert np.all(lut >= 0.0)
        assert np.all(lut <= 1.0)


def test_validation():
    """Test validation of control points."""
    # Valid control points
    ColorTransferFunction([
        (0.0, (0.0, 0.0, 0.0)), 
        (1.0, (1.0, 1.0, 1.0))
    ])  # Should not raise
    
    # Invalid RGB values
    with pytest.raises(InvalidControlPointError, match="RGB component .* outside valid range"):
        ColorTransferFunction([(0.0, (-0.1, 0.0, 0.0)), (1.0, (1.0, 1.0, 1.0))])
    
    with pytest.raises(InvalidControlPointError, match="RGB component .* outside valid range"):
        ColorTransferFunction([(0.0, (0.0, 0.0, 0.0)), (1.0, (1.5, 1.0, 1.0))])
    
    # Invalid scalar values
    with pytest.raises(ValueError, match="scalar .* outside valid range"):
        ColorTransferFunction([(-0.1, (0.0, 0.0, 0.0)), (1.0, (1.0, 1.0, 1.0))])
    
    # Wrong number of color components
    with pytest.raises(InvalidControlPointError, match="must be a 3-element sequence"):
        ColorTransferFunction([(0.0, (0.0, 0.0)), (1.0, (1.0, 1.0, 1.0))])


def test_repr():
    """Test string representation."""
    ctf = ColorTransferFunction([
        (0.0, (1.0, 0.0, 0.0)), 
        (0.5, (0.0, 1.0, 0.0)), 
        (1.0, (0.0, 0.0, 1.0))
    ], lut_size=512)
    
    repr_str = repr(ctf)
    
    assert "ColorTransferFunction" in repr_str
    assert "3 control points" in repr_str
    assert "lut_size=512" in repr_str


def test_colormap_integration_success():
    """Test successful matplotlib colormap integration.""" 
    # Test with matplotlib available
    with patch('matplotlib.colormaps') as mock_colormaps:
        # Mock a colormap
        mock_cmap = Mock()
        mock_cmap.return_value = np.array([[1.0, 0.0, 0.0, 1.0], 
                                          [0.0, 1.0, 0.0, 1.0],
                                          [0.0, 0.0, 1.0, 1.0]])
        mock_colormaps.get_cmap.return_value = mock_cmap
        
        ctf = ColorTransferFunction.from_colormap('viridis', lut_size=3)
        
        assert len(ctf.control_points) == 3
        mock_colormaps.get_cmap.assert_called_with('viridis')


def test_colormap_integration_import_error():
    """Test matplotlib import error handling."""
    # Test ImportError when matplotlib is not available
    import builtins
    original_import = builtins.__import__
    
    def mock_import(name, *args, **kwargs):
        if name == 'matplotlib':
            raise ImportError("No module named 'matplotlib'")
        return original_import(name, *args, **kwargs)
    
    with patch('builtins.__import__', side_effect=mock_import):
        with pytest.raises(ImportError, match="matplotlib is required"):
            ColorTransferFunction.from_colormap('viridis')


def test_colormap_integration_invalid_name():
    """Test invalid colormap name error handling."""
    # Test ValueError for unknown colormap
    with patch('matplotlib.colormaps') as mock_colormaps:
        mock_colormaps.get_cmap.side_effect = ValueError("Unknown colormap")
        
        with pytest.raises(ValueError, match="Unknown colormap name"):
            ColorTransferFunction.from_colormap('nonexistent_colormap')


def test_colormap_integration_other_errors():
    """Test other matplotlib errors."""
    with patch('matplotlib.colormaps') as mock_colormaps:
        mock_colormaps.get_cmap.side_effect = Exception("Some other error")
        
        with pytest.raises(ValueError, match="Unknown colormap name"):
            ColorTransferFunction.from_colormap('viridis')


if __name__ == "__main__":
    pytest.main([__file__])