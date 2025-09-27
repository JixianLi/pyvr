"""
Tests for CameraParameters class.
"""

import pytest
import numpy as np
import json
import tempfile
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from pyvr.camera.parameters import (
    CameraParameters, 
    CameraParameterError,
    validate_camera_angles,
    degrees_to_radians,
    radians_to_degrees
)


def test_camera_parameters_initialization():
    """Test basic CameraParameters initialization."""
    # Default initialization
    params = CameraParameters()
    assert np.array_equal(params.target, np.array([0.0, 0.0, 0.0]))
    assert params.azimuth == 0.0
    assert params.elevation == 0.0
    assert params.roll == 0.0
    assert params.distance == 3.0
    
    # Custom initialization
    target = np.array([1.0, 2.0, 3.0])
    params = CameraParameters(
        target=target,
        azimuth=np.pi/4,
        elevation=np.pi/6,
        roll=np.pi/8,
        distance=5.0
    )
    assert np.array_equal(params.target, target)
    assert params.azimuth == pytest.approx(np.pi/4)
    assert params.elevation == pytest.approx(np.pi/6)
    assert params.roll == pytest.approx(np.pi/8)
    assert params.distance == 5.0


def test_camera_parameters_validation():
    """Test parameter validation."""
    # Valid parameters should not raise
    CameraParameters(
        target=np.array([0.0, 0.0, 0.0]),
        distance=1.0
    )
    
    # Invalid distance
    with pytest.raises(ValueError, match="distance must be positive"):
        CameraParameters(distance=-1.0)
    
    with pytest.raises(ValueError, match="distance must be positive"):
        CameraParameters(distance=0.0)
    
    # Invalid target
    with pytest.raises(ValueError, match="target must be a 3D numpy array"):
        CameraParameters(target=[0, 0, 0])  # List instead of array
    
    with pytest.raises(ValueError, match="target must be a 3D numpy array"):
        CameraParameters(target=np.array([0, 0]))  # Wrong shape
    
    # Invalid initial vectors
    with pytest.raises(ValueError, match="init_pos must be a 3D numpy array"):
        CameraParameters(init_pos=np.array([0, 0]))
    
    with pytest.raises(ValueError, match="init_up must be a 3D numpy array"):
        CameraParameters(init_up=np.array([0, 0]))
    
    # Zero vectors
    with pytest.raises(ValueError, match="init_pos must not be at the same location as target"):
        CameraParameters(
            target=np.array([1, 0, 0]),
            init_pos=np.array([1, 0, 0])
        )
    
    with pytest.raises(ValueError, match="init_up must not be the zero vector"):
        CameraParameters(init_up=np.array([0, 0, 0]))


def test_camera_parameters_presets():
    """Test preset camera positions."""
    target = np.array([1.0, 2.0, 3.0])
    distance = 5.0
    
    # Front view
    front = CameraParameters.front_view(target=target, distance=distance)
    assert np.array_equal(front.target, target)
    assert front.azimuth == 0.0
    assert front.elevation == 0.0
    assert front.roll == 0.0
    assert front.distance == distance
    
    # Side view
    side = CameraParameters.side_view(target=target, distance=distance)
    assert side.azimuth == pytest.approx(np.pi/2)
    assert side.elevation == 0.0
    
    # Top view
    top = CameraParameters.top_view(target=target, distance=distance)
    assert top.azimuth == 0.0
    assert top.elevation == pytest.approx(np.pi/2)
    
    # Isometric view
    iso = CameraParameters.isometric_view(target=target, distance=distance)
    assert iso.azimuth == pytest.approx(np.pi/4)
    assert iso.elevation == pytest.approx(np.pi/6)


def test_camera_parameters_serialization():
    """Test serialization and deserialization."""
    original = CameraParameters(
        target=np.array([1.0, 2.0, 3.0]),
        azimuth=np.pi/3,
        elevation=np.pi/4,
        roll=np.pi/6,
        distance=4.0
    )
    
    # Test dictionary conversion
    data = original.to_dict()
    restored = CameraParameters.from_dict(data)
    
    assert np.allclose(restored.target, original.target)
    assert restored.azimuth == pytest.approx(original.azimuth)
    assert restored.elevation == pytest.approx(original.elevation)
    assert restored.roll == pytest.approx(original.roll)
    assert restored.distance == pytest.approx(original.distance)
    
    # Test file operations
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name
    
    try:
        original.save_to_file(temp_path)
        loaded = CameraParameters.load_from_file(temp_path)
        
        assert np.allclose(loaded.target, original.target)
        assert loaded.azimuth == pytest.approx(original.azimuth)
        assert loaded.distance == pytest.approx(original.distance)
    finally:
        os.unlink(temp_path)


def test_camera_parameters_copy():
    """Test copying camera parameters."""
    original = CameraParameters(
        azimuth=np.pi/4,
        elevation=np.pi/6,
        distance=2.0
    )
    
    copy = original.copy()
    
    # Should have same values
    assert copy.azimuth == original.azimuth
    assert copy.elevation == original.elevation
    assert copy.distance == original.distance
    
    # Should be different objects
    assert copy is not original
    assert copy.target is not original.target
    
    # Modifying copy shouldn't affect original
    copy.azimuth = np.pi/2
    assert original.azimuth == pytest.approx(np.pi/4)


def test_validate_camera_angles():
    """Test angle validation function."""
    # Valid angles
    validate_camera_angles(0.0, np.pi/2, -np.pi/4)  # Should not raise
    
    # Invalid angle types
    with pytest.raises(CameraParameterError, match="azimuth must be numeric"):
        validate_camera_angles("invalid", 0.0, 0.0)
    
    with pytest.raises(CameraParameterError, match="elevation must be numeric"):
        validate_camera_angles(0.0, "invalid", 0.0)
    
    with pytest.raises(CameraParameterError, match="roll must be numeric"):
        validate_camera_angles(0.0, 0.0, "invalid")
    
    # Non-finite values
    with pytest.raises(CameraParameterError, match="azimuth must be finite"):
        validate_camera_angles(np.inf, 0.0, 0.0)
    
    with pytest.raises(CameraParameterError, match="elevation must be finite"):
        validate_camera_angles(0.0, np.nan, 0.0)


def test_angle_conversion_utilities():
    """Test degree/radian conversion utilities."""
    # Degrees to radians
    deg_angles = {'azimuth': 90.0, 'elevation': 45.0, 'roll': 180.0}
    rad_angles = degrees_to_radians(**deg_angles)
    
    assert rad_angles['azimuth'] == pytest.approx(np.pi/2)
    assert rad_angles['elevation'] == pytest.approx(np.pi/4)
    assert rad_angles['roll'] == pytest.approx(np.pi)
    
    # Radians to degrees
    rad_angles = {'azimuth': np.pi/2, 'elevation': np.pi/4, 'roll': np.pi}
    deg_angles = radians_to_degrees(**rad_angles)
    
    assert deg_angles['azimuth'] == pytest.approx(90.0)
    assert deg_angles['elevation'] == pytest.approx(45.0)
    assert deg_angles['roll'] == pytest.approx(180.0)


def test_from_spherical_class_method():
    """Test from_spherical class method."""
    target = np.array([1.0, 2.0, 3.0])
    params = CameraParameters.from_spherical(
        target=target,
        azimuth=np.pi/4,
        elevation=np.pi/6,
        roll=np.pi/8,
        distance=2.0,
        fov=np.pi/3  # Additional parameter
    )
    
    assert np.array_equal(params.target, target)
    assert params.azimuth == pytest.approx(np.pi/4)
    assert params.elevation == pytest.approx(np.pi/6)
    assert params.roll == pytest.approx(np.pi/8)
    assert params.distance == 2.0
    assert params.fov == pytest.approx(np.pi/3)


def test_repr():
    """Test string representation."""
    params = CameraParameters(
        azimuth=np.pi/4,
        elevation=np.pi/6,
        distance=2.5
    )
    
    repr_str = repr(params)
    assert "CameraParameters" in repr_str
    assert "azimuth=" in repr_str
    assert "45.0°" in repr_str  # Should show degrees
    assert "distance=2.50" in repr_str


if __name__ == "__main__":
    pytest.main([__file__])