"""
Tests for camera control functionality.
"""

import pytest
import numpy as np
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from pyvr.camera.control import (
    get_camera_pos,
    get_camera_pos_from_params,
    CameraPath,
    CameraController
)
from pyvr.camera.parameters import Camera


def test_get_camera_pos_basic():
    """Test basic get_camera_pos functionality."""
    target = np.array([0.0, 0.0, 0.0])
    
    # Test front view (no rotation)
    pos, up = get_camera_pos(
        target=target,
        azimuth=0.0,
        elevation=0.0,
        roll=0.0,
        distance=3.0
    )
    
    # Should be at [0, 0, 3] looking toward origin
    assert np.allclose(pos, [0.0, 0.0, 3.0])
    assert np.allclose(up, [0.0, 1.0, 0.0])


def test_get_camera_pos_with_custom_initial_vectors():
    """Test get_camera_pos with custom initial vectors."""
    target = np.array([0.0, 0.0, 0.0])
    init_pos = np.array([1.0, 0.0, 0.0])
    init_up = np.array([0.0, 0.0, 1.0])
    
    pos, up = get_camera_pos(
        target=target,
        azimuth=0.0,
        elevation=0.0,
        roll=0.0,
        distance=2.0,
        init_pos=init_pos,
        init_up=init_up
    )
    
    # Should be at distance 2.0 in the direction of init_pos
    assert np.linalg.norm(pos - target) == pytest.approx(2.0)
    assert np.allclose(up, init_up)


def test_get_camera_pos_rotation():
    """Test camera rotation functionality."""
    target = np.array([0.0, 0.0, 0.0])
    distance = 3.0
    
    # Test azimuth rotation (around up axis)
    pos1, up1 = get_camera_pos(target, azimuth=0.0, elevation=0.0, roll=0.0, distance=distance)
    pos2, up2 = get_camera_pos(target, azimuth=np.pi/2, elevation=0.0, roll=0.0, distance=distance)
    
    # Distance should remain the same
    assert np.linalg.norm(pos1 - target) == pytest.approx(distance)
    assert np.linalg.norm(pos2 - target) == pytest.approx(distance)
    
    # Positions should be different
    assert not np.allclose(pos1, pos2)


def test_get_camera_pos_validation():
    """Test input validation for get_camera_pos."""
    target = np.array([0.0, 0.0, 0.0])
    
    # Invalid target
    with pytest.raises(ValueError, match="target must be a 3D numpy array"):
        get_camera_pos([0, 0, 0], 0.0, 0.0, 0.0, 3.0)
    
    # Invalid distance
    with pytest.raises(ValueError, match="distance must be positive"):
        get_camera_pos(target, 0.0, 0.0, 0.0, -1.0)
    
    # Invalid initial vectors
    with pytest.raises(ValueError, match="init_pos must be a 3D numpy array"):
        get_camera_pos(target, 0.0, 0.0, 0.0, 3.0, init_pos=[1, 0, 0])
    
    with pytest.raises(ValueError, match="init_up must be a 3D numpy array"):
        get_camera_pos(target, 0.0, 0.0, 0.0, 3.0, init_up=[0, 1, 0])


def test_get_camera_pos_from_params():
    """Test get_camera_pos_from_params function."""
    params = Camera(
        target=np.array([1.0, 2.0, 3.0]),
        azimuth=np.pi/4,
        elevation=np.pi/6,
        roll=0.0,
        distance=2.0
    )
    
    pos, up = get_camera_pos_from_params(params)
    
    # Should be at correct distance from target
    assert np.linalg.norm(pos - params.target) == pytest.approx(params.distance)


def test_camera_path():
    """Test CameraPath animation functionality."""
    start_params = Camera.front_view(distance=3.0)
    end_params = Camera.side_view(distance=3.0)
    
    path = CameraPath([start_params, end_params])
    
    # Test interpolation
    mid_params = path.interpolate(0.5)
    assert mid_params.distance == pytest.approx(3.0)
    assert 0.0 < mid_params.azimuth < np.pi/2  # Between front and side
    
    # Test frame generation
    frames = path.generate_frames(5)
    assert len(frames) == 5
    
    # First and last frames should match keyframes
    assert np.allclose(frames[0].target, start_params.target)
    assert frames[0].azimuth == pytest.approx(start_params.azimuth)
    assert np.allclose(frames[-1].target, end_params.target)
    assert frames[-1].azimuth == pytest.approx(end_params.azimuth)


def test_camera_path_validation():
    """Test CameraPath validation."""
    # Need at least 2 keyframes
    with pytest.raises(ValueError, match="At least 2 keyframes are required"):
        CameraPath([Camera.front_view()])
    
    # Invalid interpolation parameter
    path = CameraPath([Camera.front_view(), Camera.side_view()])
    
    with pytest.raises(ValueError, match="t must be between 0.0 and 1.0"):
        path.interpolate(-0.1)
    
    with pytest.raises(ValueError, match="t must be between 0.0 and 1.0"):
        path.interpolate(1.1)


def test_camera_path_angle_interpolation():
    """Test angle interpolation handles wraparound correctly."""
    start_params = Camera(azimuth=-np.pi*0.9)  # Near -π
    end_params = Camera(azimuth=np.pi*0.9)     # Near +π
    
    path = CameraPath([start_params, end_params])
    mid_params = path.interpolate(0.5)
    
    # Should take shortest path (around π, not through 0)
    assert abs(mid_params.azimuth) > np.pi/2


def test_camera_controller():
    """Test CameraController functionality."""
    controller = CameraController()
    
    # Test initial state
    initial_params = controller.params
    assert initial_params.azimuth == 0.0
    assert initial_params.elevation == 0.0
    assert initial_params.distance == 3.0
    
    # Test orbiting
    controller.orbit(np.pi/4, np.pi/6)
    assert controller.params.azimuth == pytest.approx(np.pi/4)
    assert controller.params.elevation == pytest.approx(np.pi/6)
    
    # Test zooming
    original_distance = controller.params.distance
    controller.zoom(0.5)  # Zoom in
    assert controller.params.distance == pytest.approx(original_distance * 0.5)
    
    controller.zoom(2.0)  # Zoom out
    assert controller.params.distance == pytest.approx(original_distance)
    
    # Test panning
    original_target = controller.params.target.copy()
    controller.pan(np.array([1.0, 2.0, 3.0]))
    expected_target = original_target + np.array([1.0, 2.0, 3.0])
    assert np.allclose(controller.params.target, expected_target)


def test_camera_controller_validation():
    """Test CameraController input validation."""
    controller = CameraController()
    
    # Invalid zoom factor
    with pytest.raises(ValueError, match="Zoom factor must be positive"):
        controller.zoom(-0.5)
    
    with pytest.raises(ValueError, match="Zoom factor must be positive"):
        controller.zoom(0.0)
    
    # Invalid pan delta
    with pytest.raises(ValueError, match="delta_target must be a 3D numpy array"):
        controller.pan([1, 2, 3])
    
    # Invalid distance
    with pytest.raises(ValueError, match="Distance must be positive"):
        controller.set_distance(-1.0)
    
    # Invalid target
    with pytest.raises(ValueError, match="target must be a 3D numpy array"):
        controller.look_at([0, 0, 0])


def test_camera_controller_presets():
    """Test camera controller preset functionality."""
    controller = CameraController()
    
    # Test preset switching
    controller.reset_to_preset('side')
    assert controller.params.azimuth == pytest.approx(np.pi/2)
    
    controller.reset_to_preset('top')
    assert controller.params.elevation == pytest.approx(np.pi/2)
    
    controller.reset_to_preset('isometric')
    assert controller.params.azimuth == pytest.approx(np.pi/4)
    assert controller.params.elevation == pytest.approx(np.pi/6)
    
    # Test invalid preset
    with pytest.raises(ValueError, match="Unknown preset"):
        controller.reset_to_preset('invalid')


def test_camera_controller_animation():
    """Test camera controller animation generation."""
    controller = CameraController()
    target_params = Camera.isometric_view(distance=5.0)
    
    frames = controller.animate_to(target_params, n_frames=10)
    assert len(frames) == 10
    
    # First frame should be current state
    assert frames[0].azimuth == pytest.approx(controller.params.azimuth)
    assert frames[0].distance == pytest.approx(controller.params.distance)
    
    # Last frame should be target state
    assert frames[-1].azimuth == pytest.approx(target_params.azimuth)
    assert frames[-1].distance == pytest.approx(target_params.distance)


def test_gimbal_lock_handling():
    """Test that gimbal lock situations are handled gracefully."""
    target = np.array([0.0, 0.0, 0.0])
    
    # Test case where init_up and view direction are parallel
    pos, up = get_camera_pos(
        target=target,
        azimuth=0.0,
        elevation=np.pi/2,  # Looking straight up
        roll=0.0,
        distance=3.0,
        init_pos=np.array([0.0, 0.0, 3.0]),
        init_up=np.array([0.0, 0.0, 1.0])  # Up vector parallel to view
    )
    
    # Should not crash and should return valid results
    assert np.linalg.norm(pos - target) == pytest.approx(3.0)
    assert np.linalg.norm(up) == pytest.approx(1.0)


def test_compatibility_with_old_interface():
    """Test that new interface works correctly."""
    initial_params = Camera(
        target=np.array([0.0, 0.0, 0.0]),
        azimuth=0.0, elevation=0.0, roll=0.0, distance=3.0,
        init_pos=np.array([0.0, 0.0, 1.0])  # Make sure init_pos != target
    )
    controller = CameraController(initial_params)
    
    # Should be able to call get_camera_pos_from_params 
    pos, up = get_camera_pos_from_params(controller.params)
    assert len(pos) == 3
    assert len(up) == 3


def test_get_camera_pos_edge_cases():
    """Test edge cases and error conditions for get_camera_pos."""
    target = np.array([0.0, 0.0, 0.0])
    
    # Test zero vector error for init_pos == target
    with pytest.raises(ValueError, match="init_pos must not be the zero vector"):
        get_camera_pos(
            target=target,
            azimuth=0.0, elevation=0.0, roll=0.0, distance=3.0,
            init_pos=target  # Same as target - should raise error
        )
    
    # Test elevation axis normalization edge case (parallel vectors)
    # This should not crash, even with edge cases
    pos, up = get_camera_pos(
        target=np.array([0.0, 0.0, 0.0]),
        azimuth=0.0, elevation=np.pi/2,  # 90 degrees elevation
        roll=0.0, distance=3.0,
        init_pos=np.array([0.0, 0.0, 1.0]),  
        init_up=np.array([0.0, 0.0, 1.0])  # Same direction as init_pos
    )
    assert len(pos) == 3
    assert len(up) == 3


def test_camera_path_edge_cases():
    """Test edge cases for CameraPath."""
    # Test single keyframe (should require at least 2)
    single_keyframe = Camera(
        target=np.array([0.0, 0.0, 0.0]),
        azimuth=0.0, elevation=0.0, roll=0.0, distance=3.0
    )
    
    with pytest.raises(ValueError, match="At least 2 keyframes are required"):
        CameraPath([single_keyframe])
    
    # Test invalid keyframe type
    invalid_keyframe = "not a Camera"
    
    with pytest.raises(ValueError, match="must be a Camera instance"):
        CameraPath([invalid_keyframe, invalid_keyframe])


def test_camera_controller_edge_cases():
    """Test edge cases for CameraController."""
    # Test with very small distance
    initial_params = Camera(
        target=np.array([0.0, 0.0, 0.0]),
        azimuth=0.0, elevation=0.0, roll=0.0, distance=0.001
    )
    controller = CameraController(initial_params)
    
    pos, up = get_camera_pos_from_params(controller.params)
    assert len(pos) == 3
    assert len(up) == 3
    
    # Test with extreme angles
    extreme_params = Camera(
        target=np.array([0.0, 0.0, 0.0]),
        azimuth=10 * np.pi,  # Multiple full rotations
        elevation=2 * np.pi,
        roll=5 * np.pi,
        distance=3.0
    )
    pos, up = get_camera_pos_from_params(extreme_params)
    assert len(pos) == 3
    assert len(up) == 3


def test_camera_path_interpolation_edge_cases():
    """Test edge cases for camera path interpolation."""
    params1 = Camera(
        target=np.array([0.0, 0.0, 0.0]),
        azimuth=0.0, elevation=0.0, roll=0.0, distance=3.0,
        init_pos=np.array([0.0, 0.0, 1.0])
    )
    params2 = Camera(
        target=np.array([1.0, 0.0, 0.0]),
        azimuth=np.pi, elevation=np.pi/4, roll=0.0, distance=5.0,
        init_pos=np.array([0.0, 0.0, 1.0])
    )
    
    path = CameraPath([params1, params2])
    
    # Test t values outside [0, 1] - should raise ValueError
    with pytest.raises(ValueError, match="t must be between 0.0 and 1.0"):
        path.interpolate(-0.5)
    
    with pytest.raises(ValueError, match="t must be between 0.0 and 1.0"):
        path.interpolate(1.5)
    
    # Test valid interpolation
    result_mid = path.interpolate(0.5)
    assert isinstance(result_mid, Camera)


if __name__ == "__main__":
    pytest.main([__file__])