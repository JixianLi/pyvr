"""
Tests for Camera matrix creation methods.
"""

import pytest
import numpy as np
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from pyvr.camera import Camera


def test_get_view_matrix_basic():
    """Test basic view matrix creation."""
    camera = Camera.front_view(distance=3.0)
    view_matrix = camera.get_view_matrix()

    # Check shape and type
    assert view_matrix.shape == (4, 4)
    assert view_matrix.dtype == np.float32

    # View matrix should be invertible (non-zero determinant)
    assert np.linalg.det(view_matrix[:3, :3]) != pytest.approx(0.0)


def test_get_view_matrix_different_views():
    """Test view matrix for different camera presets."""
    # Front view
    front = Camera.front_view(distance=3.0)
    front_view = front.get_view_matrix()

    # Side view
    side = Camera.side_view(distance=3.0)
    side_view = side.get_view_matrix()

    # Top view
    top = Camera.top_view(distance=3.0)
    top_view = top.get_view_matrix()

    # All should be different matrices
    assert not np.allclose(front_view, side_view)
    assert not np.allclose(front_view, top_view)
    assert not np.allclose(side_view, top_view)

    # All should have correct shape
    assert front_view.shape == (4, 4)
    assert side_view.shape == (4, 4)
    assert top_view.shape == (4, 4)


def test_get_view_matrix_with_target():
    """Test view matrix with different target positions."""
    target1 = np.array([0.0, 0.0, 0.0])
    camera1 = Camera(target=target1, azimuth=0.0, elevation=0.0, roll=0.0, distance=3.0)
    view1 = camera1.get_view_matrix()

    target2 = np.array([5.0, 5.0, 5.0])
    camera2 = Camera(target=target2, azimuth=0.0, elevation=0.0, roll=0.0, distance=3.0)
    view2 = camera2.get_view_matrix()

    # Matrices should be different for different targets
    assert not np.allclose(view1, view2)


def test_get_projection_matrix_basic():
    """Test basic projection matrix creation."""
    camera = Camera.front_view(distance=3.0)
    aspect_ratio = 16.0 / 9.0
    proj_matrix = camera.get_projection_matrix(aspect_ratio)

    # Check shape and type
    assert proj_matrix.shape == (4, 4)
    assert proj_matrix.dtype == np.float32

    # Projection matrix should have specific structure
    # Top-left should be non-zero (scale factors)
    assert proj_matrix[0, 0] != pytest.approx(0.0)
    assert proj_matrix[1, 1] != pytest.approx(0.0)

    # Bottom-right should be 0 for perspective projection
    assert proj_matrix[3, 3] == pytest.approx(0.0)


def test_get_projection_matrix_aspect_ratios():
    """Test projection matrix with different aspect ratios."""
    camera = Camera.front_view(distance=3.0)

    # Wide aspect ratio (16:9)
    proj_wide = camera.get_projection_matrix(16.0 / 9.0)

    # Square aspect ratio (1:1)
    proj_square = camera.get_projection_matrix(1.0)

    # Tall aspect ratio (9:16)
    proj_tall = camera.get_projection_matrix(9.0 / 16.0)

    # All should have correct shape
    assert proj_wide.shape == (4, 4)
    assert proj_square.shape == (4, 4)
    assert proj_tall.shape == (4, 4)

    # Aspect ratio affects x-scale (first element)
    assert proj_wide[0, 0] != pytest.approx(proj_square[0, 0])
    assert proj_square[0, 0] != pytest.approx(proj_tall[0, 0])

    # Y-scale should be the same (second element on diagonal)
    assert proj_wide[1, 1] == pytest.approx(proj_square[1, 1])
    assert proj_square[1, 1] == pytest.approx(proj_tall[1, 1])


def test_get_projection_matrix_fov():
    """Test projection matrix with different field of view."""
    # Narrow FOV
    camera_narrow = Camera(fov=np.pi / 6)  # 30 degrees
    proj_narrow = camera_narrow.get_projection_matrix(1.0)

    # Wide FOV
    camera_wide = Camera(fov=np.pi / 2)  # 90 degrees
    proj_wide = camera_wide.get_projection_matrix(1.0)

    # Different FOV should produce different matrices
    assert not np.allclose(proj_narrow, proj_wide)

    # Narrower FOV should have larger scale factor
    assert proj_narrow[1, 1] > proj_wide[1, 1]


def test_get_projection_matrix_near_far():
    """Test projection matrix with different near/far planes."""
    # Close near/far
    camera_close = Camera(near_plane=0.1, far_plane=10.0)
    proj_close = camera_close.get_projection_matrix(1.0)

    # Far near/far
    camera_far = Camera(near_plane=1.0, far_plane=1000.0)
    proj_far = camera_far.get_projection_matrix(1.0)

    # Different near/far should produce different matrices
    assert not np.allclose(proj_close, proj_far)


def test_matrix_consistency():
    """Test that matrices are consistent across multiple calls."""
    camera = Camera.isometric_view(distance=5.0)

    # Get matrices multiple times
    view1 = camera.get_view_matrix()
    view2 = camera.get_view_matrix()

    proj1 = camera.get_projection_matrix(1.0)
    proj2 = camera.get_projection_matrix(1.0)

    # Should produce identical matrices
    assert np.allclose(view1, view2)
    assert np.allclose(proj1, proj2)


def test_camera_vectors_integration():
    """Test that get_camera_vectors integrates correctly with matrices."""
    camera = Camera.from_spherical(
        target=np.array([0, 0, 0]),
        azimuth=np.pi / 4,
        elevation=np.pi / 6,
        roll=0.0,
        distance=3.0,
    )

    # Get camera position
    position, up = camera.get_camera_vectors()

    # Get view matrix
    view_matrix = camera.get_view_matrix()

    # Position should be at correct distance from target
    assert np.linalg.norm(position - camera.target) == pytest.approx(camera.distance)

    # Up vector should be unit length
    assert np.linalg.norm(up) == pytest.approx(1.0)

    # View matrix should be 4x4
    assert view_matrix.shape == (4, 4)


def test_matrix_transformations():
    """Test that matrices perform expected transformations."""
    camera = Camera.front_view(distance=3.0)
    view_matrix = camera.get_view_matrix()

    # Test transformation of a point
    world_point = np.array(
        [0, 0, 0, 1], dtype=np.float32
    )  # Origin in homogeneous coords
    view_point = view_matrix @ world_point

    # After transformation, point should be in camera space
    assert len(view_point) == 4
    assert view_point[3] == pytest.approx(1.0)  # Homogeneous coordinate


def test_isometric_view_matrix():
    """Test view matrix for isometric view."""
    camera = Camera.isometric_view(distance=5.0)
    view_matrix = camera.get_view_matrix()

    # Check shape and type
    assert view_matrix.shape == (4, 4)
    assert view_matrix.dtype == np.float32

    # View matrix should be valid (invertible)
    det = np.linalg.det(view_matrix[:3, :3])
    assert abs(det) > 1e-6


def test_camera_copy_preserves_matrices():
    """Test that copied cameras produce same matrices."""
    original = Camera.from_spherical(
        target=np.array([1, 2, 3]),
        azimuth=np.pi / 3,
        elevation=np.pi / 4,
        roll=0.0,
        distance=4.0,
    )

    copy = original.copy()

    # Matrices should be identical
    assert np.allclose(original.get_view_matrix(), copy.get_view_matrix())
    assert np.allclose(
        original.get_projection_matrix(1.0), copy.get_projection_matrix(1.0)
    )


if __name__ == "__main__":
    pytest.main([__file__])
