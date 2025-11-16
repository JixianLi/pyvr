"""Tests for trackball control helper functions."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from pyvr.camera import Camera
from pyvr.camera.control import (
    CameraController,
    _map_to_sphere,
    _camera_to_quaternion,
    _quaternion_to_camera_angles,
    get_camera_pos_from_params,
)


class TestMapToSphere:
    """Tests for _map_to_sphere() helper function."""

    def test_origin_maps_to_front(self):
        """Origin (0, 0) should map to point at front of sphere (0, 0, 1)."""
        point = _map_to_sphere(0.0, 0.0)
        expected = np.array([0.0, 0.0, 1.0])
        np.testing.assert_allclose(point, expected, atol=1e-6)

    def test_point_on_sphere(self):
        """Point on sphere surface should have unit length."""
        point = _map_to_sphere(0.5, 0.5)
        norm = np.linalg.norm(point)
        assert abs(norm - 1.0) < 1e-6, f"Expected unit length, got {norm}"

    def test_point_inside_sphere(self):
        """Point clearly inside sphere radius should be projected to sphere."""
        point = _map_to_sphere(0.3, 0.3)
        # Should have positive z component (in front)
        assert point[2] > 0
        # Should have unit length
        assert abs(np.linalg.norm(point) - 1.0) < 1e-6

    def test_point_outside_sphere(self):
        """Point outside sphere should use hyperbolic sheet."""
        point = _map_to_sphere(1.5, 1.5)
        # Should still have unit length
        assert abs(np.linalg.norm(point) - 1.0) < 1e-6
        # Z should be small (close to edge)
        assert point[2] < 0.5

    def test_radius_scaling(self):
        """Different radius values should scale sphere appropriately."""
        point1 = _map_to_sphere(0.5, 0.5, radius=1.0)
        point2 = _map_to_sphere(0.5, 0.5, radius=2.0)
        # Both should have unit length (normalized)
        assert abs(np.linalg.norm(point1) - 1.0) < 1e-6
        assert abs(np.linalg.norm(point2) - 1.0) < 1e-6
        # But z components should differ
        assert point2[2] > point1[2]  # Larger radius -> higher z

    def test_symmetry(self):
        """Mapping should be symmetric in x and y."""
        point1 = _map_to_sphere(0.5, 0.3)
        point2 = _map_to_sphere(0.3, 0.5)
        # x and y should be swapped, z should be same
        assert abs(point1[0] - point2[1]) < 1e-6
        assert abs(point1[1] - point2[0]) < 1e-6
        assert abs(point1[2] - point2[2]) < 1e-6


class TestCameraToQuaternion:
    """Tests for _camera_to_quaternion() helper function."""

    def test_front_view(self):
        """Front view camera should produce consistent quaternion."""
        camera = Camera.front_view(distance=3.0)
        quat = _camera_to_quaternion(camera)
        assert isinstance(quat, R)
        # Quaternion should be valid (unit length)
        quat_array = quat.as_quat()
        assert abs(np.linalg.norm(quat_array) - 1.0) < 1e-6

    def test_side_view(self):
        """Side view should produce 90° rotation from front view."""
        camera = Camera.side_view(distance=3.0)
        quat = _camera_to_quaternion(camera)
        assert isinstance(quat, R)

    def test_top_view(self):
        """Top view should produce rotation looking down."""
        camera = Camera.top_view(distance=3.0)
        quat = _camera_to_quaternion(camera)
        assert isinstance(quat, R)

    def test_isometric_view(self):
        """Isometric view should produce known orientation."""
        camera = Camera.isometric_view(distance=3.0)
        quat = _camera_to_quaternion(camera)
        assert isinstance(quat, R)

    def test_roundtrip_conversion(self):
        """Converting camera → quat → angles → camera should be consistent."""
        original = Camera.isometric_view(distance=3.0)
        quat = _camera_to_quaternion(original)

        # Convert back to angles
        az, el, roll = _quaternion_to_camera_angles(
            quat,
            original.target,
            original.distance,
            original.init_pos,
            original.init_up
        )

        # Create new camera with computed angles
        reconstructed = Camera(
            target=original.target,
            azimuth=az,
            elevation=el,
            roll=roll,
            distance=original.distance,
            init_pos=original.init_pos,
            init_up=original.init_up
        )

        # Compare camera positions and up vectors
        orig_pos, orig_up = get_camera_pos_from_params(original)
        recon_pos, recon_up = get_camera_pos_from_params(reconstructed)

        np.testing.assert_allclose(orig_pos, recon_pos, atol=1e-5)
        np.testing.assert_allclose(orig_up, recon_up, atol=1e-5)


class TestQuaternionToCameraAngles:
    """Tests for _quaternion_to_camera_angles() helper function."""

    def test_identity_rotation(self):
        """Front view camera roundtrip should preserve angles."""
        camera = Camera.front_view(distance=3.0)
        # Get quaternion from camera
        quat = _camera_to_quaternion(camera)

        # Convert back to angles
        az, el, roll = _quaternion_to_camera_angles(
            quat,
            camera.target,
            camera.distance,
            camera.init_pos,
            camera.init_up
        )

        # Should match camera's original angles
        np.testing.assert_allclose(az, camera.azimuth, atol=1e-5)
        np.testing.assert_allclose(el, camera.elevation, atol=1e-5)
        np.testing.assert_allclose(roll, camera.roll, atol=1e-5)

    def test_known_camera_angles(self):
        """Known camera orientations should decompose to correct angles."""
        # Test isometric view
        camera = Camera.isometric_view(distance=3.0)
        quat = _camera_to_quaternion(camera)

        az, el, roll = _quaternion_to_camera_angles(
            quat,
            camera.target,
            camera.distance,
            camera.init_pos,
            camera.init_up
        )

        # Should match original angles
        np.testing.assert_allclose(az, camera.azimuth, atol=1e-5)
        np.testing.assert_allclose(el, camera.elevation, atol=1e-5)
        np.testing.assert_allclose(roll, camera.roll, atol=1e-5)

    def test_gimbal_lock_elevation_90(self):
        """Elevation = π/2 (looking straight up) should be handled."""
        camera = Camera(
            target=np.array([0, 0, 0], dtype=np.float32),
            azimuth=np.pi/4,
            elevation=np.pi/2,
            roll=0.0,
            distance=3.0
        )
        quat = _camera_to_quaternion(camera)

        az, el, roll = _quaternion_to_camera_angles(
            quat,
            camera.target,
            camera.distance,
            camera.init_pos,
            camera.init_up
        )

        # Elevation should be correct
        np.testing.assert_allclose(el, np.pi/2, atol=1e-5)
        # Azimuth is undefined at gimbal lock, but shouldn't crash

    def test_gimbal_lock_elevation_neg90(self):
        """Elevation = -π/2 (looking straight down) should be handled."""
        camera = Camera(
            target=np.array([0, 0, 0], dtype=np.float32),
            azimuth=np.pi/4,
            elevation=-np.pi/2,
            roll=0.0,
            distance=3.0
        )
        quat = _camera_to_quaternion(camera)

        az, el, roll = _quaternion_to_camera_angles(
            quat,
            camera.target,
            camera.distance,
            camera.init_pos,
            camera.init_up
        )

        # Elevation should be correct
        np.testing.assert_allclose(el, -np.pi/2, atol=1e-5)

    def test_roundtrip_accuracy(self):
        """Multiple roundtrips should maintain accuracy."""
        camera = Camera.from_spherical(
            target=np.array([0, 0, 0], dtype=np.float32),
            azimuth=1.2,
            elevation=0.8,
            roll=0.3,
            distance=3.0
        )

        # Roundtrip 1: camera → quat → angles
        quat1 = _camera_to_quaternion(camera)
        az1, el1, roll1 = _quaternion_to_camera_angles(
            quat1, camera.target, camera.distance,
            camera.init_pos, camera.init_up
        )

        # Roundtrip 2: angles → camera → quat → angles
        camera2 = Camera(
            target=camera.target,
            azimuth=az1,
            elevation=el1,
            roll=roll1,
            distance=camera.distance,
            init_pos=camera.init_pos,
            init_up=camera.init_up
        )
        quat2 = _camera_to_quaternion(camera2)
        az2, el2, roll2 = _quaternion_to_camera_angles(
            quat2, camera2.target, camera2.distance,
            camera2.init_pos, camera2.init_up
        )

        # Both roundtrips should give same result
        np.testing.assert_allclose(az1, az2, atol=1e-5)
        np.testing.assert_allclose(el1, el2, atol=1e-5)
        np.testing.assert_allclose(roll1, roll2, atol=1e-5)


class TestCameraControllerTrackball:
    """Tests for CameraController.trackball() method."""

    def test_trackball_horizontal_right(self):
        """Dragging right should rotate camera left around target."""
        controller = CameraController(Camera.front_view(distance=3.0))
        initial_azimuth = controller.params.azimuth

        # Drag right
        controller.trackball(
            dx=100, dy=0,
            viewport_width=800, viewport_height=600
        )

        # Azimuth should decrease (camera rotates left)
        assert controller.params.azimuth < initial_azimuth

    def test_trackball_horizontal_left(self):
        """Dragging left should rotate camera right around target."""
        controller = CameraController(Camera.front_view(distance=3.0))
        initial_azimuth = controller.params.azimuth

        # Drag left
        controller.trackball(
            dx=-100, dy=0,
            viewport_width=800, viewport_height=600
        )

        # Azimuth should increase (camera rotates right)
        assert controller.params.azimuth > initial_azimuth

    def test_trackball_vertical_up(self):
        """Dragging up should rotate camera up (increase elevation)."""
        controller = CameraController(Camera.front_view(distance=3.0))
        initial_elevation = controller.params.elevation

        # Drag up (negative dy in screen coords)
        controller.trackball(
            dx=0, dy=-100,
            viewport_width=800, viewport_height=600
        )

        # Elevation should increase (camera looks up)
        assert controller.params.elevation > initial_elevation

    def test_trackball_vertical_down(self):
        """Dragging down should rotate camera down (decrease elevation)."""
        controller = CameraController(Camera.front_view(distance=3.0))
        initial_elevation = controller.params.elevation

        # Drag down (positive dy in screen coords)
        controller.trackball(
            dx=0, dy=100,
            viewport_width=800, viewport_height=600
        )

        # Elevation should decrease (camera looks down)
        assert controller.params.elevation < initial_elevation

    def test_trackball_diagonal(self):
        """Diagonal drag should combine horizontal and vertical rotation."""
        controller = CameraController(Camera.front_view(distance=3.0))
        initial_azimuth = controller.params.azimuth
        initial_elevation = controller.params.elevation

        # Drag diagonally (right and up)
        controller.trackball(
            dx=100, dy=-100,
            viewport_width=800, viewport_height=600
        )

        # Both angles should change
        assert controller.params.azimuth != initial_azimuth
        assert controller.params.elevation != initial_elevation

    def test_trackball_zero_movement(self):
        """Zero mouse movement should not change camera."""
        controller = CameraController(Camera.isometric_view(distance=3.0))
        initial_azimuth = controller.params.azimuth
        initial_elevation = controller.params.elevation
        initial_roll = controller.params.roll

        # No movement
        controller.trackball(
            dx=0, dy=0,
            viewport_width=800, viewport_height=600
        )

        # Camera should be unchanged
        assert controller.params.azimuth == initial_azimuth
        assert controller.params.elevation == initial_elevation
        assert controller.params.roll == initial_roll

    def test_trackball_very_small_movement(self):
        """Very small movements should be ignored for stability."""
        controller = CameraController(Camera.isometric_view(distance=3.0))
        initial_azimuth = controller.params.azimuth
        initial_elevation = controller.params.elevation

        # Very small movement (< 0.001 normalized)
        controller.trackball(
            dx=0.1, dy=0.1,
            viewport_width=800, viewport_height=600
        )

        # Camera should be unchanged (below threshold)
        assert controller.params.azimuth == initial_azimuth
        assert controller.params.elevation == initial_elevation

    def test_trackball_sensitivity_scaling(self):
        """Higher sensitivity should produce larger rotation."""
        # Test with sensitivity=1.0
        controller1 = CameraController(Camera.front_view(distance=3.0))
        controller1.trackball(
            dx=50, dy=0,
            viewport_width=800, viewport_height=600,
            sensitivity=1.0
        )
        rotation1 = abs(controller1.params.azimuth)

        # Test with sensitivity=2.0
        controller2 = CameraController(Camera.front_view(distance=3.0))
        controller2.trackball(
            dx=50, dy=0,
            viewport_width=800, viewport_height=600,
            sensitivity=2.0
        )
        rotation2 = abs(controller2.params.azimuth)

        # Higher sensitivity should produce more rotation
        assert rotation2 > rotation1
        # Should be approximately double
        assert abs(rotation2 / rotation1 - 2.0) < 0.5

    def test_trackball_invalid_viewport_width(self):
        """Invalid viewport width should raise ValueError."""
        controller = CameraController(Camera.front_view(distance=3.0))

        with pytest.raises(ValueError, match="Viewport dimensions must be positive"):
            controller.trackball(
                dx=10, dy=10,
                viewport_width=0, viewport_height=600
            )

    def test_trackball_invalid_viewport_height(self):
        """Invalid viewport height should raise ValueError."""
        controller = CameraController(Camera.front_view(distance=3.0))

        with pytest.raises(ValueError, match="Viewport dimensions must be positive"):
            controller.trackball(
                dx=10, dy=10,
                viewport_width=800, viewport_height=-100
            )

    def test_trackball_preserves_distance(self):
        """Trackball rotation should not change distance to target."""
        controller = CameraController(Camera.isometric_view(distance=5.0))
        initial_distance = controller.params.distance

        # Apply several trackball movements
        controller.trackball(dx=50, dy=30, viewport_width=800, viewport_height=600)
        controller.trackball(dx=-30, dy=50, viewport_width=800, viewport_height=600)

        # Distance should be unchanged
        assert abs(controller.params.distance - initial_distance) < 1e-6

    def test_trackball_preserves_target(self):
        """Trackball rotation should not change target position."""
        target = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        camera = Camera.isometric_view(target=target, distance=5.0)
        controller = CameraController(camera)

        # Apply trackball movement
        controller.trackball(dx=50, dy=30, viewport_width=800, viewport_height=600)

        # Target should be unchanged
        np.testing.assert_allclose(controller.params.target, target)

    def test_trackball_continuous_rotation(self):
        """Multiple small trackball movements should accumulate smoothly."""
        controller = CameraController(Camera.front_view(distance=3.0))
        initial_azimuth = controller.params.azimuth

        # Apply 5 small movements
        for _ in range(5):
            controller.trackball(
                dx=20, dy=0,
                viewport_width=800, viewport_height=600
            )

        # Total rotation should be accumulated
        total_rotation = abs(controller.params.azimuth - initial_azimuth)
        assert total_rotation > 0.1  # Significant rotation

    def test_trackball_different_viewport_sizes(self):
        """Trackball should work with different viewport aspect ratios."""
        # Wide viewport
        controller1 = CameraController(Camera.front_view(distance=3.0))
        controller1.trackball(dx=50, dy=0, viewport_width=1600, viewport_height=600)

        # Tall viewport
        controller2 = CameraController(Camera.front_view(distance=3.0))
        controller2.trackball(dx=50, dy=0, viewport_width=600, viewport_height=1600)

        # Both should produce rotation (exact values may differ due to scaling)
        assert abs(controller1.params.azimuth) > 0
        assert abs(controller2.params.azimuth) > 0

    def test_trackball_from_different_orientations(self):
        """Trackball should work correctly from various starting orientations."""
        # Test from side view
        controller = CameraController(Camera.side_view(distance=3.0))
        initial_az = controller.params.azimuth
        controller.trackball(dx=50, dy=0, viewport_width=800, viewport_height=600)
        assert controller.params.azimuth != initial_az

        # Test from top view
        controller = CameraController(Camera.top_view(distance=3.0))
        initial_el = controller.params.elevation
        controller.trackball(dx=0, dy=50, viewport_width=800, viewport_height=600)
        assert controller.params.elevation != initial_el
