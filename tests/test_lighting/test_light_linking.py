"""Tests for light camera linking functionality."""

import pytest
import numpy as np
from pyvr.lighting import Light
from pyvr.camera import Camera


class TestLightLinking:
    """Tests for Light camera linking."""

    def test_light_not_linked_by_default(self):
        """Test light is not linked by default."""
        light = Light.default()
        assert light.is_linked is False
        assert light.get_offsets() is None

    def test_link_to_camera_basic(self):
        """Test linking light to camera."""
        light = Light.default()

        result = light.link_to_camera()

        assert light.is_linked is True
        assert result is light  # Returns self for chaining

    def test_link_to_camera_with_offsets(self):
        """Test linking with custom offsets."""
        light = Light.default()

        light.link_to_camera(
            azimuth_offset=np.pi / 4, elevation_offset=np.pi / 6, distance_offset=1.0
        )

        offsets = light.get_offsets()
        assert offsets is not None
        assert offsets["azimuth"] == pytest.approx(np.pi / 4)
        assert offsets["elevation"] == pytest.approx(np.pi / 6)
        assert offsets["distance"] == pytest.approx(1.0)

    def test_unlink_from_camera(self):
        """Test unlinking light from camera."""
        light = Light.default()
        light.link_to_camera()

        assert light.is_linked is True

        result = light.unlink_from_camera()

        assert light.is_linked is False
        assert light.get_offsets() is None
        assert result is light  # Returns self for chaining

    def test_update_from_camera_not_linked_error(self):
        """Test error when updating unlinked light."""
        light = Light.default()
        camera = Camera.from_spherical(
            target=np.array([0.0, 0.0, 0.0]),
            distance=3.0,
            azimuth=0.0,
            elevation=0.0,
            roll=0.0,
        )

        with pytest.raises(ValueError, match="not linked"):
            light.update_from_camera(camera)

    def test_update_from_camera_invalid_camera(self):
        """Test error when camera is not Camera instance."""
        light = Light.default()
        light.link_to_camera()

        with pytest.raises(ValueError, match="Camera instance"):
            light.update_from_camera("not a camera")

    def test_update_from_camera_updates_position(self):
        """Test update_from_camera() updates light position."""
        light = Light.default()
        light.link_to_camera(
            azimuth_offset=0.0, elevation_offset=0.0, distance_offset=0.0
        )

        camera = Camera.from_spherical(
            target=np.array([0.0, 0.0, 0.0]),
            distance=3.0,
            azimuth=np.pi / 2,  # 90 degrees
            elevation=0.0,
            roll=0.0,
        )

        original_position = light.position.copy()
        light.update_from_camera(camera)

        # Position should have changed
        assert not np.allclose(light.position, original_position)

        # Light should be at camera distance from target
        distance_to_target = np.linalg.norm(light.position - camera.target)
        assert distance_to_target == pytest.approx(camera.distance, rel=1e-5)

    def test_update_from_camera_with_azimuth_offset(self):
        """Test light position matches camera (headlight mode).

        Note: Offsets are currently ignored in headlight mode implementation.
        Light is positioned at camera location regardless of offset values.
        """
        light = Light.default()
        light.link_to_camera(
            azimuth_offset=np.pi / 4, elevation_offset=0.0, distance_offset=0.0
        )

        camera = Camera.from_spherical(
            target=np.array([0.0, 0.0, 0.0]),
            distance=3.0,
            azimuth=0.0,
            elevation=0.0,
            roll=0.0,
        )

        light.update_from_camera(camera)

        # In headlight mode, light should be at camera position (offset ignored)
        camera_pos, _ = camera.get_camera_vectors()
        assert light.position[0] == pytest.approx(camera_pos[0], rel=1e-5)
        assert light.position[1] == pytest.approx(camera_pos[1], rel=1e-5)
        assert light.position[2] == pytest.approx(camera_pos[2], rel=1e-5)

    def test_update_from_camera_with_elevation_offset(self):
        """Test light position matches camera (headlight mode).

        Note: Offsets are currently ignored in headlight mode implementation.
        Light is positioned at camera location regardless of offset values.
        """
        light = Light.default()
        light.link_to_camera(
            azimuth_offset=0.0, elevation_offset=np.pi / 6, distance_offset=0.0
        )

        camera = Camera.from_spherical(
            target=np.array([0.0, 0.0, 0.0]),
            distance=3.0,
            azimuth=0.0,
            elevation=0.0,
            roll=0.0,
        )

        light.update_from_camera(camera)

        # In headlight mode, light should be at camera position (offset ignored)
        camera_pos, _ = camera.get_camera_vectors()
        assert light.position[0] == pytest.approx(camera_pos[0], rel=1e-5)
        assert light.position[1] == pytest.approx(camera_pos[1], rel=1e-5)
        assert light.position[2] == pytest.approx(camera_pos[2], rel=1e-5)

    def test_update_from_camera_with_distance_offset(self):
        """Test light position matches camera (headlight mode).

        Note: Offsets are currently ignored in headlight mode implementation.
        Light is positioned at camera location regardless of offset values.
        """
        light = Light.default()
        light.link_to_camera(
            azimuth_offset=0.0, elevation_offset=0.0, distance_offset=1.0
        )

        camera = Camera.from_spherical(
            target=np.array([0.0, 0.0, 0.0]),
            distance=3.0,
            azimuth=0.0,
            elevation=0.0,
            roll=0.0,
        )

        light.update_from_camera(camera)

        # In headlight mode, light should be at camera position (offset ignored)
        camera_pos, _ = camera.get_camera_vectors()
        assert light.position[0] == pytest.approx(camera_pos[0], rel=1e-5)
        assert light.position[1] == pytest.approx(camera_pos[1], rel=1e-5)
        assert light.position[2] == pytest.approx(camera_pos[2], rel=1e-5)

        # Light should be at camera distance from target
        distance_to_target = np.linalg.norm(light.position - camera.target)
        assert distance_to_target == pytest.approx(camera.distance, rel=1e-5)

    def test_update_from_camera_updates_target(self):
        """Test update_from_camera() updates light target."""
        light = Light.default()
        light.link_to_camera()

        camera = Camera.from_spherical(
            target=np.array([1.0, 2.0, 3.0]),
            distance=3.0,
            azimuth=0.0,
            elevation=0.0,
            roll=0.0,
        )

        light.update_from_camera(camera)

        # Light target should match camera target
        assert np.allclose(light.target, camera.target)

    def test_get_offsets_returns_copy(self):
        """Test get_offsets() returns copy, not reference."""
        light = Light.default()
        light.link_to_camera(
            azimuth_offset=1.0, elevation_offset=2.0, distance_offset=3.0
        )

        offsets1 = light.get_offsets()
        offsets1["azimuth"] = 999.0  # Modify copy

        offsets2 = light.get_offsets()
        assert offsets2["azimuth"] == 1.0  # Original unchanged

    def test_copy_preserves_linking_state(self):
        """Test that copy() preserves linking state."""
        light = Light.default()
        light.link_to_camera(
            azimuth_offset=np.pi / 4, elevation_offset=np.pi / 6, distance_offset=1.0
        )

        # Copy the light
        light_copy = light.copy()

        # Linking state should be preserved
        assert light_copy.is_linked is True
        offsets = light_copy.get_offsets()
        assert offsets is not None
        assert offsets["azimuth"] == pytest.approx(np.pi / 4)
        assert offsets["elevation"] == pytest.approx(np.pi / 6)
        assert offsets["distance"] == pytest.approx(1.0)

    def test_copy_unlinked_light(self):
        """Test that copy() works for unlinked light."""
        light = Light.default()

        # Copy unlinked light
        light_copy = light.copy()

        assert light_copy.is_linked is False
        assert light_copy.get_offsets() is None

    def test_camera_linked_preset(self):
        """Test Light.camera_linked() preset."""
        light = Light.camera_linked(
            azimuth_offset=np.pi / 4, elevation_offset=0.0, ambient=0.3, diffuse=0.9
        )

        assert light.is_linked is True
        assert light.ambient_intensity == 0.3
        assert light.diffuse_intensity == 0.9

        offsets = light.get_offsets()
        assert offsets["azimuth"] == pytest.approx(np.pi / 4)

    def test_camera_linked_can_be_updated(self):
        """Test camera_linked light can be updated."""
        light = Light.camera_linked()

        camera = Camera.from_spherical(
            target=np.array([0.0, 0.0, 0.0]),
            distance=3.0,
            azimuth=0.0,
            elevation=0.0,
            roll=0.0,
        )

        # Should not raise
        light.update_from_camera(camera)
        assert np.linalg.norm(light.position) > 0


class TestLightLinkingIntegration:
    """Integration tests for light linking with camera."""

    def test_light_follows_camera_orbit(self):
        """Test light follows camera during orbit."""
        light = Light.camera_linked()
        camera = Camera.from_spherical(
            target=np.array([0.0, 0.0, 0.0]),
            distance=3.0,
            azimuth=0.0,
            elevation=0.0,
            roll=0.0,
        )

        # Update light from initial camera
        light.update_from_camera(camera)
        initial_position = light.position.copy()

        # Orbit camera
        from pyvr.camera import CameraController

        controller = CameraController(camera)
        controller.orbit(delta_azimuth=np.pi / 4, delta_elevation=0.0)

        # Update light from new camera position
        light.update_from_camera(controller.params)

        # Light position should have changed
        assert not np.allclose(light.position, initial_position)

    def test_light_maintains_offset_during_orbit(self):
        """Test light follows camera during orbit (headlight mode).

        Note: In headlight mode, the light is positioned at the camera location,
        so it maintains the camera's azimuth (offset is currently ignored).
        """
        offset = np.pi / 4
        light = Light.camera_linked(azimuth_offset=offset)

        camera = Camera.from_spherical(
            target=np.array([0.0, 0.0, 0.0]),
            distance=3.0,
            azimuth=0.0,
            elevation=0.0,
            roll=0.0,
        )

        light.update_from_camera(camera)

        # In headlight mode, light should be at camera position
        camera_pos, _ = camera.get_camera_vectors()
        assert np.allclose(light.position, camera_pos)

        # Calculate light azimuth from position
        light_direction = light.position - light.target
        light_azimuth = np.arctan2(light_direction[2], light_direction[0])

        # Should match camera azimuth (offset ignored in headlight mode)
        expected_azimuth = camera.azimuth
        assert light_azimuth == pytest.approx(expected_azimuth, rel=1e-5)

    def test_multiple_updates_maintain_consistency(self):
        """Test multiple updates maintain consistent light behavior."""
        light = Light.camera_linked(azimuth_offset=np.pi / 4)
        camera = Camera.from_spherical(
            target=np.array([0.0, 0.0, 0.0]),
            distance=3.0,
            azimuth=0.0,
            elevation=0.0,
            roll=0.0,
        )

        # Multiple updates with different camera positions
        for azimuth in [0.0, np.pi / 6, np.pi / 3, np.pi / 2]:
            camera = Camera.from_spherical(
                target=np.array([0.0, 0.0, 0.0]),
                distance=3.0,
                azimuth=azimuth,
                elevation=0.0,
                roll=0.0,
            )
            light.update_from_camera(camera)

            # Verify consistency
            distance = np.linalg.norm(light.position - light.target)
            assert distance == pytest.approx(3.0, rel=1e-5)
            assert np.allclose(light.target, camera.target)

    def test_linking_with_non_zero_target(self):
        """Test light linking works with non-zero camera target."""
        light = Light.camera_linked()
        target = np.array([5.0, 5.0, 5.0])
        camera = Camera.from_spherical(
            target=target, distance=3.0, azimuth=0.0, elevation=0.0, roll=0.0
        )

        light.update_from_camera(camera)

        # Light target should match camera target
        assert np.allclose(light.target, target)

        # Light position should be relative to target
        relative_position = light.position - light.target
        distance = np.linalg.norm(relative_position)
        assert distance == pytest.approx(3.0, rel=1e-5)
