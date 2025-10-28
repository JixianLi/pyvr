"""Tests for Light class."""
import numpy as np
import pytest

from pyvr.lighting import Light, LightError


class TestLightCreation:
    """Test Light class instantiation and factory methods."""

    def test_default_light(self):
        """Default light should have standard parameters."""
        light = Light.default()

        assert light.ambient_intensity == 0.2
        assert light.diffuse_intensity == 0.8
        assert light.position.shape == (3,)
        assert light.target.shape == (3,)

    def test_directional_light(self):
        """Directional light should point in specified direction."""
        direction = np.array([1, 0, 0])
        light = Light.directional(direction=direction)

        # Direction should be normalized
        light_dir = light.get_direction()
        expected_dir = direction / np.linalg.norm(direction)

        assert np.allclose(light_dir, expected_dir, atol=0.01)

    def test_point_light(self):
        """Point light should be at specified position."""
        position = np.array([5, 5, 5])
        light = Light.point_light(position=position)

        assert np.allclose(light.position, position)

    def test_ambient_only(self):
        """Ambient-only light should have no diffuse component."""
        light = Light.ambient_only(intensity=0.5)

        assert light.ambient_intensity == 0.5
        assert light.diffuse_intensity == 0.0

    def test_custom_parameters(self):
        """Light should accept custom parameters."""
        light = Light(
            position=np.array([1, 2, 3]),
            target=np.array([4, 5, 6]),
            ambient_intensity=0.3,
            diffuse_intensity=0.7,
        )

        assert np.allclose(light.position, [1, 2, 3])
        assert np.allclose(light.target, [4, 5, 6])
        assert light.ambient_intensity == 0.3
        assert light.diffuse_intensity == 0.7


class TestLightValidation:
    """Test Light validation."""

    def test_invalid_ambient_intensity(self):
        """Ambient intensity outside [0, 1] should raise error."""
        with pytest.raises(ValueError, match="ambient_intensity"):
            Light(ambient_intensity=1.5)

        with pytest.raises(ValueError, match="ambient_intensity"):
            Light(ambient_intensity=-0.1)

    def test_invalid_diffuse_intensity(self):
        """Diffuse intensity outside [0, 1] should raise error."""
        with pytest.raises(ValueError, match="diffuse_intensity"):
            Light(diffuse_intensity=2.0)

        with pytest.raises(ValueError, match="diffuse_intensity"):
            Light(diffuse_intensity=-0.5)

    def test_invalid_position_shape(self):
        """Invalid position shape should raise error."""
        with pytest.raises(ValueError, match="position"):
            Light(position=np.array([1, 2]))  # 2D instead of 3D

    def test_invalid_target_shape(self):
        """Invalid target shape should raise error."""
        with pytest.raises(ValueError, match="target"):
            Light(target=np.array([1, 2, 3, 4]))  # 4D instead of 3D


class TestLightMethods:
    """Test Light class methods."""

    def test_get_direction(self):
        """get_direction should return normalized direction vector."""
        light = Light(
            position=np.array([0, 0, 0]),
            target=np.array([3, 4, 0]),
        )

        direction = light.get_direction()

        # Should be normalized
        assert np.isclose(np.linalg.norm(direction), 1.0)

        # Should point from position to target
        expected = np.array([3, 4, 0]) / 5.0  # normalized
        assert np.allclose(direction, expected)

    def test_get_direction_same_position_target(self):
        """get_direction with position==target should return default."""
        light = Light(
            position=np.array([1, 1, 1]),
            target=np.array([1, 1, 1]),
        )

        direction = light.get_direction()

        # Should return default direction (not raise error)
        assert direction.shape == (3,)
        assert np.isclose(np.linalg.norm(direction), 1.0)

    def test_copy(self):
        """copy should create independent instance."""
        original = Light.default()
        copy = original.copy()

        # Modify copy
        copy.ambient_intensity = 0.9

        # Original should be unchanged
        assert original.ambient_intensity == 0.2

    def test_repr(self):
        """__repr__ should return informative string."""
        light = Light.default()
        repr_str = repr(light)

        assert "Light(" in repr_str
        assert "position=" in repr_str
        assert "ambient=" in repr_str


class TestLightIntensities:
    """Test various intensity combinations."""

    def test_zero_ambient(self):
        """Light with zero ambient should be valid."""
        light = Light(ambient_intensity=0.0)
        assert light.ambient_intensity == 0.0

    def test_zero_diffuse(self):
        """Light with zero diffuse should be valid."""
        light = Light(diffuse_intensity=0.0)
        assert light.diffuse_intensity == 0.0

    def test_full_intensities(self):
        """Light with maximum intensities should be valid."""
        light = Light(ambient_intensity=1.0, diffuse_intensity=1.0)
        assert light.ambient_intensity == 1.0
        assert light.diffuse_intensity == 1.0


class TestLightDirectionalFactory:
    """Test directional light factory method variations."""

    def test_directional_custom_distance(self):
        """Directional light should respect custom distance."""
        direction = np.array([1, 0, 0])
        distance = 50.0
        light = Light.directional(direction=direction, distance=distance)

        # Light should be positioned at -direction * distance
        expected_position = -direction * distance
        assert np.allclose(light.position, expected_position)

    def test_directional_custom_intensities(self):
        """Directional light should respect custom intensities."""
        light = Light.directional(
            direction=[0, 1, 0],
            ambient=0.1,
            diffuse=0.95
        )

        assert light.ambient_intensity == 0.1
        assert light.diffuse_intensity == 0.95

    def test_directional_normalizes_direction(self):
        """Directional light should normalize input direction."""
        # Use non-normalized direction
        direction = np.array([3, 4, 0])
        light = Light.directional(direction=direction)

        # Get direction and verify it's normalized
        result_dir = light.get_direction()
        assert np.isclose(np.linalg.norm(result_dir), 1.0)


class TestLightPointFactory:
    """Test point light factory method variations."""

    def test_point_light_default_target(self):
        """Point light should default to origin target."""
        position = np.array([5, 5, 5])
        light = Light.point_light(position=position)

        assert np.allclose(light.target, [0, 0, 0])

    def test_point_light_custom_target(self):
        """Point light should accept custom target."""
        position = np.array([5, 5, 5])
        target = np.array([1, 2, 3])
        light = Light.point_light(position=position, target=target)

        assert np.allclose(light.target, target)

    def test_point_light_custom_intensities(self):
        """Point light should respect custom intensities."""
        light = Light.point_light(
            position=[0, 0, 0],
            ambient=0.15,
            diffuse=0.75
        )

        assert light.ambient_intensity == 0.15
        assert light.diffuse_intensity == 0.75
