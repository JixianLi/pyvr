"""Tests for Volume class."""
import numpy as np
import pytest

from pyvr.volume import Volume, VolumeError


class TestVolumeCreation:
    """Test Volume class instantiation."""

    def test_basic_volume(self):
        """Volume should accept 3D data."""
        data = np.zeros((64, 64, 64), dtype=np.float32)
        volume = Volume(data=data)

        assert volume.shape == (64, 64, 64)
        assert volume.has_normals is False

    def test_volume_with_normals(self):
        """Volume should accept normals."""
        data = np.zeros((64, 64, 64), dtype=np.float32)
        normals = np.zeros((64, 64, 64, 3), dtype=np.float32)
        volume = Volume(data=data, normals=normals)

        assert volume.has_normals is True

    def test_volume_with_custom_bounds(self):
        """Volume should accept custom bounds."""
        data = np.zeros((64, 64, 64))
        volume = Volume(
            data=data,
            min_bounds=np.array([-1, -1, -1]),
            max_bounds=np.array([1, 1, 1]),
        )

        assert np.allclose(volume.min_bounds, [-1, -1, -1])
        assert np.allclose(volume.max_bounds, [1, 1, 1])

    def test_volume_with_name(self):
        """Volume should accept optional name."""
        data = np.zeros((64, 64, 64))
        volume = Volume(data=data, name="test_volume")

        assert volume.name == "test_volume"


class TestVolumeValidation:
    """Test Volume validation."""

    def test_invalid_data_shape(self):
        """Non-3D data should raise error."""
        with pytest.raises(ValueError, match="3D"):
            Volume(data=np.zeros((64, 64)))  # 2D

    def test_invalid_normal_shape(self):
        """Invalid normal shape should raise error."""
        data = np.zeros((64, 64, 64))
        normals = np.zeros((64, 64, 64))  # Missing channel dimension

        with pytest.raises(ValueError, match="shape"):
            Volume(data=data, normals=normals)

    def test_invalid_bounds(self):
        """Invalid bounds should raise error."""
        data = np.zeros((64, 64, 64))

        with pytest.raises(ValueError):
            Volume(
                data=data,
                min_bounds=np.array([1, 1, 1]),
                max_bounds=np.array([-1, -1, -1]),  # max < min
            )


class TestVolumeProperties:
    """Test Volume property methods."""

    def test_shape_property(self):
        """shape should return data dimensions."""
        data = np.zeros((100, 50, 75))
        volume = Volume(data=data)

        assert volume.shape == (100, 50, 75)

    def test_dimensions_property(self):
        """dimensions should return physical size."""
        data = np.zeros((64, 64, 64))
        volume = Volume(
            data=data,
            min_bounds=np.array([0, 0, 0]),
            max_bounds=np.array([2, 3, 4]),
        )

        expected = np.array([2, 3, 4])
        assert np.allclose(volume.dimensions, expected)

    def test_center_property(self):
        """center should return bounding box center."""
        data = np.zeros((64, 64, 64))
        volume = Volume(
            data=data,
            min_bounds=np.array([-1, -2, -3]),
            max_bounds=np.array([1, 2, 3]),
        )

        expected = np.array([0, 0, 0])
        assert np.allclose(volume.center, expected)

    def test_voxel_spacing(self):
        """voxel_spacing should calculate spacing correctly."""
        data = np.zeros((100, 100, 100))
        volume = Volume(
            data=data,
            min_bounds=np.array([0, 0, 0]),
            max_bounds=np.array([1, 1, 1]),
        )

        expected = np.array([0.01, 0.01, 0.01])
        assert np.allclose(volume.voxel_spacing, expected)


class TestVolumeMethods:
    """Test Volume methods."""

    def test_compute_normals(self):
        """compute_normals should generate normal data."""
        data = np.random.rand(32, 32, 32).astype(np.float32)
        volume = Volume(data=data)

        assert not volume.has_normals

        volume.compute_normals()

        assert volume.has_normals
        assert volume.normals.shape == (32, 32, 32, 3)

    def test_normalize_minmax(self):
        """normalize with minmax should scale to [0, 1]."""
        data = np.random.rand(32, 32, 32) * 100
        volume = Volume(data=data)

        normalized = volume.normalize(method="minmax")

        assert normalized.data.min() >= 0.0
        assert normalized.data.max() <= 1.0

    def test_copy(self):
        """copy should create independent instance."""
        data = np.zeros((32, 32, 32))
        original = Volume(data=data, name="original")
        copy = original.copy()

        # Modify copy
        copy.data[0, 0, 0] = 1.0
        copy.name = "copy"

        # Original should be unchanged
        assert original.data[0, 0, 0] == 0.0
        assert original.name == "original"

    def test_repr(self):
        """__repr__ should return informative string."""
        data = np.zeros((64, 64, 64))
        volume = Volume(data=data, name="test")

        repr_str = repr(volume)

        assert "Volume" in repr_str
        assert "test" in repr_str
        assert "64" in repr_str


if __name__ == "__main__":
    pytest.main([__file__])
