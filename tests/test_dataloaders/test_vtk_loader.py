"""Tests for VTK volume loader."""

from pathlib import Path

import numpy as np
import pytest

from pyvr.dataloaders import load_vtk_volume


class TestBasicLoading:
    """Test basic VTK file loading functionality."""

    def test_load_hydrogen_vti(self):
        """Load hydrogen.vti with default scalars_name."""
        volume = load_vtk_volume("example_data/hydrogen.vti")

        # Verify basic properties
        assert volume.name == "hydrogen.vti(Scalars_)"
        assert volume.shape == (128, 128, 128)
        assert volume.data.dtype == np.float32
        assert volume.has_normals is True

        # Verify data is normalized
        assert 0.0 <= volume.data.min() <= volume.data.max() <= 1.0

    def test_load_fuel_vti(self):
        """Load fuel.vti with default scalars_name."""
        volume = load_vtk_volume("example_data/fuel.vti")

        # Verify basic properties
        assert volume.name == "fuel.vti(Scalars_)"
        assert volume.shape == (64, 64, 64)
        assert volume.data.dtype == np.float32
        assert volume.has_normals is True

        # Verify data is normalized
        assert 0.0 <= volume.data.min() <= volume.data.max() <= 1.0


class TestBoundsCalculation:
    """Test bounds calculation with aspect ratio preservation."""

    def test_bounds_centered_at_origin(self):
        """Verify bounds are centered at [0, 0, 0]."""
        volume = load_vtk_volume("example_data/hydrogen.vti")

        center = volume.center
        assert np.allclose(center, [0.0, 0.0, 0.0], atol=1e-6)

    def test_bounds_longest_dimension_maps_to_one(self):
        """Verify longest dimension maps to [-1, 1]."""
        volume = load_vtk_volume("example_data/hydrogen.vti")

        # For cubic volume with uniform spacing, all dims should be [-1, 1]
        max_extent = np.max(np.abs(volume.max_bounds))
        assert np.isclose(max_extent, 1.0, atol=1e-6)

    def test_aspect_ratio_preserved(self):
        """Verify aspect ratio matches physical dimensions."""
        volume = load_vtk_volume("example_data/hydrogen.vti")

        # hydrogen.vti has uniform spacing (1, 1, 1) and cubic dims (128, 128, 128)
        # So physical dims are (128, 128, 128) and bounds should be cubic
        dims = volume.dimensions
        ratios = dims / dims[0]  # Ratio relative to first dimension

        # For cubic volume, all ratios should be 1.0
        assert np.allclose(ratios, [1.0, 1.0, 1.0], atol=1e-6)

        # Verify bounds match this aspect ratio
        bounds_dims = volume.max_bounds - volume.min_bounds
        bounds_ratios = bounds_dims / bounds_dims[0]
        assert np.allclose(bounds_ratios, ratios, atol=1e-6)


class TestNormalization:
    """Test data normalization to [0, 1] range."""

    def test_data_normalized_to_unit_range(self):
        """Verify data is normalized to [0, 1]."""
        volume = load_vtk_volume("example_data/hydrogen.vti")

        data_min = volume.data.min()
        data_max = volume.data.max()

        # Should be normalized to [0, 1]
        assert data_min >= 0.0
        assert data_max <= 1.0

        # For real data, should actually use the full range
        # (not just be within [0, 1] but close to 0 and 1)
        assert data_min < 0.1  # Should have values near 0
        assert data_max > 0.9  # Should have values near 1

    def test_data_converted_to_float32(self):
        """Verify data is converted to float32 dtype."""
        volume_h = load_vtk_volume("example_data/hydrogen.vti")
        volume_f = load_vtk_volume("example_data/fuel.vti")

        assert volume_h.data.dtype == np.float32
        assert volume_f.data.dtype == np.float32


class TestVolumeMetadata:
    """Test Volume object metadata and properties."""

    def test_volume_name_format(self):
        """Verify volume name format is 'filename.vti(ScalarName)'."""
        volume = load_vtk_volume("example_data/hydrogen.vti")
        assert volume.name == "hydrogen.vti(Scalars_)"

        # Test with fuel.vti
        volume_f = load_vtk_volume("example_data/fuel.vti")
        assert volume_f.name == "fuel.vti(Scalars_)"

    def test_normals_computed(self):
        """Verify normals are automatically computed."""
        volume = load_vtk_volume("example_data/hydrogen.vti")
        assert volume.has_normals is True

        volume_f = load_vtk_volume("example_data/fuel.vti")
        assert volume_f.has_normals is True

    def test_normals_shape(self):
        """Verify normals have correct shape (D, H, W, 3)."""
        volume = load_vtk_volume("example_data/hydrogen.vti")
        expected_shape = volume.shape + (3,)
        assert volume.normals.shape == expected_shape

        volume_f = load_vtk_volume("example_data/fuel.vti")
        expected_shape_f = volume_f.shape + (3,)
        assert volume_f.normals.shape == expected_shape_f


class TestErrorHandling:
    """Test error handling and validation."""

    def test_file_not_found(self):
        """Non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError) as exc_info:
            load_vtk_volume("nonexistent.vti")

        assert "not found" in str(exc_info.value).lower()

    def test_invalid_scalars_name(self):
        """Wrong scalar name raises ValueError with available arrays listed."""
        with pytest.raises(ValueError) as exc_info:
            load_vtk_volume("example_data/hydrogen.vti", scalars_name="InvalidName")

        error_msg = str(exc_info.value)
        assert "not found" in error_msg
        assert "Available arrays" in error_msg
        assert "Scalars_" in error_msg  # Should list the actual available array


class TestPathHandling:
    """Test path handling (string and Path objects)."""

    def test_string_path(self):
        """Accept string paths."""
        volume = load_vtk_volume("example_data/hydrogen.vti")
        assert volume.name == "hydrogen.vti(Scalars_)"
        assert volume.shape == (128, 128, 128)

    def test_pathlib_path(self):
        """Accept pathlib.Path objects."""
        path = Path("example_data/hydrogen.vti")
        volume = load_vtk_volume(path)
        assert volume.name == "hydrogen.vti(Scalars_)"
        assert volume.shape == (128, 128, 128)
