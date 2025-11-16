# Phase 02: Comprehensive Testing

**Feature:** VTK Data Loader v0.4.0
**Phase:** 02 of 03
**Goal:** Comprehensive test suite with >85% coverage

## Scope

This phase delivers complete test coverage for the VTK loader:
- Create `tests/test_dataloaders/` test structure
- Implement all test categories from design
- Verify >85% coverage for dataloaders module
- All tests passing

**Deliverables:**
- `tests/test_dataloaders/__init__.py`
- `tests/test_dataloaders/test_vtk_loader.py`

## Implementation

### Task 1: Create Test Module Structure

**File:** `tests/test_dataloaders/__init__.py`

**Create new file with content:**
```python
"""Tests for PyVR data loaders module."""
```

**Verification:**
```bash
ls tests/test_dataloaders/__init__.py
```

Expected: File exists

---

### Task 2: Implement Basic Loading Tests

**File:** `tests/test_dataloaders/test_vtk_loader.py`

**Create new file and add imports:**
```python
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
```

**Verification:**
```bash
pytest tests/test_dataloaders/test_vtk_loader.py::TestBasicLoading -v
```

Expected: 2 tests pass

---

### Task 3: Implement Bounds Calculation Tests

**File:** `tests/test_dataloaders/test_vtk_loader.py`

**Add to existing file:**
```python


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
```

**Verification:**
```bash
pytest tests/test_dataloaders/test_vtk_loader.py::TestBoundsCalculation -v
```

Expected: 3 tests pass

---

### Task 4: Implement Normalization Tests

**File:** `tests/test_dataloaders/test_vtk_loader.py`

**Add to existing file:**
```python


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
```

**Verification:**
```bash
pytest tests/test_dataloaders/test_vtk_loader.py::TestNormalization -v
```

Expected: 2 tests pass

---

### Task 5: Implement Volume Metadata Tests

**File:** `tests/test_dataloaders/test_vtk_loader.py`

**Add to existing file:**
```python


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
```

**Verification:**
```bash
pytest tests/test_dataloaders/test_vtk_loader.py::TestVolumeMetadata -v
```

Expected: 3 tests pass

---

### Task 6: Implement Error Handling Tests

**File:** `tests/test_dataloaders/test_vtk_loader.py`

**Add to existing file:**
```python


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
```

**Verification:**
```bash
pytest tests/test_dataloaders/test_vtk_loader.py::TestErrorHandling -v
```

Expected: 2 tests pass

---

### Task 7: Implement Path Handling Tests

**File:** `tests/test_dataloaders/test_vtk_loader.py`

**Add to existing file:**
```python


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
```

**Verification:**
```bash
pytest tests/test_dataloaders/test_vtk_loader.py::TestPathHandling -v
```

Expected: 2 tests pass

---

## Validation

### Run Full Test Suite

```bash
pytest tests/test_dataloaders/test_vtk_loader.py -v
```

Expected: All 14 tests pass

### Check Test Coverage

```bash
pytest tests/test_dataloaders/ --cov=pyvr.dataloaders --cov-report=term-missing
```

Expected output should show:
- Overall coverage >85% for `pyvr/dataloaders/` module
- `vtk_loader.py` should have >90% coverage
- Key areas covered:
  - File loading and validation
  - Metadata extraction
  - Data reshaping and normalization
  - Bounds calculation
  - Error handling
  - Path handling

### Integration with Full Test Suite

```bash
pytest tests/ -v
```

Expected: All existing tests still pass + new 14 tests pass

---

## Acceptance Criteria

- [ ] `tests/test_dataloaders/__init__.py` created
- [ ] `tests/test_dataloaders/test_vtk_loader.py` created
- [ ] **Basic Loading Tests (2 tests):**
  - [ ] `test_load_hydrogen_vti` passes
  - [ ] `test_load_fuel_vti` passes
- [ ] **Bounds Calculation Tests (3 tests):**
  - [ ] `test_bounds_centered_at_origin` passes
  - [ ] `test_bounds_longest_dimension_maps_to_one` passes
  - [ ] `test_aspect_ratio_preserved` passes
- [ ] **Normalization Tests (2 tests):**
  - [ ] `test_data_normalized_to_unit_range` passes
  - [ ] `test_data_converted_to_float32` passes
- [ ] **Volume Metadata Tests (3 tests):**
  - [ ] `test_volume_name_format` passes
  - [ ] `test_normals_computed` passes
  - [ ] `test_normals_shape` passes
- [ ] **Error Handling Tests (2 tests):**
  - [ ] `test_file_not_found` passes
  - [ ] `test_invalid_scalars_name` passes
- [ ] **Path Handling Tests (2 tests):**
  - [ ] `test_string_path` passes
  - [ ] `test_pathlib_path` passes
- [ ] Total: 14 tests, all passing
- [ ] Coverage >85% for `pyvr.dataloaders` module
- [ ] No regressions in existing tests
- [ ] Code follows PyVR testing patterns (class-based organization)

---

## Git Commit

**Files to commit:**
- `tests/test_dataloaders/__init__.py`
- `tests/test_dataloaders/test_vtk_loader.py`

**Commit message:**
```
test: Add comprehensive test suite for VTK loader

Implement complete test coverage for load_vtk_volume():
- Basic loading tests (hydrogen.vti, fuel.vti)
- Bounds calculation and aspect ratio preservation
- Data normalization to [0, 1]
- Volume metadata (name, normals)
- Error handling (file not found, invalid scalars)
- Path handling (string and Path objects)

14 tests total, >85% coverage for dataloaders module

Part of v0.4.0 - Phase 02/03

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

---

**Phase Status:** Ready for implementation
**Next Phase:** Phase 03 - Documentation and Release Preparation
