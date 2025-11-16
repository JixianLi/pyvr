# Phase 01: Core VTK Loader Implementation

**Feature:** VTK Data Loader v0.4.0
**Phase:** 01 of 03
**Goal:** Implement core VTK loading functionality with full validation

## Scope

This phase delivers the working VTK loader implementation:
- Create `pyvr/dataloaders/` module structure
- Implement `load_vtk_volume()` with complete data processing pipeline
- Full validation and error handling
- Add VTK dependency
- Update package imports
- Manual verification with test data

**Deliverables:**
- `pyvr/dataloaders/__init__.py`
- `pyvr/dataloaders/vtk_loader.py`
- Updated `pyproject.toml` (VTK dependency)
- Updated `pyvr/__init__.py` (expose dataloaders)

## Implementation

### Task 1: Add VTK Dependency

**File:** `pyproject.toml`

**Changes:**
1. Locate the `[tool.poetry.dependencies]` section
2. Add `vtk` to core dependencies (not in an optional group)
3. Add after existing dependencies with version constraint

```toml
[tool.poetry.dependencies]
# ... existing dependencies ...
vtk = "^9.0"
```

**Verification:**
```bash
poetry lock --no-update
poetry install
python -c "import vtk; print(vtk.VTK_VERSION)"
```

Expected: Should print VTK version (9.x.x)

---

### Task 2: Create Module Structure

**File:** `pyvr/dataloaders/__init__.py`

**Create new file with content:**
```python
"""ABOUTME: PyVR Data Loaders Module
ABOUTME: Provides functions for loading volume data from various file formats.
"""

from .vtk_loader import load_vtk_volume

__all__ = ["load_vtk_volume"]
```

**Verification:**
```bash
python -c "from pyvr.dataloaders import load_vtk_volume; print(load_vtk_volume)"
```

Expected: Should print function object without errors

---

### Task 3: Implement VTK Loader - File and Metadata Handling

**File:** `pyvr/dataloaders/vtk_loader.py`

**Create new file with imports and function signature:**
```python
"""ABOUTME: VTK volume data loader implementation.
ABOUTME: Loads VTK ImageData (.vti) files as PyVR Volume objects.
"""

from pathlib import Path
from typing import Union

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

from pyvr.volume import Volume


def load_vtk_volume(
    file_path: Union[str, Path], scalars_name: str = "Scalars_"
) -> Volume:
    """
    Load VTK ImageData (.vti) file as PyVR Volume.

    This function loads VTK ImageData files and converts them to PyVR's
    backend-agnostic Volume format. The data is automatically normalized
    to [0, 1] range, and bounds are calculated to preserve physical aspect
    ratio while centering the volume at [0, 0, 0].

    Args:
        file_path: Path to .vti file (str or pathlib.Path)
        scalars_name: Name of scalar array to load (default: 'Scalars_')

    Returns:
        Volume object with:
        - Data normalized to [0, 1] as float32
        - Bounds rescaled to [-1, 1] space (longest dimension)
        - Aspect ratio preserved from physical dimensions (spacing × dimensions)
        - Centered at [0, 0, 0]
        - Name set to "filename.vti(ScalarName)"
        - Normals computed automatically

    Raises:
        FileNotFoundError: File doesn't exist
        ValueError: Invalid VTK data (multi-component scalars, missing array,
                    invalid dimensions/spacing)

    Example:
        >>> from pyvr.dataloaders import load_vtk_volume
        >>> volume = load_vtk_volume("example_data/hydrogen.vti")
        >>> print(volume.name)
        hydrogen.vti(Scalars_)
        >>> print(volume.shape)
        (128, 128, 128)
    """
    # Convert to Path object (handles both str and Path)
    file_path = Path(file_path)

    # Validate file exists
    if not file_path.exists():
        raise FileNotFoundError(f"VTK file not found: {file_path}")

    # Load VTK file
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(str(file_path))
    reader.Update()

    # Get image data
    image_data = reader.GetOutput()

    # Extract and validate metadata
    dims = image_data.GetDimensions()  # (nx, ny, nz) in X,Y,Z order
    spacing = image_data.GetSpacing()  # (sx, sy, sz) in X,Y,Z order

    # Validate dimensions
    if any(d <= 0 for d in dims):
        raise ValueError(f"Invalid dimensions: {dims}")

    # Validate spacing
    if any(s <= 0 for s in spacing):
        raise ValueError(f"Invalid spacing (must be positive): {spacing}")

    # Extract and validate scalar array
    point_data = image_data.GetPointData()
    scalar_array = point_data.GetArray(scalars_name)

    if scalar_array is None:
        available = [
            point_data.GetArrayName(i)
            for i in range(point_data.GetNumberOfArrays())
        ]
        raise ValueError(
            f"Scalar array '{scalars_name}' not found in {file_path}. "
            f"Available arrays: {available}"
        )

    # Validate single-component
    num_components = scalar_array.GetNumberOfComponents()
    if num_components != 1:
        raise ValueError(
            f"Multi-component scalars not supported. "
            f"Array '{scalars_name}' has {num_components} components, expected 1."
        )

    # Convert to numpy and reshape
    # VTK internal data is in Z,Y,X order
    numpy_data = vtk_to_numpy(scalar_array)
    data_3d = numpy_data.reshape((dims[2], dims[1], dims[0]))  # (nz, ny, nx)

    # Normalize data to [0, 1]
    data_float = data_3d.astype(np.float32)
    data_min, data_max = data_float.min(), data_float.max()

    if data_max - data_min < 1e-9:
        # Handle constant volume
        normalized = np.zeros_like(data_float)
    else:
        normalized = (data_float - data_min) / (data_max - data_min)

    # Calculate bounds preserving physical aspect ratio
    # Physical dimensions = voxel dimensions × spacing (in X,Y,Z order)
    physical_dims = np.array(
        [dims[0] * spacing[0], dims[1] * spacing[1], dims[2] * spacing[2]],
        dtype=np.float32,
    )

    # Scale longest dimension to [-1, 1]
    longest = np.max(physical_dims)
    scale = 2.0 / longest

    # Calculate half-extents centered at [0, 0, 0]
    half_extents = physical_dims * scale / 2.0
    min_bounds = -half_extents
    max_bounds = +half_extents

    # Create volume name
    volume_name = f"{file_path.name}({scalars_name})"

    # Create Volume object
    volume = Volume(
        data=normalized,
        normals=None,
        min_bounds=min_bounds,
        max_bounds=max_bounds,
        name=volume_name,
    )

    # Compute normals
    volume.compute_normals()

    return volume
```

**Verification:**
```python
from pyvr.dataloaders import load_vtk_volume

# Load hydrogen.vti
volume = load_vtk_volume("example_data/hydrogen.vti")
print(f"Name: {volume.name}")
print(f"Shape: {volume.shape}")
print(f"Data type: {volume.data.dtype}")
print(f"Data range: [{volume.data.min()}, {volume.data.max()}]")
print(f"Min bounds: {volume.min_bounds}")
print(f"Max bounds: {volume.max_bounds}")
print(f"Center: {volume.center}")
print(f"Has normals: {volume.has_normals}")
```

Expected output:
```
Name: hydrogen.vti(Scalars_)
Shape: (128, 128, 128)
Data type: float32
Data range: [0.0, 1.0]
Min bounds: [-1. -1. -1.]
Max bounds: [1. 1. 1.]
Center: [0. 0. 0.]
Has normals: True
```

---

### Task 4: Update Package Imports

**File:** `pyvr/__init__.py`

**Changes:**
1. Add `dataloaders` to imports (line 10)
2. Add `dataloaders` to `__all__` (line ~22)

**Before:**
```python
from . import camera, datasets, interface, lighting, moderngl_renderer, transferfunctions, volume
```

**After:**
```python
from . import camera, dataloaders, datasets, interface, lighting, moderngl_renderer, transferfunctions, volume
```

**Before:**
```python
__all__ = [
    "camera",
    "datasets",
    "interface",
    "lighting",
    "moderngl_renderer",
    "transferfunctions",
    "volume",
    "RenderConfig",
]
```

**After:**
```python
__all__ = [
    "camera",
    "dataloaders",
    "datasets",
    "interface",
    "lighting",
    "moderngl_renderer",
    "transferfunctions",
    "volume",
    "RenderConfig",
]
```

**Verification:**
```python
import pyvr
print("dataloaders" in dir(pyvr))
print(hasattr(pyvr.dataloaders, "load_vtk_volume"))
```

Expected: Both print `True`

---

## Validation

### Integration Test 1: Load hydrogen.vti and verify all properties

```python
from pyvr.dataloaders import load_vtk_volume
import numpy as np

volume = load_vtk_volume("example_data/hydrogen.vti")

# Check basic properties
assert volume.name == "hydrogen.vti(Scalars_)"
assert volume.shape == (128, 128, 128)
assert volume.data.dtype == np.float32

# Check normalization
assert volume.data.min() >= 0.0
assert volume.data.max() <= 1.0

# Check bounds (cubic volume with uniform spacing)
assert np.allclose(volume.min_bounds, [-1.0, -1.0, -1.0])
assert np.allclose(volume.max_bounds, [1.0, 1.0, 1.0])
assert np.allclose(volume.center, [0.0, 0.0, 0.0])

# Check normals
assert volume.has_normals is True
assert volume.normals.shape == (128, 128, 128, 3)

print("✓ hydrogen.vti loaded correctly")
```

### Integration Test 2: Load fuel.vti and verify all properties

```python
from pyvr.dataloaders import load_vtk_volume
import numpy as np

volume = load_vtk_volume("example_data/fuel.vti")

# Check basic properties
assert volume.name == "fuel.vti(Scalars_)"
assert volume.shape == (64, 64, 64)
assert volume.data.dtype == np.float32

# Check normalization
assert volume.data.min() >= 0.0
assert volume.data.max() <= 1.0

# Check bounds (cubic volume with uniform spacing)
assert np.allclose(volume.min_bounds, [-1.0, -1.0, -1.0])
assert np.allclose(volume.max_bounds, [1.0, 1.0, 1.0])
assert np.allclose(volume.center, [0.0, 0.0, 0.0])

# Check normals
assert volume.has_normals is True
assert volume.normals.shape == (64, 64, 64, 3)

print("✓ fuel.vti loaded correctly")
```

### Integration Test 3: Error handling verification

```python
from pyvr.dataloaders import load_vtk_volume
import pytest

# Test file not found
try:
    load_vtk_volume("nonexistent.vti")
    assert False, "Should raise FileNotFoundError"
except FileNotFoundError as e:
    assert "not found" in str(e)
    print("✓ FileNotFoundError raised correctly")

# Test invalid scalar name
try:
    load_vtk_volume("example_data/hydrogen.vti", scalars_name="InvalidName")
    assert False, "Should raise ValueError"
except ValueError as e:
    assert "not found" in str(e)
    assert "Available arrays" in str(e)
    print("✓ ValueError for invalid scalar name raised correctly")
```

### Integration Test 4: Path handling verification

```python
from pyvr.dataloaders import load_vtk_volume
from pathlib import Path

# Test with string path
volume1 = load_vtk_volume("example_data/hydrogen.vti")
assert volume1.name == "hydrogen.vti(Scalars_)"
print("✓ String path works")

# Test with Path object
volume2 = load_vtk_volume(Path("example_data/hydrogen.vti"))
assert volume2.name == "hydrogen.vti(Scalars_)"
print("✓ Path object works")
```

---

## Acceptance Criteria

- [ ] VTK dependency added to `pyproject.toml` and installed
- [ ] Module `pyvr/dataloaders/` created with `__init__.py`
- [ ] `vtk_loader.py` implemented with complete docstring
- [ ] `load_vtk_volume()` function working
- [ ] File validation raises `FileNotFoundError` for missing files
- [ ] Scalar array validation with descriptive error messages
- [ ] Multi-component validation raises `ValueError`
- [ ] Dimension validation raises `ValueError` for invalid dims
- [ ] Spacing validation raises `ValueError` for invalid spacing
- [ ] Data correctly reshaped from VTK Z,Y,X order
- [ ] Data normalized to [0, 1] as float32
- [ ] Bounds calculated preserving physical aspect ratio
- [ ] Bounds centered at [0, 0, 0]
- [ ] Volume name format: "filename.vti(ScalarName)"
- [ ] Normals computed automatically
- [ ] Both string and Path objects accepted
- [ ] `pyvr/__init__.py` updated to expose `dataloaders`
- [ ] hydrogen.vti loads successfully with correct properties
- [ ] fuel.vti loads successfully with correct properties
- [ ] All integration tests pass

---

## Git Commit

**Files to commit:**
- `pyvr/dataloaders/__init__.py`
- `pyvr/dataloaders/vtk_loader.py`
- `pyproject.toml`
- `poetry.lock`
- `pyvr/__init__.py`

**Commit message:**
```
feat: Add VTK data loader for .vti files

Implement core VTK ImageData loader with:
- load_vtk_volume() function for .vti files
- Automatic normalization to [0, 1]
- Physical space-aware bounds calculation
- Aspect ratio preservation
- Automatic normal computation
- Comprehensive validation and error handling
- Support for string and Path objects

VTK added as core dependency (>= 9.0)

Part of v0.4.0 - Phase 01/03

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

---

**Phase Status:** Ready for implementation
**Next Phase:** Phase 02 - Comprehensive Testing
