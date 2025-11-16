# VTK Data Loader Design - v0.4.0

**Feature:** VTK ImageData (.vti) file loader for PyVR
**Version:** v0.4.0
**Status:** Design Complete
**Date:** 2025-11-16

## Overview

Add support for loading real scientific volume data via VTK's ImageData format (.vti files). This enables PyVR to work with scientific datasets while maintaining the existing backend-agnostic Volume architecture.

## Goals

**Primary Goal:** Load VTK ImageData files as PyVR Volume objects

**In Scope:**
- VTK ImageData (.vti) format support
- Single-component scalar data only
- Automatic normalization to [0, 1]
- Physical space-aware bounds calculation
- Aspect ratio preservation
- Automatic normal computation

**Out of Scope (v0.4.0):**
- Legacy VTK formats (.vtk)
- Structured/unstructured grids
- Multi-component volumes (RGB, vector fields)
- Optional normalization
- Custom normal computation methods

## Architecture

### Module Structure

```
pyvr/dataloaders/
├── __init__.py          # Exports load_vtk_volume
└── vtk_loader.py        # VTK implementation
```

### Public API

```python
from pyvr.dataloaders import load_vtk_volume

def load_vtk_volume(file_path, scalars_name='Scalars_') -> Volume:
    """
    Load VTK ImageData (.vti) file as PyVR Volume.

    Args:
        file_path: Path to .vti file (str or pathlib.Path)
        scalars_name: Name of scalar array to load (default: 'Scalars_')

    Returns:
        Volume object with:
        - Data normalized to [0, 1] as float32
        - Bounds rescaled to [-1, 1] space (longest dimension)
        - Aspect ratio preserved from physical dimensions
        - Centered at [0, 0, 0]
        - Name set to "filename.vti(ScalarName)"
        - Normals computed via volume.compute_normals()

    Raises:
        FileNotFoundError: File doesn't exist
        ValueError: Invalid VTK data (multi-component, missing array,
                    invalid dimensions/spacing)
    """
```

### Usage Example

```python
from pyvr.dataloaders import load_vtk_volume
from pyvr.moderngl_renderer import VolumeRenderer

# Load VTK file - returns render-ready Volume
volume = load_vtk_volume("example_data/hydrogen.vti")

# Volume is already normalized and bounded correctly
renderer = VolumeRenderer(512, 512)
renderer.load_volume(volume)
image = renderer.render()
```

## Data Processing Pipeline

### Step-by-Step Processing

1. **File Loading**
   - Use `vtkXMLImageDataReader` to read .vti file
   - Validate file exists and is readable
   - Extract `vtkImageData` object

2. **Metadata Extraction**
   - Voxel dimensions: `image_data.GetDimensions()` → `(nx, ny, nz)` (X,Y,Z order)
   - Spacing: `image_data.GetSpacing()` → `(sx, sy, sz)` (X,Y,Z order)
   - Physical dimensions: `[nx*sx, ny*sy, nz*sz]`

3. **Scalar Array Extraction**
   - Get array by name from `GetPointData()`
   - Validate exactly one component (no RGB/vector data)
   - Convert VTK array → numpy via `vtk_to_numpy()`
   - **Critical:** VTK internal data is in Z,Y,X order despite API using X,Y,Z

4. **Data Reshaping and Normalization**
   ```python
   dims = image_data.GetDimensions()  # (nx, ny, nz) in X,Y,Z
   numpy_data = vtk_to_numpy(vtk_array)  # Flat 1D array

   # Reshape to Z,Y,X order (VTK's internal layout)
   data_3d = numpy_data.reshape((dims[2], dims[1], dims[0]))  # (nz, ny, nx)

   # Convert to float32
   data_float = data_3d.astype(np.float32)

   # Normalize to [0, 1]
   data_min, data_max = data_float.min(), data_float.max()
   if data_max - data_min < 1e-9:
       normalized = np.zeros_like(data_float)
   else:
       normalized = (data_float - data_min) / (data_max - data_min)
   ```

5. **Bounds Calculation** (Center-based, Aspect-preserving)
   ```python
   # Physical dimensions in X,Y,Z order
   physical_dims = np.array([nx*sx, ny*sy, nz*sz], dtype=np.float32)

   # Scale longest dimension to [-1, 1]
   longest = np.max(physical_dims)
   scale = 2.0 / longest

   # Calculate half-extents centered at [0,0,0]
   half_extents = physical_dims * scale / 2.0
   min_bounds = -half_extents
   max_bounds = +half_extents
   ```

6. **Volume Creation**
   ```python
   volume = Volume(
       data=normalized,
       normals=None,
       min_bounds=min_bounds,
       max_bounds=max_bounds,
       name=f"{Path(file_path).name}({scalars_name})"
   )
   volume.compute_normals()
   return volume
   ```

### Bounds Calculation Examples

**Example 1 - Cubic volume:**
- hydrogen.vti: dims=(128,128,128), spacing=(1,1,1)
- Physical dims = [128, 128, 128]
- Longest = 128, scale = 2/128 = 0.015625
- Bounds: [[-1, -1, -1], [1, 1, 1]] (perfect cube)

**Example 2 - Non-cubic volume:**
- dims=(64,64,128), spacing=(1,1,1)
- Physical dims = [64, 64, 128]
- Longest = 128, scale = 2/128 = 0.015625
- Bounds: [[-0.5, -0.5, -1.0], [0.5, 0.5, 1.0]] (aspect preserved)

**Example 3 - Non-uniform spacing:**
- dims=(100,100,100), spacing=(0.5, 1.0, 2.0)
- Physical dims = [50, 100, 200]
- Longest = 200, scale = 2/200 = 0.01
- Bounds: [[-0.25, -0.5, -1.0], [0.25, 0.5, 1.0]] (physical space preserved)

## Error Handling and Validation

### Validation Checks

All validation uses strict error-raising strategy - no warnings, no silent fallbacks.

1. **File Validation**
   ```python
   if not Path(file_path).exists():
       raise FileNotFoundError(f"VTK file not found: {file_path}")
   ```

2. **Scalar Array Validation**
   ```python
   point_data = image_data.GetPointData()
   scalar_array = point_data.GetArray(scalars_name)

   if scalar_array is None:
       available = [point_data.GetArrayName(i)
                    for i in range(point_data.GetNumberOfArrays())]
       raise ValueError(
           f"Scalar array '{scalars_name}' not found in {file_path}. "
           f"Available arrays: {available}"
       )
   ```

3. **Component Validation**
   ```python
   num_components = scalar_array.GetNumberOfComponents()
   if num_components != 1:
       raise ValueError(
           f"Multi-component scalars not supported. "
           f"Array '{scalars_name}' has {num_components} components, expected 1."
       )
   ```

4. **Dimension Validation**
   ```python
   dims = image_data.GetDimensions()
   if any(d <= 0 for d in dims):
       raise ValueError(f"Invalid dimensions: {dims}")
   ```

5. **Spacing Validation**
   ```python
   spacing = image_data.GetSpacing()
   if any(s <= 0 for s in spacing):
       raise ValueError(f"Invalid spacing (must be positive): {spacing}")
   ```

### Error Message Philosophy

- **Descriptive:** Tell users exactly what went wrong
- **Actionable:** Include available options (e.g., list available arrays)
- **No silent fallbacks:** Fail fast with clear information
- **No warnings:** All issues are errors

## Testing Strategy

### Test Organization

```
tests/test_dataloaders/
├── __init__.py
└── test_vtk_loader.py
```

### Test Coverage

**Test data:** Only `example_data/hydrogen.vti` and `example_data/fuel.vti` (both tracked in git)

**Test Cases:**

1. **Basic Loading Tests**
   - `test_load_hydrogen_vti()`: Load hydrogen.vti with default scalars_name
   - `test_load_fuel_vti()`: Load fuel.vti with default scalars_name
   - Verify: data shape, dtype, normalization range [0,1], has_normals=True

2. **Bounds Calculation Tests**
   - `test_bounds_centered_at_origin()`: Verify center == [0,0,0]
   - `test_bounds_longest_dimension_maps_to_one()`: Verify max(abs(bounds)) == 1.0
   - `test_aspect_ratio_preserved()`: Compare physical dims ratio to bounds ratio

3. **Normalization Tests**
   - `test_data_normalized_to_unit_range()`: Verify min≈0, max≈1
   - `test_data_converted_to_float32()`: Verify dtype

4. **Volume Metadata Tests**
   - `test_volume_name_format()`: Verify name == "hydrogen.vti(Scalars_)"
   - `test_normals_computed()`: Verify volume.has_normals is True
   - `test_normals_shape()`: Verify normals.shape == data.shape + (3,)

5. **Error Handling Tests**
   - `test_file_not_found()`: Non-existent file raises FileNotFoundError
   - `test_invalid_scalars_name()`: Wrong scalar name raises ValueError with available arrays

6. **Path Handling Tests**
   - `test_string_path()`: Accept string paths
   - `test_pathlib_path()`: Accept Path objects

**Deferred to v0.4.x:**
- Multi-component scalars testing
- Zero/negative spacing edge cases
- Corrupted file handling
- Different data types (uint8, uint16, float64)
- Non-uniform spacing comprehensive tests

## Implementation Details

### Dependencies

**VTK Dependency:**
- Add `vtk >= 9.0` to core dependencies in `pyproject.toml`
- Not optional - VTK becomes required for PyVR v0.4.0+

### Import Structure

```python
# pyvr/dataloaders/__init__.py
"""PyVR Data Loaders Module

This module provides functions for loading volume data from various file formats.
"""

from .vtk_loader import load_vtk_volume

__all__ = ["load_vtk_volume"]
```

```python
# pyvr/__init__.py (update)
from . import (..., dataloaders)  # Add dataloaders to imports
__all__ = [..., "dataloaders"]  # Add to __all__
```

### Key Implementation Notes

1. **VTK Dimension Ordering:**
   - VTK API methods use X,Y,Z order: `GetDimensions()`, `GetSpacing()`
   - VTK internal data is Z,Y,X order
   - When reshaping: `data.reshape((nz, ny, nx))` where `(nx,ny,nz) = GetDimensions()`

2. **Required Imports:**
   ```python
   from pathlib import Path
   from typing import Union
   import numpy as np
   import vtk
   from vtk.util.numpy_support import vtk_to_numpy
   from pyvr.volume import Volume
   ```

3. **Path Handling:**
   ```python
   file_path = Path(file_path)  # Convert early, handles both str and Path
   ```

4. **Type Hints:**
   ```python
   def load_vtk_volume(
       file_path: Union[str, Path],
       scalars_name: str = 'Scalars_'
   ) -> Volume:
   ```

5. **Volume Creation Order:**
   - Create Volume with data and bounds
   - Then call `volume.compute_normals()`
   - Return configured Volume

## Documentation

### Example Script

**File:** `examples/vtk_loading_demo.py`

```python
"""Demonstrate loading and rendering VTK volume data.

This example shows how to:
1. Load VTK .vti files using pyvr.dataloaders
2. Render with interactive interface
3. Inspect loaded volume properties
"""

from pyvr.dataloaders import load_vtk_volume
from pyvr.interface import InteractiveVolumeRenderer
from pyvr.config import RenderConfig

# Load hydrogen dataset
volume = load_vtk_volume("example_data/hydrogen.vti")

print(f"Loaded: {volume.name}")
print(f"Shape: {volume.shape}")
print(f"Bounds: {volume.min_bounds} to {volume.max_bounds}")
print(f"Center: {volume.center}")
print(f"Has normals: {volume.has_normals}")

# Create interactive renderer
interface = InteractiveVolumeRenderer(
    volume=volume,
    width=800,
    height=800,
    config=RenderConfig.balanced()
)

interface.show()
```

### README Updates

Add new section "Loading VTK Data" after the "Datasets" section:

```markdown
## Loading VTK Data

PyVR supports loading scientific volume data from VTK ImageData (.vti) files:

```python
from pyvr.dataloaders import load_vtk_volume

# Load VTK file - returns normalized, render-ready Volume
volume = load_vtk_volume("example_data/hydrogen.vti")

# Use with any renderer
from pyvr.interface import InteractiveVolumeRenderer
interface = InteractiveVolumeRenderer(volume)
interface.show()
```

**Features:**
- Automatic normalization to [0, 1] range
- Preserves physical aspect ratio from VTK spacing
- Centered at [0, 0, 0] in world space
- Normal vectors computed automatically
- Supports custom scalar array names

**Installation:**
VTK is included as a core dependency starting from v0.4.0.
```

### Version Notes

**File:** `version_notes/v0.4.0_vtk_loader.md`

Content to include:
- New `pyvr.dataloaders` module introduction
- VTK added as core dependency
- API usage examples with hydrogen.vti and fuel.vti
- Breaking changes: None (pure addition)
- Migration notes: N/A
- Known limitations: .vti only, single-component only

## Success Criteria

- ✅ Load `hydrogen.vti` and `fuel.vti` successfully
- ✅ Data normalized to [0, 1] as float32
- ✅ Bounds preserve aspect ratio, centered at [0,0,0]
- ✅ Normals computed automatically
- ✅ All tests pass with >85% coverage for new module
- ✅ Example script runs without errors
- ✅ Documentation complete (README, version notes, docstrings)
- ✅ VTK added to dependencies
- ✅ Version updated to 0.4.0

## Future Enhancements (v0.4.x)

Potential improvements for stabilization releases:

- Performance optimization for large volumes
- Support for more VTK formats (.vtk legacy)
- Multi-component volume support (RGB, vector fields)
- Optional normalization control
- Caching/lazy loading for very large datasets
- Progress callbacks for large files
- More comprehensive edge case testing

---

**Design Status:** Complete and validated
**Next Step:** Use `writing-plans` skill to create implementation plan
