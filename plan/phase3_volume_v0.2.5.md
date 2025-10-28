# Phase 3: Volume Refactoring (v0.2.5)

> ⚠️ **BREAKING CHANGES**: This phase removes backward compatibility.
> Pre-1.0 development prioritizes clean implementation over API stability.
> See [REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md) for migration guide.

## Overview
Extract volume data handling from `VolumeRenderer` into a dedicated `Volume` class. This encapsulates volume data, normals, bounds, and metadata into a single cohesive unit, improving the Application Stage organization.

**Breaking Changes**:
- Remove `load_normal_volume()` method
- Remove `set_volume_bounds()` method
- `load_volume()` requires `Volume` instance (no raw numpy arrays)
- Volume data, normals, and bounds must be in `Volume` object

## Goals
- ✅ Create `Volume` class to encapsulate volume data and metadata
- ✅ Support volume data, normals, and bounds as a single unit
- ✅ Add volume validation and property methods
- ✅ Simplify VolumeRenderer volume loading interface
- ✅ Maintain backward compatibility with existing volume API

## Files to Create

### New Module
- `pyvr/volume/__init__.py` - Module initialization and exports
- `pyvr/volume/data.py` - Volume class implementation

### Files to Modify

### Core Implementation
- `pyvr/moderngl_renderer/renderer.py` - Refactor volume loading
- `pyvr/__init__.py` - Add volume module to exports
- `pyvr/datasets/synthetic.py` - Update to return Volume instances (optional)

### Tests
- Create `tests/test_volume/__init__.py`
- Create `tests/test_volume/test_volume.py` - Unit tests for Volume class
- `tests/test_moderngl_renderer/test_volume_renderer.py` - Integration tests

### Examples
- `example/ModernglRender/enhanced_camera_demo.py`
- `example/ModernglRender/multiview_example.py`
- `example/ModernglRender/rgba_demo.py`

### Documentation
- `README.md` - Add Volume class examples
- `CLAUDE.md` - Update architecture section

## Detailed Implementation Steps

### Step 1: Create Volume Module

#### 1.1 Create `pyvr/volume/data.py`

```python
"""
Volume data management for PyVR.

This module provides the Volume class for encapsulating 3D volume data,
normal volumes, and spatial metadata.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


@dataclass
class Volume:
    """
    Encapsulates 3D volume data and associated metadata.

    This class manages volume data, optional normal vectors, spatial bounds,
    and provides utilities for volume manipulation and validation.

    Attributes:
        data: 3D numpy array with shape (D, H, W) containing volume data
        normals: Optional 3D normal vectors with shape (D, H, W, 3)
        min_bounds: Minimum corner of volume bounding box in world space
        max_bounds: Maximum corner of volume bounding box in world space
        name: Optional descriptive name for the volume
    """

    data: np.ndarray
    normals: Optional[np.ndarray] = None
    min_bounds: np.ndarray = field(
        default_factory=lambda: np.array([-0.5, -0.5, -0.5], dtype=np.float32)
    )
    max_bounds: np.ndarray = field(
        default_factory=lambda: np.array([0.5, 0.5, 0.5], dtype=np.float32)
    )
    name: Optional[str] = None

    def __post_init__(self):
        """Validate volume data after initialization."""
        self.validate()

    def validate(self) -> None:
        """
        Validate volume data and metadata.

        Raises:
            ValueError: If volume data or parameters are invalid
        """
        # Validate data shape
        if not isinstance(self.data, np.ndarray):
            raise ValueError("Volume data must be a numpy array")

        if len(self.data.shape) != 3:
            raise ValueError(
                f"Volume data must be 3D, got shape {self.data.shape}"
            )

        # Validate normals if present
        if self.normals is not None:
            if not isinstance(self.normals, np.ndarray):
                raise ValueError("Normal volume must be a numpy array")

            expected_shape = self.data.shape + (3,)
            if self.normals.shape != expected_shape:
                raise ValueError(
                    f"Normal volume must have shape {expected_shape}, "
                    f"got {self.normals.shape}"
                )

        # Validate bounds
        if not isinstance(self.min_bounds, np.ndarray) or self.min_bounds.shape != (3,):
            raise ValueError("min_bounds must be a 3D numpy array")

        if not isinstance(self.max_bounds, np.ndarray) or self.max_bounds.shape != (3,):
            raise ValueError("max_bounds must be a 3D numpy array")

        if np.any(self.max_bounds <= self.min_bounds):
            raise ValueError("max_bounds must be greater than min_bounds in all dimensions")

    @property
    def shape(self) -> Tuple[int, int, int]:
        """
        Get volume dimensions.

        Returns:
            Tuple of (depth, height, width) as integers

        Example:
            >>> volume = Volume(data=np.zeros((64, 64, 64)))
            >>> volume.shape
            (64, 64, 64)
        """
        return self.data.shape

    @property
    def dimensions(self) -> np.ndarray:
        """
        Get physical dimensions of volume bounding box.

        Returns:
            3D array of (width, height, depth) in world space units

        Example:
            >>> volume = Volume(
            ...     data=np.zeros((64, 64, 64)),
            ...     min_bounds=np.array([0, 0, 0]),
            ...     max_bounds=np.array([2, 2, 2])
            ... )
            >>> volume.dimensions
            array([2., 2., 2.])
        """
        return self.max_bounds - self.min_bounds

    @property
    def center(self) -> np.ndarray:
        """
        Get center point of volume bounding box.

        Returns:
            3D array of center coordinates in world space

        Example:
            >>> volume = Volume(
            ...     data=np.zeros((64, 64, 64)),
            ...     min_bounds=np.array([-1, -1, -1]),
            ...     max_bounds=np.array([1, 1, 1])
            ... )
            >>> volume.center
            array([0., 0., 0.])
        """
        return (self.min_bounds + self.max_bounds) / 2.0

    @property
    def has_normals(self) -> bool:
        """
        Check if volume has normal data.

        Returns:
            True if normals are present, False otherwise

        Example:
            >>> volume = Volume(data=np.zeros((64, 64, 64)))
            >>> volume.has_normals
            False
        """
        return self.normals is not None

    @property
    def voxel_spacing(self) -> np.ndarray:
        """
        Get spacing between voxels in world space.

        Returns:
            3D array of voxel spacing (dx, dy, dz)

        Example:
            >>> volume = Volume(
            ...     data=np.zeros((100, 100, 100)),
            ...     min_bounds=np.array([0, 0, 0]),
            ...     max_bounds=np.array([1, 1, 1])
            ... )
            >>> volume.voxel_spacing
            array([0.01, 0.01, 0.01])
        """
        return self.dimensions / np.array(self.shape, dtype=np.float32)

    def compute_normals(self, method: str = "gradient") -> None:
        """
        Compute normal vectors from volume data.

        Args:
            method: Method to use for normal computation (default: "gradient")
                   Currently only "gradient" is supported.

        Raises:
            ValueError: If method is not supported

        Example:
            >>> volume = Volume(data=create_sample_volume(64, 'sphere'))
            >>> volume.compute_normals()
            >>> volume.has_normals
            True
        """
        if method != "gradient":
            raise ValueError(f"Unsupported normal computation method: {method}")

        from ..datasets.synthetic import compute_normal_volume

        self.normals = compute_normal_volume(self.data)

    def set_bounds(
        self,
        min_bounds: Tuple[float, float, float],
        max_bounds: Tuple[float, float, float],
    ) -> None:
        """
        Set volume bounding box.

        Args:
            min_bounds: Minimum corner (x_min, y_min, z_min)
            max_bounds: Maximum corner (x_max, y_max, z_max)

        Raises:
            ValueError: If bounds are invalid

        Example:
            >>> volume = Volume(data=np.zeros((64, 64, 64)))
            >>> volume.set_bounds((-1, -1, -1), (1, 1, 1))
        """
        self.min_bounds = np.array(min_bounds, dtype=np.float32)
        self.max_bounds = np.array(max_bounds, dtype=np.float32)
        self.validate()

    def set_bounds_centered(self, size: float) -> None:
        """
        Set bounds as a cube centered at origin.

        Args:
            size: Side length of the cube

        Example:
            >>> volume = Volume(data=np.zeros((64, 64, 64)))
            >>> volume.set_bounds_centered(2.0)  # [-1, -1, -1] to [1, 1, 1]
        """
        half_size = size / 2.0
        self.min_bounds = np.array([-half_size] * 3, dtype=np.float32)
        self.max_bounds = np.array([half_size] * 3, dtype=np.float32)

    def normalize(self, method: str = "minmax") -> "Volume":
        """
        Create a new volume with normalized data.

        Args:
            method: Normalization method ("minmax" or "zscore")

        Returns:
            New Volume instance with normalized data

        Example:
            >>> volume = Volume(data=np.random.rand(64, 64, 64) * 100)
            >>> normalized = volume.normalize()
            >>> normalized.data.min(), normalized.data.max()
            (0.0, 1.0)
        """
        if method == "minmax":
            data_min = self.data.min()
            data_max = self.data.max()
            if data_max - data_min < 1e-9:
                normalized_data = np.zeros_like(self.data)
            else:
                normalized_data = (self.data - data_min) / (data_max - data_min)

        elif method == "zscore":
            mean = self.data.mean()
            std = self.data.std()
            if std < 1e-9:
                normalized_data = np.zeros_like(self.data)
            else:
                normalized_data = (self.data - mean) / std
        else:
            raise ValueError(f"Unsupported normalization method: {method}")

        return Volume(
            data=normalized_data.astype(np.float32),
            normals=self.normals.copy() if self.normals is not None else None,
            min_bounds=self.min_bounds.copy(),
            max_bounds=self.max_bounds.copy(),
            name=f"{self.name}_normalized" if self.name else None,
        )

    def copy(self) -> "Volume":
        """
        Create a deep copy of this volume.

        Returns:
            New Volume instance with copied data

        Example:
            >>> original = Volume(data=np.zeros((64, 64, 64)))
            >>> copy = original.copy()
            >>> copy.data[0, 0, 0] = 1.0  # Doesn't affect original
        """
        return Volume(
            data=self.data.copy(),
            normals=self.normals.copy() if self.normals is not None else None,
            min_bounds=self.min_bounds.copy(),
            max_bounds=self.max_bounds.copy(),
            name=self.name,
        )

    def __repr__(self) -> str:
        """String representation of volume."""
        name_str = f"'{self.name}'" if self.name else "unnamed"
        normals_str = "with normals" if self.has_normals else "no normals"
        return (
            f"Volume({name_str}, "
            f"shape={self.shape}, "
            f"bounds=[{self.min_bounds}, {self.max_bounds}], "
            f"{normals_str})"
        )


class VolumeError(Exception):
    """Exception raised for volume data errors."""

    pass
```

#### 1.2 Create `pyvr/volume/__init__.py`

```python
"""
Volume data management for PyVR.

Provides the Volume class for encapsulating 3D volume data and metadata.
"""

from .data import Volume, VolumeError

__all__ = [
    "Volume",
    "VolumeError",
]
```

### Step 2: Update VolumeRenderer

#### 2.1 Refactor `load_volume()` and related methods

In `pyvr/moderngl_renderer/renderer.py`:

```python
def load_volume(self, volume):
    """
    Load volume data into the renderer.

    Supports two interfaces:
    1. New interface: Pass a Volume instance
    2. Legacy interface: Pass raw numpy array

    Args:
        volume: Volume instance or 3D numpy array (legacy)

    Example:
        # New interface (recommended)
        >>> from pyvr.volume import Volume
        >>> volume = Volume(data=volume_data, normals=normal_data)
        >>> renderer.load_volume(volume)

        # Legacy interface (backward compatibility)
        >>> renderer.load_volume(volume_data)  # 3D numpy array
    """
    from ..volume import Volume

    if isinstance(volume, Volume):
        # New interface: Volume instance
        self.volume = volume

        # Load volume data texture
        texture_unit = self.gl_manager.create_volume_texture(volume.data)
        self.gl_manager.set_uniform_int("volume_texture", texture_unit)

        # Set bounds
        self.gl_manager.set_uniform_vector("volume_min_bounds", tuple(volume.min_bounds))
        self.gl_manager.set_uniform_vector("volume_max_bounds", tuple(volume.max_bounds))

        # Load normals if present
        if volume.has_normals:
            normal_unit = self.gl_manager.create_normal_texture(volume.normals)
            self.gl_manager.set_uniform_int("normal_volume", normal_unit)

    else:
        # Legacy interface: raw numpy array
        import warnings
        warnings.warn(
            "Passing raw numpy array to load_volume() is deprecated. "
            "Use Volume instance instead.",
            DeprecationWarning,
            stacklevel=2
        )

        # Validate data
        if len(volume.shape) != 3:
            raise ValueError("Volume data must be 3D")

        # Create temporary Volume instance
        self.volume = Volume(data=volume)

        # Load texture
        texture_unit = self.gl_manager.create_volume_texture(volume)
        self.gl_manager.set_uniform_int("volume_texture", texture_unit)
```

#### 2.2 Update `load_normal_volume()` (keep for compatibility)

```python
def load_normal_volume(self, normal_volume):
    """
    Load 3D normal data into a texture.

    .. deprecated:: 0.2.5
        Include normals in Volume instance instead.

    Args:
        normal_volume: 4D array with shape (D, H, W, 3)

    Example:
        >>> # Old way (deprecated)
        >>> renderer.load_normal_volume(normals)
        >>>
        >>> # New way (recommended)
        >>> volume = Volume(data=volume_data, normals=normals)
        >>> renderer.load_volume(volume)
    """
    import warnings
    warnings.warn(
        "load_normal_volume() is deprecated. Include normals in Volume instance.",
        DeprecationWarning,
        stacklevel=2
    )

    if normal_volume.shape[-1] != 3:
        raise ValueError("Normal volume must have 3 channels (last dimension).")

    # Create normal texture
    texture_unit = self.gl_manager.create_normal_texture(normal_volume)
    self.gl_manager.set_uniform_int("normal_volume", texture_unit)

    # Update volume's normals if volume exists
    if hasattr(self, 'volume') and self.volume is not None:
        self.volume.normals = normal_volume
```

#### 2.3 Update `set_volume_bounds()` (keep for compatibility)

```python
def set_volume_bounds(
    self, min_bounds=(-0.5, -0.5, -0.5), max_bounds=(0.5, 0.5, 0.5)
):
    """
    Set the world space bounding box for the volume.

    .. deprecated:: 0.2.5
        Set bounds on Volume instance instead.

    Args:
        min_bounds: Minimum corner of bounding box
        max_bounds: Maximum corner of bounding box

    Example:
        >>> # Old way (deprecated)
        >>> renderer.set_volume_bounds((-1, -1, -1), (1, 1, 1))
        >>>
        >>> # New way (recommended)
        >>> volume = Volume(data=volume_data)
        >>> volume.set_bounds((-1, -1, -1), (1, 1, 1))
        >>> renderer.load_volume(volume)
    """
    import warnings
    warnings.warn(
        "set_volume_bounds() is deprecated. Set bounds on Volume instance.",
        DeprecationWarning,
        stacklevel=2
    )

    self.gl_manager.set_uniform_vector("volume_min_bounds", tuple(min_bounds))
    self.gl_manager.set_uniform_vector("volume_max_bounds", tuple(max_bounds))

    # Update volume bounds if volume exists
    if hasattr(self, 'volume') and self.volume is not None:
        self.volume.min_bounds = np.array(min_bounds, dtype=np.float32)
        self.volume.max_bounds = np.array(max_bounds, dtype=np.float32)
```

#### 2.4 Add `get_volume()` helper method

```python
def get_volume(self):
    """
    Get current volume instance.

    Returns:
        Volume: Current volume or None if not loaded

    Example:
        >>> renderer = VolumeRenderer()
        >>> volume = renderer.get_volume()
    """
    if hasattr(self, 'volume'):
        return self.volume
    return None
```

### Step 3: Update Dataset Utilities (Optional)

#### 3.1 Add Volume wrapper functions to `pyvr/datasets/synthetic.py`

```python
def create_sample_volume_object(
    size: int = 256,
    volume_type: str = "sphere",
    with_normals: bool = False,
    bounds: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = None,
) -> "Volume":
    """
    Create a sample volume as a Volume object (recommended interface).

    Args:
        size: Volume dimensions (cube)
        volume_type: Type of volume to generate
        with_normals: Whether to compute normal vectors
        bounds: Optional (min_bounds, max_bounds) tuple

    Returns:
        Volume instance with sample data

    Example:
        >>> from pyvr.datasets import create_sample_volume_object
        >>> volume = create_sample_volume_object(128, 'sphere', with_normals=True)
        >>> print(volume)
    """
    from ..volume import Volume

    # Generate volume data using existing function
    data = create_sample_volume(size, volume_type)

    # Set bounds
    if bounds is None:
        min_bounds = np.array([-0.5, -0.5, -0.5], dtype=np.float32)
        max_bounds = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    else:
        min_bounds = np.array(bounds[0], dtype=np.float32)
        max_bounds = np.array(bounds[1], dtype=np.float32)

    # Create volume
    volume = Volume(
        data=data,
        min_bounds=min_bounds,
        max_bounds=max_bounds,
        name=f"{volume_type}_{size}",
    )

    # Compute normals if requested
    if with_normals:
        volume.compute_normals()

    return volume
```

### Step 4: Update Package Exports

#### 4.1 Update `pyvr/__init__.py`

```python
"""PyVR: Python Volume Rendering Toolkit

PyVR provides GPU-accelerated 3D volume rendering using OpenGL/ModernGL
for real-time interactive visualization.
"""

from . import camera, datasets, lighting, moderngl_renderer, transferfunctions, volume

__version__ = "0.2.5"
__all__ = [
    "camera",
    "datasets",
    "lighting",
    "moderngl_renderer",
    "transferfunctions",
    "volume",
]
```

## Testing Requirements

### Unit Tests for Volume Class

Create `tests/test_volume/test_volume.py`:

```python
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

    def test_set_bounds(self):
        """set_bounds should update bounds."""
        data = np.zeros((64, 64, 64))
        volume = Volume(data=data)

        volume.set_bounds((-2, -2, -2), (2, 2, 2))

        assert np.allclose(volume.min_bounds, [-2, -2, -2])
        assert np.allclose(volume.max_bounds, [2, 2, 2])

    def test_set_bounds_centered(self):
        """set_bounds_centered should create centered cube."""
        data = np.zeros((64, 64, 64))
        volume = Volume(data=data)

        volume.set_bounds_centered(4.0)

        assert np.allclose(volume.min_bounds, [-2, -2, -2])
        assert np.allclose(volume.max_bounds, [2, 2, 2])

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
```

### Integration Tests

Add to `tests/test_moderngl_renderer/test_volume_renderer.py`:

```python
def test_load_volume_with_volume_object(mock_moderngl_context):
    """load_volume should accept Volume instance."""
    from pyvr.volume import Volume
    from pyvr.moderngl_renderer import VolumeRenderer

    data = np.random.rand(32, 32, 32).astype(np.float32)
    volume = Volume(data=data)

    renderer = VolumeRenderer(width=256, height=256)
    renderer.load_volume(volume)

    assert renderer.get_volume() is volume


def test_load_volume_legacy_array(mock_moderngl_context):
    """load_volume should accept raw array with deprecation warning."""
    from pyvr.moderngl_renderer import VolumeRenderer

    data = np.random.rand(32, 32, 32).astype(np.float32)

    renderer = VolumeRenderer(width=256, height=256)

    with pytest.warns(DeprecationWarning):
        renderer.load_volume(data)


def test_load_volume_with_normals(mock_moderngl_context):
    """load_volume should handle normals in Volume."""
    from pyvr.volume import Volume
    from pyvr.moderngl_renderer import VolumeRenderer

    data = np.random.rand(32, 32, 32).astype(np.float32)
    normals = np.random.rand(32, 32, 32, 3).astype(np.float32)
    volume = Volume(data=data, normals=normals)

    renderer = VolumeRenderer(width=256, height=256)
    renderer.load_volume(volume)

    assert renderer.get_volume().has_normals
```

## Validation Steps

### Pre-merge Checklist

- [ ] All existing tests pass (151 tests from Phase 2)
- [ ] New volume unit tests pass (+10 tests)
- [ ] Integration tests pass
- [ ] Deprecation warnings work correctly
- [ ] Examples updated and run successfully
- [ ] Documentation updated
- [ ] No performance regression

### Manual Validation

```bash
# Run full test suite
pytest tests/ -v

# Run only volume tests
pytest tests/test_volume/ -v

# Run with coverage
pytest --cov=pyvr.volume --cov-report=term-missing tests/test_volume/

# Test examples
python example/ModernglRender/multiview_example.py
```

## Migration Guide

### Recommended Usage (New API)

```python
from pyvr.volume import Volume
from pyvr.datasets import create_sample_volume
from pyvr.moderngl_renderer import VolumeRenderer

# Create volume object
data = create_sample_volume(128, 'sphere')
volume = Volume(data=data, name='sphere')

# Optionally add normals
volume.compute_normals()

# Set custom bounds
volume.set_bounds((-1, -1, -1), (1, 1, 1))

# Load into renderer
renderer = VolumeRenderer(width=512, height=512)
renderer.load_volume(volume)

# Access volume properties
print(f"Volume shape: {volume.shape}")
print(f"Volume center: {volume.center}")
print(f"Has normals: {volume.has_normals}")
```

### Backward Compatibility

```python
# Old way still works but emits warnings
renderer.load_volume(data_array)  # DeprecationWarning
renderer.load_normal_volume(normals)  # DeprecationWarning
renderer.set_volume_bounds((-1, -1, -1), (1, 1, 1))  # DeprecationWarning
```

## Benefits Achieved

1. ✅ **Data Encapsulation**: Volume data + metadata as single unit
2. ✅ **Simplified API**: One method call vs multiple
3. ✅ **Better Validation**: Automatic validation on creation
4. ✅ **Rich Properties**: Access dimensions, center, spacing, etc.
5. ✅ **Extensibility**: Easy to add transformations, filters
6. ✅ **Type Safety**: Clear interface for volume data

## Timeline

- **Implementation**: 2 days
- **Testing**: 1 day
- **Documentation & Examples**: 0.5 day
- **Total**: 3-4 days

## Dependencies

- **Requires**: Phase 1 (Camera) and Phase 2 (Light) completed
- **Blocks**: None

## Next Phase

After Phase 3 completion, proceed to **Phase 4: RenderConfig Refactoring (v0.2.6)**
