# Phase 3: Volume & Renderer Architecture Refactoring (v0.2.5)

> ⚠️ **MAJOR BREAKING CHANGES**: This phase restructures the rendering architecture.
> Pre-1.0 development prioritizes clean architecture over API stability.

## Overview

Phase 3 addresses two critical architectural issues:

1. **Volume Data Encapsulation**: Extract volume handling into a dedicated `Volume` class
2. **Renderer Backend Independence**: Separate VolumeRenderer from OpenGL-specific implementation

Currently, `VolumeRenderer` lives in `moderngl_renderer/` and is tightly coupled to ModernGL/OpenGL. This violates separation of concerns - a volume renderer should be backend-agnostic, with specific backends (OpenGL, Vulkan, CPU) as implementations.

**Breaking Changes**:
- `VolumeRenderer` moves from `pyvr.moderngl_renderer` to `pyvr.renderer`
- `VolumeRenderer` becomes abstract base class
- `ModernGLVolumeRenderer` is the OpenGL implementation
- `load_volume()` requires `Volume` instance
- Removed methods: `load_normal_volume()`, `set_volume_bounds()` (deprecated in favor of Volume)

## Goals

### Volume Class (Application Stage)
- ✅ Create `Volume` class to encapsulate volume data and metadata
- ✅ Support volume data, normals, and bounds as a single unit
- ✅ Add volume validation and property methods
- ✅ Simplify volume loading interface

### Renderer Architecture (Backend Independence)
- ✅ Create abstract `VolumeRenderer` base class
- ✅ Move `VolumeRenderer` from `moderngl_renderer/` to `renderer/`
- ✅ Separate OpenGL-specific code into `ModernGLVolumeRenderer`
- ✅ Define clean renderer interface independent of backend
- ✅ Enable future backend implementations (Vulkan, CPU, etc.)

## Architecture Changes

### Before (v0.2.4)
```
pyvr/
└── moderngl_renderer/
    ├── renderer.py        # VolumeRenderer (OpenGL-coupled)
    └── manager.py         # ModernGLManager
```

**Problems**:
- VolumeRenderer tightly coupled to ModernGL
- Cannot add non-OpenGL backends without major refactoring
- Volume data scattered across multiple method calls
- Backend-specific code mixed with rendering logic

### After (v0.2.5)
```
pyvr/
├── volume/                # NEW: Volume data management
│   └── data.py            # Volume class (backend-agnostic)
├── renderer/              # NEW: Abstract rendering interface
│   ├── base.py            # VolumeRenderer base class (abstract)
│   └── config.py          # RenderConfig (for Phase 4)
└── moderngl_renderer/     # OpenGL-specific implementation
    ├── renderer.py        # ModernGLVolumeRenderer (concrete)
    └── manager.py         # ModernGLManager (unchanged)
```

**Benefits**:
- `Volume` is backend-agnostic (CPU, GPU, any backend)
- `VolumeRenderer` defines clean abstract interface
- `ModernGLVolumeRenderer` implements OpenGL backend
- Easy to add Vulkan, CPU, or other backends
- Clear separation of concerns

## Files to Create

### New Modules

#### Volume Module
- `pyvr/volume/__init__.py` - Volume module exports
- `pyvr/volume/data.py` - Volume class implementation

#### Renderer Module (Abstract)
- `pyvr/renderer/__init__.py` - Renderer module exports
- `pyvr/renderer/base.py` - Abstract VolumeRenderer base class

### Files to Modify

#### Core Implementation
- `pyvr/moderngl_renderer/renderer.py` - Rename to ModernGLVolumeRenderer, inherit from base
- `pyvr/moderngl_renderer/__init__.py` - Update exports
- `pyvr/__init__.py` - Add volume and renderer modules

#### Tests
- Create `tests/test_volume/test_volume.py` - Volume class tests (+12 tests)
- Create `tests/test_renderer/test_base.py` - Abstract renderer tests (+5 tests)
- Update `tests/test_moderngl_renderer/test_volume_renderer.py` - Update for new API

#### Examples
- `example/ModernglRender/*.py` - Update to use Volume and new imports

## Detailed Implementation Steps

### Step 1: Create Volume Module

#### 1.1 Create `pyvr/volume/data.py`

```python
"""
Volume data management for PyVR.

This module provides the Volume class for encapsulating 3D volume data,
normal volumes, and spatial metadata. Volume is backend-agnostic and can
be used with any renderer implementation.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


@dataclass
class Volume:
    """
    Encapsulates 3D volume data and associated metadata.

    Volume is backend-agnostic - it stores data that can be rendered
    by any backend (OpenGL, Vulkan, CPU, etc.).

    Attributes:
        data: 3D numpy array (D, H, W) containing volume data
        normals: Optional 3D normal vectors (D, H, W, 3)
        min_bounds: Minimum corner of bounding box in world space
        max_bounds: Maximum corner of bounding box in world space
        name: Optional descriptive name
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
        Validate all volume data and metadata.

        Raises:
            ValueError: If volume data or parameters are invalid
        """
        if not isinstance(self.data, np.ndarray):
            raise ValueError("Volume data must be a numpy array")

        if len(self.data.shape) != 3:
            raise ValueError(f"Volume data must be 3D, got shape {self.data.shape}")

        if self.normals is not None:
            if not isinstance(self.normals, np.ndarray):
                raise ValueError("Normal volume must be a numpy array")

            expected_shape = self.data.shape + (3,)
            if self.normals.shape != expected_shape:
                raise ValueError(
                    f"Normal volume must have shape {expected_shape}, "
                    f"got {self.normals.shape}"
                )

        if not isinstance(self.min_bounds, np.ndarray) or self.min_bounds.shape != (3,):
            raise ValueError("min_bounds must be a 3D numpy array")

        if not isinstance(self.max_bounds, np.ndarray) or self.max_bounds.shape != (3,):
            raise ValueError("max_bounds must be a 3D numpy array")

        if np.any(self.max_bounds <= self.min_bounds):
            raise ValueError("max_bounds must be greater than min_bounds")

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Get volume dimensions (D, H, W)."""
        return self.data.shape

    @property
    def dimensions(self) -> np.ndarray:
        """Get physical dimensions of bounding box."""
        return self.max_bounds - self.min_bounds

    @property
    def center(self) -> np.ndarray:
        """Get center point of bounding box."""
        return (self.min_bounds + self.max_bounds) / 2.0

    @property
    def has_normals(self) -> bool:
        """Check if volume has normal data."""
        return self.normals is not None

    @property
    def voxel_spacing(self) -> np.ndarray:
        """Get spacing between voxels in world space."""
        return self.dimensions / np.array(self.shape, dtype=np.float32)

    def compute_normals(self, method: str = "gradient") -> None:
        """
        Compute normal vectors from volume data.

        Args:
            method: Computation method (default: "gradient")
        """
        if method != "gradient":
            raise ValueError(f"Unsupported method: {method}")

        from ..datasets.synthetic import compute_normal_volume

        self.normals = compute_normal_volume(self.data)

    def normalize(self, method: str = "minmax") -> "Volume":
        """
        Create normalized volume.

        Args:
            method: Normalization method ("minmax" or "zscore")

        Returns:
            New Volume with normalized data
        """
        if method == "minmax":
            data_min, data_max = self.data.min(), self.data.max()
            if data_max - data_min < 1e-9:
                normalized_data = np.zeros_like(self.data)
            else:
                normalized_data = (self.data - data_min) / (data_max - data_min)

        elif method == "zscore":
            mean, std = self.data.mean(), self.data.std()
            if std < 1e-9:
                normalized_data = np.zeros_like(self.data)
            else:
                normalized_data = (self.data - mean) / std
        else:
            raise ValueError(f"Unsupported method: {method}")

        return Volume(
            data=normalized_data.astype(np.float32),
            normals=self.normals.copy() if self.normals is not None else None,
            min_bounds=self.min_bounds.copy(),
            max_bounds=self.max_bounds.copy(),
            name=f"{self.name}_normalized" if self.name else None,
        )

    def copy(self) -> "Volume":
        """Create deep copy of volume."""
        return Volume(
            data=self.data.copy(),
            normals=self.normals.copy() if self.normals is not None else None,
            min_bounds=self.min_bounds.copy(),
            max_bounds=self.max_bounds.copy(),
            name=self.name,
        )

    def __repr__(self) -> str:
        """String representation."""
        name_str = f"'{self.name}'" if self.name else "unnamed"
        normals_str = "with normals" if self.has_normals else "no normals"
        return (
            f"Volume({name_str}, shape={self.shape}, "
            f"bounds=[{self.min_bounds}, {self.max_bounds}], {normals_str})"
        )


class VolumeError(Exception):
    """Exception raised for volume data errors."""
    pass
```

#### 1.2 Create `pyvr/volume/__init__.py`

```python
"""Volume data management for PyVR."""

from .data import Volume, VolumeError

__all__ = ["Volume", "VolumeError"]
```

### Step 2: Create Abstract Renderer Interface

#### 2.1 Create `pyvr/renderer/base.py`

```python
"""
Abstract volume renderer interface for PyVR.

This module defines the VolumeRenderer base class that all backend
implementations must inherit from. This allows PyVR to support multiple
rendering backends (OpenGL, Vulkan, CPU, etc.) with a consistent API.
"""

from abc import ABC, abstractmethod
from typing import Optional

from ..camera import Camera
from ..lighting import Light
from ..transferfunctions import ColorTransferFunction, OpacityTransferFunction
from ..volume import Volume


class VolumeRenderer(ABC):
    """
    Abstract base class for volume renderers.

    All backend implementations (OpenGL, Vulkan, CPU, etc.) must inherit
    from this class and implement the abstract methods.

    This design enables:
    - Multiple rendering backends with consistent API
    - Backend-agnostic application code
    - Easy testing with mock renderers
    - Future extensibility
    """

    def __init__(self, width: int = 512, height: int = 512):
        """
        Initialize base renderer.

        Args:
            width: Rendering viewport width
            height: Rendering viewport height
        """
        self.width = width
        self.height = height
        self.volume: Optional[Volume] = None
        self.camera: Optional[Camera] = None
        self.light: Optional[Light] = None

    @abstractmethod
    def load_volume(self, volume: Volume) -> None:
        """
        Load volume data into the renderer.

        Args:
            volume: Volume instance with data and metadata

        Raises:
            TypeError: If volume is not a Volume instance
        """
        pass

    @abstractmethod
    def set_camera(self, camera: Camera) -> None:
        """
        Set camera configuration.

        Args:
            camera: Camera instance with position and projection parameters

        Raises:
            TypeError: If camera is not a Camera instance
        """
        pass

    @abstractmethod
    def set_light(self, light: Light) -> None:
        """
        Set lighting configuration.

        Args:
            light: Light instance with lighting parameters

        Raises:
            TypeError: If light is not a Light instance
        """
        pass

    @abstractmethod
    def set_transfer_functions(
        self,
        color_transfer_function: ColorTransferFunction,
        opacity_transfer_function: OpacityTransferFunction,
    ) -> None:
        """
        Set transfer functions for volume rendering.

        Args:
            color_transfer_function: Color transfer function
            opacity_transfer_function: Opacity transfer function
        """
        pass

    @abstractmethod
    def render(self) -> bytes:
        """
        Render the volume and return raw pixel data.

        Returns:
            Raw RGBA pixel data as bytes

        Raises:
            RuntimeError: If renderer is not properly configured
        """
        pass

    def get_volume(self) -> Optional[Volume]:
        """Get current volume."""
        return self.volume

    def get_camera(self) -> Optional[Camera]:
        """Get current camera."""
        return self.camera

    def get_light(self) -> Optional[Light]:
        """Get current light."""
        return self.light


class RendererError(Exception):
    """Exception raised for renderer errors."""
    pass
```

#### 2.2 Create `pyvr/renderer/__init__.py`

```python
"""Abstract volume renderer interface for PyVR."""

from .base import RendererError, VolumeRenderer

__all__ = ["VolumeRenderer", "RendererError"]
```

### Step 3: Refactor ModernGLVolumeRenderer

#### 3.1 Update `pyvr/moderngl_renderer/renderer.py`

Rename class and inherit from abstract base:

```python
"""
ModernGL-based volume renderer implementation.

This module provides the OpenGL/ModernGL backend implementation
of the abstract VolumeRenderer interface.
"""

import os

from PIL import Image

from ..renderer.base import VolumeRenderer as VolumeRendererBase
from ..camera import Camera
from ..lighting import Light
from ..transferfunctions import ColorTransferFunction, OpacityTransferFunction
from ..volume import Volume
from .manager import ModernGLManager


class ModernGLVolumeRenderer(VolumeRendererBase):
    """
    ModernGL/OpenGL implementation of VolumeRenderer.

    This is the concrete implementation using ModernGL for GPU-accelerated
    volume rendering. It implements all abstract methods from VolumeRendererBase.
    """

    def __init__(
        self,
        width=512,
        height=512,
        step_size=0.01,
        max_steps=200,
        light=None,
    ):
        """
        Initialize ModernGL volume renderer.

        Args:
            width: Viewport width
            height: Viewport height
            step_size: Ray marching step size
            max_steps: Maximum ray marching steps
            light: Light instance (creates default if None)
        """
        super().__init__(width, height)

        self.step_size = step_size
        self.max_steps = max_steps

        # Initialize light
        if light is None:
            from ..lighting import Light
            self.light = Light.default()
        else:
            if not isinstance(light, Light):
                raise TypeError(f"Expected Light instance, got {type(light)}")
            self.light = light

        # Create ModernGL manager
        self.gl_manager = ModernGLManager(width, height)

        # Load shaders
        pyvr_dir = os.path.dirname(os.path.dirname(__file__))
        shader_dir = os.path.join(pyvr_dir, "shaders")
        vertex_shader_path = os.path.join(shader_dir, "volume.vert.glsl")
        fragment_shader_path = os.path.join(shader_dir, "volume.frag.glsl")
        self.gl_manager.load_shaders(vertex_shader_path, fragment_shader_path)

        # Set default uniforms
        self.gl_manager.set_uniform_float("step_size", self.step_size)
        self.gl_manager.set_uniform_int("max_steps", self.max_steps)
        self.gl_manager.set_uniform_vector("volume_min_bounds", (-0.5, -0.5, -0.5))
        self.gl_manager.set_uniform_vector("volume_max_bounds", (0.5, 0.5, 0.5))

        # Set light uniforms
        self._update_light()

    def load_volume(self, volume: Volume) -> None:
        """
        Load volume data into renderer.

        Args:
            volume: Volume instance with data and metadata

        Raises:
            TypeError: If volume is not a Volume instance
        """
        if not isinstance(volume, Volume):
            raise TypeError(f"Expected Volume instance, got {type(volume)}")

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

    def set_camera(self, camera: Camera) -> None:
        """
        Set camera configuration.

        Args:
            camera: Camera instance

        Raises:
            TypeError: If camera is not a Camera instance
        """
        if not isinstance(camera, Camera):
            raise TypeError(f"Expected Camera instance, got {type(camera)}")

        self.camera = camera

        # Get matrices from camera
        aspect = self.width / self.height
        view_matrix = camera.get_view_matrix()
        projection_matrix = camera.get_projection_matrix(aspect)
        position, _ = camera.get_camera_vectors()

        # Set uniforms
        self.gl_manager.set_uniform_matrix("view_matrix", view_matrix)
        self.gl_manager.set_uniform_matrix("projection_matrix", projection_matrix)
        self.gl_manager.set_uniform_vector("camera_pos", tuple(position))

    def set_light(self, light: Light) -> None:
        """
        Set lighting configuration.

        Args:
            light: Light instance

        Raises:
            TypeError: If light is not a Light instance
        """
        if not isinstance(light, Light):
            raise TypeError(f"Expected Light instance, got {type(light)}")

        self.light = light
        self._update_light()

    def set_transfer_functions(
        self,
        color_transfer_function: ColorTransferFunction,
        opacity_transfer_function: OpacityTransferFunction,
        size=None,
    ) -> None:
        """
        Set transfer functions.

        Args:
            color_transfer_function: Color transfer function
            opacity_transfer_function: Opacity transfer function
            size: Optional LUT size override
        """
        rgba_tex_unit = self.gl_manager.create_rgba_transfer_function_texture(
            color_transfer_function, opacity_transfer_function, size
        )
        self.gl_manager.set_uniform_int("transfer_function_lut", rgba_tex_unit)

    def render(self) -> bytes:
        """
        Render volume and return raw pixel data.

        Returns:
            Raw RGBA pixel data as bytes
        """
        self.gl_manager.clear_framebuffer(0.0, 0.0, 0.0, 0.0)
        self.gl_manager.setup_blending()
        self.gl_manager.render_quad()
        return self.gl_manager.read_pixels()

    def render_to_pil(self, data=None):
        """Render and return as PIL Image."""
        if data is None:
            data = self.render()

        image = Image.frombytes("RGBA", (self.width, self.height), data)
        return image.transpose(Image.FLIP_TOP_BOTTOM)

    def set_step_size(self, step_size):
        """Set ray marching step size."""
        self.step_size = step_size
        self.gl_manager.set_uniform_float("step_size", step_size)

    def set_max_steps(self, max_steps):
        """Set maximum ray marching steps."""
        self.max_steps = max_steps
        self.gl_manager.set_uniform_int("max_steps", max_steps)

    def _update_light(self):
        """Update OpenGL uniforms from light configuration."""
        self.gl_manager.set_uniform_float("ambient_light", self.light.ambient_intensity)
        self.gl_manager.set_uniform_float("diffuse_light", self.light.diffuse_intensity)
        self.gl_manager.set_uniform_vector("light_position", tuple(self.light.position))
        self.gl_manager.set_uniform_vector("light_target", tuple(self.light.target))


# For backward compatibility
VolumeRenderer = ModernGLVolumeRenderer
```

#### 3.2 Update `pyvr/moderngl_renderer/__init__.py`

```python
"""ModernGL-based volume renderer implementation."""

from ..camera.control import CameraController
from ..camera.parameters import Camera
from ..datasets import compute_normal_volume, create_sample_volume
from ..transferfunctions.color import ColorTransferFunction
from ..transferfunctions.opacity import OpacityTransferFunction
from .manager import ModernGLManager
from .renderer import ModernGLVolumeRenderer, VolumeRenderer  # VolumeRenderer is alias

__all__ = [
    "ModernGLVolumeRenderer",
    "VolumeRenderer",  # For backward compatibility
    "ModernGLManager",
    "CameraController",
    "Camera",
    "ColorTransferFunction",
    "OpacityTransferFunction",
    "create_sample_volume",
    "compute_normal_volume",
]
```

### Step 4: Update Package Exports

#### 4.1 Update `pyvr/__init__.py`

```python
"""PyVR: Python Volume Rendering Toolkit

GPU-accelerated 3D volume rendering with backend-agnostic architecture.
"""

from . import camera, datasets, lighting, moderngl_renderer, renderer, transferfunctions, volume

__version__ = "0.2.5"
__all__ = [
    "camera",
    "datasets",
    "lighting",
    "moderngl_renderer",
    "renderer",
    "transferfunctions",
    "volume",
]
```

## Testing Requirements

### New Tests

#### Volume Tests (+12 tests)
- `tests/test_volume/test_volume.py`
  - Volume creation and validation
  - Property methods (shape, dimensions, center, etc.)
  - Normal computation
  - Normalization methods
  - Copy functionality

#### Abstract Renderer Tests (+5 tests)
- `tests/test_renderer/test_base.py`
  - Abstract method enforcement
  - Base class interface validation
  - Error handling

### Updated Tests

#### ModernGL Renderer Tests
- Update imports to use Volume
- Update to use ModernGLVolumeRenderer class name
- Ensure backward compatibility alias works

### Total Test Count
- **Before Phase 3**: 162 tests
- **After Phase 3**: ~179 tests (+17)

## Migration Guide

### Import Changes

**Before (v0.2.4)**:
```python
from pyvr.moderngl_renderer import VolumeRenderer
```

**After (v0.2.5)**:
```python
# Recommended: Use specific backend
from pyvr.moderngl_renderer import ModernGLVolumeRenderer

# Or for backward compatibility
from pyvr.moderngl_renderer import VolumeRenderer  # Alias to ModernGLVolumeRenderer

# Or use abstract interface
from pyvr.renderer import VolumeRenderer  # Abstract base class
```

### Volume Usage

**Before**:
```python
renderer = VolumeRenderer()
renderer.load_volume(volume_data)  # numpy array
renderer.load_normal_volume(normals)
renderer.set_volume_bounds((-1, -1, -1), (1, 1, 1))
```

**After**:
```python
from pyvr.volume import Volume

volume = Volume(
    data=volume_data,
    normals=normals,
    min_bounds=np.array([-1, -1, -1]),
    max_bounds=np.array([1, 1, 1])
)
renderer = ModernGLVolumeRenderer()
renderer.load_volume(volume)
```

## Benefits Achieved

### Volume Class
1. ✅ **Data Encapsulation**: Volume data + metadata as single unit
2. ✅ **Simplified API**: One method call vs multiple
3. ✅ **Backend Agnostic**: Works with any renderer implementation
4. ✅ **Rich Properties**: Easy access to dimensions, center, spacing
5. ✅ **Type Safety**: Clear interface for volume data

### Renderer Architecture
1. ✅ **Backend Independence**: Abstract interface, multiple implementations
2. ✅ **Clean Separation**: Rendering logic separate from OpenGL specifics
3. ✅ **Extensibility**: Easy to add new backends (Vulkan, CPU, etc.)
4. ✅ **Testability**: Can mock renderer without OpenGL
5. ✅ **Future-Proof**: Architecture supports advanced rendering features

## Timeline

- **Volume Implementation**: 1 day
- **Renderer Refactoring**: 2 days
- **Testing**: 1 day
- **Documentation & Examples**: 0.5 day
- **Total**: 4-5 days

## Dependencies

- **Requires**: Phase 1 (Camera) and Phase 2 (Light) completed
- **Blocks**: Phase 4 (RenderConfig)

## Next Phase

After Phase 3, proceed to **Phase 4: RenderConfig Refactoring (v0.2.6)** which will:
- Extract rendering parameters (step_size, max_steps, etc.) into RenderConfig
- Further clean up renderer initialization
- Complete the pipeline alignment

---

**Phase 3 Status**: 📋 Planning Complete - Ready for Implementation
