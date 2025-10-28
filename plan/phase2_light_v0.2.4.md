# Phase 2: Light Refactoring (v0.2.4)

> ⚠️ **BREAKING CHANGES**: This phase removes backward compatibility.
> Pre-1.0 development prioritizes clean implementation over API stability.
> See [REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md) for migration guide.

## Overview
Extract lighting parameters and logic from `VolumeRenderer` into a dedicated `Light` class. This completes the Application Stage separation by isolating lighting concerns, following the same pattern established in Phase 1 (Camera refactoring).

**Breaking Changes**:
- Remove `ambient_light`, `diffuse_light`, `light_position`, `light_target` from `VolumeRenderer.__init__()`
- Remove methods: `set_ambient_light()`, `set_diffuse_light()`, `set_light_position()`, `set_light_target()`
- Add `light` parameter to `VolumeRenderer.__init__()`
- All lighting configuration through `Light` class only

## Goals
- ✅ Create `Light` class to encapsulate lighting parameters
- ✅ Add `light` attribute to `VolumeRenderer`
- ✅ Remove lighting setter methods from `VolumeRenderer`
- ✅ Support multiple light types (directional, point)
- ✅ Maintain backward compatibility with existing lighting API

## Files to Create

### New Module
- `pyvr/lighting/__init__.py` - Module initialization and exports
- `pyvr/lighting/light.py` - Light class implementation

### Files to Modify

### Core Implementation
- `pyvr/moderngl_renderer/renderer.py` - Add light attribute, refactor lighting
- `pyvr/__init__.py` - Add lighting module to exports

### Tests
- Create `tests/test_lighting/__init__.py`
- Create `tests/test_lighting/test_light.py` - Unit tests for Light class
- `tests/test_moderngl_renderer/test_volume_renderer.py` - Integration tests

### Examples (Update to use new Light class)
- `example/ModernglRender/enhanced_camera_demo.py`
- `example/ModernglRender/multiview_example.py`
- `example/ModernglRender/rgba_demo.py`

### Documentation
- `README.md` - Add Light class examples
- `CLAUDE.md` - Update architecture section

## Detailed Implementation Steps

### Step 1: Create Light Module

#### 1.1 Create `pyvr/lighting/light.py`

```python
"""
Lighting system for PyVR volume rendering.

This module provides light classes and utilities for configuring
illumination in volume rendering scenes.
"""

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np


@dataclass
class Light:
    """
    Encapsulates lighting parameters for volume rendering.

    This class manages all lighting-related parameters including position,
    direction, and intensity values for ambient and diffuse lighting.

    Attributes:
        position: 3D position of the light source in world space
        target: 3D point the light is directed toward
        ambient_intensity: Ambient light intensity (0.0 to 1.0)
        diffuse_intensity: Diffuse light intensity (0.0 to 1.0)
    """

    position: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 1.0, 1.0], dtype=np.float32)
    )
    target: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=np.float32)
    )
    ambient_intensity: float = 0.2
    diffuse_intensity: float = 0.8

    def __post_init__(self):
        """Validate lighting parameters after initialization."""
        self.validate()

    def validate(self) -> None:
        """
        Validate lighting parameters.

        Raises:
            ValueError: If any parameter is invalid
        """
        # Validate position
        if not isinstance(self.position, np.ndarray) or self.position.shape != (3,):
            raise ValueError("position must be a 3D numpy array")

        # Validate target
        if not isinstance(self.target, np.ndarray) or self.target.shape != (3,):
            raise ValueError("target must be a 3D numpy array")

        # Validate intensities
        if not (0.0 <= self.ambient_intensity <= 1.0):
            raise ValueError("ambient_intensity must be between 0.0 and 1.0")

        if not (0.0 <= self.diffuse_intensity <= 1.0):
            raise ValueError("diffuse_intensity must be between 0.0 and 1.0")

    @classmethod
    def directional(
        cls,
        direction: np.ndarray,
        ambient: float = 0.2,
        diffuse: float = 0.8,
        distance: float = 10.0,
    ) -> "Light":
        """
        Create a directional light.

        A directional light illuminates from a specific direction, simulating
        distant light sources like the sun.

        Args:
            direction: Direction vector the light points toward (will be normalized)
            ambient: Ambient light intensity (default: 0.2)
            diffuse: Diffuse light intensity (default: 0.8)
            distance: Distance to place light source from origin (default: 10.0)

        Returns:
            Light instance configured as directional light

        Example:
            >>> light = Light.directional(
            ...     direction=np.array([1, -1, 0]),
            ...     ambient=0.3,
            ...     diffuse=0.9
            ... )
        """
        direction = np.array(direction, dtype=np.float32)
        direction = direction / np.linalg.norm(direction)

        # Position light far away in opposite direction
        position = -direction * distance
        target = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        return cls(
            position=position,
            target=target,
            ambient_intensity=ambient,
            diffuse_intensity=diffuse,
        )

    @classmethod
    def point_light(
        cls,
        position: np.ndarray,
        target: np.ndarray = None,
        ambient: float = 0.2,
        diffuse: float = 0.8,
    ) -> "Light":
        """
        Create a point light at a specific position.

        A point light radiates in all directions from a specific point in space.

        Args:
            position: 3D position of the light source
            target: Optional target point (default: origin)
            ambient: Ambient light intensity (default: 0.2)
            diffuse: Diffuse light intensity (default: 0.8)

        Returns:
            Light instance configured as point light

        Example:
            >>> light = Light.point_light(
            ...     position=np.array([5, 5, 5]),
            ...     ambient=0.1,
            ...     diffuse=0.7
            ... )
        """
        position = np.array(position, dtype=np.float32)
        if target is None:
            target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        else:
            target = np.array(target, dtype=np.float32)

        return cls(
            position=position,
            target=target,
            ambient_intensity=ambient,
            diffuse_intensity=diffuse,
        )

    @classmethod
    def default(cls) -> "Light":
        """
        Create default light configuration.

        Returns a standard light suitable for most volume rendering scenarios.

        Returns:
            Light instance with default parameters

        Example:
            >>> light = Light.default()
        """
        return cls()

    @classmethod
    def ambient_only(cls, intensity: float = 0.5) -> "Light":
        """
        Create ambient-only lighting (no directional component).

        Useful for examining internal structures without strong shadows.

        Args:
            intensity: Ambient light intensity (default: 0.5)

        Returns:
            Light instance with only ambient lighting

        Example:
            >>> light = Light.ambient_only(intensity=0.3)
        """
        return cls(
            position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            target=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            ambient_intensity=intensity,
            diffuse_intensity=0.0,
        )

    def get_direction(self) -> np.ndarray:
        """
        Calculate light direction vector (from position to target).

        Returns:
            Normalized direction vector as np.ndarray

        Example:
            >>> light = Light.directional(direction=[1, 0, 0])
            >>> direction = light.get_direction()
        """
        direction = self.target - self.position
        norm = np.linalg.norm(direction)

        if norm < 1e-9:
            # Position equals target, return default direction
            return np.array([0.0, 0.0, -1.0], dtype=np.float32)

        return direction / norm

    def copy(self) -> "Light":
        """
        Create a copy of this light.

        Returns:
            New Light instance with same parameters

        Example:
            >>> original = Light.default()
            >>> copy = original.copy()
            >>> copy.ambient_intensity = 0.5  # Doesn't affect original
        """
        return Light(
            position=self.position.copy(),
            target=self.target.copy(),
            ambient_intensity=self.ambient_intensity,
            diffuse_intensity=self.diffuse_intensity,
        )

    def __repr__(self) -> str:
        """String representation of light configuration."""
        return (
            f"Light("
            f"position={self.position}, "
            f"target={self.target}, "
            f"ambient={self.ambient_intensity:.2f}, "
            f"diffuse={self.diffuse_intensity:.2f})"
        )


class LightError(Exception):
    """Exception raised for lighting configuration errors."""

    pass
```

#### 1.2 Create `pyvr/lighting/__init__.py`

```python
"""
Lighting system for PyVR volume rendering.

Provides light classes and utilities for configuring illumination.
"""

from .light import Light, LightError

__all__ = [
    "Light",
    "LightError",
]
```

### Step 2: Update VolumeRenderer

#### 2.1 Modify `VolumeRenderer.__init__()`

In `pyvr/moderngl_renderer/renderer.py`:

```python
class VolumeRenderer:
    def __init__(
        self,
        width=512,
        height=512,
        step_size=0.01,
        max_steps=200,
        light=None,  # NEW: Changed from individual light parameters
        camera=None,
    ):
        """
        Initializes the volume renderer with specified rendering parameters.

        Parameters:
            width (int): The width of the rendering viewport (default: 512).
            height (int): The height of the rendering viewport (default: 512).
            step_size (float): The step size for ray marching (default: 0.01).
            max_steps (int): The maximum number of ray marching steps (default: 200).
            light (Light, optional): Light configuration. If None, creates default light.
            camera (Camera, optional): Camera instance. If None, creates default front view.

        Initializes OpenGL context, loads shaders, creates framebuffer, and sets up
        geometry and shader uniforms for volume rendering.
        """
        self.width = width
        self.height = height
        self.step_size = step_size
        self.max_steps = max_steps

        # Initialize camera
        if camera is None:
            from ..camera import Camera
            self.camera = Camera.front_view(distance=3.0)
        else:
            self.camera = camera

        # Initialize light
        if light is None:
            from ..lighting import Light
            self.light = Light.default()
        else:
            self.light = light

        # Create ModernGL manager
        self.gl_manager = ModernGLManager(width, height)

        # Load shaders from shared shader directory
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

    def _update_light(self):
        """Update OpenGL uniforms from current light configuration."""
        self.gl_manager.set_uniform_float("ambient_light", self.light.ambient_intensity)
        self.gl_manager.set_uniform_float("diffuse_light", self.light.diffuse_intensity)
        self.gl_manager.set_uniform_vector("light_position", tuple(self.light.position))
        self.gl_manager.set_uniform_vector("light_target", tuple(self.light.target))
```

#### 2.2 Add `set_light()` method

Add to `VolumeRenderer`:

```python
def set_light(self, light):
    """
    Set lighting configuration.

    Args:
        light: Light instance with lighting parameters

    Example:
        >>> from pyvr.lighting import Light
        >>> light = Light.directional(direction=[1, -1, 0], ambient=0.3)
        >>> renderer.set_light(light)
    """
    from ..lighting import Light

    if not isinstance(light, Light):
        raise TypeError(f"Expected Light instance, got {type(light)}")

    self.light = light
    self._update_light()
```

#### 2.3 Add `get_light()` helper method

```python
def get_light(self):
    """
    Get current light configuration.

    Returns:
        Light: Current light instance

    Example:
        >>> renderer = VolumeRenderer()
        >>> light = renderer.get_light()
        >>> print(light)
    """
    return self.light
```

#### 2.4 Add backward compatibility methods (deprecated)

```python
def set_ambient_light(self, ambient_light):
    """
    Set the ambient light intensity.

    .. deprecated:: 0.2.4
        Use set_light() with a Light instance instead.

    Args:
        ambient_light: Ambient light intensity (0.0 to 1.0)
    """
    import warnings
    warnings.warn(
        "set_ambient_light() is deprecated. Use set_light() with a Light instance.",
        DeprecationWarning,
        stacklevel=2
    )
    self.light.ambient_intensity = ambient_light
    self.gl_manager.set_uniform_float("ambient_light", ambient_light)

def set_diffuse_light(self, diffuse_light):
    """
    Set the diffuse light intensity.

    .. deprecated:: 0.2.4
        Use set_light() with a Light instance instead.

    Args:
        diffuse_light: Diffuse light intensity (0.0 to 1.0)
    """
    import warnings
    warnings.warn(
        "set_diffuse_light() is deprecated. Use set_light() with a Light instance.",
        DeprecationWarning,
        stacklevel=2
    )
    self.light.diffuse_intensity = diffuse_light
    self.gl_manager.set_uniform_float("diffuse_light", diffuse_light)

def set_light_position(self, light_position):
    """
    Set the position of the light source.

    .. deprecated:: 0.2.4
        Use set_light() with a Light instance instead.

    Args:
        light_position: 3D position of light source
    """
    import warnings
    warnings.warn(
        "set_light_position() is deprecated. Use set_light() with a Light instance.",
        DeprecationWarning,
        stacklevel=2
    )
    self.light.position = np.array(light_position, dtype=np.float32)
    self.gl_manager.set_uniform_vector("light_position", self.light.position)

def set_light_target(self, light_target):
    """
    Set the target point the light is pointing to.

    .. deprecated:: 0.2.4
        Use set_light() with a Light instance instead.

    Args:
        light_target: 3D target point
    """
    import warnings
    warnings.warn(
        "set_light_target() is deprecated. Use set_light() with a Light instance.",
        DeprecationWarning,
        stacklevel=2
    )
    self.light.target = np.array(light_target, dtype=np.float32)
    self.gl_manager.set_uniform_vector("light_target", self.light.target)
```

### Step 3: Update Package Exports

#### 3.1 Update `pyvr/__init__.py`

```python
"""PyVR: Python Volume Rendering Toolkit

PyVR provides GPU-accelerated 3D volume rendering using OpenGL/ModernGL
for real-time interactive visualization.
"""

from . import camera, datasets, lighting, moderngl_renderer, transferfunctions

__version__ = "0.2.4"
__all__ = ["camera", "datasets", "lighting", "moderngl_renderer", "transferfunctions"]
```

## Testing Requirements

### Unit Tests for Light Class

Create `tests/test_lighting/test_light.py`:

```python
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
```

### Integration Tests for VolumeRenderer

Add to `tests/test_moderngl_renderer/test_volume_renderer.py`:

```python
def test_volume_renderer_light_attribute(mock_moderngl_context):
    """VolumeRenderer should have a light attribute."""
    from pyvr.lighting import Light
    from pyvr.moderngl_renderer import VolumeRenderer

    renderer = VolumeRenderer(width=256, height=256)

    assert hasattr(renderer, 'light')
    assert isinstance(renderer.light, Light)


def test_volume_renderer_custom_light(mock_moderngl_context):
    """VolumeRenderer should accept custom light."""
    from pyvr.lighting import Light
    from pyvr.moderngl_renderer import VolumeRenderer

    custom_light = Light.directional(direction=[1, 0, 0], ambient=0.5)
    renderer = VolumeRenderer(width=256, height=256, light=custom_light)

    assert renderer.light is custom_light
    assert renderer.light.ambient_intensity == 0.5


def test_set_light(mock_moderngl_context):
    """set_light should accept Light instance."""
    from pyvr.lighting import Light
    from pyvr.moderngl_renderer import VolumeRenderer

    renderer = VolumeRenderer(width=256, height=256)
    new_light = Light.point_light(position=[5, 5, 5])

    renderer.set_light(new_light)

    assert renderer.light is new_light


def test_get_light(mock_moderngl_context):
    """get_light should return current light."""
    from pyvr.lighting import Light
    from pyvr.moderngl_renderer import VolumeRenderer

    renderer = VolumeRenderer(width=256, height=256)
    light = renderer.get_light()

    assert isinstance(light, Light)
    assert light is renderer.light


def test_deprecated_light_setters(mock_moderngl_context):
    """Deprecated light setters should work with warnings."""
    from pyvr.moderngl_renderer import VolumeRenderer

    renderer = VolumeRenderer(width=256, height=256)

    with pytest.warns(DeprecationWarning):
        renderer.set_ambient_light(0.5)

    with pytest.warns(DeprecationWarning):
        renderer.set_diffuse_light(0.9)

    with pytest.warns(DeprecationWarning):
        renderer.set_light_position([2, 2, 2])

    with pytest.warns(DeprecationWarning):
        renderer.set_light_target([0, 0, 0])
```

## Example Updates

### Update examples to use Light class

Example update for `example/ModernglRender/rgba_demo.py`:

```python
# Old way (deprecated)
renderer = VolumeRenderer(width=512, height=512)
renderer.set_ambient_light(0.3)
renderer.set_diffuse_light(0.9)
renderer.set_light_position([5, 5, 5])

# New way (recommended)
from pyvr.lighting import Light

light = Light.directional(
    direction=[1, -1, 0],
    ambient=0.3,
    diffuse=0.9
)
renderer = VolumeRenderer(width=512, height=512, light=light)

# Or set later
light = Light.point_light(position=[5, 5, 5], ambient=0.2, diffuse=0.8)
renderer.set_light(light)
```

## Validation Steps

### Pre-merge Checklist

- [ ] All existing tests pass (139 tests from Phase 1)
- [ ] New light unit tests pass (+12 tests)
- [ ] Integration tests with VolumeRenderer pass
- [ ] Deprecation warnings work correctly
- [ ] All examples updated and run successfully
- [ ] Documentation updated (README.md, CLAUDE.md)
- [ ] No performance regression

### Manual Validation

```bash
# Run full test suite
pytest tests/ -v

# Run only lighting tests
pytest tests/test_lighting/ -v

# Run with coverage
pytest --cov=pyvr.lighting --cov-report=term-missing tests/test_lighting/

# Test examples
python example/ModernglRender/multiview_example.py
python example/ModernglRender/enhanced_camera_demo.py
python example/ModernglRender/rgba_demo.py

# Check deprecation warnings
pytest tests/test_moderngl_renderer/ -W default::DeprecationWarning
```

## Migration Guide for Users

### Recommended Usage (New API)

```python
from pyvr.lighting import Light
from pyvr.moderngl_renderer import VolumeRenderer

# Create various light types
light_dir = Light.directional(direction=[1, -1, 0], ambient=0.3, diffuse=0.9)
light_point = Light.point_light(position=[5, 5, 5], ambient=0.2, diffuse=0.8)
light_ambient = Light.ambient_only(intensity=0.4)

# Use with renderer
renderer = VolumeRenderer(width=512, height=512, light=light_dir)

# Or set later
renderer.set_light(light_point)

# Modify light properties
new_light = renderer.get_light().copy()
new_light.ambient_intensity = 0.5
renderer.set_light(new_light)
```

### Backward Compatibility

Deprecated methods still work but emit warnings:
```python
# These still work but are deprecated
renderer.set_ambient_light(0.3)  # DeprecationWarning
renderer.set_diffuse_light(0.9)  # DeprecationWarning
renderer.set_light_position([5, 5, 5])  # DeprecationWarning
renderer.set_light_target([0, 0, 0])  # DeprecationWarning
```

**Deprecation timeline**:
- v0.2.4: Deprecated, emit warnings
- v0.3.0: Remove deprecated methods (breaking change)

## Benefits Achieved

1. ✅ **Pipeline Alignment**: Lighting isolated in Application Stage
2. ✅ **Consistent Architecture**: Follows Camera refactoring pattern
3. ✅ **Reduced VolumeRenderer Complexity**: Removes 4 setter methods
4. ✅ **Better Testability**: Lighting logic testable independently
5. ✅ **Extensibility**: Easy to add specular lighting, multiple lights, shadows
6. ✅ **Type Safety**: Light class provides clear interface

## Timeline

- **Implementation**: 1 day
- **Testing**: 0.5 day
- **Documentation & Examples**: 0.5 day
- **Total**: 2 days

## Dependencies

- **Requires**: Phase 1 (Camera refactoring) completed
- **Blocks**: None (can proceed in parallel with Phase 3)

## Next Phase

After Phase 2 completion, proceed to **Phase 3: Volume Refactoring (v0.2.5)**
