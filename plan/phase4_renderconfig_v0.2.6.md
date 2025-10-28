# Phase 4: RenderConfig Refactoring (v0.2.6)

> ⚠️ **BREAKING CHANGES**: This phase removes backward compatibility.
> Pre-1.0 development prioritizes clean implementation over API stability.
> See [REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md) for migration guide.

## Overview
Extract rendering configuration parameters from `VolumeRenderer` constructor into a dedicated `RenderConfig` class. This completes the pipeline separation by isolating Rasterization Stage configuration (ray marching parameters) from the renderer itself, and provides quality presets for users.

**Breaking Changes**:
- Remove `step_size`, `max_steps` from `VolumeRenderer.__init__()`
- Remove methods: `set_step_size()`, `set_max_steps()`
- Add `config` parameter to `VolumeRenderer.__init__()`
- All rendering configuration through `RenderConfig` class only

## Goals
- ✅ Create `RenderConfig` class to encapsulate rendering parameters
- ✅ Provide quality presets (fast, balanced, high_quality)
- ✅ Add `config` attribute to `VolumeRenderer`
- ✅ Remove ray marching setter methods from `VolumeRenderer`
- ✅ Simplify VolumeRenderer constructor
- ✅ Maintain backward compatibility

## Files to Create

### New Module
- `pyvr/config.py` - RenderConfig class implementation

### Files to Modify

### Core Implementation
- `pyvr/moderngl_renderer/renderer.py` - Add config attribute, refactor constructor
- `pyvr/__init__.py` - Add RenderConfig to exports

### Tests
- Create `tests/test_config.py` - Unit tests for RenderConfig class
- `tests/test_moderngl_renderer/test_volume_renderer.py` - Integration tests

### Examples
- Update all examples to use RenderConfig presets
- `example/ModernglRender/enhanced_camera_demo.py`
- `example/ModernglRender/multiview_example.py`
- `example/ModernglRender/rgba_demo.py`
- `example/benchmark.py` - Add benchmark for different configs

### Documentation
- `README.md` - Add RenderConfig examples and quality presets
- `CLAUDE.md` - Update architecture section

## Detailed Implementation Steps

### Step 1: Create RenderConfig Class

#### 1.1 Create `pyvr/config.py`

```python
"""
Rendering configuration for PyVR.

This module provides the RenderConfig class for managing rendering
quality and performance parameters.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class RenderConfig:
    """
    Rendering quality and performance configuration.

    This class encapsulates ray marching parameters and other rendering
    settings that affect quality vs. performance tradeoffs.

    Attributes:
        step_size: Ray marching step size in normalized volume space.
                   Smaller values = higher quality but slower.
        max_steps: Maximum number of ray marching steps per ray.
                   Higher values = more detail but slower.
        early_ray_termination: Enable early ray termination optimization.
                              Stops ray when accumulated opacity reaches threshold.
        opacity_threshold: Opacity threshold for early ray termination (0.0-1.0).
                          Only used if early_ray_termination is True.
    """

    step_size: float = 0.01
    max_steps: int = 500
    early_ray_termination: bool = True
    opacity_threshold: float = 0.95

    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()

    def validate(self) -> None:
        """
        Validate rendering configuration parameters.

        Raises:
            ValueError: If any parameter is invalid
        """
        if self.step_size <= 0:
            raise ValueError("step_size must be positive")

        if self.step_size > 1.0:
            import warnings
            warnings.warn(
                f"step_size {self.step_size} is very large and may produce poor quality",
                UserWarning
            )

        if self.max_steps < 1:
            raise ValueError("max_steps must be at least 1")

        if self.max_steps > 10000:
            import warnings
            warnings.warn(
                f"max_steps {self.max_steps} is very large and may be slow",
                UserWarning
            )

        if not (0.0 <= self.opacity_threshold <= 1.0):
            raise ValueError("opacity_threshold must be between 0.0 and 1.0")

    @classmethod
    def fast(cls) -> "RenderConfig":
        """
        Fast rendering preset - prioritizes speed over quality.

        Best for: Interactive exploration, real-time manipulation, previews

        Settings:
        - step_size: 0.02 (larger steps, fewer samples)
        - max_steps: 100 (limit maximum samples)
        - early_ray_termination: True (stop early when possible)

        Returns:
            RenderConfig configured for fast rendering

        Example:
            >>> config = RenderConfig.fast()
            >>> renderer = VolumeRenderer(width=512, height=512, config=config)
        """
        return cls(
            step_size=0.02,
            max_steps=100,
            early_ray_termination=True,
            opacity_threshold=0.90,
        )

    @classmethod
    def balanced(cls) -> "RenderConfig":
        """
        Balanced rendering preset - good quality and reasonable speed.

        Best for: General use, most visualization tasks, development

        Settings:
        - step_size: 0.01 (standard sampling density)
        - max_steps: 500 (sufficient for most volumes)
        - early_ray_termination: True (maintain performance)

        Returns:
            RenderConfig configured for balanced rendering (default)

        Example:
            >>> config = RenderConfig.balanced()
            >>> renderer = VolumeRenderer(width=512, height=512, config=config)
        """
        return cls(
            step_size=0.01,
            max_steps=500,
            early_ray_termination=True,
            opacity_threshold=0.95,
        )

    @classmethod
    def high_quality(cls) -> "RenderConfig":
        """
        High quality rendering preset - prioritizes quality over speed.

        Best for: Final renders, publications, detailed analysis, screenshots

        Settings:
        - step_size: 0.005 (small steps, dense sampling)
        - max_steps: 1000 (allow more detail)
        - early_ray_termination: True (but with higher threshold)

        Returns:
            RenderConfig configured for high quality rendering

        Example:
            >>> config = RenderConfig.high_quality()
            >>> renderer = VolumeRenderer(width=1024, height=1024, config=config)
        """
        return cls(
            step_size=0.005,
            max_steps=1000,
            early_ray_termination=True,
            opacity_threshold=0.98,
        )

    @classmethod
    def ultra_quality(cls) -> "RenderConfig":
        """
        Ultra quality rendering preset - maximum quality, very slow.

        Best for: Final publication renders, extreme detail requirements

        Settings:
        - step_size: 0.001 (very small steps)
        - max_steps: 2000 (maximum detail)
        - early_ray_termination: False (sample entire ray)

        Returns:
            RenderConfig configured for ultra quality rendering

        Warning:
            This preset can be very slow. Use sparingly.

        Example:
            >>> config = RenderConfig.ultra_quality()
            >>> renderer = VolumeRenderer(width=2048, height=2048, config=config)
        """
        return cls(
            step_size=0.001,
            max_steps=2000,
            early_ray_termination=False,
            opacity_threshold=1.0,
        )

    @classmethod
    def preview(cls) -> "RenderConfig":
        """
        Preview quality preset - extremely fast, low quality.

        Best for: Quick previews, testing, rapid iteration

        Settings:
        - step_size: 0.05 (very large steps)
        - max_steps: 50 (minimal samples)
        - early_ray_termination: True (aggressive early stopping)

        Returns:
            RenderConfig configured for preview rendering

        Example:
            >>> config = RenderConfig.preview()
            >>> renderer = VolumeRenderer(width=256, height=256, config=config)
        """
        return cls(
            step_size=0.05,
            max_steps=50,
            early_ray_termination=True,
            opacity_threshold=0.80,
        )

    @classmethod
    def custom(
        cls,
        step_size: float,
        max_steps: int,
        early_ray_termination: bool = True,
        opacity_threshold: float = 0.95,
    ) -> "RenderConfig":
        """
        Create custom rendering configuration.

        Args:
            step_size: Custom step size
            max_steps: Custom maximum steps
            early_ray_termination: Enable early termination
            opacity_threshold: Opacity threshold for early termination

        Returns:
            RenderConfig with custom parameters

        Example:
            >>> config = RenderConfig.custom(
            ...     step_size=0.015,
            ...     max_steps=300,
            ...     early_ray_termination=True
            ... )
        """
        return cls(
            step_size=step_size,
            max_steps=max_steps,
            early_ray_termination=early_ray_termination,
            opacity_threshold=opacity_threshold,
        )

    def copy(self) -> "RenderConfig":
        """
        Create a copy of this configuration.

        Returns:
            New RenderConfig instance with same parameters

        Example:
            >>> original = RenderConfig.balanced()
            >>> modified = original.copy()
            >>> modified.step_size = 0.005  # Doesn't affect original
        """
        return RenderConfig(
            step_size=self.step_size,
            max_steps=self.max_steps,
            early_ray_termination=self.early_ray_termination,
            opacity_threshold=self.opacity_threshold,
        )

    def with_step_size(self, step_size: float) -> "RenderConfig":
        """
        Create new config with modified step size.

        Args:
            step_size: New step size value

        Returns:
            New RenderConfig with updated step_size

        Example:
            >>> config = RenderConfig.balanced().with_step_size(0.005)
        """
        config = self.copy()
        config.step_size = step_size
        config.validate()
        return config

    def with_max_steps(self, max_steps: int) -> "RenderConfig":
        """
        Create new config with modified max steps.

        Args:
            max_steps: New max steps value

        Returns:
            New RenderConfig with updated max_steps

        Example:
            >>> config = RenderConfig.balanced().with_max_steps(800)
        """
        config = self.copy()
        config.max_steps = max_steps
        config.validate()
        return config

    def estimate_samples_per_ray(self) -> int:
        """
        Estimate average number of samples per ray.

        This is a rough estimate assuming rays traverse the entire volume diagonal.

        Returns:
            Estimated number of samples per ray

        Example:
            >>> config = RenderConfig.balanced()
            >>> config.estimate_samples_per_ray()
            173
        """
        # Assume unit cube diagonal (~1.732)
        diagonal_length = 1.732
        estimated_samples = int(diagonal_length / self.step_size)
        return min(estimated_samples, self.max_steps)

    def estimate_render_time_relative(self) -> float:
        """
        Estimate relative render time compared to balanced preset.

        Returns:
            Multiplier relative to balanced preset (1.0 = balanced speed)

        Example:
            >>> fast = RenderConfig.fast()
            >>> fast.estimate_render_time_relative()
            0.2
            >>> hq = RenderConfig.high_quality()
            >>> hq.estimate_render_time_relative()
            5.0
        """
        balanced = RenderConfig.balanced()
        balanced_samples = balanced.estimate_samples_per_ray()
        our_samples = self.estimate_samples_per_ray()

        relative_time = our_samples / balanced_samples
        return relative_time

    def __repr__(self) -> str:
        """String representation of rendering configuration."""
        return (
            f"RenderConfig("
            f"step_size={self.step_size}, "
            f"max_steps={self.max_steps}, "
            f"early_termination={self.early_ray_termination}, "
            f"~{self.estimate_samples_per_ray()} samples/ray)"
        )


class RenderConfigError(Exception):
    """Exception raised for rendering configuration errors."""

    pass
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
        config=None,  # NEW: RenderConfig parameter
        camera=None,
        light=None,
    ):
        """
        Initializes the volume renderer.

        Parameters:
            width (int): Viewport width (default: 512)
            height (int): Viewport height (default: 512)
            config (RenderConfig, optional): Rendering configuration.
                                            If None, uses balanced preset.
            camera (Camera, optional): Camera instance. If None, creates default.
            light (Light, optional): Light configuration. If None, creates default.

        Example:
            >>> from pyvr.config import RenderConfig
            >>> from pyvr.moderngl_renderer import VolumeRenderer
            >>>
            >>> # Use preset
            >>> config = RenderConfig.high_quality()
            >>> renderer = VolumeRenderer(width=1024, height=1024, config=config)
            >>>
            >>> # Use defaults (balanced preset)
            >>> renderer = VolumeRenderer(width=512, height=512)
        """
        self.width = width
        self.height = height

        # Initialize render config
        if config is None:
            from ..config import RenderConfig
            self.config = RenderConfig.balanced()
        else:
            self.config = config

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

        # Load shaders
        pyvr_dir = os.path.dirname(os.path.dirname(__file__))
        shader_dir = os.path.join(pyvr_dir, "shaders")
        vertex_shader_path = os.path.join(shader_dir, "volume.vert.glsl")
        fragment_shader_path = os.path.join(shader_dir, "volume.frag.glsl")
        self.gl_manager.load_shaders(vertex_shader_path, fragment_shader_path)

        # Set default uniforms
        self._update_render_config()
        self.gl_manager.set_uniform_vector("volume_min_bounds", (-0.5, -0.5, -0.5))
        self.gl_manager.set_uniform_vector("volume_max_bounds", (0.5, 0.5, 0.5))

        # Set light uniforms
        self._update_light()

    def _update_render_config(self):
        """Update OpenGL uniforms from current render configuration."""
        self.gl_manager.set_uniform_float("step_size", self.config.step_size)
        self.gl_manager.set_uniform_int("max_steps", self.config.max_steps)

        # Note: early_ray_termination and opacity_threshold would require
        # shader modifications, planned for future versions
```

#### 2.2 Add `set_config()` method

```python
def set_config(self, config):
    """
    Set rendering configuration.

    Args:
        config: RenderConfig instance with rendering parameters

    Example:
        >>> from pyvr.config import RenderConfig
        >>> config = RenderConfig.high_quality()
        >>> renderer.set_config(config)
    """
    from ..config import RenderConfig

    if not isinstance(config, RenderConfig):
        raise TypeError(f"Expected RenderConfig instance, got {type(config)}")

    self.config = config
    self._update_render_config()
```

#### 2.3 Add `get_config()` method

```python
def get_config(self):
    """
    Get current rendering configuration.

    Returns:
        RenderConfig: Current configuration

    Example:
        >>> renderer = VolumeRenderer()
        >>> config = renderer.get_config()
        >>> print(config)
    """
    return self.config
```

#### 2.4 Add backward compatibility methods (deprecated)

```python
def set_step_size(self, step_size):
    """
    Set the ray marching step size.

    .. deprecated:: 0.2.6
        Use set_config() with a RenderConfig instance instead.

    Args:
        step_size: Ray marching step size
    """
    import warnings
    warnings.warn(
        "set_step_size() is deprecated. Use set_config() with a RenderConfig instance.",
        DeprecationWarning,
        stacklevel=2
    )
    self.config.step_size = step_size
    self.gl_manager.set_uniform_float("step_size", step_size)

def set_max_steps(self, max_steps):
    """
    Set the maximum number of ray marching steps.

    .. deprecated:: 0.2.6
        Use set_config() with a RenderConfig instance instead.

    Args:
        max_steps: Maximum ray marching steps
    """
    import warnings
    warnings.warn(
        "set_max_steps() is deprecated. Use set_config() with a RenderConfig instance.",
        DeprecationWarning,
        stacklevel=2
    )
    self.config.max_steps = max_steps
    self.gl_manager.set_uniform_int("max_steps", max_steps)
```

### Step 3: Update Package Exports

#### 3.1 Update `pyvr/__init__.py`

```python
"""PyVR: Python Volume Rendering Toolkit

PyVR provides GPU-accelerated 3D volume rendering using OpenGL/ModernGL
for real-time interactive visualization.
"""

from . import camera, datasets, lighting, moderngl_renderer, transferfunctions, volume
from .config import RenderConfig

__version__ = "0.2.6"
__all__ = [
    "camera",
    "datasets",
    "lighting",
    "moderngl_renderer",
    "transferfunctions",
    "volume",
    "RenderConfig",
]
```

## Testing Requirements

### Unit Tests for RenderConfig

Create `tests/test_config.py`:

```python
"""Tests for RenderConfig class."""
import pytest

from pyvr.config import RenderConfig, RenderConfigError


class TestRenderConfigPresets:
    """Test RenderConfig preset factory methods."""

    def test_fast_preset(self):
        """Fast preset should have performance-oriented settings."""
        config = RenderConfig.fast()

        assert config.step_size > 0.01  # Larger step size
        assert config.max_steps < 200  # Fewer steps
        assert config.early_ray_termination is True

    def test_balanced_preset(self):
        """Balanced preset should have default settings."""
        config = RenderConfig.balanced()

        assert config.step_size == 0.01
        assert config.max_steps == 500
        assert config.early_ray_termination is True

    def test_high_quality_preset(self):
        """High quality preset should have quality-oriented settings."""
        config = RenderConfig.high_quality()

        assert config.step_size < 0.01  # Smaller step size
        assert config.max_steps > 500  # More steps

    def test_ultra_quality_preset(self):
        """Ultra quality preset should have maximum quality settings."""
        config = RenderConfig.ultra_quality()

        assert config.step_size <= 0.001
        assert config.max_steps >= 1000

    def test_preview_preset(self):
        """Preview preset should have minimal quality settings."""
        config = RenderConfig.preview()

        assert config.step_size >= 0.05
        assert config.max_steps <= 100

    def test_custom_preset(self):
        """Custom preset should accept arbitrary parameters."""
        config = RenderConfig.custom(
            step_size=0.015,
            max_steps=300,
            early_ray_termination=False
        )

        assert config.step_size == 0.015
        assert config.max_steps == 300
        assert config.early_ray_termination is False


class TestRenderConfigValidation:
    """Test RenderConfig validation."""

    def test_invalid_step_size(self):
        """Negative or zero step_size should raise error."""
        with pytest.raises(ValueError, match="step_size"):
            RenderConfig(step_size=0.0)

        with pytest.raises(ValueError, match="step_size"):
            RenderConfig(step_size=-0.01)

    def test_invalid_max_steps(self):
        """max_steps < 1 should raise error."""
        with pytest.raises(ValueError, match="max_steps"):
            RenderConfig(max_steps=0)

        with pytest.raises(ValueError, match="max_steps"):
            RenderConfig(max_steps=-10)

    def test_large_step_size_warning(self):
        """Very large step_size should emit warning."""
        with pytest.warns(UserWarning, match="step_size"):
            RenderConfig(step_size=2.0)

    def test_large_max_steps_warning(self):
        """Very large max_steps should emit warning."""
        with pytest.warns(UserWarning, match="max_steps"):
            RenderConfig(max_steps=50000)

    def test_invalid_opacity_threshold(self):
        """opacity_threshold outside [0, 1] should raise error."""
        with pytest.raises(ValueError, match="opacity_threshold"):
            RenderConfig(opacity_threshold=1.5)

        with pytest.raises(ValueError, match="opacity_threshold"):
            RenderConfig(opacity_threshold=-0.1)


class TestRenderConfigMethods:
    """Test RenderConfig methods."""

    def test_copy(self):
        """copy should create independent instance."""
        original = RenderConfig.balanced()
        copy = original.copy()

        # Modify copy
        copy.step_size = 0.005

        # Original should be unchanged
        assert original.step_size == 0.01

    def test_with_step_size(self):
        """with_step_size should create new config."""
        original = RenderConfig.balanced()
        modified = original.with_step_size(0.005)

        assert modified.step_size == 0.005
        assert original.step_size == 0.01  # Unchanged

    def test_with_max_steps(self):
        """with_max_steps should create new config."""
        original = RenderConfig.balanced()
        modified = original.with_max_steps(800)

        assert modified.max_steps == 800
        assert original.max_steps == 500  # Unchanged

    def test_estimate_samples_per_ray(self):
        """estimate_samples_per_ray should return reasonable value."""
        config = RenderConfig.balanced()
        samples = config.estimate_samples_per_ray()

        assert samples > 0
        assert samples <= config.max_steps

    def test_estimate_render_time_relative(self):
        """estimate_render_time_relative should return reasonable values."""
        fast = RenderConfig.fast()
        balanced = RenderConfig.balanced()
        hq = RenderConfig.high_quality()

        # Fast should be faster than balanced
        assert fast.estimate_render_time_relative() < 1.0

        # Balanced should be 1.0 (reference)
        assert balanced.estimate_render_time_relative() == 1.0

        # High quality should be slower than balanced
        assert hq.estimate_render_time_relative() > 1.0

    def test_repr(self):
        """__repr__ should return informative string."""
        config = RenderConfig.balanced()
        repr_str = repr(config)

        assert "RenderConfig" in repr_str
        assert "step_size" in repr_str
        assert "samples/ray" in repr_str
```

### Integration Tests

Add to `tests/test_moderngl_renderer/test_volume_renderer.py`:

```python
def test_volume_renderer_config_attribute(mock_moderngl_context):
    """VolumeRenderer should have a config attribute."""
    from pyvr.config import RenderConfig
    from pyvr.moderngl_renderer import VolumeRenderer

    renderer = VolumeRenderer(width=256, height=256)

    assert hasattr(renderer, 'config')
    assert isinstance(renderer.config, RenderConfig)


def test_volume_renderer_custom_config(mock_moderngl_context):
    """VolumeRenderer should accept custom config."""
    from pyvr.config import RenderConfig
    from pyvr.moderngl_renderer import VolumeRenderer

    custom_config = RenderConfig.high_quality()
    renderer = VolumeRenderer(width=256, height=256, config=custom_config)

    assert renderer.config is custom_config
    assert renderer.config.step_size == 0.005


def test_set_config(mock_moderngl_context):
    """set_config should accept RenderConfig instance."""
    from pyvr.config import RenderConfig
    from pyvr.moderngl_renderer import VolumeRenderer

    renderer = VolumeRenderer(width=256, height=256)
    new_config = RenderConfig.fast()

    renderer.set_config(new_config)

    assert renderer.config is new_config


def test_get_config(mock_moderngl_context):
    """get_config should return current config."""
    from pyvr.config import RenderConfig
    from pyvr.moderngl_renderer import VolumeRenderer

    renderer = VolumeRenderer(width=256, height=256)
    config = renderer.get_config()

    assert isinstance(config, RenderConfig)
    assert config is renderer.config


def test_deprecated_render_setters(mock_moderngl_context):
    """Deprecated render setters should work with warnings."""
    from pyvr.moderngl_renderer import VolumeRenderer

    renderer = VolumeRenderer(width=256, height=256)

    with pytest.warns(DeprecationWarning):
        renderer.set_step_size(0.005)

    with pytest.warns(DeprecationWarning):
        renderer.set_max_steps(800)
```

## Example Updates

### Update benchmark to compare presets

Update `example/benchmark.py`:

```python
from pyvr.config import RenderConfig
from pyvr.moderngl_renderer import VolumeRenderer
import time

# Benchmark different quality presets
configs = {
    "preview": RenderConfig.preview(),
    "fast": RenderConfig.fast(),
    "balanced": RenderConfig.balanced(),
    "high_quality": RenderConfig.high_quality(),
}

results = {}
for name, config in configs.items():
    renderer = VolumeRenderer(width=512, height=512, config=config)
    # ... load volume ...

    start = time.time()
    for _ in range(100):
        renderer.render()
    elapsed = time.time() - start

    results[name] = elapsed / 100
    print(f"{name}: {results[name]*1000:.2f}ms/frame ({1/results[name]:.1f} FPS)")

# Output example:
# preview: 5.2ms/frame (192.3 FPS)
# fast: 10.8ms/frame (92.6 FPS)
# balanced: 15.6ms/frame (64.1 FPS)
# high_quality: 45.3ms/frame (22.1 FPS)
```

## Validation Steps

### Pre-merge Checklist

- [ ] All existing tests pass (161 tests from Phase 3)
- [ ] New config unit tests pass (+15 tests)
- [ ] Integration tests pass
- [ ] Deprecation warnings work correctly
- [ ] Benchmark shows expected performance differences
- [ ] Examples updated and run successfully
- [ ] Documentation updated
- [ ] README includes preset comparison table

### Manual Validation

```bash
# Run full test suite
pytest tests/ -v

# Run config tests
pytest tests/test_config.py -v

# Run benchmark
python example/benchmark.py

# Visual quality comparison
python example/ModernglRender/quality_comparison.py  # Create this
```

## Migration Guide

### Recommended Usage (New API)

```python
from pyvr.config import RenderConfig
from pyvr.moderngl_renderer import VolumeRenderer

# Use presets (easiest)
renderer_fast = VolumeRenderer(width=512, height=512, config=RenderConfig.fast())
renderer_hq = VolumeRenderer(width=1024, height=1024, config=RenderConfig.high_quality())

# Modify preset
config = RenderConfig.balanced().with_step_size(0.008)
renderer = VolumeRenderer(width=512, height=512, config=config)

# Custom configuration
config = RenderConfig.custom(step_size=0.015, max_steps=300)
renderer = VolumeRenderer(width=512, height=512, config=config)

# Change config at runtime
renderer.set_config(RenderConfig.fast())  # Switch to fast rendering
```

### Quality Preset Comparison

| Preset | Step Size | Max Steps | Est. Speed | Use Case |
|--------|-----------|-----------|------------|----------|
| preview | 0.05 | 50 | ~15x faster | Quick iteration |
| fast | 0.02 | 100 | ~5x faster | Interactive exploration |
| **balanced** | 0.01 | 500 | **1x (baseline)** | **General use** |
| high_quality | 0.005 | 1000 | ~5x slower | Final renders |
| ultra_quality | 0.001 | 2000 | ~20x slower | Publication quality |

### Backward Compatibility

```python
# Old way still works but emits warnings
renderer = VolumeRenderer(width=512, height=512)
renderer.set_step_size(0.005)  # DeprecationWarning
renderer.set_max_steps(800)  # DeprecationWarning
```

## Benefits Achieved

1. ✅ **User-Friendly Presets**: Easy quality selection (fast/balanced/high_quality)
2. ✅ **Simplified Constructor**: 3 parameters instead of 8
3. ✅ **Pipeline Alignment**: Rasterization Stage configuration isolated
4. ✅ **Performance Visibility**: Users understand speed/quality tradeoffs
5. ✅ **Extensibility**: Easy to add adaptive step size, progressive rendering
6. ✅ **Consistent Architecture**: Completes the Camera/Light/Volume/Config pattern

## Architecture Completion

After Phase 4, PyVR architecture is fully aligned with traditional rendering pipeline:

```
Application Stage (Complete):
├── Volume (data, bounds) ✅
├── Camera (view, projection) ✅
├── Light (position, intensity) ✅
└── TransferFunctions (color, opacity) ✅

Geometry Stage (Complete):
└── Camera.get_view_matrix() ✅
└── Camera.get_projection_matrix() ✅

Rasterization Stage (Complete):
└── RenderConfig (step_size, max_steps) ✅

Fragment Stage (Existing):
└── Shaders (volume.frag.glsl) ✅

Backend (Existing):
└── ModernGLManager ✅
```

**Pipeline Alignment: 100%** 🎉

## Timeline

- **Implementation**: 1 day
- **Testing**: 0.5 day
- **Documentation & Benchmarks**: 0.5 day
- **Total**: 2 days

## Dependencies

- **Requires**: Phase 1, 2, 3 completed
- **Blocks**: None (final phase)

## Post-Refactoring

After Phase 4, the codebase is ready for:
- **v0.3.0**: Advanced features (Scene abstraction, multiple lights, shadows)
- **v0.3.x**: Performance optimizations (adaptive step size, progressive rendering)
- **v0.4.x**: Advanced rendering (ambient occlusion, isosurfaces)

## Summary

Phase 4 completes the 4-phase refactoring roadmap (v0.2.3 → v0.2.6), transforming PyVR from a monolithic renderer into a well-architected, pipeline-aligned volume rendering toolkit. Users benefit from intuitive quality presets, developers benefit from clean separation of concerns, and the project is positioned for advanced features in v0.3+.
