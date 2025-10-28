# Phase 1: Camera Refactoring (v0.2.3)

## Overview
Rename `CameraParameters` to `Camera` and move camera matrix creation logic from `VolumeRenderer` to the `Camera` class. This aligns the codebase with traditional rendering pipeline architecture where the camera owns its view/projection transformations.

**This is a breaking change** - API will change, but comprehensive tests ensure correctness.

## Goals
- ✅ Rename `CameraParameters` → `Camera` (breaking change)
- ✅ Move view/projection matrix creation to `Camera` class
- ✅ Add `camera` attribute to `VolumeRenderer`
- ✅ Comprehensive test coverage for all new functionality
- ✅ Improve separation of concerns (Geometry Stage isolation)

## Files to Modify

### Core Implementation
- `pyvr/camera/parameters.py` - Rename class, add matrix methods
- `pyvr/camera/control.py` - Update type hints and references
- `pyvr/camera/__init__.py` - Update exports
- `pyvr/moderngl_renderer/renderer.py` - Add camera attribute, refactor set_camera()

### Tests (Required)
- `tests/test_camera/test_parameters.py` - Update all tests
- `tests/test_camera/test_control.py` - Update type hint tests
- Create `tests/test_camera/test_matrix_creation.py` - **New tests for matrix methods**
- `tests/test_moderngl_renderer/test_volume_renderer.py` - Camera integration tests

### Examples (Update Required)
- `example/ModernglRender/enhanced_camera_demo.py`
- `example/ModernglRender/multiview_example.py`
- `example/ModernglRender/rgba_demo.py`
- `example/benchmark.py`

### Documentation
- `README.md` - Update API examples
- `CLAUDE.md` - Update architecture section

## Detailed Implementation Steps

### Step 1: Rename CameraParameters to Camera

#### 1.1 Update `pyvr/camera/parameters.py`

**Action**: Rename the class (clean break, no aliases)

```python
@dataclass
class Camera:
    """
    Camera with validation, matrix creation, and presets.

    This class encapsulates all camera-related parameters and provides
    validation, serialization, matrix generation, and preset management.

    Attributes:
        target: 3D point the camera is looking at
        azimuth: Horizontal rotation angle in radians
        elevation: Vertical rotation angle in radians
        roll: Roll rotation angle in radians
        distance: Distance from camera to target
        init_pos: Initial camera position (relative to target)
        init_up: Initial up vector
        fov: Field of view in radians
        near_plane: Near clipping plane distance
        far_plane: Far clipping plane distance
    """
    # Core positioning parameters
    target: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=np.float32)
    )
    azimuth: float = 0.0
    elevation: float = 0.0
    roll: float = 0.0
    distance: float = 3.0

    # Initial vectors
    init_pos: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=np.float32)
    )
    init_up: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 1.0], dtype=np.float32)
    )

    # Projection parameters
    fov: float = np.pi / 4  # 45 degrees
    near_plane: float = 0.1
    far_plane: float = 100.0

    # ... existing __post_init__, validate(), etc. unchanged
```

**Update all classmethod return types**:
```python
@classmethod
def from_spherical(cls, ...) -> "Camera":
    """Create camera from spherical coordinates."""
    ...

@classmethod
def front_view(cls, ...) -> "Camera":
    ...

# Update all preset methods similarly
```

**Update `__repr__`**:
```python
def __repr__(self) -> str:
    """String representation of camera."""
    return (
        f"Camera("
        f"target={self.target}, "
        f"azimuth={self.azimuth:.3f} rad ({np.degrees(self.azimuth):.1f}°), "
        f"elevation={self.elevation:.3f} rad ({np.degrees(self.elevation):.1f}°), "
        f"roll={self.roll:.3f} rad ({np.degrees(self.roll):.1f}°), "
        f"distance={self.distance:.2f})"
    )
```

**Rename exception class**:
```python
class CameraError(Exception):
    """Exception raised for camera errors."""
    pass
```

Update all docstrings that reference "CameraParameters" to "Camera".

#### 1.2 Update `pyvr/camera/__init__.py`

```python
"""
Camera system for PyVR volume rendering.

Provides camera management, controls, and animation utilities.
"""

from .control import (
    CameraController,
    CameraPath,
    get_camera_pos,
    get_camera_pos_from_params,
)
from .parameters import (
    Camera,
    CameraError,
    degrees_to_radians,
    radians_to_degrees,
    validate_camera_angles,
)

__all__ = [
    "Camera",
    "CameraController",
    "CameraPath",
    "get_camera_pos",
    "get_camera_pos_from_params",
    "CameraError",
    "validate_camera_angles",
    "degrees_to_radians",
    "radians_to_degrees",
]
```

#### 1.3 Update `pyvr/camera/control.py`

**Update type hints throughout**:

```python
from .parameters import Camera, CameraError, validate_camera_angles

def get_camera_pos_from_params(
    params: Camera,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate camera position and up vector from Camera object."""
    ...

class CameraPath:
    """Camera path animation utility."""

    def __init__(self, keyframes: List[Camera]):
        """Initialize with Camera keyframes."""
        ...

    def interpolate(self, t: float) -> Camera:
        """Interpolate camera at time t."""
        ...
        return Camera(  # Changed from CameraParameters
            target=self._lerp_vector(kf1.target, kf2.target, local_t),
            ...
        )

    def generate_frames(self, n_frames: int) -> List[Camera]:
        """Generate animation frames."""
        ...

class CameraController:
    """High-level camera controller."""

    def __init__(self, initial_params: Optional[Camera] = None):
        """Initialize controller."""
        if initial_params is None:
            initial_params = Camera.front_view()
        ...

    def reset_to_preset(self, preset: str, distance: Optional[float] = None) -> None:
        """Reset to preset view."""
        preset_methods = {
            "front": Camera.front_view,
            "side": Camera.side_view,
            "top": Camera.top_view,
            "isometric": Camera.isometric_view,
        }
        ...

    def animate_to(self, target_params: Camera, n_frames: int = 30) -> List[Camera]:
        """Create animation to target."""
        ...
```

### Step 2: Add Matrix Creation Methods to Camera

#### 2.1 Add `get_camera_vectors()` method

In `pyvr/camera/parameters.py`, add to `Camera` class:

```python
def get_camera_vectors(self) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get camera position and up vector from spherical parameters.

    Delegates to get_camera_pos_from_params() for quaternion-based calculation.

    Returns:
        Tuple of (position, up) vectors

    Example:
        >>> camera = Camera.from_spherical(
        ...     target=np.array([0, 0, 0]),
        ...     azimuth=np.pi/4,
        ...     elevation=np.pi/6,
        ...     roll=0.0,
        ...     distance=3.0
        ... )
        >>> position, up = camera.get_camera_vectors()
    """
    from .control import get_camera_pos_from_params
    return get_camera_pos_from_params(self)
```

#### 2.2 Add `get_view_matrix()` method

```python
def get_view_matrix(self) -> np.ndarray:
    """
    Create view matrix from camera parameters.

    Transforms from world space to camera space (Geometry Stage).

    Returns:
        4x4 view matrix as np.ndarray (float32)

    Example:
        >>> camera = Camera.isometric_view(distance=5.0)
        >>> view_matrix = camera.get_view_matrix()
        >>> view_matrix.shape
        (4, 4)
    """
    position, up = self.get_camera_vectors()

    # Look-at algorithm
    forward = self.target - position
    forward = forward / np.linalg.norm(forward)

    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)

    up_corrected = np.cross(right, forward)

    # View matrix (column-major for OpenGL)
    view_matrix = np.array(
        [
            [right[0], up_corrected[0], -forward[0], 0],
            [right[1], up_corrected[1], -forward[1], 0],
            [right[2], up_corrected[2], -forward[2], 0],
            [
                -np.dot(right, position),
                -np.dot(up_corrected, position),
                np.dot(forward, position),
                1,
            ],
        ],
        dtype=np.float32,
    )

    return view_matrix
```

#### 2.3 Add `get_projection_matrix()` method

```python
def get_projection_matrix(self, aspect_ratio: float) -> np.ndarray:
    """
    Create perspective projection matrix.

    Transforms from camera space to clip space.
    Uses camera's FOV, near plane, and far plane settings.

    Args:
        aspect_ratio: Width / height ratio of viewport

    Returns:
        4x4 projection matrix as np.ndarray (float32)

    Example:
        >>> camera = Camera.front_view(distance=3.0)
        >>> camera.fov = np.radians(60)
        >>> proj = camera.get_projection_matrix(aspect_ratio=16/9)
    """
    fov = self.fov
    near = self.near_plane
    far = self.far_plane

    f = 1.0 / np.tan(fov / 2.0)
    projection_matrix = np.array(
        [
            [f / aspect_ratio, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
            [0, 0, -1, 0],
        ],
        dtype=np.float32,
    )

    return projection_matrix
```

### Step 3: Refactor VolumeRenderer

#### 3.1 Add camera attribute to `VolumeRenderer.__init__()`

In `pyvr/moderngl_renderer/renderer.py`:

```python
class VolumeRenderer:
    def __init__(
        self,
        width=512,
        height=512,
        step_size=0.01,
        max_steps=200,
        ambient_light=0.2,
        diffuse_light=0.8,
        light_position=(1.0, 1.0, 1.0),
        light_target=(0.0, 0.0, 0.0),
        camera=None,  # NEW PARAMETER
    ):
        """
        Initialize volume renderer.

        Parameters:
            width (int): Viewport width (default: 512)
            height (int): Viewport height (default: 512)
            step_size (float): Ray marching step size (default: 0.01)
            max_steps (int): Max ray marching steps (default: 200)
            ambient_light (float): Ambient light intensity (default: 0.2)
            diffuse_light (float): Diffuse light intensity (default: 0.8)
            light_position (tuple): Light source position
            light_target (tuple): Light target point
            camera (Camera, optional): Camera instance. If None, creates default.
        """
        self.width = width
        self.height = height
        self.step_size = step_size
        self.max_steps = max_steps
        self.ambient_light = ambient_light
        self.diffuse_light = diffuse_light
        self.light_position = np.array(light_position, dtype=np.float32)
        self.light_target = np.array(light_target, dtype=np.float32)

        # Initialize camera
        if camera is None:
            from ..camera import Camera
            self.camera = Camera.front_view(distance=3.0)
        else:
            self.camera = camera

        # ... rest of __init__ unchanged
```

#### 3.2 Refactor `set_camera()` method

**Replace existing method entirely**:

```python
def set_camera(self, camera):
    """
    Set camera configuration.

    Args:
        camera: Camera instance

    Example:
        >>> from pyvr.camera import Camera
        >>> camera = Camera.isometric_view(distance=5.0)
        >>> renderer.set_camera(camera)
    """
    from ..camera import Camera

    if not isinstance(camera, Camera):
        raise TypeError(f"Expected Camera instance, got {type(camera)}")

    self.camera = camera

    # Get camera data
    position, up = camera.get_camera_vectors()
    view_matrix = camera.get_view_matrix()
    projection_matrix = camera.get_projection_matrix(self.width / self.height)

    # VolumeRenderer controls its GL state
    self.gl_manager.set_uniform_matrix("view_matrix", view_matrix)
    self.gl_manager.set_uniform_matrix("projection_matrix", projection_matrix)
    self.gl_manager.set_uniform_vector("camera_pos", tuple(position))
```

#### 3.3 Add `get_camera()` helper

```python
def get_camera(self):
    """
    Get current camera instance.

    Returns:
        Camera: Current camera configuration
    """
    return self.camera
```

## Testing Requirements

### Unit Tests for Camera Matrix Methods

Create `tests/test_camera/test_matrix_creation.py`:

```python
"""Tests for Camera matrix creation methods."""
import numpy as np
import pytest

from pyvr.camera import Camera


class TestCameraMatrixCreation:
    """Test view and projection matrix generation."""

    def test_get_view_matrix_shape(self):
        """View matrix should be 4x4 float32."""
        camera = Camera.front_view(distance=3.0)
        view_matrix = camera.get_view_matrix()

        assert view_matrix.shape == (4, 4)
        assert view_matrix.dtype == np.float32

    def test_get_projection_matrix_shape(self):
        """Projection matrix should be 4x4 float32."""
        camera = Camera.front_view(distance=3.0)
        proj_matrix = camera.get_projection_matrix(aspect_ratio=16/9)

        assert proj_matrix.shape == (4, 4)
        assert proj_matrix.dtype == np.float32

    def test_view_matrix_deterministic(self):
        """View matrix should be deterministic."""
        camera = Camera.front_view(distance=3.0)

        view1 = camera.get_view_matrix()
        view2 = camera.get_view_matrix()

        assert np.allclose(view1, view2)

    def test_view_matrix_valid(self):
        """View matrix should contain no NaN/inf."""
        camera = Camera.isometric_view(distance=5.0)
        view_matrix = camera.get_view_matrix()

        assert np.all(np.isfinite(view_matrix))
        assert view_matrix[3, 3] == 1.0  # Homogeneous coordinate

    def test_projection_matrix_fov_sensitivity(self):
        """Different FOV should produce different projection."""
        camera1 = Camera.front_view(distance=3.0)
        camera1.fov = np.radians(45)

        camera2 = Camera.front_view(distance=3.0)
        camera2.fov = np.radians(90)

        proj1 = camera1.get_projection_matrix(aspect_ratio=1.0)
        proj2 = camera2.get_projection_matrix(aspect_ratio=1.0)

        assert not np.allclose(proj1, proj2)

    def test_view_matrix_position_sensitivity(self):
        """Different positions produce different view matrices."""
        camera_front = Camera.front_view(distance=3.0)
        camera_side = Camera.side_view(distance=3.0)

        view_front = camera_front.get_view_matrix()
        view_side = camera_side.get_view_matrix()

        assert not np.allclose(view_front, view_side)

    def test_get_camera_vectors_consistency(self):
        """get_camera_vectors should be consistent."""
        camera = Camera.isometric_view(distance=5.0)

        pos1, up1 = camera.get_camera_vectors()
        pos2, up2 = camera.get_camera_vectors()

        assert np.allclose(pos1, pos2)
        assert np.allclose(up1, up2)

    def test_projection_aspect_ratio_effect(self):
        """Aspect ratio should affect projection matrix."""
        camera = Camera.front_view(distance=3.0)

        proj_square = camera.get_projection_matrix(aspect_ratio=1.0)
        proj_wide = camera.get_projection_matrix(aspect_ratio=16/9)

        assert not np.allclose(proj_square, proj_wide)
```

### Integration Tests for VolumeRenderer

Update `tests/test_moderngl_renderer/test_volume_renderer.py`:

```python
def test_volume_renderer_has_camera(mock_moderngl_context):
    """VolumeRenderer should have camera attribute."""
    from pyvr.camera import Camera
    from pyvr.moderngl_renderer import VolumeRenderer

    renderer = VolumeRenderer(width=256, height=256)

    assert hasattr(renderer, 'camera')
    assert isinstance(renderer.camera, Camera)


def test_volume_renderer_custom_camera(mock_moderngl_context):
    """VolumeRenderer should accept custom camera."""
    from pyvr.camera import Camera
    from pyvr.moderngl_renderer import VolumeRenderer

    custom_camera = Camera.isometric_view(distance=10.0)
    renderer = VolumeRenderer(width=256, height=256, camera=custom_camera)

    assert renderer.camera is custom_camera
    assert renderer.camera.distance == 10.0


def test_set_camera_with_instance(mock_moderngl_context):
    """set_camera should accept Camera instance."""
    from pyvr.camera import Camera
    from pyvr.moderngl_renderer import VolumeRenderer

    renderer = VolumeRenderer(width=256, height=256)
    new_camera = Camera.side_view(distance=5.0)

    renderer.set_camera(new_camera)

    assert renderer.camera is new_camera


def test_set_camera_type_check(mock_moderngl_context):
    """set_camera should reject non-Camera objects."""
    from pyvr.moderngl_renderer import VolumeRenderer

    renderer = VolumeRenderer(width=256, height=256)

    with pytest.raises(TypeError, match="Expected Camera"):
        renderer.set_camera([3, 3, 3])  # Wrong type


def test_get_camera(mock_moderngl_context):
    """get_camera should return current camera."""
    from pyvr.camera import Camera
    from pyvr.moderngl_renderer import VolumeRenderer

    renderer = VolumeRenderer(width=256, height=256)
    camera = renderer.get_camera()

    assert isinstance(camera, Camera)
    assert camera is renderer.camera


def test_camera_matrix_updates_uniforms(mock_moderngl_context):
    """Setting camera should update GL uniforms."""
    from pyvr.camera import Camera
    from pyvr.moderngl_renderer import VolumeRenderer

    renderer = VolumeRenderer(width=256, height=256)
    camera = Camera.top_view(distance=10.0)

    # This should not raise
    renderer.set_camera(camera)

    # Verify camera is set
    assert renderer.camera.distance == 10.0
```

### Update Existing Tests

Update `tests/test_camera/test_parameters.py`:
- Replace all `CameraParameters` with `Camera`
- Update all test names and docstrings
- Add tests for new matrix methods

Update `tests/test_camera/test_control.py`:
- Replace all `CameraParameters` with `Camera`
- Verify type hints are correct

## Validation Steps

### Pre-merge Checklist

- [ ] All existing tests pass with updated Camera name
- [ ] New matrix creation tests pass (+15 tests)
- [ ] Integration tests with VolumeRenderer pass
- [ ] All examples updated and run successfully
- [ ] Documentation updated (README.md, CLAUDE.md)
- [ ] No performance regression (run benchmark.py)
- [ ] Test coverage maintained at 85%+

### Manual Validation

```bash
# Run full test suite
pytest tests/ -v

# Run only camera tests
pytest tests/test_camera/ -v --tb=short

# Run with coverage
pytest --cov=pyvr.camera --cov-report=term-missing tests/test_camera/

# Test examples
python example/ModernglRender/multiview_example.py
python example/ModernglRender/enhanced_camera_demo.py

# Performance check
python example/benchmark.py
```

## Migration Guide for Examples

### Update Examples

```python
# Old code (v0.2.2)
from pyvr.camera import CameraParameters
camera = CameraParameters.isometric_view(distance=5.0)

# New code (v0.2.3)
from pyvr.camera import Camera
camera = Camera.isometric_view(distance=5.0)

# Use with renderer
renderer = VolumeRenderer(width=512, height=512, camera=camera)

# Or set later
renderer.set_camera(camera)

# Access matrix methods
view_matrix = camera.get_view_matrix()
proj_matrix = camera.get_projection_matrix(aspect_ratio=16/9)
```

## Benefits Achieved

1. ✅ **Pipeline Alignment**: Camera owns transformations (Geometry Stage)
2. ✅ **Clean Architecture**: No backward compatibility burden
3. ✅ **Better Testability**: Matrix creation testable without OpenGL
4. ✅ **Extensibility**: Easy to add orthographic, fisheye projections
5. ✅ **Clear Interface**: Camera returns matrices, doesn't apply them
6. ✅ **Comprehensive Tests**: All functionality verified

## Timeline

- **Implementation**: 1-2 days
- **Testing**: 0.5-1 day
- **Documentation & Examples**: 0.5 day
- **Total**: 2-3 days

## Next Phase

After Phase 1 completion, proceed to **Phase 2: Light Refactoring (v0.2.4)**
