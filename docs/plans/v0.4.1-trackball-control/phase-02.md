# Phase 02: CameraController.trackball() Method

## Scope

Implement the `trackball()` method in the `CameraController` class using the helper functions from Phase 01. This method enables intuitive 3D camera rotation based on 2D mouse movement, converting pixel deltas into smooth quaternion-based rotations.

**What will be built:**
- `CameraController.trackball()` method with input validation
- Unit tests for trackball behavior (directional movement, sensitivity, edge cases)
- Error handling for invalid inputs
- Integration with existing `CameraController` methods

**Dependencies:**
- Phase 01 helpers: `_map_to_sphere()`, `_camera_to_quaternion()`, `_quaternion_to_camera_angles()`
- Existing `CameraController` class and `params` attribute
- scipy.spatial.transform.Rotation

**Out of scope:**
- Interface integration (Phase 03)
- Mode switching UI (Phase 03)
- Documentation updates (Phase 04)

## Implementation

### Task 1: Implement `CameraController.trackball()` method

**File:** `pyvr/camera/control.py`

**Location:** Add to `CameraController` class, after the `zoom()` method (around line 320)

**Implementation:**
```python
def trackball(
    self,
    dx: float,
    dy: float,
    viewport_width: int,
    viewport_height: int,
    sensitivity: float = 1.0
) -> None:
    """
    Rotate camera using trackball/arcball control.

    Provides intuitive 3D rotation following mouse movement,
    like rotating a physical ball. Mouse movement is mapped to
    rotation on a virtual sphere centered on the target.

    Args:
        dx: Mouse delta in pixels (horizontal, right is positive)
        dy: Mouse delta in pixels (vertical, down is positive)
        viewport_width: Viewport width in pixels
        viewport_height: Viewport height in pixels
        sensitivity: Rotation sensitivity multiplier (default: 1.0)
                    Higher values = more rotation per pixel

    Raises:
        ValueError: If viewport dimensions are invalid (<= 0)

    Example:
        >>> controller = CameraController()
        >>> # User dragged mouse 50 pixels right, 30 pixels down
        >>> controller.trackball(
        ...     dx=50, dy=-30,
        ...     viewport_width=800, viewport_height=600
        ... )
        >>> # Camera has rotated smoothly

    Notes:
        - Uses quaternion-based arcball algorithm for smooth rotation
        - No gimbal lock artifacts
        - Movement is relative to current camera orientation
        - Small movements (< 0.001 normalized) are ignored for stability
    """
    # Validate viewport dimensions
    if viewport_width <= 0 or viewport_height <= 0:
        raise ValueError(
            f"Viewport dimensions must be positive, got width={viewport_width}, height={viewport_height}"
        )

    # Early exit for zero movement
    if dx == 0 and dy == 0:
        return

    # Normalize pixel deltas to [-1, 1] range
    # Use the smaller dimension for uniform scaling
    scale = min(viewport_width, viewport_height)
    dx_norm = (dx / scale) * sensitivity
    dy_norm = (dy / scale) * sensitivity

    # Invert dy for intuitive movement (drag up = rotate up)
    dy_norm = -dy_norm

    # Early exit for very small movements (avoid numerical instability)
    if abs(dx_norm) < 0.001 and abs(dy_norm) < 0.001:
        return

    # Map start and end points to sphere
    start_point = _map_to_sphere(0.0, 0.0)
    end_point = _map_to_sphere(dx_norm, dy_norm)

    # Compute rotation axis and angle
    axis = np.cross(start_point, end_point)
    axis_length = np.linalg.norm(axis)

    # Check for parallel vectors (no rotation needed)
    if axis_length < 1e-8:
        return

    axis = axis / axis_length

    # Compute rotation angle
    dot_product = np.clip(np.dot(start_point, end_point), -1.0, 1.0)
    angle = np.arccos(dot_product)

    # Create trackball rotation quaternion
    trackball_rotation = R.from_rotvec(angle * axis)

    # Get current camera orientation as quaternion
    current_rotation = _camera_to_quaternion(self.params)

    # Apply trackball rotation to current orientation
    # Important: trackball rotation is in camera-local space
    new_rotation = trackball_rotation * current_rotation

    # Decompose back to spherical angles
    new_azimuth, new_elevation, new_roll = _quaternion_to_camera_angles(
        new_rotation,
        self.params.target,
        self.params.distance,
        self.params.init_pos,
        self.params.init_up
    )

    # Update camera parameters
    self.params.azimuth = new_azimuth
    self.params.elevation = new_elevation
    self.params.roll = new_roll
```

### Task 2: Add unit tests for trackball behavior

**File:** `tests/test_camera/test_trackball.py`

**Location:** Add new test class at end of file

**Implementation:**
```python
class TestCameraControllerTrackball:
    """Tests for CameraController.trackball() method."""

    def test_trackball_horizontal_right(self):
        """Dragging right should rotate camera left around target."""
        controller = CameraController(Camera.front_view(distance=3.0))
        initial_azimuth = controller.params.azimuth

        # Drag right
        controller.trackball(
            dx=100, dy=0,
            viewport_width=800, viewport_height=600
        )

        # Azimuth should decrease (camera rotates left)
        assert controller.params.azimuth < initial_azimuth

    def test_trackball_horizontal_left(self):
        """Dragging left should rotate camera right around target."""
        controller = CameraController(Camera.front_view(distance=3.0))
        initial_azimuth = controller.params.azimuth

        # Drag left
        controller.trackball(
            dx=-100, dy=0,
            viewport_width=800, viewport_height=600
        )

        # Azimuth should increase (camera rotates right)
        assert controller.params.azimuth > initial_azimuth

    def test_trackball_vertical_up(self):
        """Dragging up should rotate camera up (increase elevation)."""
        controller = CameraController(Camera.front_view(distance=3.0))
        initial_elevation = controller.params.elevation

        # Drag up (negative dy in screen coords)
        controller.trackball(
            dx=0, dy=-100,
            viewport_width=800, viewport_height=600
        )

        # Elevation should increase (camera looks up)
        assert controller.params.elevation > initial_elevation

    def test_trackball_vertical_down(self):
        """Dragging down should rotate camera down (decrease elevation)."""
        controller = CameraController(Camera.front_view(distance=3.0))
        initial_elevation = controller.params.elevation

        # Drag down (positive dy in screen coords)
        controller.trackball(
            dx=0, dy=100,
            viewport_width=800, viewport_height=600
        )

        # Elevation should decrease (camera looks down)
        assert controller.params.elevation < initial_elevation

    def test_trackball_diagonal(self):
        """Diagonal drag should combine horizontal and vertical rotation."""
        controller = CameraController(Camera.front_view(distance=3.0))
        initial_azimuth = controller.params.azimuth
        initial_elevation = controller.params.elevation

        # Drag diagonally (right and up)
        controller.trackball(
            dx=100, dy=-100,
            viewport_width=800, viewport_height=600
        )

        # Both angles should change
        assert controller.params.azimuth != initial_azimuth
        assert controller.params.elevation != initial_elevation

    def test_trackball_zero_movement(self):
        """Zero mouse movement should not change camera."""
        controller = CameraController(Camera.isometric_view(distance=3.0))
        initial_azimuth = controller.params.azimuth
        initial_elevation = controller.params.elevation
        initial_roll = controller.params.roll

        # No movement
        controller.trackball(
            dx=0, dy=0,
            viewport_width=800, viewport_height=600
        )

        # Camera should be unchanged
        assert controller.params.azimuth == initial_azimuth
        assert controller.params.elevation == initial_elevation
        assert controller.params.roll == initial_roll

    def test_trackball_very_small_movement(self):
        """Very small movements should be ignored for stability."""
        controller = CameraController(Camera.isometric_view(distance=3.0))
        initial_azimuth = controller.params.azimuth
        initial_elevation = controller.params.elevation

        # Very small movement (< 0.001 normalized)
        controller.trackball(
            dx=0.1, dy=0.1,
            viewport_width=800, viewport_height=600
        )

        # Camera should be unchanged (below threshold)
        assert controller.params.azimuth == initial_azimuth
        assert controller.params.elevation == initial_elevation

    def test_trackball_sensitivity_scaling(self):
        """Higher sensitivity should produce larger rotation."""
        # Test with sensitivity=1.0
        controller1 = CameraController(Camera.front_view(distance=3.0))
        controller1.trackball(
            dx=50, dy=0,
            viewport_width=800, viewport_height=600,
            sensitivity=1.0
        )
        rotation1 = abs(controller1.params.azimuth)

        # Test with sensitivity=2.0
        controller2 = CameraController(Camera.front_view(distance=3.0))
        controller2.trackball(
            dx=50, dy=0,
            viewport_width=800, viewport_height=600,
            sensitivity=2.0
        )
        rotation2 = abs(controller2.params.azimuth)

        # Higher sensitivity should produce more rotation
        assert rotation2 > rotation1
        # Should be approximately double
        assert abs(rotation2 / rotation1 - 2.0) < 0.5

    def test_trackball_invalid_viewport_width(self):
        """Invalid viewport width should raise ValueError."""
        controller = CameraController(Camera.front_view(distance=3.0))

        with pytest.raises(ValueError, match="Viewport dimensions must be positive"):
            controller.trackball(
                dx=10, dy=10,
                viewport_width=0, viewport_height=600
            )

    def test_trackball_invalid_viewport_height(self):
        """Invalid viewport height should raise ValueError."""
        controller = CameraController(Camera.front_view(distance=3.0))

        with pytest.raises(ValueError, match="Viewport dimensions must be positive"):
            controller.trackball(
                dx=10, dy=10,
                viewport_width=800, viewport_height=-100
            )

    def test_trackball_preserves_distance(self):
        """Trackball rotation should not change distance to target."""
        controller = CameraController(Camera.isometric_view(distance=5.0))
        initial_distance = controller.params.distance

        # Apply several trackball movements
        controller.trackball(dx=50, dy=30, viewport_width=800, viewport_height=600)
        controller.trackball(dx=-30, dy=50, viewport_width=800, viewport_height=600)

        # Distance should be unchanged
        assert abs(controller.params.distance - initial_distance) < 1e-6

    def test_trackball_preserves_target(self):
        """Trackball rotation should not change target position."""
        target = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        camera = Camera.isometric_view(target=target, distance=5.0)
        controller = CameraController(camera)

        # Apply trackball movement
        controller.trackball(dx=50, dy=30, viewport_width=800, viewport_height=600)

        # Target should be unchanged
        np.testing.assert_allclose(controller.params.target, target)

    def test_trackball_continuous_rotation(self):
        """Multiple small trackball movements should accumulate smoothly."""
        controller = CameraController(Camera.front_view(distance=3.0))
        initial_azimuth = controller.params.azimuth

        # Apply 5 small movements
        for _ in range(5):
            controller.trackball(
                dx=20, dy=0,
                viewport_width=800, viewport_height=600
            )

        # Total rotation should be accumulated
        total_rotation = abs(controller.params.azimuth - initial_azimuth)
        assert total_rotation > 0.1  # Significant rotation

    def test_trackball_different_viewport_sizes(self):
        """Trackball should work with different viewport aspect ratios."""
        # Wide viewport
        controller1 = CameraController(Camera.front_view(distance=3.0))
        controller1.trackball(dx=50, dy=0, viewport_width=1600, viewport_height=600)

        # Tall viewport
        controller2 = CameraController(Camera.front_view(distance=3.0))
        controller2.trackball(dx=50, dy=0, viewport_width=600, viewport_height=1600)

        # Both should produce rotation (exact values may differ due to scaling)
        assert abs(controller1.params.azimuth) > 0
        assert abs(controller2.params.azimuth) > 0

    def test_trackball_from_different_orientations(self):
        """Trackball should work correctly from various starting orientations."""
        # Test from side view
        controller = CameraController(Camera.side_view(distance=3.0))
        initial_az = controller.params.azimuth
        controller.trackball(dx=50, dy=0, viewport_width=800, viewport_height=600)
        assert controller.params.azimuth != initial_az

        # Test from top view
        controller = CameraController(Camera.top_view(distance=3.0))
        initial_el = controller.params.elevation
        controller.trackball(dx=0, dy=50, viewport_width=800, viewport_height=600)
        assert controller.params.elevation != initial_el
```

## Verification

### Unit Test Execution

Run all trackball tests including new CameraController tests:

```bash
pytest tests/test_camera/test_trackball.py::TestCameraControllerTrackball -v
```

**Expected output:**
- All 16+ tests pass
- No warnings or errors
- Execution time < 5 seconds

### Interactive Testing

**Test basic trackball movement:**
```python
from pyvr.camera import Camera
from pyvr.camera.control import CameraController
import numpy as np

# Create controller
controller = CameraController(Camera.front_view(distance=3.0))
print(f"Initial: az={controller.params.azimuth:.4f}, el={controller.params.elevation:.4f}")

# Simulate drag right
controller.trackball(dx=100, dy=0, viewport_width=800, viewport_height=600)
print(f"After drag right: az={controller.params.azimuth:.4f}, el={controller.params.elevation:.4f}")

# Simulate drag up
controller.trackball(dx=0, dy=-100, viewport_width=800, viewport_height=600)
print(f"After drag up: az={controller.params.azimuth:.4f}, el={controller.params.elevation:.4f}")
```

**Test sensitivity:**
```python
# Low sensitivity
controller1 = CameraController(Camera.front_view(distance=3.0))
controller1.trackball(dx=50, dy=0, viewport_width=800, viewport_height=600, sensitivity=0.5)
print(f"Low sensitivity (0.5): rotation={abs(controller1.params.azimuth):.4f}")

# High sensitivity
controller2 = CameraController(Camera.front_view(distance=3.0))
controller2.trackball(dx=50, dy=0, viewport_width=800, viewport_height=600, sensitivity=2.0)
print(f"High sensitivity (2.0): rotation={abs(controller2.params.azimuth):.4f}")
```

**Test edge cases:**
```python
# Zero movement
controller = CameraController(Camera.front_view(distance=3.0))
initial = controller.params.azimuth
controller.trackball(dx=0, dy=0, viewport_width=800, viewport_height=600)
assert controller.params.azimuth == initial
print("✓ Zero movement handled correctly")

# Very small movement
controller = CameraController(Camera.front_view(distance=3.0))
initial = controller.params.azimuth
controller.trackball(dx=0.1, dy=0.1, viewport_width=800, viewport_height=600)
assert controller.params.azimuth == initial
print("✓ Small movement threshold works")

# Invalid viewport
try:
    controller.trackball(dx=10, dy=10, viewport_width=0, viewport_height=600)
    print("✗ Should have raised ValueError")
except ValueError as e:
    print(f"✓ ValueError raised: {e}")
```

## Validation

### Behavior Verification

**Rotation direction consistency:**
- Drag right → camera moves left (azimuth decreases)
- Drag left → camera moves right (azimuth increases)
- Drag up → camera looks up (elevation increases)
- Drag down → camera looks down (elevation decreases)

**Invariant preservation:**
- Target position never changes
- Distance to target never changes
- Camera always produces valid view matrix

**Numerical stability:**
- No NaN or Inf values
- No crashes on edge cases
- Smooth behavior with small movements

### Performance Check

```python
import time
from pyvr.camera import Camera
from pyvr.camera.control import CameraController

controller = CameraController(Camera.front_view(distance=3.0))

# Time 1000 trackball calls
start = time.time()
for i in range(1000):
    controller.trackball(
        dx=float(i % 100),
        dy=float(i % 50),
        viewport_width=800,
        viewport_height=600
    )
elapsed = time.time() - start

print(f"1000 trackball calls: {elapsed:.3f}s ({1000/elapsed:.0f} ops/sec)")
# Should be < 1 second (>1000 ops/sec)
```

### Integration with Existing Methods

**Verify trackball works alongside existing controller methods:**
```python
from pyvr.camera import Camera
from pyvr.camera.control import CameraController

controller = CameraController(Camera.isometric_view(distance=3.0))

# Use trackball
controller.trackball(dx=50, dy=30, viewport_width=800, viewport_height=600)
print(f"After trackball: az={controller.params.azimuth:.4f}")

# Use orbit
controller.orbit(delta_azimuth=0.5, delta_elevation=0.2)
print(f"After orbit: az={controller.params.azimuth:.4f}")

# Use zoom
controller.zoom(factor=1.5)
print(f"After zoom: distance={controller.params.distance:.4f}")

# All should work together without issues
```

## Acceptance Criteria

- [ ] `CameraController.trackball()` method implemented
- [ ] All unit tests pass (16+ tests, 100% pass rate)
- [ ] Rotation directions intuitive and consistent
- [ ] Sensitivity parameter works correctly
- [ ] Zero and very small movements handled properly
- [ ] Invalid viewport dimensions raise ValueError
- [ ] Distance and target preserved during rotation
- [ ] No NaN or Inf values produced
- [ ] No regression in existing controller methods (orbit, zoom, pan, etc.)
- [ ] Performance: >1000 trackball ops/sec
- [ ] Docstring complete with examples and notes
- [ ] Type hints present

## Git Commit

**Commit message:**
```
feat: Add trackball() method to CameraController

Implement intuitive trackball/arcball camera control using quaternions:
- Natural 3D rotation following mouse movement
- Sensitivity parameter for adjustable rotation speed
- Preserves distance and target during rotation
- No gimbal lock artifacts

Features:
- Maps 2D mouse deltas to 3D sphere rotation
- Uses quaternion composition for smooth rotation
- Threshold for very small movements (stability)
- Input validation for viewport dimensions
- Works with any viewport aspect ratio

Tests:
- 16 unit tests covering directional movement, sensitivity, edge cases
- Tests for zero movement, small movements, invalid inputs
- Verification of invariants (distance, target preservation)
- Performance testing (>1000 ops/sec)

Integration:
- Works alongside existing orbit(), zoom(), pan() methods
- No breaking changes to CameraController API
- Uses Phase 01 helpers for quaternion conversion

Phase 02 of trackball control implementation (v0.4.1).

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

**Files to commit:**
```
pyvr/camera/control.py (modified - add trackball() method to CameraController)
tests/test_camera/test_trackball.py (modified - add TestCameraControllerTrackball class)
```

**Pre-commit checklist:**
- [ ] All tests pass: `pytest tests/test_camera/test_trackball.py -v`
- [ ] No linting errors: `black pyvr/camera/control.py tests/test_camera/test_trackball.py`
- [ ] Type hints verified
- [ ] Docstring complete
- [ ] No regression: `pytest tests/test_camera/ -v`
