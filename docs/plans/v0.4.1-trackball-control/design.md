# PyVR v0.4.1: Trackball Control Design

## Overview

Add trackball (arcball) camera control to PyVR as an alternative to the current orbit control. Trackball provides more intuitive, natural 3D rotation that follows the mouse cursor like rotating a physical ball.

**Key Changes:**
- Add trackball control methods to `CameraController` in `pyvr/camera/control.py`
- Refactor interface to use controller methods exclusively (no direct angle computation)
- Make trackball the default for `InteractiveVolumeRenderer`
- Keep both orbit and trackball available via API

**Breaking Changes:**
- Default mouse control in interface changes from orbit to trackball
- Mitigation: Both modes remain available, can toggle or use programmatically

## Architecture

### Current State

**InteractiveVolumeRenderer** (matplotlib_interface.py):
- Computes camera angles directly from mouse deltas
- Calls `camera_controller.orbit(delta_azimuth, delta_elevation)`
- Has orbit-specific sensitivity calculations in interface code

**CameraController** (control.py):
- Provides `orbit(delta_azimuth, delta_elevation)` method
- Provides `zoom(factor)` method
- Stores `params: Camera` with spherical coordinates

### Target Architecture

```
InteractiveVolumeRenderer (interface)
    ↓ (calls methods with mouse deltas)
CameraController (control.py)
    ├── orbit(delta_azimuth, delta_elevation)  [existing]
    ├── trackball(dx, dy, width, height)       [new]
    ├── zoom(factor)                           [existing]
    └── params: Camera                         [existing]
```

**Design Principles:**
- **Single controller:** Add trackball to existing `CameraController` (not separate class)
- **Both modes available:** orbit() and trackball() coexist
- **Clean separation:** All camera manipulation logic in control.py
- **Interface simplicity:** Interface just passes mouse deltas, no angle computation

## Trackball Algorithm

### Arcball Method

The trackball algorithm maps 2D mouse movement to 3D rotation on a virtual sphere:

1. **Normalize coordinates:** Convert pixel deltas to normalized [-1, 1] range
2. **Map to sphere:** Project 2D screen points to 3D sphere surface
3. **Compute rotation:**
   - Rotation axis = cross product of start/end vectors
   - Rotation angle = angle between vectors
4. **Apply rotation:** Use quaternion rotation to update camera
5. **Update params:** Decompose back to spherical angles

### Quaternion-Based Implementation

**Why quaternions:**
- Smooth, continuous rotations without gimbal lock
- Natural composition of rotations
- Camera already uses quaternions internally (via scipy)

**Conversion strategy:**
1. Get current camera orientation → quaternion
2. Apply trackball rotation → new quaternion
3. Decompose new quaternion → azimuth/elevation/roll
4. Update `self.params` with new angles

## Implementation Details

### New Helper Functions (control.py)

```python
def _map_to_sphere(x: float, y: float, radius: float = 1.0) -> np.ndarray:
    """
    Map 2D point to 3D point on virtual sphere (arcball algorithm).

    Args:
        x: Normalized x coordinate [-1, 1]
        y: Normalized y coordinate [-1, 1]
        radius: Virtual sphere radius

    Returns:
        3D point on sphere surface or hyperbolic sheet

    Notes:
        - Points inside sphere: project to sphere (x² + y² + z² = r²)
        - Points outside: use hyperbolic sheet for smooth behavior
        - This prevents discontinuities at sphere edge
    """
```

```python
def _camera_to_quaternion(camera: Camera) -> R:
    """
    Convert camera spherical coords to orientation quaternion.

    Args:
        camera: Camera with spherical coordinates

    Returns:
        scipy Rotation representing camera orientation

    Notes:
        Uses get_camera_pos to get actual camera vectors,
        then builds rotation matrix from those vectors.
    """
```

```python
def _quaternion_to_camera_angles(
    rotation: R,
    target: np.ndarray,
    distance: float,
    init_pos: np.ndarray,
    init_up: np.ndarray
) -> Tuple[float, float, float]:
    """
    Decompose quaternion rotation to azimuth/elevation/roll angles.

    Args:
        rotation: Scipy Rotation quaternion
        target: Camera target point
        distance: Camera distance from target
        init_pos: Initial camera position (for reference frame)
        init_up: Initial up vector (for reference frame)

    Returns:
        Tuple of (azimuth, elevation, roll) in radians

    Notes:
        This is the inverse operation of camera orientation calculation.
        Handles gimbal lock cases (elevation = ±π/2) gracefully.
    """
```

### CameraController.trackball() Method

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
    like rotating a physical ball.

    Args:
        dx: Mouse delta in pixels (horizontal)
        dy: Mouse delta in pixels (vertical)
        viewport_width: Viewport width in pixels
        viewport_height: Viewport height in pixels
        sensitivity: Rotation sensitivity multiplier (default: 1.0)

    Example:
        >>> controller = CameraController()
        >>> # User dragged mouse 50 pixels right, 30 pixels down
        >>> controller.trackball(
        ...     dx=50, dy=-30,
        ...     viewport_width=800, viewport_height=600
        ... )
    """
```

**Implementation flow:**

1. **Early exit:** If dx=0 and dy=0, return immediately
2. **Normalize:** Convert pixel deltas to [-1, 1] range
3. **Map points:**
   - Start: (0, 0) → sphere point
   - End: (dx_norm, dy_norm) → sphere point
4. **Compute rotation:**
   - Axis = cross(start, end)
   - Angle = arccos(dot(start, end))
   - Create quaternion from axis-angle
5. **Get current orientation:** `_camera_to_quaternion(self.params)`
6. **Apply rotation:** `new_rot = trackball_rot * current_rot`
7. **Decompose:** Get new (azimuth, elevation, roll)
8. **Update:** `self.params.azimuth/elevation/roll = new values`

### Interface Integration

**InterfaceState changes (state.py):**

```python
camera_control_mode: str = 'trackball'  # or 'orbit'
```

**InteractiveVolumeRenderer changes (matplotlib_interface.py):**

**Before (lines 415-442):**
```python
if self.state.is_dragging_camera:
    dx = event.xdata - self.state.drag_start_pos[0]
    dy = event.ydata - self.state.drag_start_pos[1]

    # Compute angles in interface code
    sensitivity = 0.005
    delta_azimuth = -dx * sensitivity
    delta_elevation = dy * sensitivity

    # Call controller
    self.camera_controller.orbit(
        delta_azimuth=delta_azimuth,
        delta_elevation=delta_elevation
    )
```

**After:**
```python
if self.state.is_dragging_camera:
    dx = event.xdata - self.state.drag_start_pos[0]
    dy = event.ydata - self.state.drag_start_pos[1]

    # Delegate to controller based on mode
    if self.state.camera_control_mode == 'trackball':
        self.camera_controller.trackball(
            dx=dx,
            dy=dy,
            viewport_width=self.width,
            viewport_height=self.height,
            sensitivity=1.0
        )
    else:  # orbit mode
        sensitivity = 0.005
        delta_azimuth = -dx * sensitivity
        delta_elevation = dy * sensitivity
        self.camera_controller.orbit(
            delta_azimuth=delta_azimuth,
            delta_elevation=delta_elevation
        )
```

**Keyboard shortcuts (add to _on_key_press):**

```python
elif event.key == 't':
    # Toggle camera control mode
    if self.state.camera_control_mode == 'trackball':
        self.state.camera_control_mode = 'orbit'
        print("Switched to orbit control")
    else:
        self.state.camera_control_mode = 'trackball'
        print("Switched to trackball control")
    self._update_status_display()
```

**Info display update:**

Add current control mode to status text:
```python
f"  Control Mode: {self.state.camera_control_mode.capitalize()}\n"
```

## Edge Cases & Error Handling

### Input Validation

1. **Zero movement:**
   - Condition: dx=0, dy=0
   - Handling: Early return, no-op

2. **Very small movements:**
   - Condition: |dx| < threshold and |dy| < threshold
   - Handling: Threshold check (e.g., < 0.001 normalized)
   - Prevents numerical instability

3. **Large drags:**
   - Condition: Very large dx/dy values
   - Handling: Clamping or normalization to prevent over-rotation

4. **Invalid viewport size:**
   - Condition: width <= 0 or height <= 0
   - Handling: ValueError with descriptive message

### Quaternion Decomposition Edge Cases

1. **Gimbal lock:**
   - Condition: elevation = ±π/2
   - Handling: Azimuth becomes undefined, use convention (azimuth=0)
   - Document in docstring

2. **Multiple valid solutions:**
   - Condition: Spherical coords wrap (azimuth += 2π)
   - Handling: Normalize to [-π, π] range for consistency

3. **Numerical precision:**
   - Use `np.clip` for arccos/arcsin to avoid domain errors
   - Handle near-zero cross products for parallel vectors

## Testing Strategy

### Unit Tests (test_camera/test_control.py)

**Helper function tests:**

```python
class TestMapToSphere:
    def test_origin_maps_to_front():
        # (0, 0) → (0, 0, 1)

    def test_point_inside_sphere():
        # (0.5, 0.5) → point on sphere surface
        # Verify: x² + y² + z² ≈ 1

    def test_point_outside_sphere():
        # (1.5, 1.5) → hyperbolic sheet

    def test_sphere_radius_scaling():
        # radius=2.0 vs radius=1.0

class TestCameraToQuaternion:
    def test_front_view_identity():
        # Front view → known quaternion

    def test_side_view():
        # Side view → 90° rotation

    def test_top_view():
        # Top view → known orientation

    def test_roundtrip_conversion():
        # camera → quat → angles → camera (close to original)

class TestQuaternionToCameraAngles:
    def test_identity_rotation():
        # Identity quaternion → (0, 0, 0)

    def test_gimbal_lock_elevation_90():
        # elevation = π/2 handled gracefully

    def test_gimbal_lock_elevation_neg90():
        # elevation = -π/2 handled gracefully

    def test_roundtrip_accuracy():
        # quat → angles → quat (match)
```

**CameraController.trackball() tests:**

```python
class TestCameraControllerTrackball:
    def test_trackball_horizontal_right():
        # Drag right → camera orbits left

    def test_trackball_horizontal_left():
        # Drag left → camera orbits right

    def test_trackball_vertical_up():
        # Drag up → camera moves down (inverse y)

    def test_trackball_vertical_down():
        # Drag down → camera moves up

    def test_trackball_diagonal():
        # Combined rotation

    def test_trackball_zero_movement():
        # dx=0, dy=0 is no-op

    def test_trackball_small_movements():
        # Very small dx/dy doesn't crash

    def test_trackball_sensitivity():
        # sensitivity=2.0 → 2x rotation

    def test_trackball_invalid_viewport():
        # width=0 raises ValueError
```

### Integration Tests (test_interface/)

```python
class TestInterfaceTrackballControl:
    def test_trackball_mode_drag(mock_interface):
        # Set trackball mode
        # Simulate mouse drag
        # Verify camera changed

    def test_orbit_mode_drag(mock_interface):
        # Set orbit mode
        # Simulate drag
        # Verify backward compatibility

    def test_toggle_control_mode(mock_interface):
        # Press 't' key
        # Verify mode switches

    def test_default_mode_is_trackball(mock_interface):
        # New interface → trackball mode
```

### Manual Testing Checklist

- [ ] Trackball feels natural and intuitive
- [ ] No gimbal lock artifacts during rotation
- [ ] Smooth continuous rotation
- [ ] Sensitivity feels right (not too fast/slow)
- [ ] Mode toggle works correctly
- [ ] Info display shows current mode
- [ ] Backward compatibility: orbit mode still works

## Implementation Phases

### Phase 1: Core trackball algorithm
**Files:** `pyvr/camera/control.py`, `tests/test_camera/test_control.py`

- Add `_map_to_sphere()` helper
- Add `_camera_to_quaternion()` helper
- Add `_quaternion_to_camera_angles()` helper
- Unit tests for all helpers
- Verify round-trip conversions

**Acceptance criteria:**
- All helper tests pass
- Round-trip accuracy within tolerance (< 1e-6 rad)
- Gimbal lock cases handled

### Phase 2: CameraController.trackball()
**Files:** `pyvr/camera/control.py`, `tests/test_camera/test_control.py`

- Implement `CameraController.trackball()` method
- Unit tests for trackball behavior
- Edge case handling (zero movement, large values)

**Acceptance criteria:**
- All trackball tests pass
- Camera params update correctly
- No crashes on edge cases

### Phase 3: Interface integration
**Files:** `pyvr/interface/state.py`, `pyvr/interface/matplotlib_interface.py`

- Add `camera_control_mode` to InterfaceState
- Update `_on_mouse_move()` to use trackball by default
- Keep orbit code path for mode switching
- Update info display

**Acceptance criteria:**
- Interface uses trackball by default
- Camera rotation feels natural
- No regression in zoom or other controls

### Phase 4: Polish & testing
**Files:** All, `docs/`, `version_notes/`

- Add 't' key toggle for modes
- Integration tests
- Sensitivity tuning
- Documentation updates
- Version notes

**Acceptance criteria:**
- All tests pass (unit + integration)
- Documentation complete
- Trackball feels polished and responsive

## Documentation Updates

### README.md

**Features section:**
```markdown
- **Trackball camera control** - Intuitive 3D rotation (default)
- **Orbit camera control** - Traditional spherical coordinate rotation (via toggle)
```

**Mouse controls section:**
```markdown
Mouse Controls (Trackball mode - default):
  - Drag: Rotate camera like a physical ball
  - Scroll: Zoom in/out
  - Toggle mode: Press 't' to switch to orbit control

Mouse Controls (Orbit mode):
  - Drag left/right: Rotate horizontally (azimuth)
  - Drag up/down: Rotate vertically (elevation)
  - Scroll: Zoom in/out
```

### CLAUDE.md

**API Usage section:**

```python
# Camera controller with trackball
controller = CameraController(camera)

# Trackball control (for interactive UIs)
controller.trackball(
    dx=50,  # Mouse moved 50 pixels right
    dy=-30,  # Mouse moved 30 pixels down
    viewport_width=800,
    viewport_height=600,
    sensitivity=1.0
)

# Orbit control (for scripted animations)
controller.orbit(
    delta_azimuth=np.pi/4,   # 45 degrees horizontal
    delta_elevation=np.pi/6   # 30 degrees vertical
)

# Both methods update controller.params
camera = controller.params
```

### version_notes/v0.4.1_trackball_control.md

Create detailed version notes:
- New features (trackball control)
- Breaking changes (default mode)
- Migration guide
- API additions

## Breaking Changes & Migration

### Breaking Changes

**Default camera control mode:**
- **Before:** Interface used orbit control (azimuth/elevation)
- **After:** Interface uses trackball control (arcball rotation)

**Impact:**
- Users familiar with orbit control will notice different rotation behavior
- Trackball is more intuitive for most users (industry standard)

### Migration Guide

**For users:**
- Press 't' to toggle back to orbit mode if preferred
- No code changes needed

**For developers:**
- `CameraController.orbit()` still available - no API changes
- New `CameraController.trackball()` method available
- Both modes supported indefinitely

### API Stability

**No breaking API changes:**
- All existing Camera methods unchanged
- All existing CameraController methods unchanged
- New methods are additions only

## Success Criteria

**Functional:**
- ✅ Trackball control implemented and working
- ✅ Both orbit and trackball available
- ✅ Interface defaults to trackball
- ✅ Mode toggle works
- ✅ All tests pass

**Quality:**
- ✅ Trackball feels natural and intuitive
- ✅ No gimbal lock or rotation artifacts
- ✅ Smooth, continuous rotation
- ✅ Good sensitivity (configurable)

**Documentation:**
- ✅ All new functions documented
- ✅ README updated
- ✅ CLAUDE.md updated
- ✅ Version notes created

**Testing:**
- ✅ >85% code coverage maintained
- ✅ Unit tests for all helpers
- ✅ Integration tests for interface
- ✅ Edge cases covered

## Future Enhancements (Out of Scope)

**Not included in v0.4.1:**
- Pan with trackball (requires different algorithm)
- Trackball with roll support (arcball variant)
- Touch/gesture support for trackball
- Custom sensitivity per axis
- Acceleration/momentum effects

These can be addressed in future versions based on user feedback.
