# Bug Analysis: Light Linking Doesn't Follow Camera During Rotation

**Date**: 2025-11-14
**Severity**: HIGH (feature completely broken)
**Status**: Root cause identified
**Impact**: Light linking unusable - light doesn't follow camera rotation

## Summary

When light linking is enabled ('l' key), the light does NOT properly follow the camera during rotation. The camera and light use **incompatible coordinate systems**, causing them to diverge when the camera rotates.

## Bug Reproduction

1. Launch interactive interface
2. Press 'l' to enable light linking
3. Drag mouse to rotate camera around scene
4. **BUG**: Light position doesn't match camera-relative position
5. Result: Lighting appears to change as you rotate (it shouldn't)

## Root Cause: Coordinate System Mismatch

### Camera Position Calculation (Quaternion-Based)

**File**: `pyvr/camera/control.py:get_camera_pos()`

```python
# Camera uses complex quaternion-based rotation
# Start from init_pos and init_up vectors
rot_azimuth = R.from_rotvec(azimuth * init_up)
rot_elevation = R.from_rotvec(elevation * elev_axis)
rot_roll = R.from_rotvec(roll * view_dir)

rot = rot_azimuth * rot_elevation * rot_roll
position = rot.apply(rel_init_pos) + target
```

**Default init vectors**:
- `init_pos = [1.0, 0.0, 0.0]` (pointing along +X axis)
- `init_up = [0.0, 0.0, 1.0]` (Z-up)

**Result**: Camera position calculated via quaternion rotation of init_pos

### Light Position Calculation (Simple Spherical)

**File**: `pyvr/lighting/light.py:update_from_camera()`

```python
# Light uses simple spherical coordinates
azimuth = camera.azimuth + self._camera_offsets['azimuth']
elevation = camera.elevation + self._camera_offsets['elevation']
distance = camera.distance + self._camera_offsets['distance']

# Standard spherical coordinate formula
cos_elev = np.cos(elevation)
x = distance * cos_elev * np.cos(azimuth)
y = distance * np.sin(elevation)
z = distance * cos_elev * np.sin(azimuth)

self.position = camera.target + np.array([x, y, z], dtype=np.float32)
```

**Result**: Light position calculated using standard spherical coordinates

### The Mismatch

**Camera**: `position = quaternion_rotation(init_pos) + target`
**Light**: `position = spherical_coords(azimuth, elevation, distance) + target`

**These are NOT the same!** They only match when:
- `init_pos` aligns with spherical coordinate axes
- No roll is applied
- Specific combinations of azimuth/elevation

## Evidence

### Test Results

Created `tmp_dev/test_light_rotation.py` which shows:

**Initial State** (azimuth=45°, elevation=30°):
- Camera position: `[1.837, 1.837, 1.500]`
- Light position: `[1.837, 1.500, 1.837]`
- **Difference**: Y and Z swapped! (1.837 vs 1.500)

**After Rotation** (azimuth=225°, elevation=30°):
- Camera position: `[-1.837, -1.837, 1.500]`
- Light position: `[-1.837, 1.500, -1.837]`
- **Difference**: Camera Y is negative, Light Y is positive!

**Vector from Camera to Light**: `[0, 3.337, -3.337]`
**Magnitude**: 4.719 units

The camera and light are **4.7 units apart** even though they should be at the same position (with offset=0)!

### Why Sometimes It Appears to Work

For certain camera orientations (like front view with azimuth=0, elevation=0), the quaternion and spherical calculations happen to produce similar results, hiding the bug.

## Impact

### User Experience

1. **Inconsistent Lighting**: As user rotates camera, lighting direction changes unexpectedly
2. **Confusing Behavior**: Light "follows" but not in the expected way
3. **Feature Unusable**: Light linking doesn't achieve its intended purpose
4. **No Error Messages**: Silent failure - looks like it works but doesn't

### Technical Impact

- Light linking feature completely broken
- Camera-linked lighting (v0.3.1 feature) doesn't work as designed
- Any code relying on light.update_from_camera() produces wrong results

## Solution

### Option 1: Use Camera's Actual Position (RECOMMENDED)

Instead of recalculating light position from spherical coordinates, use the camera's actual calculated position with an offset.

**Implementation**:
```python
# In light.py update_from_camera()
def update_from_camera(self, camera) -> None:
    if not self._is_linked:
        raise ValueError("Light is not linked to camera. Call link_to_camera() first.")

    # Get camera's ACTUAL position (quaternion-calculated)
    camera_pos, camera_up = camera.get_camera_vectors()

    # Calculate offset direction in camera's local space
    # For now, support simple distance offset along camera-to-target vector
    view_direction = camera.target - camera_pos
    view_direction = view_direction / np.linalg.norm(view_direction)

    # Apply distance offset
    distance_offset = self._camera_offsets.get('distance', 0.0)
    offset_vector = view_direction * distance_offset

    # Light position = camera position + offset
    self.position = camera_pos + offset_vector
    self.target = camera.target.copy()
```

**Pros**:
- Uses camera's actual position (correct by definition)
- No coordinate system mismatch
- Simple implementation
- Light truly follows camera

**Cons**:
- Changes how azimuth/elevation offsets work
- Need to redefine offset semantics (currently not useful with this approach)

### Option 2: Make Camera Use Spherical Coordinates

Change camera to use simple spherical coordinates like the light.

**Pros**:
- Camera and light would match
- Simpler coordinate system

**Cons**:
- **BREAKING**: Major change to camera system
- Loses quaternion benefits (gimbal lock avoidance)
- Affects all camera code
- Would break existing camera behavior
- **NOT RECOMMENDED**

### Option 3: Convert Light to Use Quaternion System

Make light use the same quaternion rotation system as camera.

**Pros**:
- Matches camera's coordinate system
- Could support complex offsets

**Cons**:
- Complex implementation
- Light needs init_pos, init_up concepts
- Adds unnecessary complexity to light
- **NOT RECOMMENDED**

### Option 4: Headlight Mode (Simplified Option 1)

Make light always at camera position, pointing at target.

**Implementation**:
```python
def update_from_camera(self, camera) -> None:
    # Light at camera position
    self.position, _ = camera.get_camera_vectors()
    self.target = camera.target.copy()
```

**Pros**:
- **SIMPLEST** solution
- Truly "follows camera"
- No offset calculation needed
- Matches user expectation of "headlight"

**Cons**:
- Removes offset capability
- Changes feature semantics

## Recommended Solution

**Use Option 4 (Headlight Mode) - Simplest and Most Intuitive**

**Rationale**:
1. **User Expectation**: When user says "light follows camera", they expect headlight behavior
2. **Simplicity**: Dead simple implementation, no coordinate system math
3. **Correct**: Always works, no edge cases
4. **Useful**: Headlight is a common, useful lighting mode
5. **Offsets**: If offsets are needed later, can add them in camera's local space

**Implementation**:
```python
# File: pyvr/lighting/light.py
# Method: update_from_camera()

def update_from_camera(self, camera) -> None:
    """
    Update light position to follow camera.

    Light is positioned at camera location, pointing at target (headlight mode).

    Args:
        camera: Camera instance to follow
    """
    if not self._is_linked:
        raise ValueError("Light is not linked to camera. Call link_to_camera() first.")

    # Import Camera here to avoid circular dependency
    from pyvr.camera import Camera
    if not isinstance(camera, Camera):
        raise ValueError("camera must be a Camera instance")

    # Position light at camera position (headlight mode)
    camera_pos, _ = camera.get_camera_vectors()
    self.position = camera_pos.copy()
    self.target = camera.target.copy()
```

**Changes needed**:
- Modify `Light.update_from_camera()` (1 method, ~10 lines)
- Remove offset calculation (simplify)
- Offsets stored in `_camera_offsets` can be deprecated or ignored

**Migration**:
- Existing code continues to work
- Light linking behavior changes to headlight mode
- No API changes (method signatures unchanged)

## Testing

### Unit Tests

```python
def test_light_follows_camera_position():
    """Test that light is at camera position when linked."""
    camera = Camera.isometric_view(distance=3.0)
    light = Light.directional([1, -1, 0])
    light.link_to_camera()
    light.update_from_camera(camera)

    camera_pos, _ = camera.get_camera_vectors()
    assert np.allclose(light.position, camera_pos)
    assert np.allclose(light.target, camera.target)

def test_light_follows_camera_rotation():
    """Test that light updates when camera rotates."""
    camera = Camera.isometric_view(distance=3.0)
    light = Light.directional([1, -1, 0])
    light.link_to_camera()
    light.update_from_camera(camera)

    initial_pos = light.position.copy()

    # Rotate camera
    camera.azimuth += np.pi
    light.update_from_camera(camera)

    # Light position should change
    assert not np.allclose(light.position, initial_pos)

    # Light should still be at camera position
    camera_pos, _ = camera.get_camera_vectors()
    assert np.allclose(light.position, camera_pos)
```

### Integration Test

```python
def test_interactive_interface_light_linking():
    """Test light linking in interactive interface."""
    volume = Volume(data=create_sample_volume(64, 'sphere'))
    interface = InteractiveVolumeRenderer(volume=volume)

    # Enable light linking
    interface.set_camera_linked_lighting()

    # Get initial state
    camera = interface.camera_controller.params
    light = interface.renderer.get_light()
    camera_pos_initial, _ = camera.get_camera_vectors()

    # Light should be at camera position
    assert np.allclose(light.position, camera_pos_initial)

    # Rotate camera
    interface.camera_controller.orbit(delta_azimuth=np.pi/2, delta_elevation=0)
    interface._update_display(force_render=True)

    # Get updated state
    camera_pos_rotated, _ = camera.get_camera_vectors()
    light = interface.renderer.get_light()

    # Light should follow camera
    assert np.allclose(light.position, camera_pos_rotated)
```

## New Patch Assessment

**YES - This warrants a new patch (v0.3.4 or part of v0.3.3)**

**Severity**: HIGH
- Feature completely broken
- Affects main v0.3.1 feature (camera-linked lighting)
- User-visible bug with no workaround

**Effort**: LOW
- Simple fix (headlight mode)
- ~10 line change in one method
- Few tests to add

**Impact**: HIGH POSITIVE
- Makes light linking actually work
- Simpler, more intuitive behavior
- Fixes user-reported bug

**Recommendation**: Include in v0.3.3 as second bug fix
- v0.3.3 already has threading fix
- Both are critical interface bugs
- Ship together as "v0.3.3: Bug Fixes"

## Files to Modify

**Core Fix**:
- `pyvr/lighting/light.py` - Modify `update_from_camera()` method

**Tests**:
- `tests/test_lighting/test_light_linking.py` - Update existing tests
- Add new test for camera position matching

**Documentation**:
- `version_notes/v0.3.3_ray_marching_consistency.md` - Add third bug fix section

**Estimated Time**: 30 minutes

## Related Bugs

This fix also resolves the "hardcoded center" bug partially:
- Camera should be initialized to `target=volume.center` (separate fix)
- Light will automatically point to correct target once camera is fixed

## References

- Light class: `pyvr/lighting/light.py`
- Camera class: `pyvr/camera/camera.py`
- Camera control: `pyvr/camera/control.py:get_camera_pos()`
- Test script: `tmp_dev/test_light_rotation.py`
- Reproduction: User manual testing with 'l' key
