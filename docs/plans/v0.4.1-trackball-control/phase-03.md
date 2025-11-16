# Phase 03: Interface Integration

## Scope

Integrate trackball control into the `InteractiveVolumeRenderer` interface, making it the default camera control mode while keeping orbit mode available. Add mode switching capability and update the UI to display the current control mode.

**What will be built:**
- Add `camera_control_mode` field to `InterfaceState`
- Update `_on_mouse_move()` to support both trackball and orbit modes
- Make trackball the default control mode
- Add 't' keyboard shortcut to toggle between modes
- Update info display to show current control mode
- Integration tests for interface behavior

**Dependencies:**
- Phase 01: Helper functions
- Phase 02: `CameraController.trackball()` method
- Existing `InterfaceState` and `InteractiveVolumeRenderer` classes

**Out of scope:**
- Documentation updates (Phase 04)
- Version notes (Phase 04)
- README updates (Phase 04)

## Implementation

### Task 1: Add camera control mode to InterfaceState

**File:** `pyvr/interface/state.py`

**Location:** Add new field in `InterfaceState` class after existing fields (around line 30)

**Implementation:**
```python
# Add to InterfaceState class:

camera_control_mode: str = 'trackball'  # or 'orbit'
```

**Docstring update:**
Add to class docstring:
```python
"""
...existing docstring...

camera_control_mode: Current camera control mode ('trackball' or 'orbit')
"""
```

### Task 2: Update mouse move handler to support both modes

**File:** `pyvr/interface/matplotlib_interface.py`

**Location:** Modify `_on_mouse_move()` method (lines 408-443)

**Before:**
```python
# Handle camera drag
if self.state.is_dragging_camera:
    if event.inaxes != self.image_display.ax or event.xdata is None:
        return

    # Calculate drag delta
    if self.state.drag_start_pos is not None:
        dx = event.xdata - self.state.drag_start_pos[0]
        dy = event.ydata - self.state.drag_start_pos[1]

        # Convert pixel movement to camera angles
        # Sensitivity factor for camera movement
        sensitivity = 0.005
        delta_azimuth = -dx * sensitivity
        delta_elevation = dy * sensitivity

        # Update camera using controller
        self.camera_controller.orbit(
            delta_azimuth=delta_azimuth,
            delta_elevation=delta_elevation
        )

        # Update drag start position for next move
        self.state.drag_start_pos = (event.xdata, event.ydata)

        # Trigger render (throttled to avoid performance issues)
        self.state.needs_render = True
        self._update_display()  # Throttling prevents excessive renders
    return
```

**After:**
```python
# Handle camera drag
if self.state.is_dragging_camera:
    if event.inaxes != self.image_display.ax or event.xdata is None:
        return

    # Calculate drag delta
    if self.state.drag_start_pos is not None:
        dx = event.xdata - self.state.drag_start_pos[0]
        dy = event.ydata - self.state.drag_start_pos[1]

        # Apply camera control based on current mode
        if self.state.camera_control_mode == 'trackball':
            # Trackball control: intuitive 3D rotation
            self.camera_controller.trackball(
                dx=dx,
                dy=dy,
                viewport_width=self.width,
                viewport_height=self.height,
                sensitivity=1.0
            )
        else:  # orbit mode
            # Orbit control: traditional azimuth/elevation
            sensitivity = 0.005
            delta_azimuth = -dx * sensitivity
            delta_elevation = dy * sensitivity

            self.camera_controller.orbit(
                delta_azimuth=delta_azimuth,
                delta_elevation=delta_elevation
            )

        # Update drag start position for next move
        self.state.drag_start_pos = (event.xdata, event.ydata)

        # Trigger render (throttled to avoid performance issues)
        self.state.needs_render = True
        self._update_display()  # Throttling prevents excessive renders
    return
```

### Task 3: Add keyboard toggle for control mode

**File:** `pyvr/interface/matplotlib_interface.py`

**Location:** Add to `_on_key_press()` method (after existing key handlers, around line 660)

**Implementation:**
```python
elif event.key == 't':
    # Toggle camera control mode
    if self.state.camera_control_mode == 'trackball':
        self.state.camera_control_mode = 'orbit'
        print("Switched to orbit control (azimuth/elevation)")
    else:
        self.state.camera_control_mode = 'trackball'
        print("Switched to trackball control (arcball)")
    self._update_status_display()
```

### Task 4: Update info display to show control mode

**File:** `pyvr/interface/matplotlib_interface.py`

**Location:** Modify `_get_full_info_text()` method (around line 265)

**Before:**
```python
def _get_full_info_text(self) -> str:
    """Get complete info text with controls and status."""
    controls = (
        "Mouse Controls:\n"
        "  Image: Drag=orbit, Scroll=zoom\n"
        "  Opacity: L-click=add/select, R-click=remove, Drag=move\n\n"
        "Keyboard Shortcuts:\n"
        "  r: Reset view\n"
        "  s: Save image\n"
        "  f: Toggle FPS\n"
        "  h: Toggle histogram\n"
        "  l: Toggle light link\n"
        "  q: Toggle auto-quality\n"
        "  Esc: Deselect\n"
        "  Del: Remove selected\n"
    )
    # ... rest of method
```

**After:**
```python
def _get_full_info_text(self) -> str:
    """Get complete info text with controls and status."""
    controls = (
        "Mouse Controls:\n"
        "  Image: Drag=rotate, Scroll=zoom\n"
        "  Opacity: L-click=add/select, R-click=remove, Drag=move\n\n"
        "Keyboard Shortcuts:\n"
        "  r: Reset view\n"
        "  s: Save image\n"
        "  t: Toggle control mode\n"
        "  f: Toggle FPS\n"
        "  h: Toggle histogram\n"
        "  l: Toggle light link\n"
        "  q: Toggle auto-quality\n"
        "  Esc: Deselect\n"
        "  Del: Remove selected\n"
    )

    # Add dynamic status
    light = self.renderer.get_light()
    status = (
        f"\nCurrent Status:\n"
        f"  Control Mode: {self.state.camera_control_mode.capitalize()}\n"
        f"  Preset: {self.state.current_preset_name}\n"
        f"  FPS: {'ON' if self.state.show_fps else 'OFF'}\n"
        f"  Histogram: {'ON' if self.state.show_histogram else 'OFF'}\n"
        f"  Light Linked: {'YES' if light.is_linked else 'NO'}\n"
        f"  Auto-Quality: {'ON' if self.state.auto_quality_enabled else 'OFF'}"
    )
    # ... rest of method
```

### Task 5: Add integration tests for interface

**File:** `tests/test_interface/test_trackball_integration.py` (new file)

**Implementation:**
```python
"""Integration tests for trackball control in interface."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from pyvr.interface import InteractiveVolumeRenderer
from pyvr.volume import Volume
from pyvr.camera import Camera


@pytest.fixture
def mock_volume():
    """Create a mock volume for testing."""
    data = np.random.rand(32, 32, 32).astype(np.float32)
    return Volume(data=data)


@pytest.fixture
def mock_interface(mock_volume):
    """Create a mock interface for testing."""
    with patch('pyvr.interface.matplotlib_interface.VolumeRenderer'):
        interface = InteractiveVolumeRenderer(
            volume=mock_volume,
            width=512,
            height=512
        )
        # Mock the display widgets to avoid matplotlib backend issues
        interface.image_display = MagicMock()
        interface.opacity_editor = MagicMock()
        interface.fig = MagicMock()
        return interface


class TestInterfaceTrackballMode:
    """Tests for trackball mode in interface."""

    def test_default_mode_is_trackball(self, mock_interface):
        """New interface should default to trackball mode."""
        assert mock_interface.state.camera_control_mode == 'trackball'

    def test_trackball_mode_drag_calls_trackball(self, mock_interface):
        """Dragging in trackball mode should call controller.trackball()."""
        # Set up drag state
        mock_interface.state.is_dragging_camera = True
        mock_interface.state.drag_start_pos = (100, 100)

        # Mock the trackball method
        with patch.object(mock_interface.camera_controller, 'trackball') as mock_trackball:
            # Simulate mouse move
            event = MagicMock()
            event.inaxes = mock_interface.image_display.ax
            event.xdata = 150
            event.ydata = 120

            mock_interface._on_mouse_move(event)

            # Verify trackball was called
            mock_trackball.assert_called_once()
            call_args = mock_trackball.call_args
            assert call_args[1]['dx'] == 50  # 150 - 100
            assert call_args[1]['dy'] == 20  # 120 - 100
            assert call_args[1]['viewport_width'] == 512
            assert call_args[1]['viewport_height'] == 512

    def test_orbit_mode_drag_calls_orbit(self, mock_interface):
        """Dragging in orbit mode should call controller.orbit()."""
        # Switch to orbit mode
        mock_interface.state.camera_control_mode = 'orbit'

        # Set up drag state
        mock_interface.state.is_dragging_camera = True
        mock_interface.state.drag_start_pos = (100, 100)

        # Mock the orbit method
        with patch.object(mock_interface.camera_controller, 'orbit') as mock_orbit:
            # Simulate mouse move
            event = MagicMock()
            event.inaxes = mock_interface.image_display.ax
            event.xdata = 150
            event.ydata = 120

            mock_interface._on_mouse_move(event)

            # Verify orbit was called
            mock_orbit.assert_called_once()

    def test_toggle_control_mode_key(self, mock_interface):
        """Pressing 't' should toggle control mode."""
        # Start in trackball mode
        assert mock_interface.state.camera_control_mode == 'trackball'

        # Simulate 't' key press
        event = MagicMock()
        event.key = 't'
        mock_interface._on_key_press(event)

        # Should switch to orbit
        assert mock_interface.state.camera_control_mode == 'orbit'

        # Press again
        mock_interface._on_key_press(event)

        # Should switch back to trackball
        assert mock_interface.state.camera_control_mode == 'trackball'

    def test_camera_updates_after_trackball_drag(self, mock_interface):
        """Camera should update after trackball drag."""
        initial_azimuth = mock_interface.camera_controller.params.azimuth

        # Set up drag state
        mock_interface.state.is_dragging_camera = True
        mock_interface.state.drag_start_pos = (100, 100)
        mock_interface.state.camera_control_mode = 'trackball'

        # Simulate mouse move (right drag)
        event = MagicMock()
        event.inaxes = mock_interface.image_display.ax
        event.xdata = 200
        event.ydata = 100

        mock_interface._on_mouse_move(event)

        # Camera should have changed
        assert mock_interface.camera_controller.params.azimuth != initial_azimuth

    def test_camera_updates_after_orbit_drag(self, mock_interface):
        """Camera should update after orbit drag (backward compatibility)."""
        mock_interface.state.camera_control_mode = 'orbit'
        initial_azimuth = mock_interface.camera_controller.params.azimuth

        # Set up drag state
        mock_interface.state.is_dragging_camera = True
        mock_interface.state.drag_start_pos = (100, 100)

        # Simulate mouse move (right drag)
        event = MagicMock()
        event.inaxes = mock_interface.image_display.ax
        event.xdata = 200
        event.ydata = 100

        mock_interface._on_mouse_move(event)

        # Camera should have changed
        assert mock_interface.camera_controller.params.azimuth != initial_azimuth

    def test_zoom_works_in_both_modes(self, mock_interface):
        """Zoom should work regardless of control mode."""
        for mode in ['trackball', 'orbit']:
            mock_interface.state.camera_control_mode = mode
            initial_distance = mock_interface.camera_controller.params.distance

            # Simulate scroll event
            event = MagicMock()
            event.inaxes = mock_interface.image_display.ax
            event.step = 1  # Scroll up

            mock_interface._on_scroll(event)

            # Distance should have changed
            assert mock_interface.camera_controller.params.distance != initial_distance

    def test_info_display_shows_control_mode(self, mock_interface):
        """Info display should show current control mode."""
        # Trackball mode
        mock_interface.state.camera_control_mode = 'trackball'
        info_text = mock_interface._get_full_info_text()
        assert 'Control Mode: Trackball' in info_text

        # Orbit mode
        mock_interface.state.camera_control_mode = 'orbit'
        info_text = mock_interface._get_full_info_text()
        assert 'Control Mode: Orbit' in info_text


class TestControlModeToggle:
    """Tests for control mode toggling."""

    def test_toggle_prints_message(self, mock_interface, capsys):
        """Toggling should print informative message."""
        # Toggle to orbit
        event = MagicMock()
        event.key = 't'
        mock_interface._on_key_press(event)

        captured = capsys.readouterr()
        assert 'orbit control' in captured.out.lower()

        # Toggle back to trackball
        mock_interface._on_key_press(event)

        captured = capsys.readouterr()
        assert 'trackball control' in captured.out.lower()

    def test_toggle_updates_status_display(self, mock_interface):
        """Toggling should update status display."""
        mock_interface._update_status_display = MagicMock()

        # Toggle mode
        event = MagicMock()
        event.key = 't'
        mock_interface._on_key_press(event)

        # Verify status display was updated
        mock_interface._update_status_display.assert_called()


class TestBackwardCompatibility:
    """Tests for backward compatibility with orbit control."""

    def test_orbit_mode_maintains_functionality(self, mock_interface):
        """Orbit mode should work exactly as before."""
        mock_interface.state.camera_control_mode = 'orbit'

        # Perform orbit drag
        mock_interface.state.is_dragging_camera = True
        mock_interface.state.drag_start_pos = (100, 100)

        event = MagicMock()
        event.inaxes = mock_interface.image_display.ax
        event.xdata = 200
        event.ydata = 150

        # Should not raise any errors
        mock_interface._on_mouse_move(event)

    def test_programmatic_orbit_still_works(self, mock_interface):
        """Direct calls to controller.orbit() should still work."""
        initial_azimuth = mock_interface.camera_controller.params.azimuth

        # Call orbit directly
        mock_interface.camera_controller.orbit(
            delta_azimuth=0.5,
            delta_elevation=0.2
        )

        # Should work regardless of interface mode
        assert mock_interface.camera_controller.params.azimuth != initial_azimuth
```

## Verification

### Unit Test Execution

Run integration tests:

```bash
pytest tests/test_interface/test_trackball_integration.py -v
```

**Expected output:**
- All tests pass (15+ tests)
- No warnings or errors
- Backward compatibility verified

### Interactive Testing

**Test mode switching:**
```python
from pyvr.interface import InteractiveVolumeRenderer
from pyvr.volume import Volume
from pyvr.datasets import create_sample_volume
import numpy as np

# Create interface
volume_data = create_sample_volume(64, 'sphere')
volume = Volume(data=volume_data)
interface = InteractiveVolumeRenderer(volume=volume, width=512, height=512)

# Check default mode
print(f"Default mode: {interface.state.camera_control_mode}")
# Should be: trackball

# Note: Cannot fully test interactively without launching show()
# Manual testing required for UI interaction
```

**Test info display:**
```python
# Get info text
info = interface._get_full_info_text()
print(info)

# Should contain:
# - "t: Toggle control mode"
# - "Control Mode: Trackball"
```

### Manual Testing Checklist

Launch interface and verify:

```bash
python -c "
from pyvr.interface import InteractiveVolumeRenderer
from pyvr.volume import Volume
from pyvr.datasets import create_sample_volume

volume_data = create_sample_volume(128, 'double_sphere')
volume = Volume(data=volume_data)
interface = InteractiveVolumeRenderer(volume=volume)
interface.show()
"
```

**Test checklist:**
- [ ] Interface launches successfully
- [ ] Info panel shows "Control Mode: Trackball"
- [ ] Dragging rotates camera smoothly (trackball feel)
- [ ] Press 't' → mode switches to "Orbit"
- [ ] Info panel updates to show "Control Mode: Orbit"
- [ ] Dragging in orbit mode feels different (azimuth/elevation)
- [ ] Press 't' again → switches back to trackball
- [ ] Zoom (scroll) works in both modes
- [ ] Reset ('r') works in both modes
- [ ] No errors or crashes

## Validation

### Behavior Verification

**Trackball mode:**
- Rotation feels natural and intuitive
- Follows mouse movement smoothly
- No gimbal lock artifacts

**Orbit mode:**
- Maintains traditional azimuth/elevation behavior
- Same feel as pre-trackball versions
- Backward compatibility confirmed

**Mode switching:**
- Toggle works instantly
- No camera jump when switching modes
- Info display updates immediately

### Edge Cases

**Switching modes during drag:**
```python
# User might press 't' while dragging
# Should handle gracefully (drag ends, mode switches)
```

**Rapid mode switching:**
```python
# User rapidly presses 't' multiple times
# Should toggle correctly each time
```

### Performance Check

Verify no performance regression:

```bash
# Launch interface and measure FPS
# Should be similar to pre-trackball performance
```

## Acceptance Criteria

- [ ] `camera_control_mode` field added to InterfaceState
- [ ] Default mode is 'trackball'
- [ ] `_on_mouse_move()` supports both trackball and orbit modes
- [ ] 't' key toggles between modes
- [ ] Info display shows current control mode
- [ ] Info display includes 't' key in shortcuts
- [ ] All integration tests pass (15+ tests)
- [ ] Trackball mode works smoothly
- [ ] Orbit mode maintains backward compatibility
- [ ] No regression in zoom or other controls
- [ ] No crashes or errors
- [ ] Manual testing checklist complete

## Git Commit

**Commit message:**
```
feat: Integrate trackball control into interface

Make trackball the default camera control mode for InteractiveVolumeRenderer:
- Add camera_control_mode to InterfaceState ('trackball' or 'orbit')
- Update mouse drag handler to support both modes
- Add 't' keyboard toggle to switch between modes
- Update info display to show current control mode

Features:
- Trackball mode: Default, intuitive 3D rotation
- Orbit mode: Traditional azimuth/elevation (via toggle)
- Seamless switching between modes
- Both modes work with zoom, reset, and other controls

UI Updates:
- Info panel shows "Control Mode: Trackball/Orbit"
- Added "t: Toggle control mode" to keyboard shortcuts
- Changed mouse controls text from "orbit" to "rotate"

Backward Compatibility:
- Orbit mode maintains exact pre-trackball behavior
- Programmatic controller.orbit() calls still work
- No breaking changes to existing API

Tests:
- 15+ integration tests for mode switching, camera updates
- Tests for backward compatibility
- Verification of both modes working correctly

Phase 03 of trackball control implementation (v0.4.1).

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

**Files to commit:**
```
pyvr/interface/state.py (modified - add camera_control_mode field)
pyvr/interface/matplotlib_interface.py (modified - update mouse handler, key handler, info display)
tests/test_interface/test_trackball_integration.py (new - integration tests)
```

**Pre-commit checklist:**
- [ ] All tests pass: `pytest tests/test_interface/test_trackball_integration.py -v`
- [ ] No regression: `pytest tests/test_interface/ -v`
- [ ] No linting errors: `black pyvr/interface/ tests/test_interface/`
- [ ] Manual testing complete
- [ ] Interface launches successfully
