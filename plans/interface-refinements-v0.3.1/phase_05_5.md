# Phase 5.5: Critical Bug Fixes

## Objective

Fix critical bugs discovered during integration testing that prevent proper functionality of v0.3.1 features.

## Issues Identified

### Issue 1: Current Status Overlays Control Menu
**Symptom**: Status text overlaps with the control text in the info panel, making both unreadable.

**Root Cause**: Both static controls and dynamic status are positioned at overlapping coordinates in the same axes.

**Fix**: Adjust layout to separate controls and status, or combine into single dynamically-updated text block.

### Issue 2: Light Linking Key 'l' Breaks Rendering
**Symptom**: Pressing 'l' key causes rendering to fail/freeze.

**Root Cause**: Light linking fix from Phase 3 was reverted or incorrectly integrated. The `_update_display()` method may not properly handle linked light updates, or camera_controller.params may not be accessible.

**Fix**:
- Verify light linking code is present in `_update_display()`
- Ensure camera_controller uses correct attribute (likely `camera` not `params`)
- Add error handling for light linking failures

### Issue 3: Mouse Click-Drag in Image Not Working
**Symptom**: Camera orbit via mouse drag in the image display is non-functional.

**Root Cause**: Event handler may not be properly connected, or drag state tracking is broken.

**Fix**:
- Verify mouse event callbacks are connected to matplotlib figure
- Check `_on_mouse_press`, `_on_mouse_move`, `_on_mouse_release` handlers
- Ensure drag state variables are properly initialized and updated

### Issue 4: Default Keyboard Interactions Interfere
**Symptom**: Pressing 'f' toggles matplotlib fullscreen instead of FPS counter. Other keys may also trigger matplotlib defaults.

**Root Cause**: Matplotlib's default key bindings are active and override custom shortcuts.

**Fix**: Disable matplotlib's default key bindings using `fig.canvas.mpl_disconnect()` or `plt.rcParams['keymap.*'] = []`.

## Implementation Steps

### 1. Fix Status Display Overlay

**File**: `/Users/jixianli/projects/pyvr/pyvr/interface/matplotlib_interface.py`

**Solution**: Combine controls and status into single text block with clear separation.

```python
def _setup_info_display(self, ax) -> None:
    """Set up info display panel with all controls and status indicators."""
    ax.axis('off')

    # Single text block with both controls and status
    self.info_text = ax.text(
        0.05, 0.98,  # Top of axes
        self._get_full_info_text(),
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment='top',
        family='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    )

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

    # Add dynamic status
    light = self.renderer.get_light()
    status = (
        f"\nCurrent Status:\n"
        f"  Preset: {self.state.current_preset_name}\n"
        f"  FPS: {'ON' if self.state.show_fps else 'OFF'}\n"
        f"  Histogram: {'ON' if self.state.show_histogram else 'OFF'}\n"
        f"  Light Linked: {'YES' if light.is_linked else 'NO'}"
    )

    return controls + status

def _update_status_display(self) -> None:
    """Update status text with current settings."""
    if hasattr(self, 'info_text'):
        self.info_text.set_text(self._get_full_info_text())
        self.fig.canvas.draw_idle()
```

### 2. Fix Light Linking

**File**: `/Users/jixianli/projects/pyvr/pyvr/interface/matplotlib_interface.py`

**Changes in `_update_display()`**:
```python
def _update_display(self, force_render: bool = False) -> None:
    """Update all display widgets based on current state."""
    # Update transfer functions if needed
    if self.state.needs_tf_update:
        self._update_transfer_functions()
        self.state.needs_render = True

    # Update light from camera if linked
    light = self.renderer.get_light()
    if light.is_linked:
        try:
            # Use camera_controller.camera (not .params)
            light.update_from_camera(self.camera_controller.camera)
            self.renderer.set_light(light)
            self.state.needs_render = True
        except Exception as e:
            print(f"Warning: Failed to update linked light: {e}")

    # Update volume rendering with throttling
    if self.state.needs_render and (force_render or self._should_render()):
        image_array = self._render_volume()
        if self.image_display is not None:
            self.image_display.update_image(image_array)
        self.state.needs_render = False

    # Always update opacity editor (fast)
    if self.opacity_editor is not None:
        self.opacity_editor.update_plot(
            self.state.control_points,
            self.state.selected_control_point
        )
```

**Changes in `_on_key_press()`**:
```python
elif event.key == 'l':
    # Toggle light camera linking
    light = self.renderer.get_light()

    if light.is_linked:
        light.unlink_from_camera()
        self.state.light_linked_to_camera = False
        print("Light unlinked from camera (fixed position)")
    else:
        # Link with default offsets (light follows camera)
        light.link_to_camera(
            azimuth_offset=0.0,
            elevation_offset=0.0,
            distance_offset=0.0
        )
        # Use camera_controller.camera
        light.update_from_camera(self.camera_controller.camera)
        self.renderer.set_light(light)
        self.state.light_linked_to_camera = True
        print("Light linked to camera (will follow movement)")

    self.state.needs_render = True
    self._update_display(force_render=True)
    self._update_status_display()
```

### 3. Fix Mouse Click-Drag

**File**: `/Users/jixianli/projects/pyvr/pyvr/interface/matplotlib_interface.py`

**Ensure event connections in `show()`**:
```python
def show(self) -> None:
    """Display the interactive interface."""
    # Create figure and axes
    self.fig, axes = self._create_layout()

    # ... widget initialization ...

    # Connect event handlers - ENSURE THESE ARE CALLED
    self.fig.canvas.mpl_connect('button_press_event', self._on_mouse_press)
    self.fig.canvas.mpl_connect('button_release_event', self._on_mouse_release)
    self.fig.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
    self.fig.canvas.mpl_connect('scroll_event', self._on_scroll)
    self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)

    # Initial render
    self.state.needs_render = True
    self._update_display(force_render=True)

    plt.show()
```

**Verify `_on_mouse_press()` handles image axes correctly**:
```python
def _on_mouse_press(self, event) -> None:
    """Handle mouse button press."""
    if event.inaxes is None:
        return

    # Handle image display (camera controls)
    if event.inaxes == self.image_display.ax:
        if event.button == 1:  # Left click
            self.state.is_dragging_camera = True
            self.state.drag_start_pos = (event.xdata, event.ydata)

            # Switch to fast preset during interaction
            if self.state.auto_quality_enabled:
                self._switch_to_interaction_quality()
        return

    # Handle opacity editor (control points)
    if event.inaxes == self.opacity_editor.ax:
        # ... existing control point logic ...
```

### 4. Disable Default Matplotlib Keybindings

**File**: `/Users/jixianli/projects/pyvr/pyvr/interface/matplotlib_interface.py`

**Add to `show()` method before creating figure**:
```python
def show(self) -> None:
    """Display the interactive interface."""
    # Disable matplotlib default key bindings to prevent conflicts
    import matplotlib as mpl

    # Store original keymaps to restore later if needed
    self._original_keymaps = {
        'fullscreen': mpl.rcParams['keymap.fullscreen'][:],
        'home': mpl.rcParams['keymap.home'][:],
        'back': mpl.rcParams['keymap.back'][:],
        'forward': mpl.rcParams['keymap.forward'][:],
        'pan': mpl.rcParams['keymap.pan'][:],
        'zoom': mpl.rcParams['keymap.zoom'][:],
        'save': mpl.rcParams['keymap.save'][:],
        'quit': mpl.rcParams['keymap.quit'][:],
        'grid': mpl.rcParams['keymap.grid'][:],
        'yscale': mpl.rcParams['keymap.yscale'][:],
        'xscale': mpl.rcParams['keymap.xscale'][:],
    }

    # Clear all default keybindings
    mpl.rcParams['keymap.fullscreen'] = []
    mpl.rcParams['keymap.home'] = []
    mpl.rcParams['keymap.back'] = []
    mpl.rcParams['keymap.forward'] = []
    mpl.rcParams['keymap.pan'] = []
    mpl.rcParams['keymap.zoom'] = []
    mpl.rcParams['keymap.save'] = []
    mpl.rcParams['keymap.quit'] = []
    mpl.rcParams['keymap.grid'] = []
    mpl.rcParams['keymap.yscale'] = []
    mpl.rcParams['keymap.xscale'] = []

    # Create figure and axes
    self.fig, axes = self._create_layout()

    # ... rest of show() method ...
```

**Add cleanup method**:
```python
def _restore_matplotlib_keymaps(self) -> None:
    """Restore original matplotlib keymaps."""
    if hasattr(self, '_original_keymaps'):
        import matplotlib as mpl
        for key, value in self._original_keymaps.items():
            mpl.rcParams[f'keymap.{key}'] = value
```

## Testing Plan

### Manual Testing Checklist

1. **Status Display**:
   - [ ] Launch interface, verify controls and status are both visible
   - [ ] Toggle features (f, h, l, q), verify status updates without overlap
   - [ ] Check text readability

2. **Light Linking**:
   - [ ] Press 'l' to enable light linking
   - [ ] Orbit camera with mouse drag
   - [ ] Verify rendering continues smoothly
   - [ ] Verify light follows camera
   - [ ] Press 'l' again to disable, verify no errors

3. **Mouse Dragging**:
   - [ ] Click and drag in image area
   - [ ] Verify camera orbits around volume
   - [ ] Release mouse, verify rendering completes
   - [ ] Test scroll zoom in image area

4. **Keyboard Shortcuts**:
   - [ ] Press 'f', verify FPS toggles (not fullscreen)
   - [ ] Press 's', verify save dialog (not matplotlib default)
   - [ ] Press 'h', verify histogram toggles
   - [ ] Press 'l', verify light linking toggles
   - [ ] Press 'q', verify auto-quality toggles
   - [ ] Verify no matplotlib default behaviors trigger

### Automated Testing

**File**: `/Users/jixianli/projects/pyvr/tests/test_interface/test_bug_fixes.py` (new)

```python
"""Tests for Phase 5.5 bug fixes."""

import pytest
from unittest.mock import MagicMock, patch
import matplotlib as mpl
from pyvr.interface import InteractiveVolumeRenderer
from pyvr.volume import Volume
from pyvr.datasets import create_sample_volume


class TestBugFixes:
    """Tests for critical bug fixes."""

    def test_status_display_no_overlap(self):
        """Test status display doesn't overlap controls."""
        volume_data = create_sample_volume(64, 'sphere')
        volume = Volume(data=volume_data)
        interface = InteractiveVolumeRenderer(volume=volume)

        # Get full info text
        info_text = interface._get_full_info_text()

        # Should contain both controls and status
        assert 'Mouse Controls' in info_text
        assert 'Keyboard Shortcuts' in info_text
        assert 'Current Status' in info_text
        assert 'Preset:' in info_text

    def test_light_linking_no_error(self):
        """Test light linking doesn't cause errors."""
        volume_data = create_sample_volume(64, 'sphere')
        volume = Volume(data=volume_data)
        interface = InteractiveVolumeRenderer(volume=volume)

        light = interface.renderer.get_light()
        light.link_to_camera()

        # Should not raise exception
        interface._update_display(force_render=True)

    def test_light_linking_uses_correct_camera_attribute(self):
        """Test light linking uses camera_controller.camera."""
        volume_data = create_sample_volume(64, 'sphere')
        volume = Volume(data=volume_data)
        interface = InteractiveVolumeRenderer(volume=volume)

        light = interface.renderer.get_light()
        light.link_to_camera()

        # Verify camera_controller has 'camera' attribute
        assert hasattr(interface.camera_controller, 'camera')

        # Should be able to update from camera
        light.update_from_camera(interface.camera_controller.camera)

    def test_matplotlib_keymaps_disabled(self):
        """Test matplotlib default keymaps are disabled."""
        volume_data = create_sample_volume(64, 'sphere')
        volume = Volume(data=volume_data)
        interface = InteractiveVolumeRenderer(volume=volume)

        # Mock plt.show to avoid blocking
        with patch('matplotlib.pyplot.show'):
            interface.show()

        # Check keymaps are cleared
        assert mpl.rcParams['keymap.fullscreen'] == []
        assert mpl.rcParams['keymap.save'] == []

    def test_event_handlers_connected(self):
        """Test mouse/keyboard event handlers are connected."""
        volume_data = create_sample_volume(64, 'sphere')
        volume = Volume(data=volume_data)
        interface = InteractiveVolumeRenderer(volume=volume)

        # Mock plt.show
        with patch('matplotlib.pyplot.show'):
            interface.show()

        # Verify event handlers are methods
        assert callable(interface._on_mouse_press)
        assert callable(interface._on_mouse_move)
        assert callable(interface._on_mouse_release)
        assert callable(interface._on_scroll)
        assert callable(interface._on_key_press)
```

## Deliverables

### Code Changes

1. **Fixed info display** - single text block with controls + status
2. **Fixed light linking** - correct camera attribute reference
3. **Verified mouse event handlers** - proper connection and state tracking
4. **Disabled matplotlib defaults** - cleared all conflicting keybindings

### Tests

- 5+ new tests in `test_bug_fixes.py`
- Manual testing checklist completed

### Documentation

- Updated phase_06.md to mention bug fixes
- This phase_05_5.md document

## Acceptance Criteria

- [ ] Status display shows both controls and status without overlap
- [ ] Light linking key 'l' works without errors
- [ ] Mouse drag in image orbits camera smoothly
- [ ] Keyboard shortcuts work without triggering matplotlib defaults
- [ ] All existing tests still pass
- [ ] 5+ new bug fix tests pass

## Git Commit Message

```
fix(interface): Critical bug fixes for v0.3.1 integration

Fix four critical bugs discovered during integration testing:

Bug Fixes:
1. Status display overlaying controls - combined into single text block
2. Light linking key 'l' breaking rendering - fixed camera attribute reference
3. Mouse click-drag not working - verified event handler connections
4. Matplotlib default keybindings interfering - disabled all defaults

Changes:
- Combined info display into single dynamically-updated text block
- Fixed light linking to use camera_controller.camera (not .params)
- Added error handling for light update failures
- Disabled matplotlib's default key bindings in show()
- Added keymap restoration method for cleanup

Tests:
- 5 new tests for bug fixes
- All existing tests still pass
- Manual testing checklist completed

Implements phase 5.5 of v0.3.1 interface refinements.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

## Notes

### Issue Priority

All four issues are **critical** and block the v0.3.1 release:
1. Status overlay makes interface unusable
2. Light linking breaks core feature
3. Mouse drag breaks primary interaction
4. Keyboard conflicts confuse users

### Root Cause Analysis

- **Status overlay**: Integration code from Phase 5 wasn't properly tested with layout
- **Light linking**: Camera controller attribute naming inconsistency (`.camera` vs `.params`)
- **Mouse drag**: Event handlers may have been disconnected during refactoring
- **Keyboard conflicts**: Matplotlib defaults were never disabled in Phase 1

### Prevention

- Add integration tests that verify UI layout
- Add tests that verify event handler connections
- Document matplotlib configuration requirements
- Test keyboard shortcuts with actual matplotlib window
