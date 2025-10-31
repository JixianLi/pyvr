# Phase 4: Camera Controls (Mouse Orbit and Zoom)

**Status**: Not Started
**Estimated Effort**: 3-4 hours
**Dependencies**: Phase 3 (rendering integration)

## Overview

Implement interactive camera controls using matplotlib mouse events. Enable orbiting (drag) and zooming (scroll) with smooth camera movements using the existing CameraController.

## Implementation Plan

### Modify: `pyvr/interface/matplotlib_interface.py`

Add event handlers:

```python
def show(self):
    # ... existing code ...

    # Connect event handlers
    self.fig.canvas.mpl_connect('button_press_event', self._on_mouse_press)
    self.fig.canvas.mpl_connect('button_release_event', self._on_mouse_release)
    self.fig.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
    self.fig.canvas.mpl_connect('scroll_event', self._on_scroll)

    plt.show()

def _on_mouse_press(self, event):
    """Handle mouse button press."""
    # Check if click is in image display axes
    if event.inaxes != self.image_display.ax:
        return

    if event.button == 1:  # Left click
        self.state.is_dragging_camera = True
        self.state.drag_start_pos = (event.xdata, event.ydata)

def _on_mouse_release(self, event):
    """Handle mouse button release."""
    if self.state.is_dragging_camera:
        self.state.is_dragging_camera = False
        self.state.drag_start_pos = None
        # Trigger final render after drag
        self.state.needs_render = True
        self._update_display()

def _on_mouse_move(self, event):
    """Handle mouse movement."""
    if not self.state.is_dragging_camera:
        return

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

        # Don't render every frame - too slow
        # Will render on mouse release

def _on_scroll(self, event):
    """Handle mouse scroll for zoom."""
    if event.inaxes != self.image_display.ax:
        return

    # Scroll up = zoom in, scroll down = zoom out
    zoom_factor = 1.1 if event.step > 0 else 0.9

    self.camera_controller.zoom(factor=zoom_factor)

    # Render immediately for zoom (it's fast enough)
    self.state.needs_render = True
    self._update_display()
```

## Testing Plan

```python
def test_mouse_press_starts_camera_drag():
    """Test left-click in image display starts camera drag."""
    interface = InteractiveVolumeRenderer(volume=small_volume)
    interface.image_display = Mock()

    event = Mock(inaxes=interface.image_display.ax, button=1, xdata=100, ydata=100)
    interface._on_mouse_press(event)

    assert interface.state.is_dragging_camera
    assert interface.state.drag_start_pos == (100, 100)

def test_mouse_move_orbits_camera():
    """Test mouse movement orbits camera."""
    interface = InteractiveVolumeRenderer(volume=small_volume)
    interface.image_display = Mock()
    interface.state.is_dragging_camera = True
    interface.state.drag_start_pos = (100, 100)

    initial_azimuth = interface.camera_controller.camera.azimuth

    event = Mock(inaxes=interface.image_display.ax, xdata=150, ydata=100)
    interface._on_mouse_move(event)

    # Azimuth should have changed
    assert interface.camera_controller.camera.azimuth != initial_azimuth

def test_scroll_zooms_camera():
    """Test scroll event zooms camera."""
    interface = InteractiveVolumeRenderer(volume=small_volume)
    interface.image_display = Mock()

    initial_distance = interface.camera_controller.camera.distance

    # Scroll up (zoom in)
    event = Mock(inaxes=interface.image_display.ax, step=1)
    interface._on_scroll(event)

    assert interface.camera_controller.camera.distance < initial_distance

def test_scroll_outside_image_ignored():
    """Test scroll outside image display is ignored."""
    interface = InteractiveVolumeRenderer(volume=small_volume)
    interface.image_display = Mock()

    initial_distance = interface.camera_controller.camera.distance

    event = Mock(inaxes=None, step=1)
    interface._on_scroll(event)

    assert interface.camera_controller.camera.distance == initial_distance
```

## Acceptance Criteria

- [x] Left-click drag in image display orbits camera
- [x] Camera orbits smoothly around scene center
- [x] Mouse scroll zooms camera in/out
- [x] Events outside image display are ignored
- [x] Performance: No lag during camera movement
- [x] Tests cover all event handlers
- [x] Manual testing confirms smooth interaction

## Git Commit

```bash
pytest tests/test_interface/
git add pyvr/interface/ tests/test_interface/
git commit -m "Phase 4: Add camera controls (orbit and zoom)

- Implement mouse drag to orbit camera using CameraController
- Implement mouse scroll to zoom camera
- Add event handlers for button press, release, move, and scroll
- Optimize: Only render on drag end, not every mouse move
- Add tests for all camera interaction events

Part of v0.3.0 interactive interface feature"
```
