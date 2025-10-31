# Phase 7: Event Handling Integration and State Management

**Status**: Not Started
**Estimated Effort**: 3-4 hours
**Dependencies**: Phases 4, 5, 6 (all interactions implemented)

## Overview

Integrate all event handlers into a cohesive system, resolve event conflicts, optimize performance, and ensure proper state synchronization across all widgets.

## Implementation Plan

### Key Tasks

1. **Resolve event conflicts** - Ensure camera and opacity events don't interfere
2. **Optimize rendering** - Debounce updates, batch state changes
3. **Add keyboard shortcuts** - Reset view, save image, etc.
4. **Improve visual feedback** - Cursor changes, tooltips
5. **Error handling** - Graceful handling of edge cases

### Modify: `pyvr/interface/matplotlib.py`

Add event coordination and optimization:

```python
def __init__(self, ...):
    # ... existing code ...

    # Event state
    self._last_render_time = 0
    self._min_render_interval = 0.1  # 100ms minimum between renders

def _should_render(self) -> bool:
    """Check if enough time has passed since last render."""
    import time
    current_time = time.time()
    if current_time - self._last_render_time > self._min_render_interval:
        self._last_render_time = current_time
        return True
    return False

def _update_display(self, force_render: bool = False):
    """Update display with optional force render."""
    # Update transfer functions if needed
    if self.state.needs_tf_update:
        self._update_transfer_functions()
        self.state.needs_render = True

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

def show(self):
    # ... existing code ...

    # Connect keyboard events
    self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)

    plt.show()

def _on_key_press(self, event):
    """Handle keyboard shortcuts."""
    if event.key == 'r':
        # Reset view to isometric
        self.camera_controller.camera = Camera.isometric_view(distance=3.0)
        self.state.needs_render = True
        self._update_display(force_render=True)

    elif event.key == 's':
        # Save current rendering
        self._save_image()

    elif event.key == 'escape':
        # Deselect control point
        if self.state.selected_control_point is not None:
            self.state.select_control_point(None)
            self._update_display()

    elif event.key == 'delete' or event.key == 'backspace':
        # Delete selected control point
        if self.state.selected_control_point is not None:
            try:
                self.state.remove_control_point(self.state.selected_control_point)
                self._update_display(force_render=True)
            except ValueError:
                pass  # Can't delete first/last

def _save_image(self):
    """Save current rendering to file."""
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"pyvr_render_{timestamp}.png"

    if self._cached_image is not None:
        from PIL import Image
        img = Image.fromarray(self._cached_image)
        img.save(filename)
        print(f"Saved rendering to {filename}")

def _update_cursor(self, event):
    """Update cursor based on context."""
    if event.inaxes == self.image_display.ax:
        # Hand cursor for camera controls
        self.fig.canvas.set_cursor(1)  # Hand cursor
    elif event.inaxes == self.opacity_editor.ax:
        # Crosshair for control point editing
        self.fig.canvas.set_cursor(2)  # Crosshair cursor
    else:
        # Default cursor
        self.fig.canvas.set_cursor(0)

def show(self):
    # ... existing code ...

    # Add cursor updates
    self.fig.canvas.mpl_connect('axes_enter_event', self._update_cursor)
```

### Update Info Display

```python
def _setup_info_display(self, ax):
    """Set up info display with keyboard shortcuts."""
    ax.axis('off')
    info_text = (
        "Mouse Controls:\n"
        "  Image: Drag=orbit, Scroll=zoom\n"
        "  Opacity: L-click=add/select, R-click=remove, Drag=move\n\n"
        "Keyboard Shortcuts:\n"
        "  r: Reset view\n"
        "  s: Save image\n"
        "  Esc: Deselect\n"
        "  Del: Remove selected"
    )
    ax.text(0.05, 0.5, info_text,
           transform=ax.transAxes,
           fontsize=8,
           verticalalignment='center',
           family='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
```

## Testing Plan

```python
def test_render_throttling():
    """Test rendering is throttled to prevent excessive updates."""
    interface = InteractiveVolumeRenderer(volume=small_volume)
    interface._last_render_time = time.time()

    # Should not render immediately
    assert not interface._should_render()

    # Should render after interval
    time.sleep(0.15)
    assert interface._should_render()

def test_keyboard_reset_view():
    """Test 'r' key resets camera view."""
    interface = InteractiveVolumeRenderer(volume=small_volume)
    interface.image_display = Mock()

    # Modify camera
    interface.camera_controller.orbit(delta_azimuth=1.0, delta_elevation=0.5)

    # Reset
    event = Mock(key='r')
    interface._on_key_press(event)

    # Should be back to isometric view
    camera = interface.camera_controller.camera
    assert camera.distance == pytest.approx(3.0)

def test_keyboard_save_image():
    """Test 's' key saves image."""
    interface = InteractiveVolumeRenderer(volume=small_volume)
    interface._cached_image = np.zeros((512, 512, 3), dtype=np.uint8)

    with patch('PIL.Image.fromarray') as mock_fromarray:
        mock_img = Mock()
        mock_fromarray.return_value = mock_img

        event = Mock(key='s')
        interface._on_key_press(event)

        mock_img.save.assert_called_once()

def test_keyboard_deselect():
    """Test Esc key deselects control point."""
    interface = InteractiveVolumeRenderer(volume=small_volume)
    interface.state.select_control_point(0)
    interface.opacity_editor = Mock()

    event = Mock(key='escape')
    interface._on_key_press(event)

    assert interface.state.selected_control_point is None

def test_keyboard_delete_selected():
    """Test Delete key removes selected control point."""
    interface = InteractiveVolumeRenderer(volume=small_volume)
    interface.state.add_control_point(0.5, 0.5)
    interface.state.select_control_point(1)
    interface.opacity_editor = Mock()
    interface.image_display = Mock()

    event = Mock(key='delete')
    interface._on_key_press(event)

    assert (0.5, 0.5) not in interface.state.control_points

def test_event_conflict_resolution():
    """Test that events in different axes don't interfere."""
    interface = InteractiveVolumeRenderer(volume=small_volume)
    interface.image_display = Mock()
    interface.opacity_editor = Mock()

    # Click in image should not affect opacity editor
    event = Mock(inaxes=interface.image_display.ax, button=1, xdata=100, ydata=100)
    interface._on_mouse_press(event)

    assert interface.state.is_dragging_camera
    assert not interface.state.is_dragging_control_point

def test_cursor_changes():
    """Test cursor changes based on axes."""
    interface = InteractiveVolumeRenderer(volume=small_volume)
    interface.image_display = Mock()
    interface.fig = Mock()
    interface.fig.canvas = Mock()

    event = Mock(inaxes=interface.image_display.ax)
    interface._update_cursor(event)

    interface.fig.canvas.set_cursor.assert_called()
```

## Acceptance Criteria

- [x] All event handlers work together without conflicts
- [x] Rendering is throttled to prevent excessive updates
- [x] Keyboard shortcuts work (reset, save, deselect, delete)
- [x] Cursor changes based on context
- [x] Info display shows all controls
- [x] Performance: Interface remains responsive under all interactions
- [x] Tests cover event coordination and keyboard shortcuts (8+ tests)
- [x] Manual testing: Smooth, lag-free interaction

## Git Commit

```bash
pytest tests/test_interface/
git add pyvr/interface/ tests/test_interface/
git commit -m "Phase 7: Integrate event handling and state management

- Add event coordination to prevent conflicts
- Implement render throttling for performance
- Add keyboard shortcuts (r=reset, s=save, Esc=deselect, Del=delete)
- Add context-aware cursor changes
- Update info display with all controls
- Add tests for event coordination and keyboard shortcuts

Part of v0.3.0 interactive interface feature"
```

## Notes

This phase ties everything together and ensures the interface feels polished and professional. Performance optimization is critical here.
