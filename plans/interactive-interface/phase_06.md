# Phase 6: Interactive Opacity Transfer Function Editor

**Status**: Completed
**Estimated Effort**: 4-5 hours (Most complex phase)
**Dependencies**: Phase 2 (layout and widgets)

## Overview

Implement full interactive editing of opacity transfer function control points with:
- Left-click to select/add control points
- Right-click to remove control points
- Drag to move control points
- Locked endpoints (x-position fixed for first/last)

This is the most complex phase due to event coordinate transforms and interaction logic.

## Implementation Plan

### Modify: `pyvr/interface/matplotlib_interface.py`

Add opacity editor event handlers:

```python
def show(self):
    # ... existing code ...

    # Connect event handlers (in addition to Phase 4 handlers)
    self.fig.canvas.mpl_connect('button_press_event', self._on_mouse_press)
    self.fig.canvas.mpl_connect('button_release_event', self._on_mouse_release)
    self.fig.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
    self.fig.canvas.mpl_connect('scroll_event', self._on_scroll)

    plt.show()

def _on_mouse_press(self, event):
    """Handle mouse button press in any axes."""
    # Handle image display (camera controls)
    if event.inaxes == self.image_display.ax:
        if event.button == 1:
            self.state.is_dragging_camera = True
            self.state.drag_start_pos = (event.xdata, event.ydata)
        return

    # Handle opacity editor
    if event.inaxes == self.opacity_editor.ax:
        if event.button == 1:  # Left click
            self._handle_opacity_left_click(event)
        elif event.button == 3:  # Right click
            self._handle_opacity_right_click(event)

def _handle_opacity_left_click(self, event):
    """Handle left click in opacity editor."""
    click_x, click_y = event.xdata, event.ydata

    # Check if clicking near existing control point
    cp_index = self._find_control_point_near(click_x, click_y, threshold=0.05)

    if cp_index is not None:
        # Select existing control point
        self.state.select_control_point(cp_index)
        self.state.is_dragging_control_point = True
        self.state.drag_start_pos = (click_x, click_y)
    else:
        # Add new control point
        try:
            self.state.add_control_point(click_x, click_y)
            # Select the newly added point
            new_index = self.state.control_points.index((click_x, click_y))
            self.state.select_control_point(new_index)
        except ValueError:
            pass  # Out of range, ignore

    self._update_display()

def _handle_opacity_right_click(self, event):
    """Handle right click in opacity editor (remove control point)."""
    click_x, click_y = event.xdata, event.ydata

    cp_index = self._find_control_point_near(click_x, click_y, threshold=0.05)

    if cp_index is not None:
        try:
            self.state.remove_control_point(cp_index)
            self._update_display()
        except ValueError:
            pass  # First/last point, cannot remove

def _find_control_point_near(self, x: float, y: float, threshold: float = 0.05) -> Optional[int]:
    """
    Find control point near given coordinates.

    Args:
        x: X coordinate (scalar value)
        y: Y coordinate (opacity value)
        threshold: Distance threshold for "near"

    Returns:
        Index of control point if found, None otherwise
    """
    for i, (cp_x, cp_y) in enumerate(self.state.control_points):
        distance = np.sqrt((cp_x - x)**2 + (cp_y - y)**2)
        if distance < threshold:
            return i
    return None

def _on_mouse_release(self, event):
    """Handle mouse button release."""
    if self.state.is_dragging_camera:
        self.state.is_dragging_camera = False
        self.state.drag_start_pos = None
        self.state.needs_render = True
        self._update_display()

    if self.state.is_dragging_control_point:
        self.state.is_dragging_control_point = False
        self.state.drag_start_pos = None
        # Final render with new transfer function
        self._update_display()

def _on_mouse_move(self, event):
    """Handle mouse movement."""
    # Handle camera drag (Phase 4)
    if self.state.is_dragging_camera:
        # ... existing camera drag code ...
        return

    # Handle control point drag
    if self.state.is_dragging_control_point:
        if event.inaxes != self.opacity_editor.ax or event.xdata is None:
            return

        if self.state.selected_control_point is None:
            return

        # Clamp to valid range
        new_x = np.clip(event.xdata, 0.0, 1.0)
        new_y = np.clip(event.ydata, 0.0, 1.0)

        try:
            self.state.update_control_point(
                self.state.selected_control_point,
                new_x,
                new_y
            )
            # Update display (but don't re-render volume yet, too slow)
            if self.opacity_editor:
                self.opacity_editor.update_plot(
                    self.state.control_points,
                    self.state.selected_control_point
                )
        except (ValueError, IndexError):
            pass  # Invalid update, ignore
```

### Add Helper Methods

```python
def _clamp_to_range(self, value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Clamp value to range."""
    return max(min_val, min(max_val, value))
```

## Testing Plan

```python
def test_add_control_point_on_click():
    """Test clicking empty space adds control point."""
    interface = InteractiveVolumeRenderer(volume=small_volume)
    interface.opacity_editor = Mock()
    interface.opacity_editor.ax = Mock()

    initial_count = len(interface.state.control_points)

    event = Mock(inaxes=interface.opacity_editor.ax, button=1,
                xdata=0.5, ydata=0.5)
    interface._on_mouse_press(event)

    assert len(interface.state.control_points) == initial_count + 1

def test_select_control_point_on_click():
    """Test clicking near control point selects it."""
    interface = InteractiveVolumeRenderer(volume=small_volume)
    interface.state.add_control_point(0.5, 0.5)
    interface.opacity_editor = Mock()
    interface.opacity_editor.ax = Mock()

    event = Mock(inaxes=interface.opacity_editor.ax, button=1,
                xdata=0.51, ydata=0.51)  # Near (0.5, 0.5)
    interface._on_mouse_press(event)

    assert interface.state.selected_control_point is not None
    assert interface.state.is_dragging_control_point

def test_remove_control_point_on_right_click():
    """Test right-clicking control point removes it."""
    interface = InteractiveVolumeRenderer(volume=small_volume)
    interface.state.add_control_point(0.5, 0.5)
    interface.opacity_editor = Mock()
    interface.opacity_editor.ax = Mock()

    cp_index = 1  # Middle point

    event = Mock(inaxes=interface.opacity_editor.ax, button=3,
                xdata=0.5, ydata=0.5)
    interface._on_mouse_press(event)

    assert (0.5, 0.5) not in interface.state.control_points

def test_cannot_remove_first_last_on_right_click():
    """Test right-clicking first/last point doesn't remove it."""
    interface = InteractiveVolumeRenderer(volume=small_volume)
    interface.opacity_editor = Mock()
    interface.opacity_editor.ax = Mock()

    initial_count = len(interface.state.control_points)

    event = Mock(inaxes=interface.opacity_editor.ax, button=3,
                xdata=0.0, ydata=0.0)  # First point
    interface._on_mouse_press(event)

    assert len(interface.state.control_points) == initial_count

def test_drag_control_point():
    """Test dragging control point updates position."""
    interface = InteractiveVolumeRenderer(volume=small_volume)
    interface.state.add_control_point(0.5, 0.5)
    interface.opacity_editor = Mock()
    interface.opacity_editor.ax = Mock()

    # Select and start dragging
    interface.state.select_control_point(1)  # Middle point
    interface.state.is_dragging_control_point = True

    event = Mock(inaxes=interface.opacity_editor.ax,
                xdata=0.6, ydata=0.7)
    interface._on_mouse_move(event)

    # Check point moved
    cp = interface.state.control_points[1]
    assert cp[0] == pytest.approx(0.6)
    assert cp[1] == pytest.approx(0.7)

def test_drag_first_point_locks_x():
    """Test dragging first control point only changes opacity."""
    interface = InteractiveVolumeRenderer(volume=small_volume)
    interface.opacity_editor = Mock()
    interface.opacity_editor.ax = Mock()

    interface.state.select_control_point(0)  # First point
    interface.state.is_dragging_control_point = True

    event = Mock(inaxes=interface.opacity_editor.ax,
                xdata=0.5, ydata=0.3)  # Try to change x
    interface._on_mouse_move(event)

    cp = interface.state.control_points[0]
    assert cp[0] == 0.0  # X locked
    assert cp[1] == 0.3  # Y changed

def test_find_control_point_near():
    """Test finding control points near coordinates."""
    interface = InteractiveVolumeRenderer(volume=small_volume)
    interface.state.add_control_point(0.5, 0.5)

    # Find nearby point
    index = interface._find_control_point_near(0.51, 0.51, threshold=0.05)
    assert index == 1  # Should find middle point

    # Don't find far point
    index = interface._find_control_point_near(0.8, 0.8, threshold=0.05)
    assert index is None

def test_control_point_coordinates_clamped():
    """Test control point coordinates are clamped to [0, 1]."""
    interface = InteractiveVolumeRenderer(volume=small_volume)
    interface.state.add_control_point(0.5, 0.5)
    interface.opacity_editor = Mock()
    interface.opacity_editor.ax = Mock()

    interface.state.select_control_point(1)
    interface.state.is_dragging_control_point = True

    # Try to drag outside bounds
    event = Mock(inaxes=interface.opacity_editor.ax,
                xdata=1.5, ydata=-0.5)
    interface._on_mouse_move(event)

    cp = interface.state.control_points[1]
    assert 0.0 <= cp[0] <= 1.0
    assert 0.0 <= cp[1] <= 1.0
```

## Acceptance Criteria

- [x] Left-click empty space adds control point
- [x] Left-click control point selects it (visual highlight)
- [x] Right-click control point removes it (except first/last)
- [x] Drag control point moves it smoothly
- [x] First/last control points only change opacity (x locked)
- [x] Control point coordinates clamped to [0, 1]
- [x] Transfer function updates in real-time during drag
- [x] Volume re-renders with new transfer function after drag
- [x] Tests cover all interaction scenarios (12+ tests)
- [x] Manual testing: Smooth, intuitive interaction

## Git Commit

```bash
pytest tests/test_interface/
git add pyvr/interface/ tests/test_interface/
git commit -m "Phase 6: Add interactive opacity transfer function editor

- Implement left-click to add/select control points
- Implement right-click to remove control points (except first/last)
- Implement drag to move control points
- Lock first/last control points to x=0.0 and x=1.0
- Add coordinate clamping to [0, 1] range
- Add comprehensive tests for all interaction scenarios (12+ tests)

Part of v0.3.0 interactive interface feature"
```

## Notes

This is the most complex phase due to:
1. Event coordinate transforms between figure and data coordinates
2. Hit detection for control points
3. Drag state management
4. Performance optimization (update plot during drag, render volume after)

Extra testing and manual verification are critical for this phase.
