"""Tests for main InteractiveVolumeRenderer class."""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from pyvr.interface.matplotlib_interface import InteractiveVolumeRenderer
from pyvr.volume import Volume
from pyvr.datasets import create_sample_volume


@pytest.fixture
def small_volume():
    """Create a very small volume for fast testing."""
    data = create_sample_volume(32, 'sphere')
    return Volume(data=data)


@patch('pyvr.interface.matplotlib_interface.VolumeRenderer')
def test_interactive_renderer_initialization(mock_renderer_class, small_volume):
    """Test InteractiveVolumeRenderer initializes correctly."""
    # Mock the renderer instance
    mock_renderer = Mock()
    mock_renderer_class.return_value = mock_renderer

    interface = InteractiveVolumeRenderer(volume=small_volume)

    assert interface.volume == small_volume
    assert interface.width == 512
    assert interface.height == 512
    assert interface.state is not None
    assert len(interface.state.control_points) == 2  # Default control points


@patch('pyvr.interface.matplotlib_interface.VolumeRenderer')
def test_custom_dimensions(mock_renderer_class, small_volume):
    """Test custom width and height."""
    mock_renderer = Mock()
    mock_renderer_class.return_value = mock_renderer

    interface = InteractiveVolumeRenderer(volume=small_volume, width=1024, height=768)

    assert interface.width == 1024
    assert interface.height == 768
    mock_renderer_class.assert_called_once()
    call_kwargs = mock_renderer_class.call_args[1]
    assert call_kwargs['width'] == 1024
    assert call_kwargs['height'] == 768


@patch('pyvr.interface.matplotlib_interface.VolumeRenderer')
def test_update_transfer_functions(mock_renderer_class, small_volume):
    """Test transfer function updates."""
    mock_renderer = Mock()
    mock_renderer_class.return_value = mock_renderer

    interface = InteractiveVolumeRenderer(volume=small_volume)
    interface.state.add_control_point(0.5, 0.5)
    interface.state.needs_tf_update = True

    interface._update_transfer_functions()

    assert not interface.state.needs_tf_update
    mock_renderer.set_transfer_functions.assert_called()


@patch('pyvr.interface.matplotlib_interface.VolumeRenderer')
def test_render_volume(mock_renderer_class, small_volume):
    """Test volume rendering."""
    mock_renderer = Mock()
    mock_pil_image = Mock()
    mock_pil_image.__array__ = lambda **kwargs: np.zeros((512, 512, 3), dtype=np.uint8)
    mock_renderer.render_to_pil.return_value = mock_pil_image
    mock_renderer_class.return_value = mock_renderer

    interface = InteractiveVolumeRenderer(volume=small_volume)
    image_array = interface._render_volume()

    assert isinstance(image_array, np.ndarray)
    assert image_array.shape == (512, 512, 3)
    mock_renderer.render_to_pil.assert_called_once()


@patch('pyvr.interface.matplotlib_interface.VolumeRenderer')
@patch('pyvr.interface.matplotlib_interface.GridSpec')
@patch('pyvr.interface.matplotlib_interface.plt')
def test_show_creates_layout(mock_plt, mock_gridspec, mock_renderer_class, small_volume):
    """Test show() creates figure and layout."""
    mock_renderer = Mock()
    mock_renderer.render_to_pil.return_value = Mock()
    mock_renderer.render_to_pil.return_value.__array__ = lambda **kwargs: np.zeros((512, 512, 3), dtype=np.uint8)
    mock_renderer_class.return_value = mock_renderer

    # Mock figure and axes with proper plot() and scatter() returns
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_ax.plot.return_value = [MagicMock()]  # plot() returns list
    mock_ax.scatter.return_value = MagicMock()  # scatter() returns PathCollection
    mock_fig.add_subplot.return_value = mock_ax
    mock_plt.figure.return_value = mock_fig

    interface = InteractiveVolumeRenderer(volume=small_volume)
    interface.show()

    # Verify figure was created
    mock_plt.figure.assert_called_once()

    # Verify widgets were initialized
    assert interface.image_display is not None
    assert interface.opacity_editor is not None
    assert interface.color_selector is not None


@patch('pyvr.interface.matplotlib_interface.VolumeRenderer')
def test_colormap_change_callback(mock_renderer_class, small_volume):
    """Test colormap change callback."""
    mock_renderer = Mock()
    mock_renderer.render_to_pil.return_value = Mock()
    mock_renderer.render_to_pil.return_value.__array__ = lambda **kwargs: np.zeros((512, 512, 3), dtype=np.uint8)
    mock_renderer_class.return_value = mock_renderer

    interface = InteractiveVolumeRenderer(volume=small_volume)
    # Mock widgets so display update works
    interface.image_display = Mock()
    interface.opacity_editor = Mock()

    interface._on_colormap_change('plasma')

    assert interface.state.current_colormap == 'plasma'
    # After _update_display() runs, needs_render is reset to False
    assert not interface.state.needs_render
    # But the image should have been updated
    interface.image_display.update_image.assert_called_once()


@patch('pyvr.interface.matplotlib_interface.VolumeRenderer')
def test_manual_update(mock_renderer_class, small_volume):
    """Test manual update() method."""
    mock_renderer = Mock()
    mock_renderer.render_to_pil.return_value = Mock()
    mock_renderer.render_to_pil.return_value.__array__ = lambda **kwargs: np.zeros((512, 512, 3), dtype=np.uint8)
    mock_renderer_class.return_value = mock_renderer

    interface = InteractiveVolumeRenderer(volume=small_volume)
    interface.state.needs_render = True

    # Mock widgets
    interface.image_display = Mock()
    interface.opacity_editor = Mock()

    interface.update()

    # Verify update was called
    interface.image_display.update_image.assert_called_once()
    interface.opacity_editor.update_plot.assert_called_once()


@patch('pyvr.interface.matplotlib_interface.VolumeRenderer')
def test_render_caching(mock_renderer_class, small_volume):
    """Test that rendering is cached when state doesn't change."""
    mock_renderer = Mock()
    mock_pil_image = Mock()
    mock_pil_image.__array__ = lambda **kwargs: np.zeros((512, 512, 3), dtype=np.uint8)
    mock_renderer.render_to_pil.return_value = mock_pil_image
    mock_renderer_class.return_value = mock_renderer

    interface = InteractiveVolumeRenderer(volume=small_volume)

    # First render
    interface.state.needs_render = True
    img1 = interface._render_volume()

    # Second call with needs_render = False should return cached image
    interface.state.needs_render = False
    img2 = interface._render_volume()

    # Should be the same object (cached)
    assert img1 is img2
    # Renderer should only be called once
    assert mock_renderer.render_to_pil.call_count == 1


@patch('pyvr.interface.matplotlib_interface.VolumeRenderer')
def test_render_cache_invalidation(mock_renderer_class, small_volume):
    """Test that cache is invalidated when needs_render is True."""
    mock_renderer = Mock()
    mock_pil_image = Mock()
    mock_pil_image.__array__ = lambda **kwargs: np.zeros((512, 512, 3), dtype=np.uint8)
    mock_renderer.render_to_pil.return_value = mock_pil_image
    mock_renderer_class.return_value = mock_renderer

    interface = InteractiveVolumeRenderer(volume=small_volume)

    # First render
    interface.state.needs_render = True
    img1 = interface._render_volume()

    # Second render with needs_render = True should re-render
    interface.state.needs_render = True
    img2 = interface._render_volume()

    # Should be different objects (re-rendered) - np.array() creates new objects
    assert img1 is not img2
    # Renderer should be called twice
    assert mock_renderer.render_to_pil.call_count == 2


@patch('pyvr.interface.matplotlib_interface.VolumeRenderer')
def test_render_error_handling(mock_renderer_class, small_volume):
    """Test rendering handles errors gracefully."""
    mock_renderer = Mock()
    mock_renderer.render_to_pil.side_effect = Exception("OpenGL error")
    mock_renderer_class.return_value = mock_renderer

    interface = InteractiveVolumeRenderer(volume=small_volume)

    # Should not crash, returns placeholder
    img = interface._render_volume()

    assert isinstance(img, np.ndarray)
    assert img.shape == (512, 512, 3)
    # Placeholder is all zeros
    assert np.all(img == 0)


@patch('pyvr.interface.matplotlib_interface.VolumeRenderer')
def test_mouse_press_starts_camera_drag(mock_renderer_class, small_volume):
    """Test left-click in image display starts camera drag."""
    mock_renderer = Mock()
    mock_renderer_class.return_value = mock_renderer

    interface = InteractiveVolumeRenderer(volume=small_volume)
    interface.image_display = Mock()

    event = Mock(inaxes=interface.image_display.ax, button=1, xdata=100, ydata=100)
    interface._on_mouse_press(event)

    assert interface.state.is_dragging_camera
    assert interface.state.drag_start_pos == (100, 100)


@patch('pyvr.interface.matplotlib_interface.VolumeRenderer')
def test_mouse_press_outside_image_ignored(mock_renderer_class, small_volume):
    """Test click outside image display is ignored."""
    mock_renderer = Mock()
    mock_renderer_class.return_value = mock_renderer

    interface = InteractiveVolumeRenderer(volume=small_volume)
    interface.image_display = Mock()

    event = Mock(inaxes=None, button=1, xdata=100, ydata=100)
    interface._on_mouse_press(event)

    assert not interface.state.is_dragging_camera
    assert interface.state.drag_start_pos is None


@patch('pyvr.interface.matplotlib_interface.VolumeRenderer')
def test_mouse_release_ends_camera_drag(mock_renderer_class, small_volume):
    """Test mouse release ends camera drag and triggers render."""
    mock_renderer = Mock()
    mock_renderer.render_to_pil.return_value = Mock()
    mock_renderer.render_to_pil.return_value.__array__ = lambda **kwargs: np.zeros((512, 512, 3), dtype=np.uint8)
    mock_renderer_class.return_value = mock_renderer

    interface = InteractiveVolumeRenderer(volume=small_volume)
    interface.image_display = Mock()
    interface.opacity_editor = Mock()
    interface.state.is_dragging_camera = True
    interface.state.drag_start_pos = (100, 100)

    event = Mock()
    interface._on_mouse_release(event)

    assert not interface.state.is_dragging_camera
    assert interface.state.drag_start_pos is None
    # Should have triggered render
    interface.image_display.update_image.assert_called_once()


@patch('pyvr.interface.matplotlib_interface.VolumeRenderer')
def test_mouse_move_orbits_camera(mock_renderer_class, small_volume):
    """Test mouse movement orbits camera."""
    mock_renderer = Mock()
    mock_renderer_class.return_value = mock_renderer

    interface = InteractiveVolumeRenderer(volume=small_volume)
    interface.image_display = Mock()
    interface.state.is_dragging_camera = True
    interface.state.drag_start_pos = (100, 100)

    initial_azimuth = interface.camera_controller.params.azimuth

    event = Mock(inaxes=interface.image_display.ax, xdata=150, ydata=100)
    interface._on_mouse_move(event)

    # Azimuth should have changed
    assert interface.camera_controller.params.azimuth != initial_azimuth
    # Drag start position should be updated
    assert interface.state.drag_start_pos == (150, 100)


@patch('pyvr.interface.matplotlib_interface.VolumeRenderer')
def test_mouse_move_when_not_dragging_ignored(mock_renderer_class, small_volume):
    """Test mouse move when not dragging is ignored."""
    mock_renderer = Mock()
    mock_renderer_class.return_value = mock_renderer

    interface = InteractiveVolumeRenderer(volume=small_volume)
    interface.image_display = Mock()
    interface.state.is_dragging_camera = False

    initial_azimuth = interface.camera_controller.params.azimuth

    event = Mock(inaxes=interface.image_display.ax, xdata=150, ydata=100)
    interface._on_mouse_move(event)

    # Camera should not have changed
    assert interface.camera_controller.params.azimuth == initial_azimuth


@patch('pyvr.interface.matplotlib_interface.VolumeRenderer')
def test_mouse_move_outside_image_ignored(mock_renderer_class, small_volume):
    """Test mouse move outside image display is ignored."""
    mock_renderer = Mock()
    mock_renderer_class.return_value = mock_renderer

    interface = InteractiveVolumeRenderer(volume=small_volume)
    interface.image_display = Mock()
    interface.state.is_dragging_camera = True
    interface.state.drag_start_pos = (100, 100)

    initial_azimuth = interface.camera_controller.params.azimuth

    event = Mock(inaxes=None, xdata=None, ydata=None)
    interface._on_mouse_move(event)

    # Camera should not have changed
    assert interface.camera_controller.params.azimuth == initial_azimuth


@patch('pyvr.interface.matplotlib_interface.VolumeRenderer')
def test_scroll_zooms_camera(mock_renderer_class, small_volume):
    """Test scroll event zooms camera."""
    mock_renderer = Mock()
    mock_renderer.render_to_pil.return_value = Mock()
    mock_renderer.render_to_pil.return_value.__array__ = lambda **kwargs: np.zeros((512, 512, 3), dtype=np.uint8)
    mock_renderer_class.return_value = mock_renderer

    interface = InteractiveVolumeRenderer(volume=small_volume)
    interface.image_display = Mock()
    interface.opacity_editor = Mock()

    initial_distance = interface.camera_controller.params.distance

    # Scroll up (zoom in)
    event = Mock(inaxes=interface.image_display.ax, step=1)
    interface._on_scroll(event)

    assert interface.camera_controller.params.distance < initial_distance
    # Should have triggered render
    interface.image_display.update_image.assert_called_once()


@patch('pyvr.interface.matplotlib_interface.VolumeRenderer')
def test_scroll_zoom_out(mock_renderer_class, small_volume):
    """Test scroll down zooms out."""
    mock_renderer = Mock()
    mock_renderer.render_to_pil.return_value = Mock()
    mock_renderer.render_to_pil.return_value.__array__ = lambda **kwargs: np.zeros((512, 512, 3), dtype=np.uint8)
    mock_renderer_class.return_value = mock_renderer

    interface = InteractiveVolumeRenderer(volume=small_volume)
    interface.image_display = Mock()
    interface.opacity_editor = Mock()

    initial_distance = interface.camera_controller.params.distance

    # Scroll down (zoom out)
    event = Mock(inaxes=interface.image_display.ax, step=-1)
    interface._on_scroll(event)

    assert interface.camera_controller.params.distance > initial_distance


@patch('pyvr.interface.matplotlib_interface.VolumeRenderer')
def test_scroll_outside_image_ignored(mock_renderer_class, small_volume):
    """Test scroll outside image display is ignored."""
    mock_renderer = Mock()
    mock_renderer_class.return_value = mock_renderer

    interface = InteractiveVolumeRenderer(volume=small_volume)
    interface.image_display = Mock()

    initial_distance = interface.camera_controller.params.distance

    event = Mock(inaxes=None, step=1)
    interface._on_scroll(event)

    assert interface.camera_controller.params.distance == initial_distance


# ========== Opacity Editor Interaction Tests (Phase 6) ==========

@patch('pyvr.interface.matplotlib_interface.VolumeRenderer')
def test_add_control_point_on_click(mock_renderer_class, small_volume):
    """Test clicking empty space adds control point."""
    mock_renderer = Mock()
    mock_renderer.render_to_pil.return_value = Mock()
    mock_renderer.render_to_pil.return_value.__array__ = lambda **kwargs: np.zeros((512, 512, 3), dtype=np.uint8)
    mock_renderer_class.return_value = mock_renderer

    interface = InteractiveVolumeRenderer(volume=small_volume)
    interface.opacity_editor = Mock()
    interface.opacity_editor.ax = Mock()
    interface.image_display = Mock()

    initial_count = len(interface.state.control_points)

    # Click in empty space
    event = Mock(inaxes=interface.opacity_editor.ax, button=1,
                xdata=0.5, ydata=0.5)
    interface._on_mouse_press(event)

    assert len(interface.state.control_points) == initial_count + 1


@patch('pyvr.interface.matplotlib_interface.VolumeRenderer')
def test_select_control_point_on_click(mock_renderer_class, small_volume):
    """Test clicking near control point selects it."""
    mock_renderer = Mock()
    mock_renderer.render_to_pil.return_value = Mock()
    mock_renderer.render_to_pil.return_value.__array__ = lambda **kwargs: np.zeros((512, 512, 3), dtype=np.uint8)
    mock_renderer_class.return_value = mock_renderer

    interface = InteractiveVolumeRenderer(volume=small_volume)
    interface.state.add_control_point(0.5, 0.5)
    interface.opacity_editor = Mock()
    interface.opacity_editor.ax = Mock()
    interface.image_display = Mock()

    # Click near (0.5, 0.5)
    event = Mock(inaxes=interface.opacity_editor.ax, button=1,
                xdata=0.51, ydata=0.51)
    interface._on_mouse_press(event)

    assert interface.state.selected_control_point is not None
    assert interface.state.is_dragging_control_point


@patch('pyvr.interface.matplotlib_interface.VolumeRenderer')
def test_remove_control_point_on_right_click(mock_renderer_class, small_volume):
    """Test right-clicking control point removes it."""
    mock_renderer = Mock()
    mock_renderer.render_to_pil.return_value = Mock()
    mock_renderer.render_to_pil.return_value.__array__ = lambda **kwargs: np.zeros((512, 512, 3), dtype=np.uint8)
    mock_renderer_class.return_value = mock_renderer

    interface = InteractiveVolumeRenderer(volume=small_volume)
    interface.state.add_control_point(0.5, 0.5)
    interface.opacity_editor = Mock()
    interface.opacity_editor.ax = Mock()
    interface.image_display = Mock()

    # Right-click near (0.5, 0.5)
    event = Mock(inaxes=interface.opacity_editor.ax, button=3,
                xdata=0.5, ydata=0.5)
    interface._on_mouse_press(event)

    # Check that the point was removed
    assert (0.5, 0.5) not in interface.state.control_points


@patch('pyvr.interface.matplotlib_interface.VolumeRenderer')
def test_cannot_remove_first_last_on_right_click(mock_renderer_class, small_volume):
    """Test right-clicking first/last point doesn't remove it."""
    mock_renderer = Mock()
    mock_renderer.render_to_pil.return_value = Mock()
    mock_renderer.render_to_pil.return_value.__array__ = lambda **kwargs: np.zeros((512, 512, 3), dtype=np.uint8)
    mock_renderer_class.return_value = mock_renderer

    interface = InteractiveVolumeRenderer(volume=small_volume)
    interface.opacity_editor = Mock()
    interface.opacity_editor.ax = Mock()
    interface.image_display = Mock()

    initial_count = len(interface.state.control_points)

    # Right-click on first point (0.0, 0.0)
    event = Mock(inaxes=interface.opacity_editor.ax, button=3,
                xdata=0.0, ydata=0.0)
    interface._on_mouse_press(event)

    assert len(interface.state.control_points) == initial_count


@patch('pyvr.interface.matplotlib_interface.VolumeRenderer')
def test_drag_control_point(mock_renderer_class, small_volume):
    """Test dragging control point updates position."""
    mock_renderer = Mock()
    mock_renderer_class.return_value = mock_renderer

    interface = InteractiveVolumeRenderer(volume=small_volume)
    interface.state.add_control_point(0.5, 0.5)
    interface.opacity_editor = Mock()
    interface.opacity_editor.ax = Mock()
    interface.image_display = Mock()

    # Select and start dragging
    interface.state.select_control_point(1)  # Middle point
    interface.state.is_dragging_control_point = True

    # Drag to new position
    event = Mock(inaxes=interface.opacity_editor.ax,
                xdata=0.6, ydata=0.7)
    interface._on_mouse_move(event)

    # Check point moved
    cp = interface.state.control_points[1]
    assert cp[0] == pytest.approx(0.6)
    assert cp[1] == pytest.approx(0.7)


@patch('pyvr.interface.matplotlib_interface.VolumeRenderer')
def test_drag_first_point_locks_x(mock_renderer_class, small_volume):
    """Test dragging first control point only changes opacity."""
    mock_renderer = Mock()
    mock_renderer_class.return_value = mock_renderer

    interface = InteractiveVolumeRenderer(volume=small_volume)
    interface.opacity_editor = Mock()
    interface.opacity_editor.ax = Mock()
    interface.image_display = Mock()

    # Select first point
    interface.state.select_control_point(0)
    interface.state.is_dragging_control_point = True

    # Try to drag to new position
    event = Mock(inaxes=interface.opacity_editor.ax,
                xdata=0.5, ydata=0.3)
    interface._on_mouse_move(event)

    cp = interface.state.control_points[0]
    assert cp[0] == 0.0  # X locked
    assert cp[1] == 0.3  # Y changed


@patch('pyvr.interface.matplotlib_interface.VolumeRenderer')
def test_drag_last_point_locks_x(mock_renderer_class, small_volume):
    """Test dragging last control point only changes opacity."""
    mock_renderer = Mock()
    mock_renderer_class.return_value = mock_renderer

    interface = InteractiveVolumeRenderer(volume=small_volume)
    interface.opacity_editor = Mock()
    interface.opacity_editor.ax = Mock()
    interface.image_display = Mock()

    # Select last point
    last_index = len(interface.state.control_points) - 1
    interface.state.select_control_point(last_index)
    interface.state.is_dragging_control_point = True

    # Try to drag to new position
    event = Mock(inaxes=interface.opacity_editor.ax,
                xdata=0.5, ydata=0.8)
    interface._on_mouse_move(event)

    cp = interface.state.control_points[last_index]
    assert cp[0] == 1.0  # X locked
    assert cp[1] == 0.8  # Y changed


@patch('pyvr.interface.matplotlib_interface.VolumeRenderer')
def test_find_control_point_near(mock_renderer_class, small_volume):
    """Test finding control points near coordinates."""
    mock_renderer = Mock()
    mock_renderer_class.return_value = mock_renderer

    interface = InteractiveVolumeRenderer(volume=small_volume)
    interface.state.add_control_point(0.5, 0.5)

    # Find nearby point
    index = interface._find_control_point_near(0.51, 0.51, threshold=0.05)
    assert index == 1  # Should find middle point

    # Don't find far point
    index = interface._find_control_point_near(0.8, 0.8, threshold=0.05)
    assert index is None


@patch('pyvr.interface.matplotlib_interface.VolumeRenderer')
def test_control_point_coordinates_clamped(mock_renderer_class, small_volume):
    """Test control point coordinates are clamped to [0, 1]."""
    mock_renderer = Mock()
    mock_renderer_class.return_value = mock_renderer

    interface = InteractiveVolumeRenderer(volume=small_volume)
    interface.state.add_control_point(0.5, 0.5)
    interface.opacity_editor = Mock()
    interface.opacity_editor.ax = Mock()
    interface.image_display = Mock()

    interface.state.select_control_point(1)
    interface.state.is_dragging_control_point = True

    # Try to drag outside bounds
    event = Mock(inaxes=interface.opacity_editor.ax,
                xdata=1.5, ydata=-0.5)
    interface._on_mouse_move(event)

    cp = interface.state.control_points[1]
    assert 0.0 <= cp[0] <= 1.0
    assert 0.0 <= cp[1] <= 1.0
    assert cp[0] == 1.0  # Clamped to max
    assert cp[1] == 0.0  # Clamped to min


@patch('pyvr.interface.matplotlib_interface.VolumeRenderer')
def test_mouse_release_ends_control_point_drag(mock_renderer_class, small_volume):
    """Test mouse release ends control point dragging."""
    mock_renderer = Mock()
    mock_renderer.render_to_pil.return_value = Mock()
    mock_renderer.render_to_pil.return_value.__array__ = lambda **kwargs: np.zeros((512, 512, 3), dtype=np.uint8)
    mock_renderer_class.return_value = mock_renderer

    interface = InteractiveVolumeRenderer(volume=small_volume)
    interface.opacity_editor = Mock()
    interface.image_display = Mock()
    interface.state.is_dragging_control_point = True
    interface.state.drag_start_pos = (0.5, 0.5)

    event = Mock()
    interface._on_mouse_release(event)

    assert not interface.state.is_dragging_control_point
    assert interface.state.drag_start_pos is None
    # Should have triggered render
    interface.image_display.update_image.assert_called_once()


@patch('pyvr.interface.matplotlib_interface.VolumeRenderer')
def test_drag_control_point_outside_axes_ignored(mock_renderer_class, small_volume):
    """Test dragging control point outside axes is ignored."""
    mock_renderer = Mock()
    mock_renderer_class.return_value = mock_renderer

    interface = InteractiveVolumeRenderer(volume=small_volume)
    interface.state.add_control_point(0.5, 0.5)
    interface.opacity_editor = Mock()
    interface.opacity_editor.ax = Mock()
    interface.image_display = Mock()

    interface.state.select_control_point(1)
    interface.state.is_dragging_control_point = True

    initial_cp = interface.state.control_points[1]

    # Move outside axes
    event = Mock(inaxes=None, xdata=None, ydata=None)
    interface._on_mouse_move(event)

    # Control point should not have changed
    assert interface.state.control_points[1] == initial_cp


@patch('pyvr.interface.matplotlib_interface.VolumeRenderer')
def test_opacity_editor_updates_during_drag(mock_renderer_class, small_volume):
    """Test opacity editor updates during control point drag."""
    mock_renderer = Mock()
    mock_renderer_class.return_value = mock_renderer

    interface = InteractiveVolumeRenderer(volume=small_volume)
    interface.state.add_control_point(0.5, 0.5)
    interface.opacity_editor = Mock()
    interface.opacity_editor.ax = Mock()
    interface.image_display = Mock()

    interface.state.select_control_point(1)
    interface.state.is_dragging_control_point = True

    event = Mock(inaxes=interface.opacity_editor.ax,
                xdata=0.6, ydata=0.7)
    interface._on_mouse_move(event)

    # Opacity editor should have been updated
    interface.opacity_editor.update_plot.assert_called_once()


# ========== Event Coordination and Keyboard Shortcuts Tests (Phase 7) ==========

@patch('pyvr.interface.matplotlib_interface.VolumeRenderer')
def test_render_throttling(mock_renderer_class, small_volume):
    """Test rendering is throttled to prevent excessive updates."""
    import time
    mock_renderer = Mock()
    mock_renderer_class.return_value = mock_renderer

    interface = InteractiveVolumeRenderer(volume=small_volume)
    interface._last_render_time = time.time()

    # Should not render immediately
    assert not interface._should_render()

    # Should render after interval
    time.sleep(0.15)
    assert interface._should_render()


@patch('pyvr.interface.matplotlib_interface.VolumeRenderer')
def test_force_render_bypasses_throttling(mock_renderer_class, small_volume):
    """Test force_render bypasses throttling."""
    import time
    mock_renderer = Mock()
    mock_renderer.render_to_pil.return_value = Mock()
    mock_renderer.render_to_pil.return_value.__array__ = lambda **kwargs: np.zeros((512, 512, 3), dtype=np.uint8)
    mock_renderer_class.return_value = mock_renderer

    interface = InteractiveVolumeRenderer(volume=small_volume)
    interface.image_display = Mock()
    interface.opacity_editor = Mock()
    interface._last_render_time = time.time()
    interface.state.needs_render = True

    # Force render should work even with recent render
    interface._update_display(force_render=True)

    # Image should have been updated
    interface.image_display.update_image.assert_called()


@patch('pyvr.interface.matplotlib_interface.VolumeRenderer')
def test_keyboard_reset_view(mock_renderer_class, small_volume):
    """Test 'r' key resets camera view."""
    mock_renderer = Mock()
    mock_renderer.render_to_pil.return_value = Mock()
    mock_renderer.render_to_pil.return_value.__array__ = lambda **kwargs: np.zeros((512, 512, 3), dtype=np.uint8)
    mock_renderer_class.return_value = mock_renderer

    interface = InteractiveVolumeRenderer(volume=small_volume)
    interface.image_display = Mock()
    interface.opacity_editor = Mock()

    # Modify camera
    interface.camera_controller.orbit(delta_azimuth=1.0, delta_elevation=0.5)
    initial_distance = interface.camera_controller.params.distance

    # Reset
    event = Mock(key='r')
    interface._on_key_press(event)

    # Should be back to isometric view
    camera = interface.camera_controller.params
    assert camera.distance == pytest.approx(3.0)


@patch('pyvr.interface.matplotlib_interface.VolumeRenderer')
@patch('PIL.Image')
def test_keyboard_save_image(mock_image, mock_renderer_class, small_volume):
    """Test 's' key saves image."""
    mock_renderer = Mock()
    mock_renderer_class.return_value = mock_renderer

    interface = InteractiveVolumeRenderer(volume=small_volume)
    interface._cached_image = np.zeros((512, 512, 3), dtype=np.uint8)

    mock_img = Mock()
    mock_image.fromarray.return_value = mock_img

    event = Mock(key='s')
    interface._on_key_press(event)

    mock_img.save.assert_called_once()


@patch('pyvr.interface.matplotlib_interface.VolumeRenderer')
def test_keyboard_deselect(mock_renderer_class, small_volume):
    """Test Esc key deselects control point."""
    mock_renderer = Mock()
    mock_renderer_class.return_value = mock_renderer

    interface = InteractiveVolumeRenderer(volume=small_volume)
    interface.state.select_control_point(0)
    interface.opacity_editor = Mock()
    interface.image_display = Mock()

    event = Mock(key='escape')
    interface._on_key_press(event)

    assert interface.state.selected_control_point is None


@patch('pyvr.interface.matplotlib_interface.VolumeRenderer')
def test_keyboard_delete_selected(mock_renderer_class, small_volume):
    """Test Delete key removes selected control point."""
    mock_renderer = Mock()
    mock_renderer.render_to_pil.return_value = Mock()
    mock_renderer.render_to_pil.return_value.__array__ = lambda **kwargs: np.zeros((512, 512, 3), dtype=np.uint8)
    mock_renderer_class.return_value = mock_renderer

    interface = InteractiveVolumeRenderer(volume=small_volume)
    interface.state.add_control_point(0.5, 0.5)
    interface.state.select_control_point(1)
    interface.opacity_editor = Mock()
    interface.image_display = Mock()

    event = Mock(key='delete')
    interface._on_key_press(event)

    assert (0.5, 0.5) not in interface.state.control_points


@patch('pyvr.interface.matplotlib_interface.VolumeRenderer')
def test_keyboard_backspace_deletes_selected(mock_renderer_class, small_volume):
    """Test Backspace key also removes selected control point."""
    mock_renderer = Mock()
    mock_renderer.render_to_pil.return_value = Mock()
    mock_renderer.render_to_pil.return_value.__array__ = lambda **kwargs: np.zeros((512, 512, 3), dtype=np.uint8)
    mock_renderer_class.return_value = mock_renderer

    interface = InteractiveVolumeRenderer(volume=small_volume)
    interface.state.add_control_point(0.5, 0.5)
    interface.state.select_control_point(1)
    interface.opacity_editor = Mock()
    interface.image_display = Mock()

    event = Mock(key='backspace')
    interface._on_key_press(event)

    assert (0.5, 0.5) not in interface.state.control_points


@patch('pyvr.interface.matplotlib_interface.VolumeRenderer')
def test_event_conflict_resolution(mock_renderer_class, small_volume):
    """Test that events in different axes don't interfere."""
    mock_renderer = Mock()
    mock_renderer_class.return_value = mock_renderer

    interface = InteractiveVolumeRenderer(volume=small_volume)
    interface.image_display = Mock()
    interface.opacity_editor = Mock()

    # Click in image should not affect opacity editor
    event = Mock(inaxes=interface.image_display.ax, button=1, xdata=100, ydata=100)
    interface._on_mouse_press(event)

    assert interface.state.is_dragging_camera
    assert not interface.state.is_dragging_control_point


@patch('pyvr.interface.matplotlib_interface.VolumeRenderer')
def test_cursor_changes(mock_renderer_class, small_volume):
    """Test cursor changes based on axes."""
    mock_renderer = Mock()
    mock_renderer_class.return_value = mock_renderer

    interface = InteractiveVolumeRenderer(volume=small_volume)
    interface.image_display = Mock()
    interface.opacity_editor = Mock()
    interface.fig = Mock()
    interface.fig.canvas = Mock()

    # Test cursor in image display
    event = Mock(inaxes=interface.image_display.ax)
    interface._update_cursor(event)

    interface.fig.canvas.set_cursor.assert_called_with(1)  # Hand cursor

    # Test cursor in opacity editor
    event = Mock(inaxes=interface.opacity_editor.ax)
    interface._update_cursor(event)

    interface.fig.canvas.set_cursor.assert_called_with(2)  # Crosshair cursor


@patch('pyvr.interface.matplotlib_interface.VolumeRenderer')
def test_save_image_no_cache(mock_renderer_class, small_volume):
    """Test save image with no cached image."""
    mock_renderer = Mock()
    mock_renderer_class.return_value = mock_renderer

    interface = InteractiveVolumeRenderer(volume=small_volume)
    interface._cached_image = None

    # Should not crash, just print message
    interface._save_image()
