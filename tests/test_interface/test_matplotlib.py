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
