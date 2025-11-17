"""Integration tests for interactive interface."""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from pyvr.interface import InteractiveVolumeRenderer
from pyvr.volume import Volume
from pyvr.datasets import create_sample_volume, compute_normal_volume


@pytest.fixture
def full_interface():
    """Create a fully initialized interface for integration testing."""
    volume_data = create_sample_volume(64, "double_sphere")
    normals = compute_normal_volume(volume_data)
    volume = Volume(data=volume_data, normals=normals)

    with patch(
        "pyvr.interface.matplotlib_interface.VolumeRenderer"
    ) as mock_renderer_class:
        mock_renderer = Mock()
        mock_renderer.render_to_pil.return_value = Mock()
        mock_renderer.render_to_pil.return_value.__array__ = lambda **kwargs: np.zeros(
            (512, 512, 3), dtype=np.uint8
        )
        mock_renderer_class.return_value = mock_renderer

        interface = InteractiveVolumeRenderer(volume=volume)
        yield interface


def test_full_workflow_camera_and_transfer_function(full_interface):
    """Test complete workflow: camera movement + transfer function editing."""
    interface = full_interface
    interface.image_display = Mock()
    interface.opacity_editor = Mock()
    interface.opacity_editor.ax = Mock()

    # 1. Orbit camera
    event = Mock(inaxes=interface.image_display.ax, button=1, xdata=100, ydata=100)
    interface._on_mouse_press(event)
    assert interface.state.is_dragging_camera

    event = Mock(inaxes=interface.image_display.ax, xdata=150, ydata=100)
    interface._on_mouse_move(event)

    event = Mock()
    interface._on_mouse_release(event)
    assert not interface.state.is_dragging_camera

    # 2. Zoom camera
    event = Mock(inaxes=interface.image_display.ax, step=1)
    interface._on_scroll(event)

    # 3. Add control point
    event = Mock(inaxes=interface.opacity_editor.ax, button=1, xdata=0.5, ydata=0.5)
    interface._on_mouse_press(event)
    assert len(interface.state.control_points) == 3

    # 4. Change colormap
    interface._on_colormap_change("plasma")
    assert interface.state.current_colormap == "plasma"

    # Verify state is consistent
    assert interface.state.needs_render or interface.state.needs_tf_update


def test_multiple_control_point_operations(full_interface):
    """Test adding, selecting, moving, and removing control points."""
    interface = full_interface
    interface.opacity_editor = Mock()
    interface.opacity_editor.ax = Mock()
    interface.image_display = Mock()

    # Add three control points
    for x, y in [(0.3, 0.2), (0.5, 0.8), (0.7, 0.4)]:
        event = Mock(inaxes=interface.opacity_editor.ax, button=1, xdata=x, ydata=y)
        interface._on_mouse_press(event)

    assert len(interface.state.control_points) == 5  # 2 default + 3 added

    # Select and move one
    interface.state.select_control_point(2)
    interface.state.is_dragging_control_point = True
    event = Mock(inaxes=interface.opacity_editor.ax, xdata=0.6, ydata=0.9)
    interface._on_mouse_move(event)

    # Remove one (find a point and remove it)
    event = Mock(inaxes=interface.opacity_editor.ax, button=3, xdata=0.7, ydata=0.41)
    interface._on_mouse_press(event)

    assert len(interface.state.control_points) == 4


def test_error_recovery(full_interface):
    """Test interface handles errors gracefully."""
    interface = full_interface

    # Simulate rendering error
    interface.renderer.render_to_pil = Mock(side_effect=Exception("OpenGL error"))

    # Should not crash
    image = interface._render_volume()
    assert image is not None
    assert image.shape == (512, 512, 3)  # Placeholder image


def test_state_persistence_across_operations(full_interface):
    """Test state remains consistent across multiple operations."""
    interface = full_interface
    interface.image_display = Mock()
    interface.opacity_editor = Mock()
    interface.opacity_editor.ax = Mock()

    # Perform multiple operations
    interface._on_colormap_change("hot")
    interface.state.add_control_point(0.4, 0.6)

    event = Mock(inaxes=interface.image_display.ax, step=-1)
    interface._on_scroll(event)

    # State should be valid
    assert len(interface.state.control_points) >= 2
    assert interface.state.current_colormap == "hot"
    assert all(
        0 <= cp[0] <= 1 and 0 <= cp[1] <= 1 for cp in interface.state.control_points
    )


def test_keyboard_shortcuts_integration(full_interface):
    """Test keyboard shortcuts work in integrated workflow."""
    interface = full_interface
    interface.image_display = Mock()
    interface.opacity_editor = Mock()

    # Add a control point
    interface.state.add_control_point(0.5, 0.5)
    interface.state.select_control_point(1)

    # Delete with keyboard
    event = Mock(key="delete")
    interface._on_key_press(event)
    assert (0.5, 0.5) not in interface.state.control_points

    # Reset camera
    interface.camera_controller.orbit(delta_azimuth=1.0, delta_elevation=0.5)
    event = Mock(key="r")
    interface._on_key_press(event)
    # Camera should be reset


def test_concurrent_camera_and_opacity_editing(full_interface):
    """Test camera and opacity editing can be done in sequence without conflicts."""
    interface = full_interface
    interface.image_display = Mock()
    interface.opacity_editor = Mock()
    interface.opacity_editor.ax = Mock()

    # Start with camera orbit
    event = Mock(inaxes=interface.image_display.ax, button=1, xdata=100, ydata=100)
    interface._on_mouse_press(event)
    event = Mock(inaxes=interface.image_display.ax, xdata=150, ydata=100)
    interface._on_mouse_move(event)
    event = Mock()
    interface._on_mouse_release(event)

    # Switch to opacity editing
    event = Mock(inaxes=interface.opacity_editor.ax, button=1, xdata=0.5, ydata=0.5)
    interface._on_mouse_press(event)

    # Verify states are independent
    assert not interface.state.is_dragging_camera
    # New control point should have been added
    assert len(interface.state.control_points) == 3


def test_preset_change_triggers_rerender(full_interface):
    """Test changing preset triggers re-render."""
    interface = full_interface

    # Change preset
    interface.state.set_preset("high_quality")

    # Should flag for re-render
    assert interface.state.needs_render is True


def test_preset_updates_renderer_config(full_interface):
    """Test preset change updates renderer config."""
    interface = full_interface

    # Simulate preset change callback
    interface._on_preset_change("high_quality")

    # Renderer set_config should be called
    interface.renderer.set_config.assert_called_once()


def test_preset_change_prints_feedback(full_interface, capsys):
    """Test preset change prints feedback to console."""
    interface = full_interface

    # Change preset
    interface._on_preset_change("balanced")

    # Should print feedback
    captured = capsys.readouterr()
    assert "Switched to 'balanced' preset" in captured.out
    assert "samples/ray" in captured.out


def test_preset_integration_with_transfer_functions(full_interface):
    """Test preset changes work alongside transfer function edits."""
    interface = full_interface

    # Change colormap
    interface._on_colormap_change("plasma")
    assert interface.state.current_colormap == "plasma"

    # Change preset
    interface.state.set_preset("high_quality")
    assert interface.state.current_preset_name == "high_quality"

    # Both should trigger renders
    assert interface.state.needs_render
