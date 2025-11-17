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
    with patch("pyvr.interface.matplotlib_interface.VolumeRenderer"):
        interface = InteractiveVolumeRenderer(volume=mock_volume, width=512, height=512)
        # Mock the display widgets to avoid matplotlib backend issues
        interface.image_display = MagicMock()
        interface.opacity_editor = MagicMock()
        interface.fig = MagicMock()
        return interface


class TestInterfaceTrackballMode:
    """Tests for trackball mode in interface."""

    def test_default_mode_is_trackball(self, mock_interface):
        """New interface should default to trackball mode."""
        assert mock_interface.state.camera_control_mode == "trackball"

    def test_trackball_mode_drag_calls_trackball(self, mock_interface):
        """Dragging in trackball mode should call controller.trackball()."""
        # Set up drag state
        mock_interface.state.is_dragging_camera = True
        mock_interface.state.drag_start_pos = (100, 100)

        # Mock the trackball method
        with patch.object(
            mock_interface.camera_controller, "trackball"
        ) as mock_trackball:
            # Simulate mouse move
            event = MagicMock()
            event.inaxes = mock_interface.image_display.ax
            event.xdata = 150
            event.ydata = 120

            mock_interface._on_mouse_move(event)

            # Verify trackball was called
            mock_trackball.assert_called_once()
            call_args = mock_trackball.call_args
            assert call_args[1]["dx"] == 50  # 150 - 100
            assert call_args[1]["dy"] == 20  # 120 - 100
            assert call_args[1]["viewport_width"] == 512
            assert call_args[1]["viewport_height"] == 512

    def test_orbit_mode_drag_calls_orbit(self, mock_interface):
        """Dragging in orbit mode should call controller.orbit()."""
        # Switch to orbit mode
        mock_interface.state.camera_control_mode = "orbit"

        # Set up drag state
        mock_interface.state.is_dragging_camera = True
        mock_interface.state.drag_start_pos = (100, 100)

        # Mock the orbit method
        with patch.object(mock_interface.camera_controller, "orbit") as mock_orbit:
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
        assert mock_interface.state.camera_control_mode == "trackball"

        # Simulate 't' key press
        event = MagicMock()
        event.key = "t"
        mock_interface._on_key_press(event)

        # Should switch to orbit
        assert mock_interface.state.camera_control_mode == "orbit"

        # Press again
        mock_interface._on_key_press(event)

        # Should switch back to trackball
        assert mock_interface.state.camera_control_mode == "trackball"

    def test_camera_updates_after_trackball_drag(self, mock_interface):
        """Camera should update after trackball drag."""
        initial_azimuth = mock_interface.camera_controller.params.azimuth

        # Set up drag state
        mock_interface.state.is_dragging_camera = True
        mock_interface.state.drag_start_pos = (100, 100)
        mock_interface.state.camera_control_mode = "trackball"

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
        mock_interface.state.camera_control_mode = "orbit"
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
        for mode in ["trackball", "orbit"]:
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
        mock_interface.state.camera_control_mode = "trackball"
        info_text = mock_interface._get_full_info_text()
        assert "Control Mode: Trackball" in info_text

        # Orbit mode
        mock_interface.state.camera_control_mode = "orbit"
        info_text = mock_interface._get_full_info_text()
        assert "Control Mode: Orbit" in info_text


class TestControlModeToggle:
    """Tests for control mode toggling."""

    def test_toggle_prints_message(self, mock_interface, capsys):
        """Toggling should print informative message."""
        # Toggle to orbit
        event = MagicMock()
        event.key = "t"
        mock_interface._on_key_press(event)

        captured = capsys.readouterr()
        assert "orbit control" in captured.out.lower()

        # Toggle back to trackball
        mock_interface._on_key_press(event)

        captured = capsys.readouterr()
        assert "trackball control" in captured.out.lower()

    def test_toggle_updates_status_display(self, mock_interface):
        """Toggling should update status display."""
        mock_interface._update_status_display = MagicMock()

        # Toggle mode
        event = MagicMock()
        event.key = "t"
        mock_interface._on_key_press(event)

        # Verify status display was updated
        mock_interface._update_status_display.assert_called()


class TestBackwardCompatibility:
    """Tests for backward compatibility with orbit control."""

    def test_orbit_mode_maintains_functionality(self, mock_interface):
        """Orbit mode should work exactly as before."""
        mock_interface.state.camera_control_mode = "orbit"

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
        mock_interface.camera_controller.orbit(delta_azimuth=0.5, delta_elevation=0.2)

        # Should work regardless of interface mode
        assert mock_interface.camera_controller.params.azimuth != initial_azimuth
