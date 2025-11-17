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
        volume_data = create_sample_volume(64, "sphere")
        volume = Volume(data=volume_data)
        interface = InteractiveVolumeRenderer(volume=volume)

        # Get full info text
        info_text = interface._get_full_info_text()

        # Should contain both controls and status
        assert "Mouse Controls" in info_text
        assert "Keyboard Shortcuts" in info_text
        assert "Current Status" in info_text
        assert "Preset:" in info_text

    def test_light_linking_no_error(self):
        """Test light linking doesn't cause errors."""
        volume_data = create_sample_volume(64, "sphere")
        volume = Volume(data=volume_data)
        interface = InteractiveVolumeRenderer(volume=volume)

        light = interface.renderer.get_light()
        light.link_to_camera()

        # Should not raise exception
        interface._update_display(force_render=True)

    def test_light_linking_uses_correct_camera_attribute(self):
        """Test light linking uses camera_controller.params."""
        volume_data = create_sample_volume(64, "sphere")
        volume = Volume(data=volume_data)
        interface = InteractiveVolumeRenderer(volume=volume)

        light = interface.renderer.get_light()
        light.link_to_camera()

        # Verify camera_controller has 'params' attribute
        assert hasattr(interface.camera_controller, "params")

        # Should be able to update from camera
        light.update_from_camera(interface.camera_controller.params)

    def test_matplotlib_keymaps_disabled(self):
        """Test matplotlib default keymaps are disabled."""
        volume_data = create_sample_volume(64, "sphere")
        volume = Volume(data=volume_data)
        interface = InteractiveVolumeRenderer(volume=volume)

        # Mock plt.show to avoid blocking
        with patch("matplotlib.pyplot.show"):
            interface.show()

        # Check keymaps are cleared
        assert mpl.rcParams["keymap.fullscreen"] == []
        assert mpl.rcParams["keymap.save"] == []

    def test_event_handlers_connected(self):
        """Test mouse/keyboard event handlers are connected."""
        volume_data = create_sample_volume(64, "sphere")
        volume = Volume(data=volume_data)
        interface = InteractiveVolumeRenderer(volume=volume)

        # Mock plt.show
        with patch("matplotlib.pyplot.show"):
            interface.show()

        # Verify event handlers are methods
        assert callable(interface._on_mouse_press)
        assert callable(interface._on_mouse_move)
        assert callable(interface._on_mouse_release)
        assert callable(interface._on_scroll)
        assert callable(interface._on_key_press)

    def test_light_linking_error_handling(self):
        """Test light linking handles errors gracefully."""
        volume_data = create_sample_volume(64, "sphere")
        volume = Volume(data=volume_data)
        interface = InteractiveVolumeRenderer(volume=volume)

        # Save original controller
        from pyvr.camera import CameraController

        original_controller = interface.camera_controller

        # Mock camera_controller to raise exception
        class FailingCamera:
            @property
            def params(self):
                raise RuntimeError("Camera error")

        interface.camera_controller = FailingCamera()

        light = interface.renderer.get_light()
        light.link_to_camera()

        # Should handle error without crashing
        interface._update_display(force_render=True)

        # Light should be automatically unlinked after error
        assert not light.is_linked

        # Restore for cleanup
        interface.camera_controller = original_controller

    def test_update_status_display_safe(self):
        """Test status display update handles missing attributes."""
        volume_data = create_sample_volume(64, "sphere")
        volume = Volume(data=volume_data)
        interface = InteractiveVolumeRenderer(volume=volume)

        # Should not raise even without info_text
        interface._update_status_display()

    def test_restore_matplotlib_keymaps(self):
        """Test matplotlib keymaps can be restored."""
        volume_data = create_sample_volume(64, "sphere")
        volume = Volume(data=volume_data)
        interface = InteractiveVolumeRenderer(volume=volume)

        # Save original
        original_fullscreen = mpl.rcParams["keymap.fullscreen"][:]

        # Mock plt.show
        with patch("matplotlib.pyplot.show"):
            interface.show()

        # Should be cleared
        assert mpl.rcParams["keymap.fullscreen"] == []

        # Restore
        interface._restore_matplotlib_keymaps()

        # Should be restored
        assert mpl.rcParams["keymap.fullscreen"] == original_fullscreen
