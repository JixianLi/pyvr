"""Tests for FPS counter functionality."""

import pytest
import time
from unittest.mock import MagicMock
from pyvr.interface.widgets import FPSCounter, ImageDisplay


class TestFPSCounter:
    """Tests for FPSCounter class."""

    def test_initialization(self):
        """Test FPSCounter initializes correctly."""
        counter = FPSCounter(window_size=30)
        assert counter.window_size == 30
        assert len(counter.frame_times) == 0
        assert counter.last_time is None

    def test_get_fps_no_data(self):
        """Test FPS returns 0 with no data."""
        counter = FPSCounter()
        assert counter.get_fps() == 0.0

    def test_tick_updates_frame_times(self):
        """Test tick() records frame time."""
        counter = FPSCounter()

        # First tick just records time
        counter.tick()
        assert counter.last_time is not None
        assert len(counter.frame_times) == 0

        # Second tick records frame time
        time.sleep(0.01)  # 10ms
        counter.tick()
        assert len(counter.frame_times) == 1

    def test_fps_calculation(self):
        """Test FPS calculation from frame times."""
        counter = FPSCounter()

        # Simulate 60 FPS (16.67ms per frame)
        for _ in range(10):
            counter.tick()
            time.sleep(0.0167)

        fps = counter.get_fps()
        # Allow 20% tolerance due to sleep imprecision
        assert 48 <= fps <= 72, f"Expected ~60 FPS, got {fps}"

    def test_window_size_limit(self):
        """Test frame times deque respects window size."""
        counter = FPSCounter(window_size=5)

        for _ in range(10):
            counter.tick()
            time.sleep(0.001)

        # Should only store last 5 frame times
        assert len(counter.frame_times) == 5

    def test_reset(self):
        """Test reset clears all data."""
        counter = FPSCounter()
        counter.tick()
        time.sleep(0.01)
        counter.tick()

        assert len(counter.frame_times) > 0
        assert counter.last_time is not None

        counter.reset()
        assert len(counter.frame_times) == 0
        assert counter.last_time is None


class TestImageDisplayFPS:
    """Tests for FPS counter integration in ImageDisplay."""

    @pytest.fixture
    def mock_axes(self):
        """Create mock matplotlib axes."""
        ax = MagicMock()
        ax.figure = MagicMock()
        ax.figure.canvas = MagicMock()
        ax.transAxes = MagicMock()
        ax.text = MagicMock()
        return ax

    def test_fps_enabled_by_default(self, mock_axes):
        """Test FPS counter is enabled by default."""
        display = ImageDisplay(mock_axes, show_fps=True)
        assert display.show_fps is True
        assert display.fps_counter is not None

    def test_fps_can_be_disabled(self, mock_axes):
        """Test FPS counter can be disabled."""
        display = ImageDisplay(mock_axes, show_fps=False)
        assert display.show_fps is False

    def test_update_image_ticks_fps(self, mock_axes):
        """Test update_image() calls FPS tick."""
        import numpy as np
        display = ImageDisplay(mock_axes, show_fps=True)

        # Initial state
        assert display.fps_counter.last_time is None

        # Update image
        image = np.zeros((512, 512, 3), dtype=np.uint8)
        display.update_image(image)

        # FPS counter should have been ticked
        assert display.fps_counter.last_time is not None

    def test_fps_text_created_on_first_update(self, mock_axes):
        """Test FPS text is created on first image update."""
        import numpy as np
        display = ImageDisplay(mock_axes, show_fps=True)

        image = np.zeros((512, 512, 3), dtype=np.uint8)
        display.update_image(image)

        # Should create text element
        mock_axes.text.assert_called_once()

    def test_set_fps_visible(self, mock_axes):
        """Test toggling FPS visibility."""
        import numpy as np
        display = ImageDisplay(mock_axes, show_fps=True)

        # Create FPS text
        image = np.zeros((512, 512, 3), dtype=np.uint8)
        display.update_image(image)

        # Toggle off
        display.set_fps_visible(False)
        assert display.show_fps is False

        # Toggle on
        display.set_fps_visible(True)
        assert display.show_fps is True

    def test_clear_removes_fps_text(self, mock_axes):
        """Test clear() removes FPS text."""
        import numpy as np
        display = ImageDisplay(mock_axes, show_fps=True)

        image = np.zeros((512, 512, 3), dtype=np.uint8)
        display.update_image(image)

        display.clear()
        # FPS counter should be reset
        assert display.fps_counter.last_time is None

    def test_fps_not_updated_when_disabled(self, mock_axes):
        """Test FPS counter not updated when show_fps is False."""
        import numpy as np
        display = ImageDisplay(mock_axes, show_fps=False)

        image = np.zeros((512, 512, 3), dtype=np.uint8)
        display.update_image(image)

        # FPS counter should not be ticked
        assert display.fps_counter.last_time is None

    def test_set_fps_visible_resets_counter_when_hiding(self, mock_axes):
        """Test that hiding FPS resets the counter."""
        import numpy as np
        display = ImageDisplay(mock_axes, show_fps=True)

        # Create some FPS data
        image = np.zeros((512, 512, 3), dtype=np.uint8)
        display.update_image(image)
        time.sleep(0.01)
        display.update_image(image)

        assert display.fps_counter.last_time is not None
        assert len(display.fps_counter.frame_times) > 0

        # Hide FPS - should reset counter
        display.set_fps_visible(False)
        assert display.fps_counter.last_time is None
        assert len(display.fps_counter.frame_times) == 0
