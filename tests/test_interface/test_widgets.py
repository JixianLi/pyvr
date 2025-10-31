"""Tests for widget components."""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock

from pyvr.interface.widgets import ImageDisplay, OpacityEditor, ColorSelector


@pytest.fixture
def mock_axes():
    """Create mock matplotlib axes."""
    ax = MagicMock()
    ax.figure = MagicMock()
    ax.figure.canvas = MagicMock()
    ax.transAxes = MagicMock()

    # Mock plot() to return a list containing a line object
    mock_line = MagicMock()
    ax.plot.return_value = [mock_line]

    # Mock scatter() to return a PathCollection
    mock_scatter = MagicMock()
    ax.scatter.return_value = mock_scatter

    # Mock get_position() for ColorSelector preview axes
    mock_position = MagicMock()
    mock_position.x0 = 0.5
    mock_position.y1 = 0.9
    mock_position.width = 0.4
    ax.get_position.return_value = mock_position

    # Mock add_axes for preview axes
    mock_preview_ax = MagicMock()
    mock_preview_ax.imshow = MagicMock()
    ax.figure.add_axes.return_value = mock_preview_ax

    return ax


class TestImageDisplay:
    """Tests for ImageDisplay widget."""

    def test_initialization(self, mock_axes):
        """Test ImageDisplay initializes correctly."""
        display = ImageDisplay(mock_axes)
        assert display.ax == mock_axes
        assert display.image is None

    def test_update_image_first_time(self, mock_axes):
        """Test updating image for the first time."""
        display = ImageDisplay(mock_axes)
        image_array = np.zeros((512, 512, 3), dtype=np.uint8)

        display.update_image(image_array)

        assert display.image is not None
        mock_axes.imshow.assert_called_once()

    def test_update_image_subsequent(self, mock_axes):
        """Test updating image after first time."""
        display = ImageDisplay(mock_axes)
        image_array = np.zeros((512, 512, 3), dtype=np.uint8)

        # First update
        display.update_image(image_array)

        # Second update
        image_array2 = np.ones((512, 512, 3), dtype=np.uint8) * 255
        display.update_image(image_array2)

        display.image.set_data.assert_called_once()

    def test_update_image_invalid_channels(self, mock_axes):
        """Test error on invalid channel count."""
        display = ImageDisplay(mock_axes)
        image_array = np.zeros((512, 512, 2), dtype=np.uint8)  # Invalid: 2 channels

        with pytest.raises(ValueError, match="3 or 4 channels"):
            display.update_image(image_array)


class TestOpacityEditor:
    """Tests for OpacityEditor widget."""

    def test_initialization(self, mock_axes):
        """Test OpacityEditor initializes correctly."""
        editor = OpacityEditor(mock_axes)
        assert editor.ax == mock_axes
        assert editor.line is None
        assert editor.points is None

    def test_update_plot_first_time(self, mock_axes):
        """Test updating plot for the first time."""
        editor = OpacityEditor(mock_axes)
        control_points = [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)]

        editor.update_plot(control_points)

        assert editor.line is not None
        assert editor.points is not None

    def test_update_plot_with_selection(self, mock_axes):
        """Test updating plot with selected control point."""
        editor = OpacityEditor(mock_axes)
        control_points = [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)]

        editor.update_plot(control_points, selected_index=1)

        # Selected point should be rendered differently
        assert editor.points is not None

    def test_update_plot_empty_list(self, mock_axes):
        """Test updating with empty control points list."""
        editor = OpacityEditor(mock_axes)
        editor.update_plot([])

        # Should not crash, line and points should remain None
        assert editor.line is None
        assert editor.points is None


class TestColorSelector:
    """Tests for ColorSelector widget."""

    def test_initialization(self, mock_axes):
        """Test ColorSelector initializes correctly."""
        selector = ColorSelector(mock_axes)
        assert selector.current_colormap == "viridis"
        assert selector.on_change is None
        assert hasattr(selector, 'radio')
        assert hasattr(selector, 'colormap_preview')

    def test_initialization_with_callback(self, mock_axes):
        """Test initialization with callback."""
        callback = Mock()
        selector = ColorSelector(mock_axes, on_change=callback)
        assert selector.on_change == callback

    def test_set_colormap(self, mock_axes):
        """Test setting colormap programmatically."""
        callback = Mock()
        selector = ColorSelector(mock_axes, on_change=callback)

        # Mock the radio.set_active method to avoid matplotlib internals
        selector.radio.set_active = MagicMock()

        selector.set_colormap('plasma')

        assert selector.current_colormap == 'plasma'
        # Verify set_active was called with correct index
        expected_index = ColorSelector.AVAILABLE_COLORMAPS.index('plasma')
        selector.radio.set_active.assert_called_once_with(expected_index)

    def test_set_invalid_colormap(self, mock_axes):
        """Test setting invalid colormap."""
        selector = ColorSelector(mock_axes)

        with pytest.raises(ValueError, match="not available"):
            selector.set_colormap('invalid_colormap')

    def test_available_colormaps(self):
        """Test that AVAILABLE_COLORMAPS is properly defined."""
        assert len(ColorSelector.AVAILABLE_COLORMAPS) > 0
        assert 'viridis' in ColorSelector.AVAILABLE_COLORMAPS
        assert 'plasma' in ColorSelector.AVAILABLE_COLORMAPS

    def test_on_selection_callback(self, mock_axes):
        """Test colormap selection triggers callback."""
        callback = Mock()
        selector = ColorSelector(mock_axes, on_change=callback)

        selector._on_selection('inferno')

        assert selector.current_colormap == 'inferno'
        callback.assert_called_once_with('inferno')

    def test_colormap_preview_updates(self, mock_axes):
        """Test colormap preview updates when selection changes."""
        selector = ColorSelector(mock_axes)

        # Mock the preview image
        selector.colormap_preview = MagicMock()

        selector._on_selection('hot')

        assert selector.current_colormap == 'hot'
        selector.colormap_preview.set_cmap.assert_called_once_with('hot')

    def test_radio_buttons_created(self, mock_axes):
        """Test that RadioButtons widget is created."""
        selector = ColorSelector(mock_axes)

        assert selector.radio is not None
        # Radio has on_clicked method for connecting callbacks
        assert hasattr(selector.radio, 'on_clicked')

    def test_set_colormap_updates_radio(self, mock_axes):
        """Test programmatic colormap change updates radio buttons."""
        selector = ColorSelector(mock_axes)

        # Mock the radio widget
        selector.radio.set_active = MagicMock()

        selector.set_colormap('inferno')

        # inferno is at index 2 in AVAILABLE_COLORMAPS
        expected_index = ColorSelector.AVAILABLE_COLORMAPS.index('inferno')
        selector.radio.set_active.assert_called_once_with(expected_index)
