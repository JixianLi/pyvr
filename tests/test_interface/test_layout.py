"""Tests for interface layout configuration."""

import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from pyvr.interface.matplotlib_interface import InteractiveVolumeRenderer
from pyvr.volume import Volume


@pytest.fixture
def mock_volume():
    """Create mock volume for testing."""
    data = np.random.rand(32, 32, 32).astype(np.float32)
    return Volume(data=data)


@pytest.fixture
def mock_renderer_context():
    """Mock OpenGL context and renderer components."""
    with patch('pyvr.interface.matplotlib_interface.VolumeRenderer') as mock_vr, \
         patch('matplotlib.pyplot.show'):  # Prevent window from opening

        # Mock renderer instance
        mock_renderer_instance = MagicMock()
        mock_renderer_instance.render_to_pil.return_value = MagicMock()
        mock_renderer_instance.get_light.return_value = MagicMock(is_linked=False)
        mock_vr.return_value = mock_renderer_instance

        yield mock_vr, mock_renderer_instance


def test_create_layout_returns_correct_structure(mock_volume, mock_renderer_context):
    """Test that _create_layout returns figure and axes dict."""
    interface = InteractiveVolumeRenderer(volume=mock_volume)

    fig, axes = interface._create_layout()

    # Should return figure and dict
    assert fig is not None
    assert isinstance(axes, dict)

    # Should have exactly 5 axes
    assert len(axes) == 5

    # Should have correct keys
    expected_keys = {'image', 'opacity', 'color', 'preset', 'info'}
    assert set(axes.keys()) == expected_keys


def test_create_layout_figure_size(mock_volume, mock_renderer_context):
    """Test that figure has correct size (16, 8)."""
    interface = InteractiveVolumeRenderer(volume=mock_volume)

    fig, axes = interface._create_layout()

    # Check figure size
    figsize = fig.get_size_inches()
    assert figsize[0] == 16.0  # Width
    assert figsize[1] == 8.0   # Height


def test_create_layout_gridspec_dimensions(mock_volume, mock_renderer_context):
    """Test that GridSpec has correct dimensions (3 rows, 3 columns)."""
    interface = InteractiveVolumeRenderer(volume=mock_volume)

    fig, axes = interface._create_layout()

    # Get GridSpec from first axes
    ax = axes['image']
    gs = ax.get_gridspec()

    # Check dimensions
    assert gs.nrows == 3
    assert gs.ncols == 3


def test_create_layout_width_ratios(mock_volume, mock_renderer_context):
    """Test that GridSpec has correct width ratios [2.5, 1.0, 0.8]."""
    interface = InteractiveVolumeRenderer(volume=mock_volume)

    fig, axes = interface._create_layout()

    # Get GridSpec from first axes
    ax = axes['image']
    gs = ax.get_gridspec()

    # Check width ratios
    expected_width_ratios = [2.5, 1.0, 0.8]
    actual_width_ratios = gs.get_width_ratios()

    assert len(actual_width_ratios) == len(expected_width_ratios)
    for actual, expected in zip(actual_width_ratios, expected_width_ratios):
        assert abs(actual - expected) < 0.001


def test_create_layout_height_ratios(mock_volume, mock_renderer_context):
    """Test that GridSpec has correct height ratios [3, 1, 1]."""
    interface = InteractiveVolumeRenderer(volume=mock_volume)

    fig, axes = interface._create_layout()

    # Get GridSpec from first axes
    ax = axes['image']
    gs = ax.get_gridspec()

    # Check height ratios
    expected_height_ratios = [3, 1, 1]
    actual_height_ratios = gs.get_height_ratios()

    assert len(actual_height_ratios) == len(expected_height_ratios)
    for actual, expected in zip(actual_height_ratios, expected_height_ratios):
        assert abs(actual - expected) < 0.001


def test_create_layout_axes_positions(mock_volume, mock_renderer_context):
    """Test that axes are positioned in correct grid cells."""
    interface = InteractiveVolumeRenderer(volume=mock_volume)

    fig, axes = interface._create_layout()

    # Get subplot specs for each axes
    image_spec = axes['image'].get_subplotspec()
    opacity_spec = axes['opacity'].get_subplotspec()
    color_spec = axes['color'].get_subplotspec()
    preset_spec = axes['preset'].get_subplotspec()
    info_spec = axes['info'].get_subplotspec()

    # Image should span all rows, column 0
    assert image_spec.rowspan.start == 0
    assert image_spec.rowspan.stop == 3
    assert image_spec.colspan.start == 0
    assert image_spec.colspan.stop == 1

    # Opacity should be row 0, column 1
    assert opacity_spec.rowspan.start == 0
    assert opacity_spec.rowspan.stop == 1
    assert opacity_spec.colspan.start == 1
    assert opacity_spec.colspan.stop == 2

    # Color should be row 1, column 1
    assert color_spec.rowspan.start == 1
    assert color_spec.rowspan.stop == 2
    assert color_spec.colspan.start == 1
    assert color_spec.colspan.stop == 2

    # Preset should be row 2, column 1
    assert preset_spec.rowspan.start == 2
    assert preset_spec.rowspan.stop == 3
    assert preset_spec.colspan.start == 1
    assert preset_spec.colspan.stop == 2

    # Info should span all rows, column 2
    assert info_spec.rowspan.start == 0
    assert info_spec.rowspan.stop == 3
    assert info_spec.colspan.start == 2
    assert info_spec.colspan.stop == 3


def test_create_layout_all_axes_valid(mock_volume, mock_renderer_context):
    """Test that all returned axes are valid matplotlib Axes objects."""
    interface = InteractiveVolumeRenderer(volume=mock_volume)

    fig, axes = interface._create_layout()

    # All axes should be matplotlib Axes instances
    import matplotlib.axes
    for key, ax in axes.items():
        assert isinstance(ax, matplotlib.axes.Axes), f"axes['{key}'] is not a valid Axes object"
