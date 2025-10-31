"""Tests for preset selector widget."""

import pytest
from unittest.mock import MagicMock, Mock
from pyvr.interface.widgets import PresetSelector


@pytest.fixture
def mock_axes():
    """Create mock matplotlib axes."""
    ax = MagicMock()
    ax.figure = MagicMock()
    ax.figure.canvas = MagicMock()

    # Mock RadioButtons
    from matplotlib.widgets import RadioButtons
    mock_radio = MagicMock(spec=RadioButtons)
    mock_radio.labels = [MagicMock() for _ in range(5)]

    # Patch RadioButtons in the module
    import pyvr.interface.widgets
    original_radio = pyvr.interface.widgets.RadioButtons
    pyvr.interface.widgets.RadioButtons = Mock(return_value=mock_radio)

    yield ax

    # Restore
    pyvr.interface.widgets.RadioButtons = original_radio


class TestPresetSelector:
    """Tests for PresetSelector widget."""

    def test_initialization_default(self, mock_axes):
        """Test PresetSelector initializes with default preset."""
        selector = PresetSelector(mock_axes, initial_preset='fast')
        assert selector.current_preset == 'fast'
        assert selector.ax == mock_axes

    def test_initialization_custom_preset(self, mock_axes):
        """Test PresetSelector with custom initial preset."""
        selector = PresetSelector(mock_axes, initial_preset='high_quality')
        assert selector.current_preset == 'high_quality'

    def test_initialization_invalid_preset(self, mock_axes):
        """Test error on invalid preset."""
        with pytest.raises(ValueError, match="Invalid preset"):
            PresetSelector(mock_axes, initial_preset='invalid')

    def test_available_presets_list(self, mock_axes):
        """Test AVAILABLE_PRESETS contains expected presets."""
        selector = PresetSelector(mock_axes)
        expected = ['preview', 'fast', 'balanced', 'high_quality', 'ultra_quality']
        assert selector.AVAILABLE_PRESETS == expected

    def test_preset_labels_match_presets(self, mock_axes):
        """Test preset labels list matches presets."""
        selector = PresetSelector(mock_axes)
        assert len(selector.PRESET_LABELS) == len(selector.AVAILABLE_PRESETS)

    def test_set_preset(self, mock_axes):
        """Test programmatically setting preset."""
        selector = PresetSelector(mock_axes, initial_preset='fast')

        selector.set_preset('high_quality')
        assert selector.current_preset == 'high_quality'

    def test_set_preset_invalid(self, mock_axes):
        """Test error on invalid preset in set_preset."""
        selector = PresetSelector(mock_axes)

        with pytest.raises(ValueError, match="Invalid preset"):
            selector.set_preset('invalid')

    def test_get_preset(self, mock_axes):
        """Test getting current preset."""
        selector = PresetSelector(mock_axes, initial_preset='balanced')
        assert selector.get_preset() == 'balanced'

    def test_on_change_callback(self, mock_axes):
        """Test on_change callback is called."""
        callback = Mock()
        selector = PresetSelector(mock_axes, on_change=callback)

        # Simulate selection change
        selector._on_selection('Fast')

        callback.assert_called_once_with('fast')

    def test_on_selection_updates_current_preset(self, mock_axes):
        """Test _on_selection updates current_preset."""
        selector = PresetSelector(mock_axes, initial_preset='fast')

        selector._on_selection('High Quality')
        assert selector.current_preset == 'high_quality'

    def test_label_to_preset_mapping(self, mock_axes):
        """Test all labels map correctly to preset names."""
        selector = PresetSelector(mock_axes)

        label_preset_pairs = [
            ('Preview (fastest)', 'preview'),
            ('Fast', 'fast'),
            ('Balanced', 'balanced'),
            ('High Quality', 'high_quality'),
            ('Ultra (slowest)', 'ultra_quality'),
        ]

        for label, expected_preset in label_preset_pairs:
            selector._on_selection(label)
            assert selector.current_preset == expected_preset

    def test_radio_button_styled(self, mock_axes):
        """Test that radio button labels are styled."""
        selector = PresetSelector(mock_axes)
        # Verify radio buttons were created
        assert selector.radio is not None

    def test_on_change_not_called_if_none(self, mock_axes):
        """Test no error when on_change callback is None."""
        selector = PresetSelector(mock_axes, on_change=None)
        # Should not raise error
        selector._on_selection('Fast')
        assert selector.current_preset == 'fast'
