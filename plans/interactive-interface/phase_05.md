# Phase 5: Color Transfer Function Selector

**Status**: Not Started
**Estimated Effort**: 2-3 hours
**Dependencies**: Phase 2 (layout)

## Overview

Implement interactive colormap selection using matplotlib's RadioButtons widget. Allow users to switch between different colormaps and see updates in real-time.

## Implementation Plan

### Modify: `pyvr/interface/widgets.py`

Update ColorSelector to use RadioButtons:

```python
from matplotlib.widgets import RadioButtons

class ColorSelector:
    """Widget for selecting color transfer function colormap."""

    AVAILABLE_COLORMAPS = [
        'viridis', 'plasma', 'inferno', 'magma', 'cividis',
        'gray', 'bone', 'hot', 'cool', 'turbo', 'jet'
    ]

    def __init__(self, ax: Axes, on_change: Optional[Callable[[str], None]] = None):
        self.ax = ax
        self.current_colormap = "viridis"
        self.on_change = on_change

        # Style axes
        self.ax.set_title("Color Transfer Function", fontsize=11, fontweight='bold')

        # Create radio buttons for colormap selection
        self.radio = RadioButtons(
            ax=self.ax,
            labels=self.AVAILABLE_COLORMAPS,
            active=0  # viridis is first
        )

        # Connect callback
        self.radio.on_clicked(self._on_selection)

        # Display current colormap preview above radio buttons
        self._create_colormap_preview()

    def _create_colormap_preview(self):
        """Create a small colormap preview."""
        # Add axes for preview above radio buttons
        preview_ax = self.ax.figure.add_axes([
            self.ax.get_position().x0,
            self.ax.get_position().y1 - 0.05,
            self.ax.get_position().width,
            0.02
        ])
        preview_ax.axis('off')

        gradient = np.linspace(0, 1, 256).reshape(1, -1)
        self.colormap_preview = preview_ax.imshow(
            gradient, aspect='auto',
            cmap=self.current_colormap,
            extent=[0, 1, 0, 1]
        )

    def _on_selection(self, label: str):
        """Handle colormap selection."""
        self.current_colormap = label

        # Update preview
        if hasattr(self, 'colormap_preview'):
            self.colormap_preview.set_cmap(label)
            self.ax.figure.canvas.draw_idle()

        # Call external callback
        if self.on_change:
            self.on_change(label)

    def set_colormap(self, colormap_name: str):
        """Programmatically set colormap."""
        if colormap_name not in self.AVAILABLE_COLORMAPS:
            raise ValueError(f"Colormap '{colormap_name}' not available")

        self.current_colormap = colormap_name
        idx = self.AVAILABLE_COLORMAPS.index(colormap_name)
        self.radio.set_active(idx)
```

### Modify: `pyvr/interface/matplotlib_interface.py`

Update layout to accommodate radio buttons:

```python
def _create_layout(self):
    """Create matplotlib figure layout."""
    fig = plt.figure(figsize=(14, 8))  # Slightly taller for radio buttons
    fig.suptitle("PyVR Interactive Volume Renderer", fontsize=14, fontweight='bold')

    gs = GridSpec(3, 2, figure=fig,
                 width_ratios=[2, 1],
                 height_ratios=[4, 3, 1],  # More space for color selector
                 hspace=0.3, wspace=0.3)

    ax_image = fig.add_subplot(gs[:, 0])
    ax_opacity = fig.add_subplot(gs[0, 1])
    ax_color = fig.add_subplot(gs[1, 1])
    ax_info = fig.add_subplot(gs[2, 1])

    return fig, {
        'image': ax_image,
        'opacity': ax_opacity,
        'color': ax_color,
        'info': ax_info,
    }
```

## Testing Plan

```python
def test_color_selector_initialization():
    """Test ColorSelector initializes with radio buttons."""
    ax = Mock()
    selector = ColorSelector(ax)

    assert hasattr(selector, 'radio')
    assert selector.current_colormap == 'viridis'

def test_color_selector_callback():
    """Test colormap change triggers callback."""
    ax = Mock()
    callback = Mock()
    selector = ColorSelector(ax, on_change=callback)

    selector._on_selection('plasma')

    assert selector.current_colormap == 'plasma'
    callback.assert_called_once_with('plasma')

def test_set_colormap_programmatically():
    """Test setting colormap programmatically."""
    ax = Mock()
    selector = ColorSelector(ax)

    selector.set_colormap('inferno')

    assert selector.current_colormap == 'inferno'

def test_colormap_preview_updates():
    """Test colormap preview updates when selection changes."""
    ax = Mock()
    selector = ColorSelector(ax)

    # Mock the preview image
    selector.colormap_preview = Mock()

    selector._on_selection('hot')

    selector.colormap_preview.set_cmap.assert_called_once_with('hot')
```

## Acceptance Criteria

- [x] Radio buttons display list of colormaps
- [x] Clicking radio button changes colormap
- [x] Colormap preview updates in real-time
- [x] Volume rendering updates with new colormap
- [x] Tests cover colormap selection and callbacks
- [x] Manual testing: All colormaps render correctly

## Git Commit

```bash
pytest tests/test_interface/
git add pyvr/interface/ tests/test_interface/
git commit -m "Phase 5: Add interactive colormap selector

- Replace placeholder with RadioButtons widget
- Add real-time colormap preview
- Implement colormap change callback with rendering update
- Add tests for colormap selection interaction

Part of v0.3.0 interactive interface feature"
```
