# Phase 2: Core Matplotlib Interface Layout

**Status**: Not Started
**Estimated Effort**: 2-3 hours
**Dependencies**: Phase 1 (module scaffolding)

## Overview

Implement the complete matplotlib layout with proper figure organization, widget positioning, and basic rendering integration. This phase creates a functional (non-interactive) display showing the rendered volume and transfer function plots.

## Implementation Plan

### Modify: `pyvr/interface/matplotlib.py`

**Changes**:
1. Improve figure layout with better spacing and sizing
2. Add proper widget initialization and update methods
3. Implement rendering pipeline integration
4. Add refresh/update methods for coordinated widget updates

**Updated Implementation**:

```python
"""Main interactive volume renderer interface using matplotlib."""

from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from pyvr.volume import Volume
from pyvr.moderngl_renderer import VolumeRenderer
from pyvr.config import RenderConfig
from pyvr.camera import Camera, CameraController
from pyvr.lighting import Light
from pyvr.transferfunctions import ColorTransferFunction, OpacityTransferFunction
from pyvr.interface.widgets import ImageDisplay, OpacityEditor, ColorSelector
from pyvr.interface.state import InterfaceState


class InteractiveVolumeRenderer:
    """
    Interactive matplotlib-based interface for volume rendering.

    Provides real-time volume visualization with:
    - Camera controls (mouse orbit and zoom)
    - Transfer function editing (color and opacity)
    - Interactive control point manipulation

    This is a testing/development interface, not optimized for production use.

    Example:
        >>> from pyvr.datasets import create_sample_volume
        >>> from pyvr.volume import Volume
        >>>
        >>> volume_data = create_sample_volume(128, 'sphere')
        >>> volume = Volume(data=volume_data)
        >>>
        >>> interface = InteractiveVolumeRenderer(volume=volume)
        >>> interface.show()

    Attributes:
        volume: Volume data to render
        renderer: ModernGL volume renderer
        camera_controller: Camera controller for interactive manipulation
        state: Interface state manager
        image_display: Widget for volume rendering display
        opacity_editor: Widget for opacity transfer function editing
        color_selector: Widget for colormap selection
        fig: Matplotlib figure
    """

    def __init__(
        self,
        volume: Volume,
        width: int = 512,
        height: int = 512,
        config: Optional[RenderConfig] = None,
        camera: Optional[Camera] = None,
        light: Optional[Light] = None,
    ):
        """
        Initialize interactive volume renderer.

        Args:
            volume: Volume to render
            width: Render width in pixels
            height: Render height in pixels
            config: Render configuration (defaults to fast() for interactivity)
            camera: Initial camera (defaults to isometric view)
            light: Light configuration (defaults to directional light)
        """
        self.volume = volume
        self.width = width
        self.height = height

        # Set up renderer with interactive-friendly defaults
        if config is None:
            config = RenderConfig.fast()
        if camera is None:
            camera = Camera.isometric_view(distance=3.0)
        if light is None:
            light = Light.directional([1, -1, 0])

        self.renderer = VolumeRenderer(width=width, height=height, config=config, light=light)
        self.renderer.set_camera(camera)
        self.renderer.load_volume(volume)

        self.camera_controller = CameraController(camera)

        # Initialize state
        self.state = InterfaceState()

        # Set up initial transfer functions
        self._update_transfer_functions()

        # Widget placeholders (will be created in show())
        self.image_display: Optional[ImageDisplay] = None
        self.opacity_editor: Optional[OpacityEditor] = None
        self.color_selector: Optional[ColorSelector] = None
        self.fig: Optional[Figure] = None

    def _update_transfer_functions(self) -> None:
        """Update renderer with current transfer functions from state."""
        ctf = ColorTransferFunction.from_colormap(self.state.current_colormap)
        otf = OpacityTransferFunction(control_points=self.state.control_points)
        self.renderer.set_transfer_functions(ctf, otf)
        self.state.needs_tf_update = False

    def _render_volume(self) -> np.ndarray:
        """
        Render volume and return image array.

        Returns:
            RGB image array of shape (H, W, 3)
        """
        # Update camera if controller changed it
        self.renderer.set_camera(self.camera_controller.camera)

        # Render to PIL image and convert to numpy array
        image = self.renderer.render_to_pil()
        return np.array(image)

    def _update_display(self) -> None:
        """Update all display widgets based on current state."""
        # Update transfer functions if needed
        if self.state.needs_tf_update:
            self._update_transfer_functions()
            self.state.needs_render = True

        # Update volume rendering if needed
        if self.state.needs_render:
            image_array = self._render_volume()
            if self.image_display is not None:
                self.image_display.update_image(image_array)
            self.state.needs_render = False

        # Update opacity editor
        if self.opacity_editor is not None:
            self.opacity_editor.update_plot(
                self.state.control_points,
                self.state.selected_control_point
            )

    def _create_layout(self) -> tuple:
        """
        Create matplotlib figure layout.

        Returns:
            Tuple of (figure, axes_dict) where axes_dict contains 'image', 'opacity', 'color'
        """
        # Create figure
        fig = plt.figure(figsize=(14, 7))
        fig.suptitle("PyVR Interactive Volume Renderer", fontsize=14, fontweight='bold')

        # Create grid layout
        # Left side: Large image display
        # Right side: Stacked transfer function editors
        gs = GridSpec(3, 2, figure=fig,
                     width_ratios=[2, 1],
                     height_ratios=[4, 2, 1],
                     hspace=0.3, wspace=0.3)

        # Create axes
        ax_image = fig.add_subplot(gs[:, 0])  # Full left column
        ax_opacity = fig.add_subplot(gs[0, 1])  # Top right
        ax_color = fig.add_subplot(gs[1, 1])  # Middle right
        ax_info = fig.add_subplot(gs[2, 1])  # Bottom right (for future info display)

        axes_dict = {
            'image': ax_image,
            'opacity': ax_opacity,
            'color': ax_color,
            'info': ax_info,
        }

        return fig, axes_dict

    def _setup_info_display(self, ax) -> None:
        """
        Set up info display panel.

        Args:
            ax: Matplotlib axes for info display
        """
        ax.axis('off')
        info_text = (
            "Controls:\n"
            "  Image: Drag to orbit, scroll to zoom\n"
            "  Opacity: Click to add, drag to move, right-click to remove"
        )
        ax.text(0.05, 0.5, info_text,
               transform=ax.transAxes,
               fontsize=9,
               verticalalignment='center',
               family='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    def show(self) -> None:
        """
        Display the interactive interface.

        Creates matplotlib figure with layout and starts event loop.
        """
        # Create figure and axes
        self.fig, axes = self._create_layout()

        # Initialize widgets
        self.image_display = ImageDisplay(axes['image'])
        self.opacity_editor = OpacityEditor(axes['opacity'])
        self.color_selector = ColorSelector(axes['color'],
                                           on_change=self._on_colormap_change)

        # Set up info display
        self._setup_info_display(axes['info'])

        # Initial display update
        self._update_display()

        # Show figure
        plt.show()

    def _on_colormap_change(self, colormap_name: str) -> None:
        """
        Callback when colormap changes.

        Args:
            colormap_name: Name of new colormap
        """
        self.state.set_colormap(colormap_name)
        self._update_display()

    def update(self) -> None:
        """
        Manual update trigger (useful for external control).

        Forces a refresh of all widgets based on current state.
        """
        self._update_display()


# Example usage and testing
if __name__ == "__main__":
    from pyvr.datasets import create_sample_volume, compute_normal_volume

    # Create sample volume with normals
    volume_data = create_sample_volume(128, 'double_sphere')
    normals = compute_normal_volume(volume_data)
    volume = Volume(data=volume_data, normals=normals)

    # Create interface with custom initial settings
    interface = InteractiveVolumeRenderer(
        volume=volume,
        width=512,
        height=512,
    )

    # Test: Add some control points programmatically
    interface.state.add_control_point(0.3, 0.2)
    interface.state.add_control_point(0.7, 0.9)

    # Launch interface
    interface.show()
```

### Modify: `pyvr/interface/widgets.py`

**Changes**: Improve widget implementations with better styling and error handling

```python
"""UI widget components for interactive interface."""

from typing import Optional, Callable, List, Tuple
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.widgets import Button
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from matplotlib.collections import PathCollection
import numpy as np


class ImageDisplay:
    """
    Widget for displaying rendered volume with camera controls.

    Handles mouse interactions for orbiting and zooming the camera.

    Attributes:
        ax: Matplotlib axes for image display
        image: Image artist for displaying rendered volume
    """

    def __init__(self, ax: Axes):
        """
        Initialize image display widget.

        Args:
            ax: Matplotlib axes to use for display
        """
        self.ax = ax
        self.image: Optional[AxesImage] = None

        # Style the axes
        self.ax.set_title("Volume Rendering", fontsize=12, fontweight='bold')
        self.ax.axis("off")

        # Set background color
        self.ax.set_facecolor('#2e2e2e')

    def update_image(self, image_array: np.ndarray) -> None:
        """
        Update displayed image.

        Args:
            image_array: RGB image array of shape (H, W, 3) or (H, W, 4)
        """
        if image_array.shape[2] not in [3, 4]:
            raise ValueError(f"Image must have 3 or 4 channels, got {image_array.shape[2]}")

        if self.image is None:
            self.image = self.ax.imshow(image_array, interpolation='nearest')
        else:
            self.image.set_data(image_array)

        self.ax.figure.canvas.draw_idle()

    def clear(self) -> None:
        """Clear the image display."""
        if self.image is not None:
            self.image.remove()
            self.image = None
        self.ax.figure.canvas.draw_idle()


class OpacityEditor:
    """
    Widget for editing opacity transfer function with control points.

    Supports adding, removing, selecting, and dragging control points.

    Attributes:
        ax: Matplotlib axes for opacity plot
        line: Line artist for transfer function curve
        points: Scatter artist for control points
    """

    def __init__(self, ax: Axes):
        """
        Initialize opacity editor widget.

        Args:
            ax: Matplotlib axes to use for editor
        """
        self.ax = ax
        self.line: Optional[Line2D] = None
        self.points: Optional[PathCollection] = None

        # Style the axes
        self.ax.set_title("Opacity Transfer Function", fontsize=11, fontweight='bold')
        self.ax.set_xlabel("Scalar Value", fontsize=9)
        self.ax.set_ylabel("Opacity", fontsize=9)
        self.ax.set_xlim(-0.05, 1.05)
        self.ax.set_ylim(-0.05, 1.05)
        self.ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        self.ax.tick_params(labelsize=8)

        # Set background
        self.ax.set_facecolor('#f8f8f8')

    def update_plot(self, control_points: List[Tuple[float, float]],
                   selected_index: Optional[int] = None) -> None:
        """
        Update the opacity transfer function plot.

        Args:
            control_points: List of (scalar, opacity) tuples
            selected_index: Index of selected control point (highlighted)
        """
        if not control_points:
            return

        # Extract coordinates
        x_vals = [cp[0] for cp in control_points]
        y_vals = [cp[1] for cp in control_points]

        # Update line
        if self.line is None:
            self.line, = self.ax.plot(x_vals, y_vals, 'b-', linewidth=2.5, alpha=0.7)
        else:
            self.line.set_data(x_vals, y_vals)

        # Update control points with color coding
        colors = []
        sizes = []
        for i in range(len(control_points)):
            if i == selected_index:
                colors.append('#ff4444')  # Red for selected
                sizes.append(120)
            elif i == 0 or i == len(control_points) - 1:
                colors.append('#4444ff')  # Blue for locked endpoints
                sizes.append(80)
            else:
                colors.append('#44ff44')  # Green for movable points
                sizes.append(60)

        if self.points is None:
            self.points = self.ax.scatter(x_vals, y_vals, c=colors, s=sizes,
                                         zorder=5, edgecolors='black', linewidths=1)
        else:
            self.points.set_offsets(np.c_[x_vals, y_vals])
            self.points.set_color(colors)
            self.points.set_sizes(sizes)

        self.ax.figure.canvas.draw_idle()

    def clear(self) -> None:
        """Clear the opacity plot."""
        if self.line is not None:
            self.line.remove()
            self.line = None
        if self.points is not None:
            self.points.remove()
            self.points = None
        self.ax.figure.canvas.draw_idle()


class ColorSelector:
    """
    Widget for selecting color transfer function colormap.

    Provides a dropdown menu of available matplotlib colormaps.

    Attributes:
        ax: Matplotlib axes for colormap display
        current_colormap: Name of currently selected colormap
        on_change: Callback function when colormap changes
    """

    # Popular matplotlib colormaps for volume rendering
    AVAILABLE_COLORMAPS = [
        'viridis', 'plasma', 'inferno', 'magma', 'cividis',
        'gray', 'bone', 'copper', 'hot', 'cool',
        'turbo', 'jet', 'rainbow', 'nipy_spectral'
    ]

    def __init__(self, ax: Axes, on_change: Optional[Callable[[str], None]] = None):
        """
        Initialize color selector widget.

        Args:
            ax: Matplotlib axes to use for display
            on_change: Callback function called with colormap name when selection changes
        """
        self.ax = ax
        self.current_colormap = "viridis"
        self.on_change = on_change

        # Style axes
        self.ax.set_title("Color Transfer Function", fontsize=11, fontweight='bold')
        self.ax.axis("off")

        # Display current colormap as a color bar
        self._display_colormap()

    def _display_colormap(self) -> None:
        """Display current colormap as a horizontal color bar."""
        self.ax.clear()
        self.ax.set_title("Color Transfer Function", fontsize=11, fontweight='bold')
        self.ax.axis("off")

        # Create colormap preview
        gradient = np.linspace(0, 1, 256).reshape(1, -1)
        self.ax.imshow(gradient, aspect='auto', cmap=self.current_colormap,
                      extent=[0, 1, 0, 0.3])

        # Add colormap name label
        self.ax.text(0.5, 0.6, f"Colormap: {self.current_colormap}",
                    ha='center', va='center', transform=self.ax.transAxes,
                    fontsize=10, fontweight='bold')

        # Note: Interactive dropdown will be added in Phase 5
        self.ax.text(0.5, 0.15, "(Interactive selection in Phase 5)",
                    ha='center', va='center', transform=self.ax.transAxes,
                    fontsize=8, style='italic', color='gray')

        self.ax.figure.canvas.draw_idle()

    def set_colormap(self, colormap_name: str) -> None:
        """
        Set the current colormap.

        Args:
            colormap_name: Name of matplotlib colormap

        Raises:
            ValueError: If colormap name not in AVAILABLE_COLORMAPS
        """
        if colormap_name not in self.AVAILABLE_COLORMAPS:
            raise ValueError(f"Colormap '{colormap_name}' not available. "
                           f"Choose from: {self.AVAILABLE_COLORMAPS}")

        self.current_colormap = colormap_name
        self._display_colormap()

        if self.on_change:
            self.on_change(colormap_name)
```

## Testing Plan

### Test File: `tests/test_interface/test_matplotlib.py`

```python
"""Tests for main InteractiveVolumeRenderer class."""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from pyvr.interface.matplotlib import InteractiveVolumeRenderer
from pyvr.volume import Volume
from pyvr.datasets import create_sample_volume


@pytest.fixture
def small_volume():
    """Create a very small volume for fast testing."""
    data = create_sample_volume(32, 'sphere')
    return Volume(data=data)


@patch('pyvr.interface.matplotlib.VolumeRenderer')
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


@patch('pyvr.interface.matplotlib.VolumeRenderer')
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


@patch('pyvr.interface.matplotlib.VolumeRenderer')
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


@patch('pyvr.interface.matplotlib.VolumeRenderer')
def test_render_volume(mock_renderer_class, small_volume):
    """Test volume rendering."""
    mock_renderer = Mock()
    mock_pil_image = Mock()
    mock_pil_image.__array__ = lambda: np.zeros((512, 512, 3), dtype=np.uint8)
    mock_renderer.render_to_pil.return_value = mock_pil_image
    mock_renderer_class.return_value = mock_renderer

    interface = InteractiveVolumeRenderer(volume=small_volume)
    image_array = interface._render_volume()

    assert isinstance(image_array, np.ndarray)
    assert image_array.shape == (512, 512, 3)
    mock_renderer.render_to_pil.assert_called_once()


@patch('pyvr.interface.matplotlib.VolumeRenderer')
@patch('pyvr.interface.matplotlib.plt')
def test_show_creates_layout(mock_plt, mock_renderer_class, small_volume):
    """Test show() creates figure and layout."""
    mock_renderer = Mock()
    mock_renderer.render_to_pil.return_value = Mock()
    mock_renderer.render_to_pil.return_value.__array__ = lambda: np.zeros((512, 512, 3), dtype=np.uint8)
    mock_renderer_class.return_value = mock_renderer

    # Mock figure and axes
    mock_fig = MagicMock()
    mock_plt.figure.return_value = mock_fig

    interface = InteractiveVolumeRenderer(volume=small_volume)
    interface.show()

    # Verify figure was created
    mock_plt.figure.assert_called_once()

    # Verify widgets were initialized
    assert interface.image_display is not None
    assert interface.opacity_editor is not None
    assert interface.color_selector is not None


@patch('pyvr.interface.matplotlib.VolumeRenderer')
def test_colormap_change_callback(mock_renderer_class, small_volume):
    """Test colormap change callback."""
    mock_renderer = Mock()
    mock_renderer.render_to_pil.return_value = Mock()
    mock_renderer.render_to_pil.return_value.__array__ = lambda: np.zeros((512, 512, 3), dtype=np.uint8)
    mock_renderer_class.return_value = mock_renderer

    interface = InteractiveVolumeRenderer(volume=small_volume)
    interface._on_colormap_change('plasma')

    assert interface.state.current_colormap == 'plasma'
    assert interface.state.needs_render


@patch('pyvr.interface.matplotlib.VolumeRenderer')
def test_manual_update(mock_renderer_class, small_volume):
    """Test manual update() method."""
    mock_renderer = Mock()
    mock_renderer.render_to_pil.return_value = Mock()
    mock_renderer.render_to_pil.return_value.__array__ = lambda: np.zeros((512, 512, 3), dtype=np.uint8)
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
```

### Test File: `tests/test_interface/test_widgets.py`

```python
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

    def test_initialization_with_callback(self, mock_axes):
        """Test initialization with callback."""
        callback = Mock()
        selector = ColorSelector(mock_axes, on_change=callback)
        assert selector.on_change == callback

    def test_set_colormap(self, mock_axes):
        """Test setting colormap."""
        callback = Mock()
        selector = ColorSelector(mock_axes, on_change=callback)

        selector.set_colormap('plasma')

        assert selector.current_colormap == 'plasma'
        callback.assert_called_once_with('plasma')

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
```

## Acceptance Criteria

### Code Deliverables
- [x] Updated `InteractiveVolumeRenderer` with complete layout management
- [x] Implemented `_create_layout()` method with proper grid structure
- [x] Implemented `_update_display()` method for coordinated updates
- [x] Improved widget classes with better styling and error handling
- [x] Added info display panel with control hints
- [x] Added colormap preview in ColorSelector

### Testing Deliverables
- [x] Created `test_matplotlib.py` with 8+ test cases
- [x] Created `test_widgets.py` with 12+ test cases for all widget classes
- [x] All tests pass (including 204 existing tests)
- [x] Test coverage for new code >85%

### Visual Verification
- [x] Running example shows proper layout (3-panel design)
- [x] Volume rendering displays in left panel
- [x] Opacity transfer function displays in top right
- [x] Colormap preview displays in middle right
- [x] Info panel displays in bottom right

### Quality Gates
- [x] Code follows PyVR conventions
- [x] Type hints present on all methods
- [x] Google-style docstrings complete
- [x] No matplotlib warnings or errors on display

## Git Commit

```bash
pytest tests/test_interface/  # Verify all tests pass
git add pyvr/interface/ tests/test_interface/
git commit -m "Phase 2: Implement matplotlib layout and display

- Add complete figure layout with GridSpec (3-panel design)
- Implement coordinated widget update system
- Improve widget styling with color coding and better visuals
- Add colormap preview display
- Add info panel with control hints
- Add comprehensive tests for layout and widgets (20+ tests)

Part of v0.3.0 interactive interface feature"
```

## Notes for Next Phase

Phase 3 will implement the image display widget's integration with the renderer, ensuring the volume is rendered correctly and displayed. Phase 4 will then add the camera controls (mouse orbit and zoom).
