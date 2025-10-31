# Phase 1: Project Structure and Module Scaffolding

**Status**: Not Started
**Estimated Effort**: 1-2 hours
**Dependencies**: None (first phase)

## Overview

Set up the foundational directory structure and skeleton files for the interactive interface module. This phase creates the `pyvr/interface/` module with placeholder implementations that will be filled in during subsequent phases.

## Implementation Plan

### Directory Structure

Create the following structure:
```
pyvr/interface/
├── __init__.py          # Public API exports
├── matplotlib.py        # Main interface class
├── widgets.py           # UI widget components
└── state.py             # State management
```

### File: `pyvr/interface/__init__.py`

**Purpose**: Public API exports for the interface module

**Content**:
```python
"""
Interactive matplotlib-based interface for PyVR volume rendering.

This module provides testing/development interfaces for real-time volume
visualization with transfer function editing and camera controls.

Example:
    >>> from pyvr.interface import InteractiveVolumeRenderer
    >>> from pyvr.datasets import create_sample_volume
    >>> from pyvr.volume import Volume
    >>>
    >>> volume_data = create_sample_volume(128, 'sphere')
    >>> volume = Volume(data=volume_data)
    >>>
    >>> interface = InteractiveVolumeRenderer(volume=volume)
    >>> interface.show()  # Launch interactive GUI
"""

from pyvr.interface.matplotlib import InteractiveVolumeRenderer
from pyvr.interface.widgets import ImageDisplay, OpacityEditor, ColorSelector
from pyvr.interface.state import InterfaceState

__all__ = [
    "InteractiveVolumeRenderer",
    "ImageDisplay",
    "OpacityEditor",
    "ColorSelector",
    "InterfaceState",
]
```

### File: `pyvr/interface/state.py`

**Purpose**: Centralized state management for interface components

**Content**:
```python
"""State management for interactive interface."""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class InterfaceState:
    """
    Manages state for the interactive volume renderer interface.

    This class centralizes all mutable state to make the interface
    easier to reason about and test.

    Attributes:
        control_points: List of (scalar, opacity) tuples for opacity transfer function
        selected_control_point: Index of currently selected control point (None if none selected)
        current_colormap: Name of currently selected matplotlib colormap
        is_dragging_camera: Whether user is currently dragging to orbit camera
        is_dragging_control_point: Whether user is currently dragging a control point
        drag_start_pos: (x, y) position where drag started (in axes coordinates)
        needs_render: Flag indicating volume needs to be re-rendered
        needs_tf_update: Flag indicating transfer function needs update
    """

    # Transfer function state
    control_points: List[Tuple[float, float]] = field(default_factory=lambda: [(0.0, 0.0), (1.0, 1.0)])
    selected_control_point: Optional[int] = None
    current_colormap: str = "viridis"

    # Interaction state
    is_dragging_camera: bool = False
    is_dragging_control_point: bool = False
    drag_start_pos: Optional[Tuple[float, float]] = None

    # Update flags
    needs_render: bool = True
    needs_tf_update: bool = False

    def __post_init__(self):
        """Validate initial state."""
        if len(self.control_points) < 2:
            raise ValueError("Must have at least 2 control points")

        # Ensure control points are sorted by scalar value
        self.control_points = sorted(self.control_points, key=lambda cp: cp[0])

        # Validate control points are in [0, 1] range
        for scalar, opacity in self.control_points:
            if not (0.0 <= scalar <= 1.0):
                raise ValueError(f"Control point scalar {scalar} out of range [0, 1]")
            if not (0.0 <= opacity <= 1.0):
                raise ValueError(f"Control point opacity {opacity} out of range [0, 1]")

    def add_control_point(self, scalar: float, opacity: float) -> None:
        """
        Add a control point and maintain sorted order.

        Args:
            scalar: Scalar value in [0, 1]
            opacity: Opacity value in [0, 1]

        Raises:
            ValueError: If scalar or opacity out of range
        """
        if not (0.0 <= scalar <= 1.0):
            raise ValueError(f"Scalar {scalar} out of range [0, 1]")
        if not (0.0 <= opacity <= 1.0):
            raise ValueError(f"Opacity {opacity} out of range [0, 1]")

        self.control_points.append((scalar, opacity))
        self.control_points = sorted(self.control_points, key=lambda cp: cp[0])
        self.needs_tf_update = True

    def remove_control_point(self, index: int) -> None:
        """
        Remove a control point by index.

        Args:
            index: Index of control point to remove

        Raises:
            ValueError: If trying to remove first or last control point
            IndexError: If index out of range
        """
        if index == 0 or index == len(self.control_points) - 1:
            raise ValueError("Cannot remove first or last control point")
        if index < 0 or index >= len(self.control_points):
            raise IndexError(f"Control point index {index} out of range")

        del self.control_points[index]
        if self.selected_control_point == index:
            self.selected_control_point = None
        elif self.selected_control_point is not None and self.selected_control_point > index:
            self.selected_control_point -= 1

        self.needs_tf_update = True

    def update_control_point(self, index: int, scalar: float, opacity: float) -> None:
        """
        Update a control point's position and opacity.

        For first and last control points, only opacity can be changed.

        Args:
            index: Index of control point to update
            scalar: New scalar value in [0, 1]
            opacity: New opacity value in [0, 1]

        Raises:
            ValueError: If scalar or opacity out of range
            IndexError: If index out of range
        """
        if index < 0 or index >= len(self.control_points):
            raise IndexError(f"Control point index {index} out of range")
        if not (0.0 <= opacity <= 1.0):
            raise ValueError(f"Opacity {opacity} out of range [0, 1]")

        # Lock first and last control points to x=0.0 and x=1.0
        if index == 0:
            scalar = 0.0
        elif index == len(self.control_points) - 1:
            scalar = 1.0
        else:
            if not (0.0 <= scalar <= 1.0):
                raise ValueError(f"Scalar {scalar} out of range [0, 1]")

        self.control_points[index] = (scalar, opacity)

        # Re-sort if middle control points changed
        if index != 0 and index != len(self.control_points) - 1:
            self.control_points = sorted(self.control_points, key=lambda cp: cp[0])
            # Update selected index if it changed due to re-sorting
            if self.selected_control_point == index:
                self.selected_control_point = self.control_points.index((scalar, opacity))

        self.needs_tf_update = True

    def select_control_point(self, index: Optional[int]) -> None:
        """
        Select a control point by index.

        Args:
            index: Index to select, or None to deselect
        """
        if index is not None and (index < 0 or index >= len(self.control_points)):
            raise IndexError(f"Control point index {index} out of range")
        self.selected_control_point = index

    def set_colormap(self, colormap_name: str) -> None:
        """
        Change the current colormap.

        Args:
            colormap_name: Name of matplotlib colormap
        """
        self.current_colormap = colormap_name
        self.needs_tf_update = True
        self.needs_render = True
```

### File: `pyvr/interface/widgets.py`

**Purpose**: Widget component classes (placeholder implementations)

**Content**:
```python
"""UI widget components for interactive interface."""

from typing import Optional, Callable
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.widgets import Button
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
        self.image = None
        self.ax.set_title("Volume Rendering")
        self.ax.axis("off")

    def update_image(self, image_array: np.ndarray) -> None:
        """
        Update displayed image.

        Args:
            image_array: RGB image array of shape (H, W, 3) or (H, W, 4)
        """
        if self.image is None:
            self.image = self.ax.imshow(image_array, interpolation='nearest')
        else:
            self.image.set_data(image_array)
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
        self.line = None
        self.points = None

        self.ax.set_title("Opacity Transfer Function")
        self.ax.set_xlabel("Scalar Value")
        self.ax.set_ylabel("Opacity")
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.grid(True, alpha=0.3)

    def update_plot(self, control_points: list, selected_index: Optional[int] = None) -> None:
        """
        Update the opacity transfer function plot.

        Args:
            control_points: List of (scalar, opacity) tuples
            selected_index: Index of selected control point (highlighted)
        """
        # Extract x and y coordinates
        if not control_points:
            return

        x_vals = [cp[0] for cp in control_points]
        y_vals = [cp[1] for cp in control_points]

        # Update line
        if self.line is None:
            self.line, = self.ax.plot(x_vals, y_vals, 'b-', linewidth=2)
        else:
            self.line.set_data(x_vals, y_vals)

        # Update control points
        colors = ['red' if i == selected_index else 'blue' for i in range(len(control_points))]
        sizes = [100 if i == selected_index else 50 for i in range(len(control_points))]

        if self.points is None:
            self.points = self.ax.scatter(x_vals, y_vals, c=colors, s=sizes, zorder=5)
        else:
            self.points.set_offsets(np.c_[x_vals, y_vals])
            self.points.set_color(colors)
            self.points.set_sizes(sizes)

        self.ax.figure.canvas.draw_idle()


class ColorSelector:
    """
    Widget for selecting color transfer function colormap.

    Provides a dropdown menu of available matplotlib colormaps.

    Attributes:
        ax: Matplotlib axes for dropdown button
        current_colormap: Name of currently selected colormap
    """

    def __init__(self, ax: Axes, on_change: Optional[Callable[[str], None]] = None):
        """
        Initialize color selector widget.

        Args:
            ax: Matplotlib axes to use for dropdown
            on_change: Callback function when colormap changes
        """
        self.ax = ax
        self.current_colormap = "viridis"
        self.on_change = on_change

        # Placeholder - will implement dropdown in Phase 5
        self.ax.text(0.5, 0.5, f"Colormap: {self.current_colormap}",
                     ha='center', va='center', transform=self.ax.transAxes)
        self.ax.axis("off")

    def set_colormap(self, colormap_name: str) -> None:
        """
        Set the current colormap.

        Args:
            colormap_name: Name of matplotlib colormap
        """
        self.current_colormap = colormap_name
        if self.on_change:
            self.on_change(colormap_name)
```

### File: `pyvr/interface/matplotlib.py`

**Purpose**: Main interface class (placeholder implementation)

**Content**:
```python
"""Main interactive volume renderer interface using matplotlib."""

from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

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
        """Render volume and return image array."""
        image = self.renderer.render_to_pil()
        return np.array(image)

    def show(self) -> None:
        """
        Display the interactive interface.

        Creates matplotlib figure with layout and starts event loop.
        """
        # Create figure with grid layout
        self.fig = plt.figure(figsize=(12, 6))
        gs = self.fig.add_gridspec(2, 2, height_ratios=[3, 1], width_ratios=[2, 1])

        # Create widget axes
        ax_image = self.fig.add_subplot(gs[:, 0])
        ax_opacity = self.fig.add_subplot(gs[0, 1])
        ax_color = self.fig.add_subplot(gs[1, 1])

        # Initialize widgets
        self.image_display = ImageDisplay(ax_image)
        self.opacity_editor = OpacityEditor(ax_opacity)
        self.color_selector = ColorSelector(ax_color)

        # Initial render
        image_array = self._render_volume()
        self.image_display.update_image(image_array)
        self.opacity_editor.update_plot(self.state.control_points)

        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    from pyvr.datasets import create_sample_volume

    # Create sample volume
    volume_data = create_sample_volume(128, 'sphere')
    volume = Volume(data=volume_data)

    # Launch interface
    interface = InteractiveVolumeRenderer(volume=volume)
    interface.show()
```

## Testing Plan

### Test File: `tests/test_interface/conftest.py`

Create shared fixtures for interface testing:

```python
"""Shared fixtures for interface tests."""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock

from pyvr.volume import Volume
from pyvr.datasets import create_sample_volume


@pytest.fixture
def sample_volume():
    """Create a small sample volume for testing."""
    data = create_sample_volume(32, 'sphere')
    return Volume(data=data)


@pytest.fixture
def mock_renderer():
    """Create a mock renderer for testing without OpenGL."""
    renderer = Mock()
    renderer.render_to_pil.return_value = Mock()
    renderer.render_to_pil.return_value.__array__ = lambda: np.zeros((512, 512, 3), dtype=np.uint8)
    return renderer


@pytest.fixture
def mock_axes():
    """Create a mock matplotlib axes."""
    ax = MagicMock()
    ax.figure = MagicMock()
    ax.figure.canvas = MagicMock()
    return ax
```

### Test File: `tests/test_interface/test_state.py`

Test state management:

```python
"""Tests for interface state management."""

import pytest
from pyvr.interface.state import InterfaceState


def test_interface_state_initialization():
    """Test default initialization."""
    state = InterfaceState()
    assert len(state.control_points) == 2
    assert state.control_points[0] == (0.0, 0.0)
    assert state.control_points[1] == (1.0, 1.0)
    assert state.selected_control_point is None
    assert state.current_colormap == "viridis"
    assert not state.is_dragging_camera
    assert not state.is_dragging_control_point


def test_add_control_point():
    """Test adding control points."""
    state = InterfaceState()
    state.add_control_point(0.5, 0.5)
    assert len(state.control_points) == 3
    assert (0.5, 0.5) in state.control_points
    assert state.needs_tf_update


def test_add_control_point_maintains_order():
    """Test control points are kept sorted."""
    state = InterfaceState()
    state.add_control_point(0.7, 0.3)
    state.add_control_point(0.3, 0.7)
    scalars = [cp[0] for cp in state.control_points]
    assert scalars == sorted(scalars)


def test_add_control_point_validation():
    """Test validation of control point values."""
    state = InterfaceState()
    with pytest.raises(ValueError):
        state.add_control_point(-0.1, 0.5)  # Scalar out of range
    with pytest.raises(ValueError):
        state.add_control_point(0.5, 1.5)  # Opacity out of range


def test_remove_control_point():
    """Test removing middle control points."""
    state = InterfaceState(control_points=[(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)])
    state.remove_control_point(1)
    assert len(state.control_points) == 2
    assert (0.5, 0.5) not in state.control_points


def test_cannot_remove_first_last_control_point():
    """Test first and last control points cannot be removed."""
    state = InterfaceState()
    with pytest.raises(ValueError):
        state.remove_control_point(0)
    with pytest.raises(ValueError):
        state.remove_control_point(1)


def test_update_control_point():
    """Test updating control point position."""
    state = InterfaceState(control_points=[(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)])
    state.update_control_point(1, 0.6, 0.8)
    assert state.control_points[1] == (0.6, 0.8)


def test_update_first_last_locks_x():
    """Test first and last control points have locked x positions."""
    state = InterfaceState()
    state.update_control_point(0, 0.5, 0.3)  # Try to change first x
    assert state.control_points[0][0] == 0.0  # X locked to 0.0
    assert state.control_points[0][1] == 0.3  # Opacity changed

    state.update_control_point(1, 0.5, 0.7)  # Try to change last x
    assert state.control_points[1][0] == 1.0  # X locked to 1.0
    assert state.control_points[1][1] == 0.7  # Opacity changed


def test_select_control_point():
    """Test control point selection."""
    state = InterfaceState()
    state.select_control_point(0)
    assert state.selected_control_point == 0
    state.select_control_point(None)
    assert state.selected_control_point is None


def test_set_colormap():
    """Test colormap changes."""
    state = InterfaceState()
    state.set_colormap("plasma")
    assert state.current_colormap == "plasma"
    assert state.needs_tf_update
    assert state.needs_render
```

### Test File: `tests/test_interface/__init__.py`

Empty init file for test module.

## Acceptance Criteria

### Code Deliverables
- [x] Created `pyvr/interface/` module directory
- [x] Implemented `state.py` with `InterfaceState` class
- [x] Implemented `widgets.py` with placeholder widget classes
- [x] Implemented `matplotlib.py` with `InteractiveVolumeRenderer` skeleton
- [x] Implemented `__init__.py` with public API exports

### Testing Deliverables
- [x] Created `tests/test_interface/` directory
- [x] Implemented `conftest.py` with shared fixtures
- [x] Implemented `test_state.py` with comprehensive state tests (15+ test cases)
- [x] All state management tests pass
- [x] Test coverage for `state.py` >90%

### Quality Gates
- [x] Code follows PyVR conventions (dataclasses, type hints, Google-style docstrings)
- [x] All existing tests still pass (204 tests)
- [x] No breaking changes to existing APIs
- [x] Module imports correctly: `from pyvr.interface import InteractiveVolumeRenderer`

## Git Commit

After completing this phase:

```bash
git add pyvr/interface/ tests/test_interface/
git commit -m "Phase 1: Add interface module scaffolding

- Create pyvr/interface/ module structure
- Implement InterfaceState class for state management
- Add placeholder widget classes (ImageDisplay, OpacityEditor, ColorSelector)
- Add InteractiveVolumeRenderer skeleton with basic layout
- Add comprehensive tests for state management (15+ tests)

Part of v0.3.0 interactive interface feature"
```

## Notes for Next Phase

Phase 2 will implement the matplotlib layout and basic figure setup, connecting the widgets together into a functional (but not yet interactive) display.
