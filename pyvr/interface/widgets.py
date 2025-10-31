"""UI widget components for interactive interface."""

from typing import Optional, Callable, List, Tuple
from matplotlib.axes import Axes
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

    def update_plot(self, control_points: List[Tuple[float, float]], selected_index: Optional[int] = None) -> None:
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
