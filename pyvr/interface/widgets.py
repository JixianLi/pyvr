"""UI widget components for interactive interface."""

from typing import Optional, Callable, List, Tuple
from matplotlib.axes import Axes
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
