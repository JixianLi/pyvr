"""UI widget components for interactive interface."""

from typing import Optional, Callable, List, Tuple
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from matplotlib.collections import PathCollection
from matplotlib.widgets import RadioButtons
import matplotlib.text
import numpy as np


class ImageDisplay:
    """
    Widget for displaying rendered volume with camera controls and FPS counter.

    Handles mouse interactions for orbiting and zooming the camera.

    Attributes:
        ax: Matplotlib axes for image display
        image: Image artist for displaying rendered volume
        show_fps: Whether to show FPS counter
        fps_text: Text artist for FPS display
        fps_counter: FPSCounter instance for tracking FPS
    """

    def __init__(self, ax: Axes, show_fps: bool = True):
        """
        Initialize image display widget.

        Args:
            ax: Matplotlib axes to use for display
            show_fps: Whether to show FPS counter (default: True)
        """
        self.ax = ax
        self.image: Optional[AxesImage] = None
        self.show_fps = show_fps
        self.fps_text: Optional[matplotlib.text.Text] = None
        self.fps_counter = FPSCounter(window_size=30)

        # Style the axes
        self.ax.set_title("Volume Rendering", fontsize=12, fontweight='bold')
        self.ax.axis("off")

        # Set background color
        self.ax.set_facecolor('#2e2e2e')

    def update_image(self, image_array: np.ndarray) -> None:
        """
        Update displayed image and FPS counter.

        Args:
            image_array: RGB image array of shape (H, W, 3) or (H, W, 4)
        """
        if image_array.shape[2] not in [3, 4]:
            raise ValueError(f"Image must have 3 or 4 channels, got {image_array.shape[2]}")

        # Update image
        if self.image is None:
            self.image = self.ax.imshow(image_array, interpolation='nearest')
        else:
            self.image.set_data(image_array)

        # Update FPS counter
        if self.show_fps:
            self.fps_counter.tick()
            self._update_fps_display()

        self.ax.figure.canvas.draw_idle()

    def _update_fps_display(self) -> None:
        """Update FPS text overlay."""
        fps = self.fps_counter.get_fps()
        fps_string = f"FPS: {fps:.1f}"

        if self.fps_text is None:
            # Create FPS text in top-left corner
            self.fps_text = self.ax.text(
                0.02, 0.98, fps_string,
                transform=self.ax.transAxes,
                fontsize=10,
                verticalalignment='top',
                color='#00ff00',  # Bright green
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7)
            )
        else:
            self.fps_text.set_text(fps_string)

    def set_fps_visible(self, visible: bool) -> None:
        """
        Toggle FPS counter visibility.

        Args:
            visible: Whether FPS counter should be visible
        """
        self.show_fps = visible
        if self.fps_text is not None:
            self.fps_text.set_visible(visible)
        if not visible:
            self.fps_counter.reset()

    def clear(self) -> None:
        """Clear the image display."""
        if self.image is not None:
            self.image.remove()
            self.image = None
        if self.fps_text is not None:
            self.fps_text.remove()
            self.fps_text = None
        self.fps_counter.reset()
        self.ax.figure.canvas.draw_idle()


class OpacityEditor:
    """
    Widget for editing opacity transfer function with control points and histogram background.

    Supports adding, removing, selecting, and dragging control points.

    Attributes:
        ax: Matplotlib axes for opacity plot
        line: Line artist for transfer function curve
        points: Scatter artist for control points
        histogram_bars: BarContainer for histogram background (optional)
        show_histogram: Whether to show histogram background
    """

    def __init__(self, ax: Axes, show_histogram: bool = True):
        """
        Initialize opacity editor widget.

        Args:
            ax: Matplotlib axes to use for editor
            show_histogram: Whether to show histogram background (default: True)
        """
        self.ax = ax
        self.line: Optional[Line2D] = None
        self.points: Optional[PathCollection] = None
        self.histogram_bars = None
        self.show_histogram = show_histogram

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

    def set_histogram(self, bin_edges: np.ndarray, log_counts: np.ndarray) -> None:
        """
        Set histogram background data.

        Args:
            bin_edges: Histogram bin edges (length: num_bins + 1)
            log_counts: Log-scale histogram counts (length: num_bins)

        Example:
            >>> from pyvr.interface.cache import get_or_compute_histogram
            >>> edges, counts = get_or_compute_histogram(volume.data)
            >>> editor.set_histogram(edges, counts)
        """
        if not self.show_histogram:
            return

        # Normalize counts to [0, 1] for display
        max_count = np.max(log_counts)
        if max_count > 0:
            normalized_counts = log_counts / max_count
        else:
            normalized_counts = log_counts

        # Compute bin centers for bar plot
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = bin_edges[1] - bin_edges[0]

        # Remove old histogram if exists
        if self.histogram_bars is not None:
            self.histogram_bars.remove()

        # Create bar plot for histogram (subtle blue-gray color)
        self.histogram_bars = self.ax.bar(
            bin_centers,
            normalized_counts,
            width=bin_width,
            color='#a0b0c0',  # Subtle blue-gray
            alpha=0.3,
            zorder=1,  # Behind control points and line
            edgecolor='none'
        )

        self.ax.figure.canvas.draw_idle()

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
            self.line, = self.ax.plot(x_vals, y_vals, 'b-', linewidth=2.5, alpha=0.7, zorder=3)
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

    def set_histogram_visible(self, visible: bool) -> None:
        """
        Toggle histogram visibility.

        Args:
            visible: Whether histogram should be visible
        """
        self.show_histogram = visible
        if self.histogram_bars is not None:
            for bar in self.histogram_bars:
                bar.set_visible(visible)
        self.ax.figure.canvas.draw_idle()

    def clear(self) -> None:
        """Clear the opacity plot including histogram."""
        if self.line is not None:
            self.line.remove()
            self.line = None
        if self.points is not None:
            self.points.remove()
            self.points = None
        if self.histogram_bars is not None:
            self.histogram_bars.remove()
            self.histogram_bars = None
        self.ax.figure.canvas.draw_idle()


class ColorSelector:
    """
    Widget for selecting color transfer function colormap.

    Provides interactive RadioButtons for colormap selection with real-time preview.

    Attributes:
        ax: Matplotlib axes for colormap display
        current_colormap: Name of currently selected colormap
        on_change: Callback function when colormap changes
        radio: RadioButtons widget for colormap selection
        colormap_preview: AxesImage for colormap preview
        preview_ax: Axes for colormap preview display
    """

    # Popular matplotlib colormaps for volume rendering
    AVAILABLE_COLORMAPS = [
        'viridis', 'plasma', 'inferno', 'magma', 'cividis',
        'gray', 'bone', 'copper', 'hot', 'cool',
        'turbo', 'jet'
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

        # Create colormap preview at the top
        self._create_colormap_preview()

        # Create radio buttons for colormap selection
        self.radio = RadioButtons(
            ax=self.ax,
            labels=self.AVAILABLE_COLORMAPS,
            active=0  # viridis is first
        )

        # Style radio buttons
        for label in self.radio.labels:
            label.set_fontsize(9)

        # Connect callback
        self.radio.on_clicked(self._on_selection)

    def _create_colormap_preview(self) -> None:
        """Create a colormap preview bar above radio buttons."""
        # Create a small axes for preview at the top of the color selector area
        fig = self.ax.figure
        ax_pos = self.ax.get_position()

        # Position preview at top of the color selector area
        self.preview_ax = fig.add_axes([
            ax_pos.x0,
            ax_pos.y1 - 0.04,
            ax_pos.width,
            0.025
        ])
        self.preview_ax.axis('off')

        # Create gradient for colormap preview
        gradient = np.linspace(0, 1, 256).reshape(1, -1)
        self.colormap_preview = self.preview_ax.imshow(
            gradient,
            aspect='auto',
            cmap=self.current_colormap,
            extent=[0, 1, 0, 1]
        )

    def _on_selection(self, label: str) -> None:
        """
        Handle colormap selection from radio buttons.

        Args:
            label: Name of selected colormap
        """
        self.current_colormap = label

        # Update preview
        if hasattr(self, 'colormap_preview'):
            self.colormap_preview.set_cmap(label)
            self.ax.figure.canvas.draw_idle()

        # Call external callback
        if self.on_change:
            self.on_change(label)

    def set_colormap(self, colormap_name: str) -> None:
        """
        Programmatically set the current colormap.

        Args:
            colormap_name: Name of matplotlib colormap

        Raises:
            ValueError: If colormap name not in AVAILABLE_COLORMAPS
        """
        if colormap_name not in self.AVAILABLE_COLORMAPS:
            raise ValueError(f"Colormap '{colormap_name}' not available. "
                           f"Choose from: {self.AVAILABLE_COLORMAPS}")

        self.current_colormap = colormap_name

        # Update radio button selection
        idx = self.AVAILABLE_COLORMAPS.index(colormap_name)
        self.radio.set_active(idx)

        # Update preview
        if hasattr(self, 'colormap_preview'):
            self.colormap_preview.set_cmap(colormap_name)
            self.ax.figure.canvas.draw_idle()


class PresetSelector:
    """
    Widget for selecting RenderConfig quality presets.

    Provides interactive RadioButtons for preset selection with real-time quality switching.

    Attributes:
        ax: Matplotlib axes for preset display
        current_preset: Name of currently selected preset
        on_change: Callback function when preset changes
        radio: RadioButtons widget for preset selection
    """

    # RenderConfig presets in order from fastest to highest quality
    AVAILABLE_PRESETS = [
        'preview',      # Extremely fast, low quality
        'fast',         # Fast, interactive
        'balanced',     # Default, good balance
        'high_quality', # High quality, slower
        'ultra_quality' # Maximum quality, very slow
    ]

    # Human-readable labels with performance hints
    PRESET_LABELS = [
        'Preview (fastest)',
        'Fast',
        'Balanced',
        'High Quality',
        'Ultra (slowest)'
    ]

    def __init__(self, ax: Axes, initial_preset: str = 'fast',
                 on_change: Optional[Callable[[str], None]] = None):
        """
        Initialize preset selector widget.

        Args:
            ax: Matplotlib axes to use for display
            initial_preset: Initial preset name (default: 'fast')
            on_change: Callback function called with preset name when selection changes

        Raises:
            ValueError: If initial_preset not in AVAILABLE_PRESETS
        """
        self.ax = ax
        self.on_change = on_change

        if initial_preset not in self.AVAILABLE_PRESETS:
            raise ValueError(f"Invalid preset '{initial_preset}'. "
                           f"Choose from: {self.AVAILABLE_PRESETS}")

        self.current_preset = initial_preset

        # Style axes
        self.ax.set_title("Rendering Quality", fontsize=11, fontweight='bold')
        self.ax.axis('off')

        # Create radio buttons for preset selection
        initial_index = self.AVAILABLE_PRESETS.index(initial_preset)
        self.radio = RadioButtons(
            ax=self.ax,
            labels=self.PRESET_LABELS,
            active=initial_index
        )

        # Style radio buttons
        for label in self.radio.labels:
            label.set_fontsize(9)

        # Connect callback
        self.radio.on_clicked(self._on_selection)

    def _on_selection(self, label: str) -> None:
        """
        Handle preset selection from radio buttons.

        Args:
            label: Display label of selected preset
        """
        # Map display label back to preset name
        label_index = self.PRESET_LABELS.index(label)
        preset_name = self.AVAILABLE_PRESETS[label_index]

        self.current_preset = preset_name

        # Call external callback
        if self.on_change:
            self.on_change(preset_name)

    def set_preset(self, preset_name: str) -> None:
        """
        Programmatically set the current preset.

        Args:
            preset_name: Name of RenderConfig preset

        Raises:
            ValueError: If preset_name not in AVAILABLE_PRESETS
        """
        if preset_name not in self.AVAILABLE_PRESETS:
            raise ValueError(f"Invalid preset '{preset_name}'. "
                           f"Choose from: {self.AVAILABLE_PRESETS}")

        self.current_preset = preset_name

        # Update radio button selection
        preset_index = self.AVAILABLE_PRESETS.index(preset_name)
        self.radio.set_active(preset_index)

    def get_preset(self) -> str:
        """
        Get currently selected preset name.

        Returns:
            Current preset name (e.g., 'balanced')
        """
        return self.current_preset


class FPSCounter:
    """
    Helper class for calculating and tracking frames per second.

    Uses a rolling window average for stable FPS display.

    Attributes:
        window_size: Number of frames to average over
        frame_times: Deque of recent frame timestamps
        last_time: Timestamp of last frame
    """

    def __init__(self, window_size: int = 30):
        """
        Initialize FPS counter.

        Args:
            window_size: Number of frames to average (default: 30)
        """
        from collections import deque
        self.window_size = window_size
        self.frame_times = deque(maxlen=window_size)
        self.last_time = None

    def tick(self) -> None:
        """Record a frame render event."""
        import time
        current_time = time.perf_counter()

        if self.last_time is not None:
            frame_time = current_time - self.last_time
            self.frame_times.append(frame_time)

        self.last_time = current_time

    def get_fps(self) -> float:
        """
        Get current FPS value.

        Returns:
            Current FPS (frames per second), or 0.0 if insufficient data
        """
        if len(self.frame_times) == 0:
            return 0.0

        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        if avg_frame_time == 0:
            return 0.0

        return 1.0 / avg_frame_time

    def reset(self) -> None:
        """Reset FPS tracking."""
        self.frame_times.clear()
        self.last_time = None
