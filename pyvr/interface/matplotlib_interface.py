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

        # Render caching
        self._cached_image: Optional[np.ndarray] = None

    def _update_transfer_functions(self) -> None:
        """Update renderer with current transfer functions from state."""
        ctf = ColorTransferFunction.from_colormap(self.state.current_colormap)
        otf = OpacityTransferFunction(control_points=self.state.control_points)
        self.renderer.set_transfer_functions(ctf, otf)
        self.state.needs_tf_update = False

    def _render_volume(self) -> np.ndarray:
        """
        Render volume and return image array with caching.

        Returns:
            RGB image array of shape (H, W, 3)
        """
        # Return cached image if no re-render is needed
        if not self.state.needs_render and self._cached_image is not None:
            return self._cached_image

        try:
            # Update camera if controller changed it
            self.renderer.set_camera(self.camera_controller.params)

            # Render to PIL image and convert to numpy array
            image = self.renderer.render_to_pil()
            self._cached_image = np.array(image)
            return self._cached_image

        except Exception as e:
            # Handle OpenGL errors gracefully
            print(f"Rendering error: {e}")
            # Return a placeholder image (black)
            placeholder = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            return placeholder

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
            Tuple of (figure, axes_dict) where axes_dict contains 'image', 'opacity', 'color', 'info'
        """
        # Create figure - slightly taller to accommodate radio buttons
        fig = plt.figure(figsize=(14, 8))
        fig.suptitle("PyVR Interactive Volume Renderer", fontsize=14, fontweight='bold')

        # Create grid layout
        # Left side: Large image display
        # Right side: Stacked transfer function editors
        gs = GridSpec(3, 2, figure=fig,
                     width_ratios=[2, 1],
                     height_ratios=[4, 3, 1],  # More space for color selector with radio buttons
                     hspace=0.3, wspace=0.3)

        # Create axes
        ax_image = fig.add_subplot(gs[:, 0])  # Full left column
        ax_opacity = fig.add_subplot(gs[0, 1])  # Top right
        ax_color = fig.add_subplot(gs[1, 1])  # Middle right - more space for radio buttons
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

    def _on_mouse_press(self, event) -> None:
        """
        Handle mouse button press.

        Args:
            event: Matplotlib mouse button press event
        """
        # Check if click is in image display axes
        if event.inaxes != self.image_display.ax:
            return

        if event.button == 1:  # Left click
            self.state.is_dragging_camera = True
            self.state.drag_start_pos = (event.xdata, event.ydata)

    def _on_mouse_release(self, event) -> None:
        """
        Handle mouse button release.

        Args:
            event: Matplotlib mouse button release event
        """
        if self.state.is_dragging_camera:
            self.state.is_dragging_camera = False
            self.state.drag_start_pos = None
            # Trigger final render after drag
            self.state.needs_render = True
            self._update_display()

    def _on_mouse_move(self, event) -> None:
        """
        Handle mouse movement.

        Args:
            event: Matplotlib mouse motion event
        """
        if not self.state.is_dragging_camera:
            return

        if event.inaxes != self.image_display.ax or event.xdata is None:
            return

        # Calculate drag delta
        if self.state.drag_start_pos is not None:
            dx = event.xdata - self.state.drag_start_pos[0]
            dy = event.ydata - self.state.drag_start_pos[1]

            # Convert pixel movement to camera angles
            # Sensitivity factor for camera movement
            sensitivity = 0.005
            delta_azimuth = -dx * sensitivity
            delta_elevation = dy * sensitivity

            # Update camera using controller
            self.camera_controller.orbit(
                delta_azimuth=delta_azimuth,
                delta_elevation=delta_elevation
            )

            # Update drag start position for next move
            self.state.drag_start_pos = (event.xdata, event.ydata)

            # Don't render every frame - too slow
            # Will render on mouse release

    def _on_scroll(self, event) -> None:
        """
        Handle mouse scroll for zoom.

        Args:
            event: Matplotlib scroll event
        """
        if event.inaxes != self.image_display.ax:
            return

        # Scroll up = zoom in (decrease distance), scroll down = zoom out (increase distance)
        zoom_factor = 0.9 if event.step > 0 else 1.1

        self.camera_controller.zoom(factor=zoom_factor)

        # Render immediately for zoom (it's fast enough)
        self.state.needs_render = True
        self._update_display()

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

        # Connect event handlers
        self.fig.canvas.mpl_connect('button_press_event', self._on_mouse_press)
        self.fig.canvas.mpl_connect('button_release_event', self._on_mouse_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
        self.fig.canvas.mpl_connect('scroll_event', self._on_scroll)

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
