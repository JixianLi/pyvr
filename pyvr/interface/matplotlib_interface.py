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
from pyvr.interface.widgets import ImageDisplay, OpacityEditor, ColorSelector, PresetSelector
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
        preset_selector: Widget for rendering quality preset selection
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

        # Compute/load histogram for opacity editor background
        self._load_histogram()

        # Widget placeholders (will be created in show())
        self.image_display: Optional[ImageDisplay] = None
        self.opacity_editor: Optional[OpacityEditor] = None
        self.color_selector: Optional[ColorSelector] = None
        self.preset_selector: Optional[PresetSelector] = None
        self.fig: Optional[Figure] = None

        # Render caching and throttling
        self._cached_image: Optional[np.ndarray] = None
        self._last_render_time: float = 0.0
        self._min_render_interval: float = 0.1  # 100ms minimum between renders

    def _update_transfer_functions(self) -> None:
        """Update renderer with current transfer functions from state."""
        ctf = ColorTransferFunction.from_colormap(self.state.current_colormap)
        otf = OpacityTransferFunction(control_points=self.state.control_points)
        self.renderer.set_transfer_functions(ctf, otf)
        self.state.needs_tf_update = False

    def _load_histogram(self) -> None:
        """Load or compute volume histogram for opacity editor."""
        from pyvr.interface.cache import get_or_compute_histogram

        print("Loading histogram...")
        self.histogram_bin_edges, self.histogram_log_counts = get_or_compute_histogram(
            self.volume.data, num_bins=256
        )
        print("Histogram loaded.")

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

    def _should_render(self) -> bool:
        """
        Check if enough time has passed since last render.

        Returns:
            True if rendering should proceed, False otherwise
        """
        import time
        current_time = time.time()
        if current_time - self._last_render_time > self._min_render_interval:
            self._last_render_time = current_time
            return True
        return False

    def _update_display(self, force_render: bool = False) -> None:
        """
        Update all display widgets based on current state.

        Args:
            force_render: If True, bypass render throttling
        """
        # Update transfer functions if needed
        if self.state.needs_tf_update:
            self._update_transfer_functions()
            self.state.needs_render = True

        # Update light from camera if linked
        light = self.renderer.get_light()
        if light.is_linked:
            light.update_from_camera(self.camera_controller.params)
            self.renderer.set_light(light)
            self.state.needs_render = True

        # Update volume rendering with throttling
        if self.state.needs_render and (force_render or self._should_render()):
            image_array = self._render_volume()
            if self.image_display is not None:
                self.image_display.update_image(image_array)
            self.state.needs_render = False

        # Always update opacity editor (fast)
        if self.opacity_editor is not None:
            self.opacity_editor.update_plot(
                self.state.control_points,
                self.state.selected_control_point
            )

    def _create_layout(self) -> tuple:
        """
        Create matplotlib figure layout.

        Returns:
            Tuple of (figure, axes_dict) with keys: 'image', 'opacity', 'color', 'preset', 'info'
        """
        # Create figure
        fig = plt.figure(figsize=(14, 8))
        fig.suptitle("PyVR Interactive Volume Renderer", fontsize=14, fontweight='bold')

        # Create grid layout
        # Left side: Large image display
        # Right side: Stacked control panels
        gs = GridSpec(4, 2, figure=fig,
                     width_ratios=[2, 1],
                     height_ratios=[4, 2, 2, 1],  # Image, Opacity, Color, Preset+Info
                     hspace=0.3, wspace=0.3)

        # Create axes
        ax_image = fig.add_subplot(gs[:, 0])         # Full left column
        ax_opacity = fig.add_subplot(gs[0, 1])       # Top right
        ax_color = fig.add_subplot(gs[1, 1])         # Middle-top right
        ax_preset = fig.add_subplot(gs[2, 1])        # Middle-bottom right
        ax_info = fig.add_subplot(gs[3, 1])          # Bottom right

        axes_dict = {
            'image': ax_image,
            'opacity': ax_opacity,
            'color': ax_color,
            'preset': ax_preset,
            'info': ax_info,
        }

        return fig, axes_dict

    def _setup_info_display(self, ax) -> None:
        """
        Set up info display panel with all controls.

        Args:
            ax: Matplotlib axes for info display
        """
        ax.axis('off')
        info_text = (
            "Mouse Controls:\n"
            "  Image: Drag=orbit, Scroll=zoom\n"
            "  Opacity: L-click=add/select, R-click=remove, Drag=move\n\n"
            "Keyboard Shortcuts:\n"
            "  r: Reset view\n"
            "  s: Save image\n"
            "  f: Toggle FPS counter\n"
            "  h: Toggle histogram\n"
            "  l: Toggle light linking\n"
            "  Esc: Deselect\n"
            "  Del: Remove selected"
        )
        ax.text(0.05, 0.5, info_text,
               transform=ax.transAxes,
               fontsize=8,
               verticalalignment='center',
               family='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    def _on_mouse_press(self, event) -> None:
        """
        Handle mouse button press.

        Args:
            event: Matplotlib mouse button press event
        """
        # Handle image display (camera controls)
        if event.inaxes == self.image_display.ax:
            if event.button == 1:  # Left click
                self.state.is_dragging_camera = True
                self.state.drag_start_pos = (event.xdata, event.ydata)
            return

        # Handle opacity editor
        if self.opacity_editor and event.inaxes == self.opacity_editor.ax:
            if event.button == 1:  # Left click
                self._handle_opacity_left_click(event)
            elif event.button == 3:  # Right click
                self._handle_opacity_right_click(event)

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

        if self.state.is_dragging_control_point:
            self.state.is_dragging_control_point = False
            self.state.drag_start_pos = None
            # Final render with new transfer function
            self.state.needs_render = True
            self._update_display()

    def _on_mouse_move(self, event) -> None:
        """
        Handle mouse movement.

        Args:
            event: Matplotlib mouse motion event
        """
        # Handle camera drag
        if self.state.is_dragging_camera:
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
            return

        # Handle control point drag
        if self.state.is_dragging_control_point:
            if not self.opacity_editor or event.inaxes != self.opacity_editor.ax or event.xdata is None:
                return

            if self.state.selected_control_point is None:
                return

            # Clamp to valid range
            new_x = np.clip(event.xdata, 0.0, 1.0)
            new_y = np.clip(event.ydata, 0.0, 1.0)

            try:
                self.state.update_control_point(
                    self.state.selected_control_point,
                    new_x,
                    new_y
                )
                # Update opacity editor display (but don't re-render volume yet, too slow)
                if self.opacity_editor:
                    self.opacity_editor.update_plot(
                        self.state.control_points,
                        self.state.selected_control_point
                    )
            except (ValueError, IndexError):
                pass  # Invalid update, ignore

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

    def _handle_opacity_left_click(self, event) -> None:
        """
        Handle left click in opacity editor.

        Args:
            event: Matplotlib mouse button press event
        """
        click_x, click_y = event.xdata, event.ydata

        # Check if clicking near existing control point
        cp_index = self._find_control_point_near(click_x, click_y, threshold=0.05)

        if cp_index is not None:
            # Select existing control point
            self.state.select_control_point(cp_index)
            self.state.is_dragging_control_point = True
            self.state.drag_start_pos = (click_x, click_y)
        else:
            # Add new control point
            try:
                self.state.add_control_point(click_x, click_y)
                # Find and select the newly added point
                # (add_control_point sorts the list, so we need to find it)
                for i, (cp_x, cp_y) in enumerate(self.state.control_points):
                    if abs(cp_x - click_x) < 0.001 and abs(cp_y - click_y) < 0.001:
                        self.state.select_control_point(i)
                        break
            except ValueError:
                pass  # Out of range, ignore

        self._update_display()

    def _handle_opacity_right_click(self, event) -> None:
        """
        Handle right click in opacity editor (remove control point).

        Args:
            event: Matplotlib mouse button press event
        """
        click_x, click_y = event.xdata, event.ydata

        cp_index = self._find_control_point_near(click_x, click_y, threshold=0.05)

        if cp_index is not None:
            try:
                self.state.remove_control_point(cp_index)
                self._update_display()
            except ValueError:
                pass  # First/last point, cannot remove

    def _find_control_point_near(self, x: float, y: float, threshold: float = 0.05) -> Optional[int]:
        """
        Find control point near given coordinates.

        Args:
            x: X coordinate (scalar value)
            y: Y coordinate (opacity value)
            threshold: Distance threshold for "near"

        Returns:
            Index of control point if found, None otherwise
        """
        for i, (cp_x, cp_y) in enumerate(self.state.control_points):
            distance = np.sqrt((cp_x - x)**2 + (cp_y - y)**2)
            if distance < threshold:
                return i
        return None

    def _on_key_press(self, event) -> None:
        """
        Handle keyboard shortcuts.

        Args:
            event: Matplotlib key press event
        """
        if event.key == 'r':
            # Reset view to isometric
            self.camera_controller.params = Camera.isometric_view(distance=3.0)
            self.state.needs_render = True
            self._update_display(force_render=True)

        elif event.key == 's':
            # Save current rendering
            self._save_image()

        elif event.key == 'f':
            # Toggle FPS display
            self.state.show_fps = not self.state.show_fps
            if self.image_display is not None:
                self.image_display.set_fps_visible(self.state.show_fps)
            self.fig.canvas.draw_idle()

        elif event.key == 'h':
            # Toggle histogram display
            self.state.show_histogram = not self.state.show_histogram
            if self.opacity_editor is not None:
                self.opacity_editor.set_histogram_visible(self.state.show_histogram)
            print(f"Histogram {'visible' if self.state.show_histogram else 'hidden'}")

        elif event.key == 'l':
            # Toggle light camera linking
            light = self.renderer.get_light()

            if light.is_linked:
                light.unlink_from_camera()
                self.state.light_linked_to_camera = False
                print("Light unlinked from camera (fixed position)")
            else:
                # Link with default offsets (light follows camera)
                light.link_to_camera(
                    azimuth_offset=0.0,
                    elevation_offset=0.0,
                    distance_offset=0.0
                )
                light.update_from_camera(self.camera_controller.params)
                self.renderer.set_light(light)
                self.state.light_linked_to_camera = True
                print("Light linked to camera (will follow movement)")

            self.state.needs_render = True
            self._update_display(force_render=True)

        elif event.key == 'escape':
            # Deselect control point
            if self.state.selected_control_point is not None:
                self.state.select_control_point(None)
                self._update_display()

        elif event.key == 'delete' or event.key == 'backspace':
            # Delete selected control point
            if self.state.selected_control_point is not None:
                try:
                    self.state.remove_control_point(self.state.selected_control_point)
                    self._update_display(force_render=True)
                except ValueError:
                    pass  # Can't delete first/last

    def _save_image(self) -> None:
        """Save current rendering to file."""
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pyvr_render_{timestamp}.png"

        if self._cached_image is not None:
            from PIL import Image
            img = Image.fromarray(self._cached_image)
            img.save(filename)
            print(f"Saved rendering to {filename}")
        else:
            print("No rendered image to save")

    def _update_cursor(self, event) -> None:
        """
        Update cursor based on context.

        Args:
            event: Matplotlib axes enter event
        """
        if self.fig is None:
            return

        try:
            if event.inaxes == self.image_display.ax:
                # Hand cursor for camera controls
                self.fig.canvas.set_cursor(1)  # Hand cursor
            elif self.opacity_editor and event.inaxes == self.opacity_editor.ax:
                # Crosshair for control point editing
                self.fig.canvas.set_cursor(2)  # Crosshair cursor
            else:
                # Default cursor
                self.fig.canvas.set_cursor(0)
        except (SystemError, RuntimeError):
            # macOS backend sometimes returns NULL without setting an exception
            # This is a known matplotlib issue - cursor change is non-critical
            pass

    def show(self) -> None:
        """
        Display the interactive interface.

        Creates matplotlib figure with layout and starts event loop.
        """
        # Create figure and axes
        self.fig, axes = self._create_layout()

        # Initialize widgets - pass show_fps to ImageDisplay, show_histogram to OpacityEditor
        self.image_display = ImageDisplay(axes['image'], show_fps=self.state.show_fps)
        self.opacity_editor = OpacityEditor(axes['opacity'], show_histogram=self.state.show_histogram)
        self.color_selector = ColorSelector(axes['color'],
                                           on_change=self._on_colormap_change)
        self.preset_selector = PresetSelector(axes['preset'],
                                             initial_preset=self.state.current_preset_name,
                                             on_change=self._on_preset_change)

        # Set histogram background
        self.opacity_editor.set_histogram(
            self.histogram_bin_edges,
            self.histogram_log_counts
        )

        # Set up info display
        self._setup_info_display(axes['info'])

        # Connect event handlers
        self.fig.canvas.mpl_connect('button_press_event', self._on_mouse_press)
        self.fig.canvas.mpl_connect('button_release_event', self._on_mouse_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
        self.fig.canvas.mpl_connect('scroll_event', self._on_scroll)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        self.fig.canvas.mpl_connect('axes_enter_event', self._update_cursor)

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

    def _on_preset_change(self, preset_name: str) -> None:
        """
        Callback when rendering preset changes.

        Args:
            preset_name: Name of new preset
        """
        # Update state
        self.state.set_preset(preset_name)

        # Get new config based on preset name
        preset_map = {
            'preview': RenderConfig.preview,
            'fast': RenderConfig.fast,
            'balanced': RenderConfig.balanced,
            'high_quality': RenderConfig.high_quality,
            'ultra_quality': RenderConfig.ultra_quality,
        }

        new_config = preset_map[preset_name]()

        # Update renderer config
        self.renderer.set_config(new_config)

        # Trigger re-render
        self.state.needs_render = True
        self._update_display(force_render=True)

        # Print feedback
        samples = new_config.estimate_samples_per_ray()
        print(f"Switched to '{preset_name}' preset (~{samples} samples/ray)")

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
