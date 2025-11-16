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
            try:
                light.update_from_camera(self.camera_controller.params)
                self.renderer.set_light(light)
                self.state.needs_render = True
            except Exception as e:
                print(f"Warning: Failed to update linked light: {e}")
                # Unlink light to prevent continued errors
                light.unlink_from_camera()
                self.state.light_linked_to_camera = False

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

        Uses 3-column layout:
        - Left (2.0): Image display (full height)
        - Middle (1.0): Opacity and color transfer function editors
        - Right (1.0): Info panel (top) and rendering quality preset selector (bottom)

        Returns:
            Tuple of (figure, axes_dict) with keys: 'image', 'opacity', 'color', 'preset', 'info'
        """
        # Create figure
        fig = plt.figure(figsize=(18, 8))
        fig.suptitle("PyVR Interactive Volume Renderer", fontsize=14, fontweight='bold')

        # Create grid layout
        # 3-column layout: image (2.0) | controls (1.0) | info+preset (1.0)
        # Middle: Opacity (top), Color (bottom)
        # Right: Info (top, rows 0-1), Preset (bottom, row 2)
        gs = GridSpec(3, 3, figure=fig,
                     width_ratios=[2.0, 1.0, 1.0],
                     height_ratios=[3, 1, 1],
                     hspace=0.3, wspace=0.3)

        # Create axes
        ax_image = fig.add_subplot(gs[:, 0])      # Full left column (all rows)
        ax_opacity = fig.add_subplot(gs[0, 1])    # Top middle
        ax_color = fig.add_subplot(gs[1:, 1])     # Bottom middle (rows 1-2 combined)
        ax_info = fig.add_subplot(gs[0:2, 2])     # Top right (rows 0-1)
        ax_preset = fig.add_subplot(gs[2, 2])     # Bottom right (row 2)

        axes_dict = {
            'image': ax_image,
            'opacity': ax_opacity,
            'color': ax_color,
            'preset': ax_preset,
            'info': ax_info,
        }

        return fig, axes_dict

    def _setup_info_display(self, ax) -> None:
        """Set up info display panel with all controls and status indicators."""
        ax.axis('off')

        # Single text block with both controls and status (no overlap)
        self.info_text = ax.text(
            0.05, 0.98,  # Top of axes
            self._get_full_info_text(),
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment='top',
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3)
        )

    def _get_full_info_text(self) -> str:
        """Get complete info text with controls and status."""
        controls = (
            "Mouse Controls:\n"
            "  Image: Drag=rotate, Scroll=zoom\n"
            "  Opacity: L-click=add/select, R-click=remove, Drag=move\n\n"
            "Keyboard Shortcuts:\n"
            "  r: Reset view\n"
            "  s: Save image\n"
            "  f: Toggle FPS\n"
            "  h: Toggle histogram\n"
            "  l: Toggle light link\n"
            "  q: Toggle auto-quality\n"
            "  t: Toggle control mode\n"
            "  Esc: Deselect\n"
            "  Del: Remove selected\n"
        )

        # Add dynamic status
        light = self.renderer.get_light()
        status = (
            f"\nCurrent Status:\n"
            f"  Control Mode: {self.state.camera_control_mode.capitalize()}\n"
            f"  Preset: {self.state.current_preset_name}\n"
            f"  FPS: {'ON' if self.state.show_fps else 'OFF'}\n"
            f"  Histogram: {'ON' if self.state.show_histogram else 'OFF'}\n"
            f"  Light Linked: {'YES' if light.is_linked else 'NO'}\n"
            f"  Auto-Quality: {'ON' if self.state.auto_quality_enabled else 'OFF'}"
        )

        # Add light offset info if linked
        if light.is_linked:
            offsets = light.get_offsets()
            if offsets and isinstance(offsets, dict):
                status += f"\n    (offsets: az={offsets['azimuth']:.2f}, el={offsets['elevation']:.2f})"

        return controls + status

    def _update_status_display(self) -> None:
        """Update status text with current settings."""
        if hasattr(self, 'info_text') and self.info_text is not None:
            self.info_text.set_text(self._get_full_info_text())
            if self.fig is not None:
                self.fig.canvas.draw_idle()

    def _switch_to_interaction_quality(self) -> None:
        """Switch to fast preset for responsive interaction."""
        from pyvr.config import RenderConfig

        # Save current preset if not already saved
        if self.state.saved_preset_name is None:
            self.state.saved_preset_name = self.state.current_preset_name

        # Switch to fast preset if not already
        if self.state.current_preset_name != 'fast':
            fast_config = RenderConfig.fast()
            self.renderer.set_config(fast_config)
            self.state.current_preset_name = 'fast'
            # Don't update preset selector UI during interaction

    def _restore_quality_after_interaction(self) -> None:
        """Restore previous quality preset after interaction."""
        from pyvr.config import RenderConfig

        if self.state.saved_preset_name is None:
            return

        # Restore saved preset if different from current
        if self.state.current_preset_name != self.state.saved_preset_name:
            preset_map = {
                'preview': RenderConfig.preview,
                'fast': RenderConfig.fast,
                'balanced': RenderConfig.balanced,
                'high_quality': RenderConfig.high_quality,
                'ultra_quality': RenderConfig.ultra_quality,
            }

            restored_config = preset_map[self.state.saved_preset_name]()
            self.renderer.set_config(restored_config)
            self.state.current_preset_name = self.state.saved_preset_name

            # Update preset selector UI
            if self.preset_selector:
                self.preset_selector.set_preset(self.state.saved_preset_name)

            # Trigger re-render with restored quality
            self.state.needs_render = True
            self._update_display(force_render=True)

        # Clear saved preset
        self.state.saved_preset_name = None

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

                # Switch to fast preset during interaction
                if self.state.auto_quality_enabled:
                    self._switch_to_interaction_quality()
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

            # Restore quality after interaction
            if self.state.auto_quality_enabled and self.state.saved_preset_name is not None:
                self._restore_quality_after_interaction()
            else:
                # Trigger final render after drag
                self.state.needs_render = True
                self._update_display()
                # Clear saved preset if we had one
                self.state.saved_preset_name = None

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

                # Apply camera control based on current mode
                if self.state.camera_control_mode == 'trackball':
                    # Trackball control: intuitive 3D rotation
                    self.camera_controller.trackball(
                        dx=dx,
                        dy=dy,
                        viewport_width=self.width,
                        viewport_height=self.height,
                        sensitivity=1.0
                    )
                else:  # orbit mode
                    # Orbit control: traditional azimuth/elevation
                    sensitivity = 0.005
                    delta_azimuth = -dx * sensitivity
                    delta_elevation = dy * sensitivity

                    self.camera_controller.orbit(
                        delta_azimuth=delta_azimuth,
                        delta_elevation=delta_elevation
                    )

                # Update drag start position for next move
                self.state.drag_start_pos = (event.xdata, event.ydata)

                # Trigger render (throttled to avoid performance issues)
                self.state.needs_render = True
                self._update_display()  # Throttling prevents excessive renders
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

        # Switch to fast preset temporarily
        if self.state.auto_quality_enabled:
            self._switch_to_interaction_quality()

        # Scroll up = zoom in (decrease distance), scroll down = zoom out (increase distance)
        zoom_factor = 0.9 if event.step > 0 else 1.1

        self.camera_controller.zoom(factor=zoom_factor)

        # Render immediately for zoom (it's fast enough)
        self.state.needs_render = True
        self._update_display()

        # Restore quality after short delay
        if self.state.auto_quality_enabled and self.fig is not None:
            # Cancel any existing timer
            if hasattr(self, '_scroll_restore_timer') and self._scroll_restore_timer is not None:
                self._scroll_restore_timer.stop()

            # Create matplotlib timer (thread-safe: executes on main thread)
            # Using fig.canvas.new_timer instead of threading.Timer to avoid
            # thread-safety violations when updating matplotlib widgets
            self._scroll_restore_timer = self.fig.canvas.new_timer(interval=500)  # 500ms
            self._scroll_restore_timer.add_callback(self._restore_quality_after_interaction)
            self._scroll_restore_timer.single_shot = True
            self._scroll_restore_timer.start()

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
            self._update_status_display()

        elif event.key == 'h':
            # Toggle histogram display
            self.state.show_histogram = not self.state.show_histogram
            if self.opacity_editor is not None:
                self.opacity_editor.set_histogram_visible(self.state.show_histogram)
            print(f"Histogram {'visible' if self.state.show_histogram else 'hidden'}")
            self._update_status_display()

        elif event.key == 'l':
            # Toggle light camera linking
            try:
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
                self._update_status_display()
            except Exception as e:
                print(f"Error toggling light linking: {e}")
                import traceback
                traceback.print_exc()

        elif event.key == 'q':
            # Toggle automatic quality adjustment
            self.state.auto_quality_enabled = not self.state.auto_quality_enabled
            status = 'enabled' if self.state.auto_quality_enabled else 'disabled'
            print(f"Automatic quality adjustment {status}")
            self._update_status_display()

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

        elif event.key == 't':
            # Toggle camera control mode
            if self.state.camera_control_mode == 'trackball':
                self.state.camera_control_mode = 'orbit'
                print("Switched to orbit control (azimuth/elevation)")
            else:
                self.state.camera_control_mode = 'trackball'
                print("Switched to trackball control (arcball)")
            self._update_status_display()

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

    def set_high_quality_mode(self) -> None:
        """
        Switch to high quality rendering mode.

        Convenience method for final renders.
        """
        from pyvr.config import RenderConfig

        self.state.set_preset('high_quality')
        self.renderer.set_config(RenderConfig.high_quality())

        if self.preset_selector:
            self.preset_selector.set_preset('high_quality')

        self.state.needs_render = True
        self._update_display(force_render=True)
        self._update_status_display()

        print("Switched to high quality mode")

    def set_camera_linked_lighting(self, azimuth_offset: float = 0.0,
                                   elevation_offset: float = 0.0) -> None:
        """
        Enable camera-linked lighting with offsets.

        Args:
            azimuth_offset: Horizontal angle offset in radians
            elevation_offset: Vertical angle offset in radians

        Example:
            >>> interface.set_camera_linked_lighting(azimuth_offset=np.pi/4)
        """
        light = self.renderer.get_light()
        light.link_to_camera(azimuth_offset=azimuth_offset,
                            elevation_offset=elevation_offset)
        light.update_from_camera(self.camera_controller.params)
        self.renderer.set_light(light)
        self.state.light_linked_to_camera = True
        self.state.needs_render = True
        self._update_display(force_render=True)
        self._update_status_display()

        print(f"Light linked to camera (az_offset={azimuth_offset:.2f}, el_offset={elevation_offset:.2f})")

    def capture_high_quality_image(self, filename: Optional[str] = None) -> str:
        """
        Capture a high-quality rendering of current view.

        Temporarily switches to ultra_quality preset, renders, and restores
        previous preset.

        Args:
            filename: Optional filename (default: auto-generated with timestamp)

        Returns:
            Path to saved image

        Example:
            >>> path = interface.capture_high_quality_image("my_render.png")
        """
        from pyvr.config import RenderConfig
        import datetime

        # Save current state
        original_preset = self.state.current_preset_name
        original_auto_quality = self.state.auto_quality_enabled

        # Disable auto-quality and switch to ultra
        self.state.auto_quality_enabled = False
        self.renderer.set_config(RenderConfig.ultra_quality())

        # Render high quality
        self.state.needs_render = True
        image_array = self._render_volume()

        # Save to file
        from PIL import Image
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pyvr_hq_render_{timestamp}.png"

        img = Image.fromarray(image_array)
        img.save(filename)

        # Restore original state
        preset_map = {
            'preview': RenderConfig.preview,
            'fast': RenderConfig.fast,
            'balanced': RenderConfig.balanced,
            'high_quality': RenderConfig.high_quality,
            'ultra_quality': RenderConfig.ultra_quality,
        }
        self.renderer.set_config(preset_map[original_preset]())
        self.state.auto_quality_enabled = original_auto_quality

        print(f"High quality image saved to {filename}")
        return filename

    def show(self) -> None:
        """
        Display the interactive interface.

        Creates matplotlib figure with layout and starts event loop.
        """
        # Disable matplotlib default key bindings to prevent conflicts
        import matplotlib as mpl

        # Store original keymaps to restore later if needed
        self._original_keymaps = {
            'fullscreen': mpl.rcParams['keymap.fullscreen'][:],
            'home': mpl.rcParams['keymap.home'][:],
            'back': mpl.rcParams['keymap.back'][:],
            'forward': mpl.rcParams['keymap.forward'][:],
            'pan': mpl.rcParams['keymap.pan'][:],
            'zoom': mpl.rcParams['keymap.zoom'][:],
            'save': mpl.rcParams['keymap.save'][:],
            'quit': mpl.rcParams['keymap.quit'][:],
            'grid': mpl.rcParams['keymap.grid'][:],
            'yscale': mpl.rcParams['keymap.yscale'][:],
            'xscale': mpl.rcParams['keymap.xscale'][:],
        }

        # Clear all default keybindings
        mpl.rcParams['keymap.fullscreen'] = []
        mpl.rcParams['keymap.home'] = []
        mpl.rcParams['keymap.back'] = []
        mpl.rcParams['keymap.forward'] = []
        mpl.rcParams['keymap.pan'] = []
        mpl.rcParams['keymap.zoom'] = []
        mpl.rcParams['keymap.save'] = []
        mpl.rcParams['keymap.quit'] = []
        mpl.rcParams['keymap.grid'] = []
        mpl.rcParams['keymap.yscale'] = []
        mpl.rcParams['keymap.xscale'] = []

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

    def _restore_matplotlib_keymaps(self) -> None:
        """Restore original matplotlib keymaps."""
        if hasattr(self, '_original_keymaps'):
            import matplotlib as mpl
            for key, value in self._original_keymaps.items():
                mpl.rcParams[f'keymap.{key}'] = value

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

        # Update status display
        self._update_status_display()

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
