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
