"""
ModernGL-based volume renderer implementation.

This module provides GPU-accelerated volume rendering using OpenGL/ModernGL.
"""

import os
from typing import Optional

from PIL import Image

from ..camera import Camera
from ..lighting import Light
from ..transferfunctions import ColorTransferFunction, OpacityTransferFunction
from ..volume import Volume
from .manager import ModernGLManager


class ModernGLVolumeRenderer:
    """
    GPU-accelerated volume renderer using ModernGL/OpenGL.

    This renderer provides real-time volume rendering with ray marching,
    transfer functions, and advanced lighting.

    Example:
        >>> from pyvr.moderngl_renderer import VolumeRenderer
        >>> from pyvr.config import RenderConfig
        >>>
        >>> config = RenderConfig.balanced()
        >>> renderer = VolumeRenderer(width=512, height=512, config=config)
    """

    def __init__(
        self,
        width=512,
        height=512,
        config=None,
        light=None,
    ):
        """
        Initialize ModernGL volume renderer.

        Args:
            width: Viewport width
            height: Viewport height
            config: RenderConfig instance (uses balanced preset if None)
            light: Light instance (creates default if None)

        Example:
            >>> from pyvr.config import RenderConfig
            >>> from pyvr.moderngl_renderer import VolumeRenderer
            >>>
            >>> # Use preset
            >>> config = RenderConfig.high_quality()
            >>> renderer = VolumeRenderer(width=1024, height=1024, config=config)
            >>>
            >>> # Use defaults (balanced preset)
            >>> renderer = VolumeRenderer(width=512, height=512)
        """
        # Initialize dimensions
        self.width = width
        self.height = height

        # Initialize optional attributes (set by respective methods)
        self.volume: Optional[Volume] = None
        self.camera: Optional[Camera] = None

        # Initialize render config
        if config is None:
            from ..config import RenderConfig

            self.config = RenderConfig.balanced()
        else:
            from ..config import RenderConfig

            if not isinstance(config, RenderConfig):
                raise TypeError(f"Expected RenderConfig instance, got {type(config)}")
            self.config = config

        # Initialize light
        if light is None:
            self.light = Light.default()
        else:
            if not isinstance(light, Light):
                raise TypeError(f"Expected Light instance, got {type(light)}")
            self.light = light

        # Create ModernGL manager
        self.gl_manager = ModernGLManager(width, height)

        # Load shaders
        pyvr_dir = os.path.dirname(os.path.dirname(__file__))
        shader_dir = os.path.join(pyvr_dir, "shaders")
        vertex_shader_path = os.path.join(shader_dir, "volume.vert.glsl")
        fragment_shader_path = os.path.join(shader_dir, "volume.frag.glsl")
        self.gl_manager.load_shaders(vertex_shader_path, fragment_shader_path)

        # Set default uniforms
        self._update_render_config()
        self.gl_manager.set_uniform_vector("volume_min_bounds", (-0.5, -0.5, -0.5))
        self.gl_manager.set_uniform_vector("volume_max_bounds", (0.5, 0.5, 0.5))

        # Set light uniforms
        self._update_light()

    def load_volume(self, volume: Volume) -> None:
        """
        Load volume data into renderer.

        Args:
            volume: Volume instance containing data, normals, and bounds

        Raises:
            TypeError: If volume is not a Volume instance

        Example:
            >>> from pyvr.volume import Volume
            >>> vol = Volume(data=volume_data, normals=normals)
            >>> renderer.load_volume(vol)
        """
        if not isinstance(volume, Volume):
            raise TypeError(
                f"Expected Volume instance, got {type(volume)}. "
                "Create a Volume instance: from pyvr.volume import Volume; "
                "volume = Volume(data=your_array)"
            )

        self.volume = volume

        # Load volume data texture
        texture_unit = self.gl_manager.create_volume_texture(volume.data)
        self.gl_manager.set_uniform_int("volume_texture", texture_unit)

        # Set bounds
        self.gl_manager.set_uniform_vector(
            "volume_min_bounds", tuple(volume.min_bounds)
        )
        self.gl_manager.set_uniform_vector(
            "volume_max_bounds", tuple(volume.max_bounds)
        )

        # Load normals if present
        if volume.has_normals:
            normal_unit = self.gl_manager.create_normal_texture(volume.normals)
            self.gl_manager.set_uniform_int("normal_volume", normal_unit)

    def set_camera(self, camera: Camera) -> None:
        """
        Set camera configuration.

        Args:
            camera: Camera instance

        Raises:
            TypeError: If camera is not a Camera instance
        """
        if not isinstance(camera, Camera):
            raise TypeError(f"Expected Camera instance, got {type(camera)}")

        self.camera = camera

        # Get matrices from camera
        aspect = self.width / self.height
        view_matrix = camera.get_view_matrix()
        projection_matrix = camera.get_projection_matrix(aspect)
        position, _ = camera.get_camera_vectors()

        # Set uniforms
        self.gl_manager.set_uniform_matrix("view_matrix", view_matrix)
        self.gl_manager.set_uniform_matrix("projection_matrix", projection_matrix)
        self.gl_manager.set_uniform_vector("camera_pos", tuple(position))

    def set_light(self, light: Light) -> None:
        """
        Set lighting configuration.

        Args:
            light: Light instance

        Raises:
            TypeError: If light is not a Light instance
        """
        if not isinstance(light, Light):
            raise TypeError(f"Expected Light instance, got {type(light)}")

        self.light = light
        self._update_light()

    def set_transfer_functions(
        self,
        color_transfer_function: ColorTransferFunction,
        opacity_transfer_function: OpacityTransferFunction,
        size: Optional[int] = None,
    ) -> None:
        """
        Set transfer functions.

        Args:
            color_transfer_function: Color transfer function
            opacity_transfer_function: Opacity transfer function
            size: Optional LUT size override
        """
        rgba_tex_unit = self.gl_manager.create_rgba_transfer_function_texture(
            color_transfer_function, opacity_transfer_function, size
        )
        self.gl_manager.set_uniform_int("transfer_function_lut", rgba_tex_unit)

    def render(self) -> bytes:
        """
        Render volume and return raw pixel data.

        Returns:
            Raw RGBA pixel data as bytes
        """
        self.gl_manager.clear_framebuffer(0.0, 0.0, 0.0, 0.0)
        self.gl_manager.setup_blending()
        self.gl_manager.render_quad()
        return self.gl_manager.read_pixels()

    def render_to_pil(self, data=None):
        """Render and return as PIL Image."""
        if data is None:
            data = self.render()

        image = Image.frombytes("RGBA", (self.width, self.height), data)
        return image.transpose(Image.FLIP_TOP_BOTTOM)

    def set_config(self, config):
        """
        Set rendering configuration.

        Args:
            config: RenderConfig instance with rendering parameters

        Raises:
            TypeError: If config is not a RenderConfig instance

        Example:
            >>> from pyvr.config import RenderConfig
            >>> config = RenderConfig.high_quality()
            >>> renderer.set_config(config)
        """
        from ..config import RenderConfig

        if not isinstance(config, RenderConfig):
            raise TypeError(f"Expected RenderConfig instance, got {type(config)}")

        self.config = config
        self._update_render_config()

    def get_config(self):
        """
        Get current rendering configuration.

        Returns:
            RenderConfig: Current configuration

        Example:
            >>> renderer = VolumeRenderer()
            >>> config = renderer.get_config()
            >>> print(config)
        """
        return self.config

    def get_light(self):
        """
        Get current light configuration.

        Returns:
            Light: Current light instance
        """
        return self.light

    def get_volume(self) -> Optional[Volume]:
        """
        Get current volume.

        Returns:
            Current Volume instance or None if not loaded

        Example:
            >>> volume = renderer.get_volume()
            >>> if volume:
            ...     print(f"Volume shape: {volume.shape}")
        """
        return self.volume

    def get_camera(self) -> Optional[Camera]:
        """
        Get current camera.

        Returns:
            Current Camera instance or None if not set

        Example:
            >>> camera = renderer.get_camera()
            >>> if camera:
            ...     print(f"Camera position: {camera.get_camera_vectors()[0]}")
        """
        return self.camera

    def _update_render_config(self):
        """Update OpenGL uniforms from current render configuration."""
        self.gl_manager.set_uniform_float("step_size", self.config.step_size)
        self.gl_manager.set_uniform_int("max_steps", self.config.max_steps)
        self.gl_manager.set_uniform_float(
            "reference_step_size", self.config.reference_step_size
        )

    def _update_light(self):
        """Update OpenGL uniforms from light configuration."""
        self.gl_manager.set_uniform_float("ambient_light", self.light.ambient_intensity)
        self.gl_manager.set_uniform_float("diffuse_light", self.light.diffuse_intensity)
        self.gl_manager.set_uniform_vector("light_position", tuple(self.light.position))
        self.gl_manager.set_uniform_vector("light_target", tuple(self.light.target))


# For backward compatibility
VolumeRenderer = ModernGLVolumeRenderer
