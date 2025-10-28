"""
ModernGL-based volume renderer implementation.

This module provides the OpenGL/ModernGL backend implementation
of the abstract VolumeRenderer interface.
"""

import os
from typing import Optional

from PIL import Image

from ..renderer.base import VolumeRenderer as VolumeRendererBase
from ..camera import Camera
from ..lighting import Light
from ..transferfunctions import ColorTransferFunction, OpacityTransferFunction
from ..volume import Volume
from .manager import ModernGLManager


class ModernGLVolumeRenderer(VolumeRendererBase):
    """
    ModernGL/OpenGL implementation of VolumeRenderer.

    This is the concrete implementation using ModernGL for GPU-accelerated
    volume rendering. It implements all abstract methods from VolumeRendererBase.
    """

    def __init__(
        self,
        width=512,
        height=512,
        step_size=0.01,
        max_steps=200,
        light=None,
    ):
        """
        Initialize ModernGL volume renderer.

        Args:
            width: Viewport width
            height: Viewport height
            step_size: Ray marching step size
            max_steps: Maximum ray marching steps
            light: Light instance (creates default if None)
        """
        super().__init__(width, height)

        self.step_size = step_size
        self.max_steps = max_steps

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
        self.gl_manager.set_uniform_float("step_size", self.step_size)
        self.gl_manager.set_uniform_int("max_steps", self.max_steps)
        self.gl_manager.set_uniform_vector("volume_min_bounds", (-0.5, -0.5, -0.5))
        self.gl_manager.set_uniform_vector("volume_max_bounds", (0.5, 0.5, 0.5))

        # Set light uniforms
        self._update_light()

    def load_volume(self, volume) -> None:
        """
        Load volume data into renderer.

        Supports both new Volume instances and legacy numpy arrays for
        backward compatibility.

        Args:
            volume: Volume instance (recommended) or 3D numpy array (legacy)

        Raises:
            TypeError: If volume is invalid type
            ValueError: If volume data is invalid

        Example:
            >>> from pyvr.volume import Volume
            >>> vol = Volume(data=volume_data, normals=normals)
            >>> renderer.load_volume(vol)
        """
        if isinstance(volume, Volume):
            # New interface: Volume instance
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

        else:
            # Legacy interface: raw numpy array (backward compatibility)
            import warnings
            import numpy as np

            if not isinstance(volume, np.ndarray):
                raise TypeError(
                    f"Expected Volume instance or numpy array, got {type(volume)}"
                )

            warnings.warn(
                "Passing raw numpy array to load_volume() is deprecated. "
                "Use Volume instance instead: "
                "from pyvr.volume import Volume; "
                "volume = Volume(data=your_array); "
                "renderer.load_volume(volume)",
                DeprecationWarning,
                stacklevel=2,
            )

            # Validate data
            if len(volume.shape) != 3:
                raise ValueError("Volume data must be 3D")

            # Create temporary Volume instance for internal tracking
            self.volume = Volume(data=volume)

            # Load texture
            texture_unit = self.gl_manager.create_volume_texture(volume)
            self.gl_manager.set_uniform_int("volume_texture", texture_unit)

    def load_normal_volume(self, normal_volume):
        """
        Load 3D normal data into a texture.

        .. deprecated:: 0.2.5
            Include normals in Volume instance instead.

        Args:
            normal_volume: 4D array with shape (D, H, W, 3)

        Example:
            >>> # Old way (deprecated)
            >>> renderer.load_normal_volume(normals)
            >>>
            >>> # New way (recommended)
            >>> volume = Volume(data=volume_data, normals=normals)
            >>> renderer.load_volume(volume)
        """
        import warnings

        warnings.warn(
            "load_normal_volume() is deprecated. Include normals in Volume instance: "
            "from pyvr.volume import Volume; "
            "volume = Volume(data=your_data, normals=your_normals); "
            "renderer.load_volume(volume)",
            DeprecationWarning,
            stacklevel=2,
        )

        if normal_volume.shape[-1] != 3:
            raise ValueError("Normal volume must have 3 channels (last dimension).")

        # Create normal texture
        texture_unit = self.gl_manager.create_normal_texture(normal_volume)
        self.gl_manager.set_uniform_int("normal_volume", texture_unit)

        # Update volume's normals if volume exists
        if self.volume is not None:
            self.volume.normals = normal_volume

    def set_volume_bounds(
        self, min_bounds=(-0.5, -0.5, -0.5), max_bounds=(0.5, 0.5, 0.5)
    ):
        """
        Set the world space bounding box for the volume.

        .. deprecated:: 0.2.5
            Set bounds on Volume instance instead.

        Args:
            min_bounds: Minimum corner of bounding box
            max_bounds: Maximum corner of bounding box

        Example:
            >>> # Old way (deprecated)
            >>> renderer.set_volume_bounds((-1, -1, -1), (1, 1, 1))
            >>>
            >>> # New way (recommended)
            >>> volume = Volume(data=volume_data,
            ...                 min_bounds=np.array([-1, -1, -1]),
            ...                 max_bounds=np.array([1, 1, 1]))
            >>> renderer.load_volume(volume)
        """
        import warnings
        import numpy as np

        warnings.warn(
            "set_volume_bounds() is deprecated. Set bounds on Volume instance: "
            "from pyvr.volume import Volume; "
            "volume = Volume(data=your_data, min_bounds=..., max_bounds=...); "
            "renderer.load_volume(volume)",
            DeprecationWarning,
            stacklevel=2,
        )

        self.gl_manager.set_uniform_vector("volume_min_bounds", tuple(min_bounds))
        self.gl_manager.set_uniform_vector("volume_max_bounds", tuple(max_bounds))

        # Update volume bounds if volume exists
        if self.volume is not None:
            self.volume.min_bounds = np.array(min_bounds, dtype=np.float32)
            self.volume.max_bounds = np.array(max_bounds, dtype=np.float32)

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

    def set_step_size(self, step_size):
        """Set ray marching step size."""
        self.step_size = step_size
        self.gl_manager.set_uniform_float("step_size", step_size)

    def set_max_steps(self, max_steps):
        """Set maximum ray marching steps."""
        self.max_steps = max_steps
        self.gl_manager.set_uniform_int("max_steps", max_steps)

    def get_light(self):
        """
        Get current light configuration.

        Returns:
            Light: Current light instance
        """
        return self.light

    def _update_light(self):
        """Update OpenGL uniforms from light configuration."""
        self.gl_manager.set_uniform_float("ambient_light", self.light.ambient_intensity)
        self.gl_manager.set_uniform_float("diffuse_light", self.light.diffuse_intensity)
        self.gl_manager.set_uniform_vector("light_position", tuple(self.light.position))
        self.gl_manager.set_uniform_vector("light_target", tuple(self.light.target))


# For backward compatibility
VolumeRenderer = ModernGLVolumeRenderer
