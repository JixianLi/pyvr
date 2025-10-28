import os
from typing import Optional

from PIL import Image

from ..transferfunctions.color import ColorTransferFunction
from ..transferfunctions.opacity import OpacityTransferFunction
from .manager import ModernGLManager


class VolumeRenderer:
    def __init__(
        self,
        width=512,
        height=512,
        step_size=0.01,
        max_steps=200,
        light=None,
    ):
        """
        Initializes the volume renderer with specified rendering parameters and OpenGL resources.

        Parameters:
            width (int):       # The width of the rendering viewport (default: 512).
            height (int):      # The height of the rendering viewport (default: 512).
            step_size (float): # The step size for ray marching in the volume (default: 0.01).
            max_steps (int):   # The maximum number of steps for ray marching (default: 200).
            light (Light):     # Light configuration. If None, creates default light.

        Initializes OpenGL context, loads shaders, creates framebuffer, and sets up geometry and shader uniforms for volume rendering.
        """
        self.width = width
        self.height = height
        self.step_size = step_size
        self.max_steps = max_steps

        # Initialize light
        if light is None:
            from ..lighting import Light
            self.light = Light.default()
        else:
            self.light = light

        # Create ModernGL manager
        self.gl_manager = ModernGLManager(width, height)

        # Load shaders from shared shader directory
        # Navigate from moderngl_renderer directory to the shared shaders directory
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

    def load_volume(self, volume_data):
        """Load 3D volume data into a texture. volume_data should always be in shape (D, H, W)"""
        if len(volume_data.shape) != 3:
            raise ValueError("Volume data must be 3D")

        # Create volume texture and bind to texture unit 0
        texture_unit = self.gl_manager.create_volume_texture(volume_data)
        self.gl_manager.set_uniform_int("volume_texture", texture_unit)

    def load_normal_volume(self, normal_volume):
        """Load 3D normal data into a texture (shape: D, H, W, 3)"""
        if normal_volume.shape[-1] != 3:
            raise ValueError("Normal volume must have 3 channels (last dimension).")

        # Create normal texture and bind to texture unit 1
        texture_unit = self.gl_manager.create_normal_texture(normal_volume)
        self.gl_manager.set_uniform_int("normal_volume", texture_unit)

    def set_camera(self, camera):
        """
        Set camera configuration using Camera instance.

        Args:
            camera: Camera instance with position and projection parameters

        Example:
            >>> from pyvr.camera import Camera
            >>> camera = Camera.isometric_view(distance=3.0)
            >>> renderer.set_camera(camera)
        """
        from ..camera import Camera

        if not isinstance(camera, Camera):
            raise TypeError(f"Expected Camera instance, got {type(camera)}")

        # Get view and projection matrices from camera
        aspect = self.width / self.height
        view_matrix = camera.get_view_matrix()
        projection_matrix = camera.get_projection_matrix(aspect)

        # Get camera position for lighting calculations
        position, _ = camera.get_camera_vectors()

        # Set uniforms
        self.gl_manager.set_uniform_matrix("view_matrix", view_matrix)
        self.gl_manager.set_uniform_matrix("projection_matrix", projection_matrix)
        self.gl_manager.set_uniform_vector("camera_pos", tuple(position))

    def render(self):
        """Render the volume and return raw framebuffer data"""
        self.gl_manager.clear_framebuffer(0.0, 0.0, 0.0, 0.0)
        self.gl_manager.setup_blending()
        self.gl_manager.render_quad()

        # Read pixels from framebuffer and return raw data
        return self.gl_manager.read_pixels()

    def render_to_pil(self, data=None):
        """Render the volume and return as PIL Image"""
        if data is None:
            data = self.render()

        # Convert to PIL Image
        image = Image.frombytes("RGBA", (self.width, self.height), data)
        image = image.transpose(Image.FLIP_TOP_BOTTOM)  # Flip vertically

        return image

    def set_volume_bounds(
        self, min_bounds=(-0.5, -0.5, -0.5), max_bounds=(0.5, 0.5, 0.5)
    ):
        """Set the world space bounding box for the volume"""
        self.gl_manager.set_uniform_vector("volume_min_bounds", tuple(min_bounds))
        self.gl_manager.set_uniform_vector("volume_max_bounds", tuple(max_bounds))

    def set_step_size(self, step_size):
        """Set the ray marching step size."""
        self.step_size = step_size
        self.gl_manager.set_uniform_float("step_size", step_size)

    def set_max_steps(self, max_steps):
        """Set the maximum number of ray marching steps."""
        self.max_steps = max_steps
        self.gl_manager.set_uniform_int("max_steps", max_steps)

    def _update_light(self):
        """Update OpenGL uniforms from current light configuration."""
        self.gl_manager.set_uniform_float("ambient_light", self.light.ambient_intensity)
        self.gl_manager.set_uniform_float("diffuse_light", self.light.diffuse_intensity)
        self.gl_manager.set_uniform_vector("light_position", tuple(self.light.position))
        self.gl_manager.set_uniform_vector("light_target", tuple(self.light.target))

    def set_light(self, light):
        """
        Set lighting configuration.

        Args:
            light: Light instance with lighting parameters

        Example:
            >>> from pyvr.lighting import Light
            >>> light = Light.directional(direction=[1, -1, 0], ambient=0.3)
            >>> renderer.set_light(light)
        """
        from ..lighting import Light

        if not isinstance(light, Light):
            raise TypeError(f"Expected Light instance, got {type(light)}")

        self.light = light
        self._update_light()

    def get_light(self):
        """
        Get current light configuration.

        Returns:
            Light: Current light instance

        Example:
            >>> renderer = VolumeRenderer()
            >>> light = renderer.get_light()
            >>> print(light)
        """
        return self.light

    def set_transfer_functions(
        self,
        color_transfer_function,
        opacity_transfer_function,
        size: Optional[int] = None,
    ):
        """
        Set transfer functions for volume rendering using combined RGBA texture.

        Args:
            color_transfer_function: ColorTransferFunction for RGB mapping
            opacity_transfer_function: OpacityTransferFunction for alpha mapping
            size: Optional LUT size override (uses maximum of both TF sizes if None)

        Example:
            renderer.set_transfer_functions(ctf, otf)
        """
        rgba_tex_unit = self.gl_manager.create_rgba_transfer_function_texture(
            color_transfer_function, opacity_transfer_function, size
        )
        self.gl_manager.set_uniform_int("transfer_function_lut", rgba_tex_unit)
