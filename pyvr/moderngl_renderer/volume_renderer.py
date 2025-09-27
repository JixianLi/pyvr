import os
import numpy as np
from PIL import Image

from .moderngl_manager import ModernGLManager
from .transfer_functions import OpacityTransferFunction, ColorTransferFunction


class VolumeRenderer:
    def __init__(self, width=512, height=512, step_size=0.01, max_steps=200, ambient_light=0.2, diffuse_light=0.8, light_position=(1.0, 1.0, 1.0), light_target=(0.0, 0.0, 0.0)):
        """
        Initializes the volume renderer with specified rendering parameters and OpenGL resources.

        Parameters:
            width (int):            # The width of the rendering viewport (default: 512).
            height (int):           # The height of the rendering viewport (default: 512).
            step_size (float):      # The step size for ray marching in the volume (default: 0.01).
            max_steps (int):        # The maximum number of steps for ray marching (default: 200).
            ambient_light (float):  # The intensity of ambient lighting (default: 0.2).
            diffuse_light (float):  # The intensity of diffuse lighting (default: 0.8).
            light_position (tuple): # The (x, y, z) position of the light source (default: (1.0, 1.0, 1.0)).
            light_target (tuple):   # The (x, y, z) target point the light is directed at (default: (0.0, 0.0, 0.0)).

        Initializes OpenGL context, loads shaders, creates framebuffer, and sets up geometry and shader uniforms for volume rendering.
        """
        self.width = width
        self.height = height
        self.step_size = step_size
        self.max_steps = max_steps
        self.ambient_light = ambient_light
        self.diffuse_light = diffuse_light
        self.light_position = np.array(light_position, dtype=np.float32)
        self.light_target = np.array(light_target, dtype=np.float32)

        # Create ModernGL manager
        self.gl_manager = ModernGLManager(width, height)

        # Load shaders from external files
        shader_dir = os.path.join(os.path.dirname(__file__), "shaders")
        vertex_shader_path = os.path.join(shader_dir, "volume.vert.glsl")
        fragment_shader_path = os.path.join(shader_dir, "volume.frag.glsl")
        self.gl_manager.load_shaders(vertex_shader_path, fragment_shader_path)

        # Set default uniforms
        self.gl_manager.set_uniform_float('step_size', self.step_size)
        self.gl_manager.set_uniform_int('max_steps', self.max_steps)
        self.gl_manager.set_uniform_vector('volume_min_bounds', (-0.5, -0.5, -0.5))
        self.gl_manager.set_uniform_vector('volume_max_bounds', (0.5, 0.5, 0.5))
        self.gl_manager.set_uniform_float('ambient_light', self.ambient_light)
        self.gl_manager.set_uniform_float('diffuse_light', self.diffuse_light)
        self.gl_manager.set_uniform_vector('light_position', self.light_position)
        self.gl_manager.set_uniform_vector('light_target', self.light_target)

    def load_volume(self, volume_data):
        """Load 3D volume data into a texture. volume_data should always be in shape (D, H, W)"""
        if len(volume_data.shape) != 3:
            raise ValueError("Volume data must be 3D")

        # Create volume texture and bind to texture unit 0
        texture_unit = self.gl_manager.create_volume_texture(volume_data)
        self.gl_manager.set_uniform_int('volume_texture', texture_unit)

    def load_normal_volume(self, normal_volume):
        """Load 3D normal data into a texture (shape: D, H, W, 3)"""
        if normal_volume.shape[-1] != 3:
            raise ValueError(
                "Normal volume must have 3 channels (last dimension).")
        
        # Create normal texture and bind to texture unit 1
        texture_unit = self.gl_manager.create_normal_texture(normal_volume)
        self.gl_manager.set_uniform_int('normal_volume', texture_unit)

    def set_camera(self, position, target=(0, 0, 0), up=(0, 1, 0)):
        """Set camera position and orientation"""
        position = np.array(position, dtype=np.float32)
        target = np.array(target, dtype=np.float32)
        up = np.array(up, dtype=np.float32)

        # Create view matrix
        forward = target - position
        forward = forward / np.linalg.norm(forward)

        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)

        up = np.cross(right, forward)

        view_matrix = np.array([
            [right[0], up[0], -forward[0], 0],
            [right[1], up[1], -forward[1], 0],
            [right[2], up[2], -forward[2], 0],
            [-np.dot(right, position), -np.dot(up, position),
             np.dot(forward, position), 1]
        ], dtype=np.float32)

        # Create projection matrix (perspective)
        fov = np.radians(45)
        aspect = self.width / self.height
        near, far = 0.1, 100.0

        f = 1.0 / np.tan(fov / 2.0)
        projection_matrix = np.array([
            [f / aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far + near) / (near - far),
             (2 * far * near) / (near - far)],
            [0, 0, -1, 0]
        ], dtype=np.float32)

        # Set uniforms
        self.gl_manager.set_uniform_matrix('view_matrix', view_matrix)
        self.gl_manager.set_uniform_matrix('projection_matrix', projection_matrix)
        self.gl_manager.set_uniform_vector('camera_pos', tuple(position))

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
        image = Image.frombytes('RGBA', (self.width, self.height), data)
        image = image.transpose(Image.FLIP_TOP_BOTTOM)  # Flip vertically

        return image

    def set_volume_bounds(self, min_bounds=(-0.5, -0.5, -0.5), max_bounds=(0.5, 0.5, 0.5)):
        """Set the world space bounding box for the volume"""
        self.gl_manager.set_uniform_vector('volume_min_bounds', tuple(min_bounds))
        self.gl_manager.set_uniform_vector('volume_max_bounds', tuple(max_bounds))

    def set_step_size(self, step_size):
        """Set the ray marching step size."""
        self.step_size = step_size
        self.gl_manager.set_uniform_float('step_size', step_size)

    def set_max_steps(self, max_steps):
        """Set the maximum number of ray marching steps."""
        self.max_steps = max_steps
        self.gl_manager.set_uniform_int('max_steps', max_steps)

    def set_ambient_light(self, ambient_light):
        """Set the ambient light intensity."""
        self.ambient_light = ambient_light
        self.gl_manager.set_uniform_float('ambient_light', ambient_light)

    def set_diffuse_light(self, diffuse_light):
        """Set the diffuse light intensity."""
        self.diffuse_light = diffuse_light
        self.gl_manager.set_uniform_float('diffuse_light', diffuse_light)

    def set_light_position(self, light_position):
        """Set the position of the light source."""
        self.light_position = np.array(light_position, dtype=np.float32)
        self.gl_manager.set_uniform_vector('light_position', self.light_position)

    def set_light_target(self, light_target):
        """Set the target point the light is pointing to."""
        self.light_target = np.array(light_target, dtype=np.float32)
        self.gl_manager.set_uniform_vector('light_target', self.light_target)

    @property
    def ctx(self):
        """Get the ModernGL context for backward compatibility."""
        return self.gl_manager.get_context()
    
    @property 
    def program(self):
        """Get the shader program for backward compatibility."""
        return self.gl_manager.program


