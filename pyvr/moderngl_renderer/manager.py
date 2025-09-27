import os

import moderngl
import numpy as np


class ModernGLManager:
    """
    Manages ModernGL resources and operations for volume rendering.
    Handles OpenGL context, shaders, textures, framebuffers, and rendering operations.
    """

    def __init__(self, width=512, height=512):
        """
        Initialize the ModernGL manager with specified viewport dimensions.

        Parameters:
            width (int): The width of the rendering viewport (default: 512).
            height (int): The height of the rendering viewport (default: 512).
        """
        self.width = width
        self.height = height

        # Create OpenGL context
        self.ctx = moderngl.create_context(standalone=True)

        # Create framebuffer for offscreen rendering
        self.color_texture = self.ctx.texture((width, height), 4)
        self.depth_texture = self.ctx.depth_texture((width, height))
        self.fbo = self.ctx.framebuffer([self.color_texture], self.depth_texture)

        # Initialize shader program and geometry
        self.program = None
        self.vao = None
        self.vbo = None
        self.ibo = None

        # Track texture units
        self._next_texture_unit = 0
        self._texture_bindings = {}

    def load_shaders(self, vertex_shader_path, fragment_shader_path):
        """
        Load and compile vertex and fragment shaders from files.

        Parameters:
            vertex_shader_path (str): Path to the vertex shader file.
            fragment_shader_path (str): Path to the fragment shader file.
        """
        with open(vertex_shader_path, "r") as f:
            vertex_shader = f.read()
        with open(fragment_shader_path, "r") as f:
            fragment_shader = f.read()

        self.program = self.ctx.program(
            vertex_shader=vertex_shader, fragment_shader=fragment_shader
        )

        # Create fullscreen quad geometry
        self._create_fullscreen_quad()

    def _create_fullscreen_quad(self):
        """Create vertex array object for a fullscreen quad."""
        vertices = np.array(
            [-1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0], dtype=np.float32
        )

        indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)

        self.vbo = self.ctx.buffer(vertices.tobytes())
        self.ibo = self.ctx.buffer(indices.tobytes())
        self.vao = self.ctx.vertex_array(
            self.program, [(self.vbo, "2f", "position")], self.ibo
        )

    def create_volume_texture(self, volume_data):
        """
        Create a 3D texture from volume data.

        Parameters:
            volume_data (np.ndarray): 3D volume data with shape (D, H, W).

        Returns:
            int: The texture unit where the texture is bound.
        """
        if len(volume_data.shape) != 3:
            raise ValueError("Volume data must be 3D")

        # Ensure data is float32
        if volume_data.dtype != np.float32:
            volume_data = volume_data.astype(np.float32)

        # Create 3D texture
        texture = self.ctx.texture3d(
            volume_data.shape, 1, volume_data.tobytes(), dtype="f4"
        )
        texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        texture.repeat_x = False
        texture.repeat_y = False
        texture.repeat_z = False

        # Bind to next available texture unit
        texture_unit = self._get_next_texture_unit()
        texture.use(texture_unit)
        self._texture_bindings["volume"] = (texture, texture_unit)

        return texture_unit

    def create_normal_texture(self, normal_data):
        """
        Create a 3D texture from normal data.

        Parameters:
            normal_data (np.ndarray): 3D normal data with shape (D, H, W, 3).

        Returns:
            int: The texture unit where the texture is bound.
        """
        if normal_data.shape[-1] != 3:
            raise ValueError("Normal volume must have 3 channels (last dimension).")

        texture = self.ctx.texture3d(
            normal_data.shape[:3], 3, normal_data.tobytes(), dtype="f4"
        )
        texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        texture.repeat_x = False
        texture.repeat_y = False
        texture.repeat_z = False

        texture_unit = self._get_next_texture_unit()
        texture.use(texture_unit)
        self._texture_bindings["normal"] = (texture, texture_unit)

        return texture_unit

    def create_lut_texture(self, lut_data, channels=1):
        """
        Create a 1D LUT texture from lookup table data.

        Parameters:
            lut_data (np.ndarray): 1D or 2D lookup table data.
            channels (int): Number of channels (1 for opacity, 3 for color).

        Returns:
            int: The texture unit where the texture is bound.
        """
        if channels == 1:
            # Opacity LUT - ensure proper shape
            if lut_data.ndim == 1:
                data = lut_data.reshape((len(lut_data), 1)).astype(np.float32)
            else:
                data = lut_data.astype(np.float32)
            texture = self.ctx.texture(
                (data.shape[0], 1), 1, data.tobytes(), dtype="f4"
            )
        elif channels == 3:
            # Color LUT - ensure proper shape
            if lut_data.ndim == 2 and lut_data.shape[1] == 3:
                data = lut_data.reshape((lut_data.shape[0], 1, 3)).astype(np.float32)
            else:
                raise ValueError("Color LUT must have shape (N, 3)")
            texture = self.ctx.texture(
                (data.shape[0], 1), 3, data.tobytes(), dtype="f4"
            )
        else:
            raise ValueError("Channels must be 1 or 3")

        texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        texture.repeat_x = False
        texture.repeat_y = False

        texture_unit = self._get_next_texture_unit()
        texture.use(texture_unit)

        return texture_unit

    def set_uniform_matrix(self, name, matrix):
        """Set a matrix uniform in the shader program."""
        if self.program is None:
            raise RuntimeError("Shader program not loaded")
        self.program[name].write(matrix.tobytes())

    def set_uniform_vector(self, name, vector):
        """Set a vector uniform in the shader program."""
        if self.program is None:
            raise RuntimeError("Shader program not loaded")
        self.program[name] = tuple(vector)

    def set_uniform_float(self, name, value):
        """Set a float uniform in the shader program."""
        if self.program is None:
            raise RuntimeError("Shader program not loaded")
        self.program[name] = float(value)

    def set_uniform_int(self, name, value):
        """Set an integer uniform in the shader program."""
        if self.program is None:
            raise RuntimeError("Shader program not loaded")
        self.program[name] = int(value)

    def clear_framebuffer(self, r=0.0, g=0.0, b=0.0, a=0.0):
        """Clear the framebuffer with specified color."""
        self.fbo.use()
        self.ctx.clear(r, g, b, a)

    def setup_blending(self):
        """Set up alpha blending for volume rendering."""
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

    def render_quad(self):
        """Render the fullscreen quad."""
        if self.vao is None:
            raise RuntimeError("Vertex array object not created")
        self.vao.render()

    def read_pixels(self):
        """Read pixels from the framebuffer and return raw data."""
        return self.fbo.read(components=4)

    def _get_next_texture_unit(self):
        """Get the next available texture unit."""
        unit = self._next_texture_unit
        self._next_texture_unit += 1
        return unit

    def cleanup(self):
        """Clean up OpenGL resources."""
        if self.vao:
            self.vao.release()
        if self.vbo:
            self.vbo.release()
        if self.ibo:
            self.ibo.release()
        if self.program:
            self.program.release()

        # Clean up textures
        for name, (texture, unit) in self._texture_bindings.items():
            texture.release()

        self.fbo.release()
        self.color_texture.release()
        self.depth_texture.release()
        self.ctx.release()
