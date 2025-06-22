import os
import moderngl
import numpy as np
from PIL import Image

from transfer_functions import OpacityTransferFunction, ColorTransferFunction


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

        # Create OpenGL context
        self.ctx = moderngl.create_context(standalone=True)

        # Create framebuffer for offscreen rendering
        self.color_texture = self.ctx.texture((width, height), 4)
        self.depth_texture = self.ctx.depth_texture((width, height))
        self.fbo = self.ctx.framebuffer(
            [self.color_texture], self.depth_texture)

        # Load shaders from external files
        shader_dir = os.path.join(os.path.dirname(__file__), "shaders")
        with open(os.path.join(shader_dir, "volume.vert.glsl"), "r") as f:
            vertex_shader = f.read()
        with open(os.path.join(shader_dir, "volume.frag.glsl"), "r") as f:
            fragment_shader = f.read()

        # Create shader program
        self.program = self.ctx.program(
            vertex_shader=vertex_shader, fragment_shader=fragment_shader)

        # Create fullscreen quad
        vertices = np.array([
            -1.0, -1.0,
            1.0, -1.0,
            1.0,  1.0,
            -1.0,  1.0
        ], dtype=np.float32)

        indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)

        self.vbo = self.ctx.buffer(vertices.tobytes())
        self.ibo = self.ctx.buffer(indices.tobytes())
        self.vao = self.ctx.vertex_array(
            self.program, [(self.vbo, '2f', 'position')], self.ibo)

        # Set default uniforms
        self.program['step_size'] = self.step_size
        self.program['max_steps'] = self.max_steps
        self.program['volume_min_bounds'] = (-0.5, -0.5, -0.5)
        self.program['volume_max_bounds'] = (0.5, 0.5, 0.5)
        self.program['ambient_light'] = self.ambient_light
        self.program['diffuse_light'] = self.diffuse_light
        self.program['light_position'] = self.light_position
        self.program['light_target'] = self.light_target

    def load_volume(self, volume_data):
        """Load 3D volume data into a texture. volume_data should always be in shape (D, H, W)"""
        if len(volume_data.shape) != 3:
            raise ValueError("Volume data must be 3D")

        # Ensure data is float32
        if volume_data.dtype != np.float32:
            volume_data = volume_data.astype(np.float32)

        # Create 3D texture
        self.volume_texture = self.ctx.texture3d(
            volume_data.shape, 1, volume_data.tobytes(), dtype='f4')
        self.volume_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.volume_texture.repeat_x = False
        self.volume_texture.repeat_y = False
        self.volume_texture.repeat_z = False

        # Bind texture to uniform
        self.volume_texture.use(0)
        self.program['volume_texture'] = 0

    def load_normal_volume(self, normal_volume):
        """Load 3D normal data into a texture (shape: D, H, W, 3)"""
        if normal_volume.shape[-1] != 3:
            raise ValueError(
                "Normal volume must have 3 channels (last dimension).")
        self.normal_texture = self.ctx.texture3d(
            normal_volume.shape[:3], 3, normal_volume.tobytes(), dtype='f4')
        self.normal_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.normal_texture.repeat_x = False
        self.normal_texture.repeat_y = False
        self.normal_texture.repeat_z = False
        self.normal_texture.use(1)
        self.program['normal_volume'] = 1

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
        self.program['view_matrix'].write(view_matrix.tobytes())
        self.program['projection_matrix'].write(projection_matrix.tobytes())
        self.program['camera_pos'] = tuple(position)

    def render(self):
        """Render the volume and return raw framebuffer data"""
        self.fbo.use()
        self.ctx.clear(0.0, 0.0, 0.0, 0.0)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        self.vao.render()

        # Read pixels from framebuffer and return raw data
        data = self.fbo.read(components=4)
        return data

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
        self.program['volume_min_bounds'] = tuple(min_bounds)
        self.program['volume_max_bounds'] = tuple(max_bounds)

    def set_step_size(self, step_size):
        """Set the ray marching step size."""
        self.step_size = step_size
        self.program['step_size'] = step_size

    def set_max_steps(self, max_steps):
        """Set the maximum number of ray marching steps."""
        self.max_steps = max_steps
        self.program['max_steps'] = max_steps

    def set_ambient_light(self, ambient_light):
        """Set the ambient light intensity."""
        self.ambient_light = ambient_light
        self.program['ambient_light'] = ambient_light

    def set_diffuse_light(self, diffuse_light):
        """Set the diffuse light intensity."""
        self.diffuse_light = diffuse_light
        self.program['diffuse_light'] = diffuse_light

    def set_light_position(self, light_position):
        """Set the position of the light source."""
        self.light_position = np.array(light_position, dtype=np.float32)
        self.program['light_position'] = self.light_position

    def set_light_target(self, light_target):
        """Set the target point the light is pointing to."""
        self.light_target = np.array(light_target, dtype=np.float32)
        self.program['light_target'] = self.light_target


def compute_normal_volume(volume):
    """Compute normalized gradient (normal) for a 3D volume, shape: (D, H, W) -> (D, H, W, 3)"""
    gx, gy, gz = np.gradient(volume)
    normals = np.stack((gx, gy, gz), axis=-1)
    norm = np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-8
    normals = normals / norm
    return normals.astype(np.float32)


# Multiple view example
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib 
    from datasets import create_sample_volume
    from camera_control import get_camera_pos
    import time

    STEP_SIZE = 1e-3
    MAX_STEPS = int(1e3)
    VOLUME_SIZE = 256
    IMAGE_RES = 224

    # Create renderer
    renderer = VolumeRenderer(IMAGE_RES, IMAGE_RES,
                              step_size=1/VOLUME_SIZE, max_steps=MAX_STEPS)

    # Load helix volume and normals
    volume = create_sample_volume(VOLUME_SIZE, 'double_sphere')
    normals = compute_normal_volume(volume)
    renderer.load_volume(volume)
    renderer.load_normal_volume(normals)
    renderer.set_volume_bounds((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))

    # Use viridis colormap
    ctf = ColorTransferFunction.from_matplotlib_colormap(
        matplotlib.colormaps.get_cmap('plasma'))

    # Linear opacity from 0 to 0.1
    otf = OpacityTransferFunction.linear(0.0, 0.1)

    renderer.set_diffuse_light(1.0);
    renderer.set_ambient_light(0.0);

    # Camera parameter sets to test
    # Camera parameter sets to test
    camera_params = [
        {"azimuth": 0, "elevation": 0, "roll": 0, "distance": 3},
        {
            "azimuth": np.random.uniform(0, 2 * np.pi),
            "elevation": np.random.uniform(-np.pi/4, np.pi/4),
            "roll": np.random.uniform(-np.pi/6, np.pi/6),
            "distance": np.random.uniform(2.5, 3.5)
        },
        {
            "azimuth": np.random.uniform(0, 2 * np.pi),
            "elevation": np.random.uniform(-np.pi/4, np.pi/4),
            "roll": np.random.uniform(-np.pi/6, np.pi/6),
            "distance": np.random.uniform(2.5, 3.5)
        },
        {
            "azimuth": np.random.uniform(0, 2 * np.pi),
            "elevation": np.random.uniform(-np.pi/4, np.pi/4),
            "roll": np.random.uniform(-np.pi/6, np.pi/6),
            "distance": np.random.uniform(2.5, 3.5)
        },
    ]

    # --- Prepare figure: images on top, transfer functions below ---
    n_views = len(camera_params)
    fig_height = 8
    tf_rel_height = 0.25  # Transfer function plot height relative to image
    image_height = fig_height * (1 - tf_rel_height)
    tf_height = fig_height * tf_rel_height
    fig, axes = plt.subplots(
        2, n_views,
        figsize=(6 * n_views, fig_height),
        gridspec_kw={'height_ratios': [image_height, tf_height]}
    )

    # Rendered images
    for i, params in enumerate(camera_params):
        # Compute camera position and up vector
        position, up = get_camera_pos(
            target=np.array([0, 0, 0], dtype=np.float32),
            azimuth=params["azimuth"],
            elevation=params["elevation"],
            roll=params["roll"],
            distance=params["distance"],
            init_pos=np.array([1, 0, 0], dtype=np.float32),
            init_up=np.array([0, 0, 1], dtype=np.float32)
        )
        renderer.set_camera(position=position, target=(0, 0, 0), up=up)

        # Upload LUTs
        opacity_tex = otf.to_texture(renderer.ctx)
        opacity_tex.use(location=2)
        renderer.program['opacity_lut'] = 2

        color_tex = ctf.to_texture(renderer.ctx)
        color_tex.use(location=3)
        renderer.program['color_lut'] = 3

        # Render
        start_ns = time.perf_counter_ns()
        data = renderer.render()
        end_ns = time.perf_counter_ns()
        print(f"Render time: {(end_ns - start_ns) / 1e6:.2f} ms")
        data = np.frombuffer(data, dtype=np.uint8).reshape(
            (renderer.height, renderer.width, 4))
        axes[0, i].imshow(data, origin='lower')
        axes[0, i].set_title(
            f"Az: {np.degrees(params['azimuth']):.1f}°, El: {np.degrees(params['elevation']):.1f}°, Roll: {np.degrees(params['roll']):.1f}°")
        axes[0, i].axis('off')

    # --- Plot transfer functions below each image ---
    lut_size = 256
    x = np.linspace(0, 1, lut_size)
    color_lut = ctf.to_lut(lut_size)
    opacity_lut = otf.to_lut(lut_size)

    for i in range(n_views):
        ax = axes[1, i]
        # Color bar
        ax.imshow(color_lut[np.newaxis, :, :],
                  aspect='auto', extent=[0, 1, 0, 1])
        # Opacity curve
        ax.plot(x, opacity_lut, color='black')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Normalized Intensity")
        ax.set_yticks([0, 0.05, 0.1, 0.5, 1.0])
        ax.set_ylabel("Opacity / Color")
        ax.set_title("Transfer Functions")

    plt.tight_layout()
    plt.show()
