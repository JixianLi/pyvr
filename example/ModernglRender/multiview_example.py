import numpy as np
import matplotlib.pyplot as plt
import matplotlib 
import time

from pyvr.moderngl_renderer.volume_renderer import VolumeRenderer
from pyvr.datasets import compute_normal_volume, create_sample_volume
from pyvr.moderngl_renderer.transfer_functions import ColorTransferFunction, OpacityTransferFunction
from pyvr.moderngl_renderer.camera_control import get_camera_pos

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