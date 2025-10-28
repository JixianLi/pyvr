"""
Multi-view volume rendering example using PyVR RGBA transfer function textures

Updated for PyVR v0.2.4:
- Uses new Camera class (v0.2.3)
- Uses new Light class (v0.2.4)
"""

import time

import matplotlib.pyplot as plt
import numpy as np

from pyvr.camera import Camera
from pyvr.datasets import compute_normal_volume, create_sample_volume
from pyvr.lighting import Light
from pyvr.moderngl_renderer import VolumeRenderer
from pyvr.transferfunctions import ColorTransferFunction, OpacityTransferFunction

STEP_SIZE = 1e-3
MAX_STEPS = int(1e3)
VOLUME_SIZE = 256
IMAGE_RES = 224

# Configure lighting (v0.2.4)
light = Light.directional(direction=[1, -1, 0], ambient=0.0, diffuse=1.0)

# Create renderer with light
renderer = VolumeRenderer(
    IMAGE_RES, IMAGE_RES, step_size=1 / VOLUME_SIZE, max_steps=MAX_STEPS, light=light
)

# Load helix volume and normals
volume = create_sample_volume(VOLUME_SIZE, "double_sphere")
normals = compute_normal_volume(volume)
renderer.load_volume(volume)
renderer.load_normal_volume(normals)
renderer.set_volume_bounds((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))

# Use plasma colormap
ctf = ColorTransferFunction.from_colormap("plasma")

# Linear opacity from 0 to 0.1
otf = OpacityTransferFunction.linear(0.0, 0.1)

# Camera parameter sets to test
# Camera parameter sets to test
camera_params = [
    {"azimuth": 0, "elevation": 0, "roll": 0, "distance": 3},
    {
        "azimuth": np.random.uniform(0, 2 * np.pi),
        "elevation": np.random.uniform(-np.pi / 4, np.pi / 4),
        "roll": np.random.uniform(-np.pi / 6, np.pi / 6),
        "distance": np.random.uniform(2.5, 3.5),
    },
    {
        "azimuth": np.random.uniform(0, 2 * np.pi),
        "elevation": np.random.uniform(-np.pi / 4, np.pi / 4),
        "roll": np.random.uniform(-np.pi / 6, np.pi / 6),
        "distance": np.random.uniform(2.5, 3.5),
    },
    {
        "azimuth": np.random.uniform(0, 2 * np.pi),
        "elevation": np.random.uniform(-np.pi / 4, np.pi / 4),
        "roll": np.random.uniform(-np.pi / 6, np.pi / 6),
        "distance": np.random.uniform(2.5, 3.5),
    },
]

# --- Prepare figure: 2x2 grid for images with transfer function at bottom ---
n_views = len(camera_params)
fig_width = 6
fig_height = 7

# Create figure with 2x2 grid for images and one subplot for transfer function
fig = plt.figure(figsize=(fig_width, fig_height))
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.4], hspace=0.3)

# Create 2x2 grid for images
axes_images = [
    fig.add_subplot(gs[0, 0]),  # Top left
    fig.add_subplot(gs[0, 1]),  # Top right
    fig.add_subplot(gs[1, 0]),  # Bottom left
    fig.add_subplot(gs[1, 1]),  # Bottom right
]

# Create subplot for transfer function spanning both columns at bottom
ax_tf = fig.add_subplot(gs[2, :])

# Rendered images
for i, params in enumerate(camera_params):
    # Create camera using Camera class (v0.2.3)
    camera = Camera.from_spherical(
        target=np.array([0, 0, 0], dtype=np.float32),
        distance=params["distance"],
        azimuth=params["azimuth"],
        elevation=params["elevation"],
        roll=params["roll"],
        init_up=np.array([0, 0, 1], dtype=np.float32),
    )
    renderer.set_camera(camera)

    # Set transfer functions using RGBA texture API
    renderer.set_transfer_functions(ctf, otf)

    # Render
    start_ns = time.perf_counter_ns()
    data = renderer.render()
    end_ns = time.perf_counter_ns()
    print(f"Render time: {(end_ns - start_ns) / 1e6:.2f} ms")
    data = np.frombuffer(data, dtype=np.uint8).reshape(
        (renderer.height, renderer.width, 4)
    )
    axes_images[i].imshow(data, origin="lower")
    axes_images[i].set_title(
        f"Az: {np.degrees(params['azimuth']):.1f}°, El: {np.degrees(params['elevation']):.1f}°, Roll: {np.degrees(params['roll']):.1f}°"
    )
    axes_images[i].axis("off")

# --- Plot transfer function at the bottom ---
lut_size = 256
x = np.linspace(0, 1, lut_size)
color_lut = ctf.to_lut(lut_size)
opacity_lut = otf.to_lut(lut_size)

# Color bar
ax_tf.imshow(color_lut[np.newaxis, :, :], aspect="auto", extent=[0, 1, 0, 1])
# Opacity curve
ax_tf.plot(x, opacity_lut, color="black", linewidth=2)
ax_tf.set_xlim(0, 1)
ax_tf.set_ylim(0, 1)
ax_tf.set_xlabel("Scalar Value")
ax_tf.set_yticks([0, 0.05, 0.1, 0.5, 1.0])
ax_tf.set_ylabel("Opacity / Color")
ax_tf.set_title("Transfer Functions (used for all 4 views)")

# Show the figure
plt.subplots_adjust(hspace=0.3, wspace=0.2)
plt.show()
