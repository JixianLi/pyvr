"""
Demo of RGBA transfer function texture functionality in PyVR.
This example shows how to use combined RGBA textures for improved performance.
"""

import time

import matplotlib.pyplot as plt
import numpy as np

from pyvr.camera import CameraParameters, get_camera_pos_from_params
from pyvr.datasets import compute_normal_volume, create_sample_volume
from pyvr.moderngl_renderer import VolumeRenderer
from pyvr.transferfunctions import ColorTransferFunction, OpacityTransferFunction

# Rendering parameters
STEP_SIZE = 1e-3
MAX_STEPS = int(1e3)
VOLUME_SIZE = 256
IMAGE_RES = 224

# Create renderer
renderer = VolumeRenderer(
    IMAGE_RES, IMAGE_RES, step_size=STEP_SIZE, max_steps=MAX_STEPS
)

# Load volume data
volume = create_sample_volume(VOLUME_SIZE, "double_sphere")
normals = compute_normal_volume(volume)
renderer.load_volume(volume)
renderer.load_normal_volume(normals)
renderer.set_volume_bounds((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))

# Create transfer functions
ctf = ColorTransferFunction.from_colormap("plasma")
otf = OpacityTransferFunction.linear(0.0, 0.1)

# Configure lighting
renderer.set_diffuse_light(1.0)
renderer.set_ambient_light(0.0)

# Camera positions for multi-view demo
camera_positions = [
    {"azimuth": 0, "elevation": 0, "roll": 0, "distance": 3, "title": "Front View"},
    {
        "azimuth": np.pi / 2,
        "elevation": np.pi / 6,
        "roll": 0,
        "distance": 3,
        "title": "Side View",
    },
    {"azimuth": np.pi, "elevation": 0, "roll": 0, "distance": 3, "title": "Back View"},
    {
        "azimuth": 3 * np.pi / 2,
        "elevation": -np.pi / 6,
        "roll": 0,
        "distance": 3,
        "title": "Other Side",
    },
]

# Create figure for multi-view display
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("PyVR: RGBA Transfer Function Texture Demo", fontsize=16)

for i, camera_params in enumerate(camera_positions):
    # Set camera position
    camera = CameraParameters.from_spherical(
        target=np.array([0, 0, 0], dtype=np.float32),
        distance=camera_params["distance"],
        azimuth=camera_params["azimuth"],
        elevation=camera_params["elevation"],
        roll=camera_params["roll"],
        init_up=np.array([0, 0, 1], dtype=np.float32),
    )
    position, up = get_camera_pos_from_params(camera)
    renderer.set_camera(position=position, target=(0, 0, 0), up=up)

    # Set transfer functions using new RGBA texture API
    renderer.set_transfer_functions(ctf, otf)

    # Render with timing
    start_ns = time.perf_counter_ns()
    data = renderer.render()
    end_ns = time.perf_counter_ns()
    render_time = (end_ns - start_ns) / 1e6

    # Convert to image
    data = np.frombuffer(data, dtype=np.uint8).reshape(
        (renderer.height, renderer.width, 4)
    )

    # Display
    row, col = i // 2, i % 2
    ax = axes[row, col]
    ax.imshow(data, origin="lower")
    ax.set_title(f'{camera_params["title"]}\\n{render_time:.2f} ms')
    ax.axis("off")

# Add transfer function visualization at the bottom
fig.text(
    0.5,
    0.02,
    "PyVR v0.2.2 introduces combined RGBA transfer function textures for improved performance",
    ha="center",
    fontsize=12,
)

plt.tight_layout()
plt.subplots_adjust(top=0.93, bottom=0.07)
plt.show()

print("=== PyVR Demo Complete ===")
print("RGBA transfer function features:")
print("• Combined RGBA transfer function textures for better performance")
print("• Simplified API: renderer.set_transfer_functions(ctf, otf)")
print("• Single shader texture lookup instead of two separate lookups")
print("• Cleaner, more efficient volume rendering pipeline")
