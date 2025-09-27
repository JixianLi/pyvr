import matplotlib.pyplot as plt
import time

from pyvr.torch_renderer.volume_renderer import VolumeRenderer
from pyvr.torch_renderer.camera import Camera
from pyvr.torch_renderer.transfer_functions import ColorTransferFunction, OpacityTransferFunction
from pyvr.datasets import create_test_volume

# Configuration variables
DEVICE = 'cpu'
VOLUME_SIZE = 256
CAMERA_DISTANCE_FACTOR = 1.5  # Multiplier for camera distance from volume center
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
N_SAMPLES = 1000
CAMERA_FOV = 45

# VOLUME_TYPE
VOLUME_TYPE = 'torus'  # Options: 'simple', 'complex', 'medical_phantom'

# Lighting parameters
LIGHT_DIRECTION = (1.0, 1.0, 1.0)
AMBIENT_INTENSITY = 0.2
DIFFUSE_INTENSITY = 0.8

# Transfer function parameters
COLORMAP = 'hot' 
OPACITY_MIN = 0.0
OPACITY_MAX = 0.1

# Rendering options
USE_OPACITY = True
USE_COLOR = True
USE_LIGHTING = True
VALUE_RANGE = (0, 1)

# Display options
FIGURE_SIZE = (8, 8)

print("Volume Renderer Performance Test")
print("=" * 40)

# Create renderer with configurable volume size
volume_shape = (VOLUME_SIZE, VOLUME_SIZE, VOLUME_SIZE)
renderer = VolumeRenderer(volume_shape=volume_shape, device=DEVICE)
print(f"Device: {renderer.device}")
print(f"Volume size: {VOLUME_SIZE}³ = {VOLUME_SIZE**3:,} voxels")

# Create camera with configurable parameters
volume_center = VOLUME_SIZE // 2
camera_distance = VOLUME_SIZE * CAMERA_DISTANCE_FACTOR
camera = Camera(
    position=(camera_distance, camera_distance//2, camera_distance),
    target=(volume_center, volume_center, volume_center),
    fov=CAMERA_FOV,
    device=renderer.device
)

# Create test volume
print("Creating test volume...")
volume = create_test_volume(volume_type=VOLUME_TYPE, shape=volume_shape)

# Set up rendering parameters
renderer.set_lighting(
    light_direction=LIGHT_DIRECTION, 
    ambient=AMBIENT_INTENSITY, 
    diffuse=DIFFUSE_INTENSITY
)
renderer.set_color_transfer_function(
    ColorTransferFunction(COLORMAP, device=renderer.device)
)
renderer.set_opacity_transfer_function(
    OpacityTransferFunction.linear(OPACITY_MIN, OPACITY_MAX, device=renderer.device)
)

# Render image with configurable parameters
print(f"\nRendering {IMAGE_WIDTH}×{IMAGE_HEIGHT} image...")
start_time = time.time()

rendered = renderer.render(
    volume=volume,
    camera=camera,
    image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    n_samples=N_SAMPLES,
    use_opacity=USE_OPACITY,
    use_color=USE_COLOR,
    use_lighting=USE_LIGHTING,
    value_range=VALUE_RANGE
)

end_time = time.time()
render_time_ms = (end_time - start_time) * 1000

# Calculate performance metrics
total_samples = IMAGE_WIDTH * IMAGE_HEIGHT * N_SAMPLES
samples_per_ms = total_samples / render_time_ms

print(f"\nPerformance Results:")
print(f"Render time: {render_time_ms:.1f} ms")
print(f"Total samples: {total_samples:,}")
print(f"Samples/ms: {samples_per_ms:,.0f}")
print(f"Resolution: {IMAGE_WIDTH}×{IMAGE_HEIGHT} pixels")
print(f"Ray samples: {N_SAMPLES} per ray")
print(f"Volume size: {VOLUME_SIZE}³ = {VOLUME_SIZE**3:,} voxels")

# Display the result with transfer function maps
fig = plt.figure(figsize=(FIGURE_SIZE[0], FIGURE_SIZE[1] * 1.5))

# Create a grid layout: main image on top, transfer functions below
gs = fig.add_gridspec(3, 1, height_ratios=[4, 0.5, 0.5], hspace=0.3)

# Main rendered image
ax_main = fig.add_subplot(gs[0])
ax_main.imshow(rendered.cpu().numpy())
ax_main.set_title(f'Volume Rendering ({VOLUME_SIZE}³ volume)\n{IMAGE_WIDTH}×{IMAGE_HEIGHT}, {N_SAMPLES} samples, {render_time_ms:.1f}ms')
ax_main.axis('off')

# Color transfer function map
ax_color = fig.add_subplot(gs[1])
color_tf = renderer.color_transfer_function
colors = color_tf.colors.cpu().numpy()
colors_reshaped = colors.reshape(1, -1, 3)
ax_color.imshow(colors_reshaped, aspect='auto', extent=[0, 1, 0, 1])
ax_color.set_xlim(0, 1)
ax_color.set_title(f'Color Transfer Function ({COLORMAP})')
ax_color.set_ylabel('Color')
ax_color.set_yticks([])
ax_color.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax_color.set_xticklabels([])  # Remove x-labels from color map

# Opacity transfer function map
ax_opacity = fig.add_subplot(gs[2])
opacity_tf = renderer.opacity_transfer_function
opacity_tf.plot(ax=ax_opacity)

plt.show()