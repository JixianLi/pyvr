"""
Multi-view volume rendering example - Updated for PyVR v0.2.0

This example demonstrates the new modular transfer function API introduced in v0.2.0.
Key changes from v0.1.0:
- Transfer functions imported from pyvr.transferfunctions (instead of pyvr.moderngl_renderer.transfer_functions)
- Camera functions imported from pyvr.camera (instead of pyvr.moderngl_renderer.camera_control) - will be available in Phase 3
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib 
import time

from pyvr.moderngl_renderer.volume_renderer import VolumeRenderer
from pyvr.datasets import compute_normal_volume, create_sample_volume
# NEW v0.2.0: Import transfer functions from the new modular location
from pyvr.transferfunctions import ColorTransferFunction, OpacityTransferFunction
# NEW v0.2.0: Import camera functions from the new modular location
from pyvr.camera import get_camera_pos

STEP_SIZE = 1e-3
MAX_STEPS = int(1e3)
VOLUME_SIZE = 256
IMAGE_RES = 224

# Create renderer (now uses ModernGLManager internally for better architecture)
# VolumeRenderer handles high-level operations, ModernGLManager handles OpenGL resources
renderer = VolumeRenderer(IMAGE_RES, IMAGE_RES,
                            step_size=1/VOLUME_SIZE, max_steps=MAX_STEPS)

# Load helix volume and normals
volume = create_sample_volume(VOLUME_SIZE, 'double_sphere')
normals = compute_normal_volume(volume)
renderer.load_volume(volume)
renderer.load_normal_volume(normals)
renderer.set_volume_bounds((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))

# NEW v0.2.0: Use the new transfer function API
# Create color transfer function from matplotlib colormap
ctf = ColorTransferFunction.from_matplotlib_colormap(
    matplotlib.colormaps.get_cmap('plasma'))

# Create linear opacity transfer function
otf = OpacityTransferFunction.linear(0.0, 0.1)

# Alternative: Create opacity transfer function with peaks for interesting effects
# otf = OpacityTransferFunction.peaks([0.3, 0.7], opacity=0.2, eps=0.05, base=0.0)

# Configure lighting (high-level operations handled by VolumeRenderer)
renderer.set_diffuse_light(1.0)
renderer.set_ambient_light(0.0)

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
    fig.add_subplot(gs[1, 1])   # Bottom right
]

# Create subplot for transfer function spanning both columns at bottom
ax_tf = fig.add_subplot(gs[2, :])

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

    # NEW v0.2.0: Upload LUTs using the new ModernGLManager architecture
    # This is the preferred method for v0.2.0
    opacity_tex_unit = otf.to_texture(moderngl_manager=renderer.gl_manager)
    renderer.gl_manager.set_uniform_int('opacity_lut', opacity_tex_unit)

    color_tex_unit = ctf.to_texture(moderngl_manager=renderer.gl_manager)
    renderer.gl_manager.set_uniform_int('color_lut', color_tex_unit)
    
    # Legacy interface (still works but not recommended):
    # opacity_tex = otf.to_texture(renderer.ctx)
    # opacity_tex.use(location=2)
    # renderer.program['opacity_lut'] = 2

    # Render
    start_ns = time.perf_counter_ns()
    data = renderer.render()
    end_ns = time.perf_counter_ns()
    print(f"Render time: {(end_ns - start_ns) / 1e6:.2f} ms")
    data = np.frombuffer(data, dtype=np.uint8).reshape(
        (renderer.height, renderer.width, 4))
    axes_images[i].imshow(data, origin='lower')
    axes_images[i].set_title(
        f"Az: {np.degrees(params['azimuth']):.1f}°, El: {np.degrees(params['elevation']):.1f}°, Roll: {np.degrees(params['roll']):.1f}°")
    axes_images[i].axis('off')

# --- Plot transfer function at the bottom ---
lut_size = 256
x = np.linspace(0, 1, lut_size)
color_lut = ctf.to_lut(lut_size)
opacity_lut = otf.to_lut(lut_size)

# Color bar
ax_tf.imshow(color_lut[np.newaxis, :, :],
            aspect='auto', extent=[0, 1, 0, 1])
# Opacity curve
ax_tf.plot(x, opacity_lut, color='black', linewidth=2)
ax_tf.set_xlim(0, 1)
ax_tf.set_ylim(0, 1)
ax_tf.set_xlabel("Scalar Value")
ax_tf.set_yticks([0, 0.05, 0.1, 0.5, 1.0])
ax_tf.set_ylabel("Opacity / Color")
ax_tf.set_title("Transfer Functions - PyVR v0.2.0 (New Modular API)")

# Show the figure
plt.subplots_adjust(hspace=0.3, wspace=0.2)
plt.show()