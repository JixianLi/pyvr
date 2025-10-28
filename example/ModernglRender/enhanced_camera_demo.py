"""
Enhanced Multi-view volume rendering example - PyVR with Advanced Camera and Light System

This example demonstrates the enhanced camera and lighting system:
- RenderConfig for quality presets (v0.2.6)
- Camera parameter management with validation and presets (v0.2.3)
- Camera animation and path interpolation
- Interactive camera controller
- Light configuration with presets (v0.2.4)
- Volume data management with Volume class (v0.2.5)
- RGBA transfer function textures for improved performance

Key features (updated for v0.2.6):
1. Rendering configuration with quality presets: pyvr.config (v0.2.6)
2. Modular transfer functions: pyvr.transferfunctions
3. Enhanced camera system: pyvr.camera with Camera class (v0.2.3)
4. Enhanced lighting system: pyvr.lighting with Light class (v0.2.4)
5. Volume data management: pyvr.volume with Volume class (v0.2.5)
6. Parameter validation and preset views
7. Camera animation capabilities
8. Single RGBA texture lookup for better performance
"""

import time

import matplotlib.pyplot as plt
import numpy as np

from pyvr.camera import Camera, CameraController, CameraPath
from pyvr.config import RenderConfig
from pyvr.datasets import compute_normal_volume, create_sample_volume
from pyvr.lighting import Light
from pyvr.moderngl_renderer import VolumeRenderer
from pyvr.transferfunctions import ColorTransferFunction, OpacityTransferFunction
from pyvr.volume import Volume

VOLUME_SIZE = 256
IMAGE_RES = 224

# Configure lighting (v0.2.4)
light = Light.directional(direction=[1, -1, 0], ambient=0.0, diffuse=1.0)

# Create renderer with high quality config and light (v0.2.6)
config = RenderConfig.high_quality()
renderer = VolumeRenderer(IMAGE_RES, IMAGE_RES, config=config, light=light)

# Create Volume with data, normals, and bounds (v0.2.5)
volume_data = create_sample_volume(VOLUME_SIZE, "double_sphere")
normals = compute_normal_volume(volume_data)
volume = Volume(
    data=volume_data,
    normals=normals,
    min_bounds=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
    max_bounds=np.array([1.0, 1.0, 1.0], dtype=np.float32),
)
renderer.load_volume(volume)

# Enhanced transfer functions
# Create color transfer function from colormap
ctf = ColorTransferFunction.from_colormap("plasma")

# Create interesting opacity transfer function with peaks
otf = OpacityTransferFunction.peaks(
    peaks=[0.3, 0.7],  # Peaks at specific density values
    opacity=0.3,  # Peak opacity
    eps=0.05,  # Peak width
    base=0.0,  # Base opacity
)

# Alternative: Linear opacity (simpler)
# otf = OpacityTransferFunction.linear(0.0, 0.1)

# Demonstrate enhanced camera system
print("=== PyVR Enhanced Camera & Light System Demo (v0.2.5) ===")

# Method 1: Using camera presets (v0.2.3 Camera class)
camera_presets = [
    Camera.front_view(distance=3.0),
    Camera.side_view(distance=3.0),
    Camera.top_view(distance=3.0),
    Camera.isometric_view(distance=3.0),
]

preset_names = ["Front View", "Side View", "Top View", "Isometric View"]

# Method 2: Using CameraController for interactive manipulation
controller = CameraController(Camera.front_view(distance=3.0))
controller.orbit(np.pi / 6, np.pi / 8)  # Slightly angled front view
controlled_params = controller.params.copy()
controlled_params.distance = 3.5  # Zoom out a bit

# Method 3: Camera animation between views
start_view = Camera.front_view(distance=2.5)
end_view = Camera(
    target=np.array([0.0, 0.0, 0.0]),
    azimuth=3 * np.pi / 4,  # 135 degrees
    elevation=np.pi / 6,  # 30 degrees
    roll=0.0,
    distance=4.0,
)
path = CameraPath([start_view, end_view])
animated_view = path.interpolate(0.7)  # 70% along the path

# Combine all views for rendering
all_camera_params = camera_presets + [controlled_params, animated_view]
view_names = preset_names + ["Controller View", "Animated View"]

# Ensure we have exactly 6 views for 2x3 grid
all_camera_params = all_camera_params[:6]
view_names = view_names[:6]

print(f"Rendering {len(all_camera_params)} views with enhanced camera system...")

# --- Setup figure: 2x3 grid for images with transfer function at bottom ---
fig_width = 9
fig_height = 8

fig = plt.figure(figsize=(fig_width, fig_height))
gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.4], hspace=0.3, wspace=0.2)

# Create 2x3 grid for images
axes_images = [
    fig.add_subplot(gs[0, 0]),  # Top left
    fig.add_subplot(gs[0, 1]),  # Top center
    fig.add_subplot(gs[0, 2]),  # Top right
    fig.add_subplot(gs[1, 0]),  # Bottom left
    fig.add_subplot(gs[1, 1]),  # Bottom center
    fig.add_subplot(gs[1, 2]),  # Bottom right
]

# Create subplot for transfer function spanning all columns at bottom
ax_tf = fig.add_subplot(gs[2, :])

# Render all views
for i, (params, view_name) in enumerate(zip(all_camera_params, view_names)):
    print(f"Rendering {view_name}...")

    # Use Camera class (v0.2.3)
    renderer.set_camera(params)

    # Set transfer functions using RGBA texture API
    renderer.set_transfer_functions(ctf, otf)

    # Render
    start_ns = time.perf_counter_ns()
    data = renderer.render()
    end_ns = time.perf_counter_ns()
    render_time = (end_ns - start_ns) / 1e6

    # Display
    data = np.frombuffer(data, dtype=np.uint8).reshape(
        (renderer.height, renderer.width, 4)
    )
    axes_images[i].imshow(data, origin="lower")

    # Title with camera info
    title = f"{view_name}\\n"
    title += f"Az: {np.degrees(params.azimuth):.0f}°, El: {np.degrees(params.elevation):.0f}°"
    title += f"\\nDist: {params.distance:.1f}, Time: {render_time:.1f}ms"
    axes_images[i].set_title(title, fontsize=9)
    axes_images[i].axis("off")

# --- Plot enhanced transfer functions at the bottom ---
lut_size = 256
x = np.linspace(0, 1, lut_size)
color_lut = ctf.to_lut(lut_size)
opacity_lut = otf.to_lut(lut_size)

# Color bar
ax_tf.imshow(color_lut[np.newaxis, :, :], aspect="auto", extent=[0, 1, 0, 1])

# Opacity curve - highlight the peaks
ax_tf.plot(x, opacity_lut, color="white", linewidth=3, label="Opacity")
ax_tf.plot(x, opacity_lut, color="black", linewidth=2)

# Mark the peaks
peak_positions = [0.3, 0.7]
for peak in peak_positions:
    peak_idx = int(peak * lut_size)
    ax_tf.plot(
        peak,
        opacity_lut[peak_idx],
        "ro",
        markersize=8,
        markeredgecolor="white",
        markeredgewidth=2,
    )

ax_tf.set_xlim(0, 1)
ax_tf.set_ylim(0, 1)
ax_tf.set_xlabel("Scalar Value", fontsize=12)
ax_tf.set_ylabel("Opacity / Color", fontsize=12)
ax_tf.set_title(
    "Enhanced Transfer Functions\\n" "Plasma Colormap + Opacity Peaks at 0.3 and 0.7",
    fontsize=12,
)
ax_tf.grid(True, alpha=0.3)

# Add legend
ax_tf.text(
    0.02,
    0.85,
    "Opacity Peaks",
    transform=ax_tf.transAxes,
    fontsize=10,
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
)

# Show camera system information
info_text = "Camera System Features:\\n"
info_text += "• Preset views (Front, Side, Top, Isometric)\\n"
info_text += "• Interactive controller with orbit/zoom/pan\\n"
info_text += "• Camera path animation and interpolation\\n"
info_text += "• Parameter validation and serialization\\n"
info_text += "• RGBA transfer function textures"

fig.text(
    0.02,
    0.02,
    info_text,
    fontsize=9,
    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
)

plt.suptitle(
    "PyVR - Enhanced Camera System & RGBA Transfer Functions",
    fontsize=14,
    fontweight="bold",
)

# Adjust layout and show
plt.tight_layout()
plt.subplots_adjust(top=0.92, bottom=0.12)
plt.show()

print("\\n=== Camera System Demo Complete ===")
print("Key PyVR features demonstrated:")
print("- Modular transfer functions with RGBA textures")
print("- Advanced camera parameter management")
print("- Camera animation and interpolation")
print("- Preset views and interactive controller")
print("- Comprehensive validation and error handling")
