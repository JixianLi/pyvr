# ABOUTME: Camera system demonstration with presets, controller, and animation
# ABOUTME: Shows 6 different views in 2x3 grid using camera capabilities

"""
Camera System Demonstration

This example demonstrates PyVR's camera system capabilities:
1. Camera presets (front, side, top, isometric views)
2. CameraController for manual manipulation (orbit, zoom, pan)
3. CameraPath for animation and interpolation

The example renders the same volume from 6 different camera positions
to showcase the flexibility of the camera system.
"""

import numpy as np
import matplotlib.pyplot as plt

from pyvr.camera import Camera, CameraController, CameraPath
from pyvr.config import RenderConfig
from pyvr.datasets import create_sample_volume, compute_normal_volume
from pyvr.lighting import Light
from pyvr.moderngl_renderer import VolumeRenderer
from pyvr.transferfunctions import ColorTransferFunction, OpacityTransferFunction
from pyvr.volume import Volume

# Rendering parameters
VOLUME_SIZE = 128  # Volume dimensions (128x128x128)
IMAGE_RES = 384    # Output resolution per view (smaller for multi-view)


def main():
    """Run the camera system demo."""
    print("Camera System Demo")
    print("=" * 50)

    # Step 1: Create volume with normals for lighting
    # Using double_sphere dataset as it looks interesting from multiple angles
    print("Creating volume...")
    volume_data = create_sample_volume(VOLUME_SIZE, 'double_sphere')
    normals = compute_normal_volume(volume_data)
    volume = Volume(
        data=volume_data,
        normals=normals,
        min_bounds=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
        max_bounds=np.array([1.0, 1.0, 1.0], dtype=np.float32)
    )

    # Step 2: Create renderer
    # Using balanced preset for good performance with multiple renders
    print("Setting up renderer...")
    config = RenderConfig.balanced()
    light = Light.default()
    renderer = VolumeRenderer(
        width=IMAGE_RES,
        height=IMAGE_RES,
        config=config,
        light=light
    )
    renderer.load_volume(volume)

    # Step 3: Set up transfer functions (same for all views)
    ctf = ColorTransferFunction.from_colormap('plasma')
    otf = OpacityTransferFunction.linear(0.0, 0.1)
    renderer.set_transfer_functions(ctf, otf)

    # Step 4: Demonstrate camera presets
    # PyVR provides convenient preset views for common angles
    print("Creating camera views...")

    # Preset 1: Front view (looking along Z-axis)
    camera_front = Camera.front_view(distance=3.0)

    # Preset 2: Side view (looking along X-axis)
    camera_side = Camera.side_view(distance=3.0)

    # Preset 3: Top view (looking down along Y-axis)
    camera_top = Camera.top_view(distance=3.0)

    # Preset 4: Isometric view (45° angles for 3D perspective)
    camera_iso = Camera.isometric_view(distance=3.0)

    # Step 5: Demonstrate CameraController
    # Controller allows programmatic camera manipulation

    # Start from front view and apply transformations
    controller = CameraController(Camera.front_view(distance=3.0))

    # Orbit: rotate horizontally by 45° and up by 30°
    controller.orbit(
        delta_azimuth=np.pi / 4,    # 45 degrees horizontal
        delta_elevation=np.pi / 6    # 30 degrees vertical
    )

    # Get the manipulated camera
    camera_controlled = controller.params

    # Step 6: Demonstrate CameraPath animation
    # CameraPath interpolates between keyframe cameras

    # Define start and end positions for animation
    start_camera = Camera.front_view(distance=2.5)
    end_camera = Camera(
        target=np.array([0.0, 0.0, 0.0]),
        azimuth=3 * np.pi / 4,   # 135 degrees
        elevation=np.pi / 6,      # 30 degrees
        roll=0.0,
        distance=4.0
    )

    # Create path and interpolate to 70% between start and end
    path = CameraPath([start_camera, end_camera])
    camera_animated = path.interpolate(0.7)  # 70% along the path

    # Collect all cameras for rendering
    cameras = [
        (camera_front, "Front View (Preset)"),
        (camera_side, "Side View (Preset)"),
        (camera_top, "Top View (Preset)"),
        (camera_iso, "Isometric View (Preset)"),
        (camera_controlled, "Controlled View\\n(Orbit 45°/30°)"),
        (camera_animated, "Animated View\\n(70% Interpolated)")
    ]

    # Step 7: Render all views and display in 2x3 grid
    print(f"Rendering {len(cameras)} views...")

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()  # Convert 2x3 to flat list for easy indexing

    for idx, (camera, title) in enumerate(cameras):
        # Set camera for this view
        renderer.set_camera(camera)

        # Render the volume
        data = renderer.render()
        image = np.frombuffer(data, dtype=np.uint8).reshape(
            (IMAGE_RES, IMAGE_RES, 4)
        )

        # Display in subplot
        axes[idx].imshow(image, origin='lower')

        # Add title with camera parameters
        param_text = (
            f"Az: {np.degrees(camera.azimuth):.0f}°, "
            f"El: {np.degrees(camera.elevation):.0f}°\\n"
            f"Dist: {camera.distance:.1f}"
        )
        axes[idx].set_title(f"{title}\\n{param_text}", fontsize=9)
        axes[idx].axis('off')

        print(f"  Rendered: {title}")

    plt.suptitle('Camera System Capabilities', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    print("=" * 50)
    print("Demo complete!")


if __name__ == "__main__":
    main()
