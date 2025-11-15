# ABOUTME: Minimal volume rendering pipeline from data creation to display
# ABOUTME: Demonstrates basic PyVR workflow with sphere dataset and default settings

"""
Basic Volume Rendering Example

This example demonstrates the minimal PyVR pipeline:
1. Create a synthetic volume dataset
2. Set up the renderer with a quality preset
3. Configure camera and lighting
4. Apply transfer functions for color and opacity
5. Render and display the result

This is the simplest complete PyVR workflow.
"""

import numpy as np
import matplotlib.pyplot as plt

from pyvr.camera import Camera
from pyvr.config import RenderConfig
from pyvr.datasets import create_sample_volume
from pyvr.lighting import Light
from pyvr.moderngl_renderer import VolumeRenderer
from pyvr.transferfunctions import ColorTransferFunction, OpacityTransferFunction
from pyvr.volume import Volume

# Rendering parameters
VOLUME_SIZE = 128  # Volume dimensions (128x128x128)
IMAGE_RES = 512    # Output image resolution (512x512 pixels)


def main():
    """Run the basic rendering demo."""
    # Step 1: Create a simple sphere volume
    # Using the sphere dataset as it renders quickly and shows basic features
    print("Creating volume data...")
    volume_data = create_sample_volume(VOLUME_SIZE, 'sphere')

    # Wrap the data in a Volume object
    # Volume handles data, bounds, and optional normal vectors
    volume = Volume(data=volume_data)
    print(f"Volume created: {volume.shape}")

    # Step 2: Create the renderer
    # Using balanced preset for good quality/performance tradeoff
    print("Setting up renderer...")
    config = RenderConfig.balanced()
    light = Light.default()  # Directional light with ambient component
    renderer = VolumeRenderer(
        width=IMAGE_RES,
        height=IMAGE_RES,
        config=config,
        light=light
    )

    # Load the volume into the renderer
    renderer.load_volume(volume)

    # Step 3: Configure the camera
    # Using front view preset at distance 3.0 for good framing
    camera = Camera.front_view(distance=3.0)
    renderer.set_camera(camera)
    print("Renderer configured")

    # Step 4: Set up transfer functions
    # Transfer functions map volume data values to colors and opacity

    # Color transfer function: viridis colormap (purple to yellow)
    ctf = ColorTransferFunction.from_colormap('viridis')

    # Opacity transfer function: linear from transparent to semi-opaque
    # Values: min_opacity=0.0 (fully transparent), max_opacity=0.1 (10% opaque)
    otf = OpacityTransferFunction.linear(0.0, 0.1)

    renderer.set_transfer_functions(ctf, otf)

    # Step 5: Render the volume
    print("Rendering...")
    data = renderer.render()

    # Convert raw bytes to numpy array for display
    # Renderer returns RGBA data as bytes, reshape to image dimensions
    image = np.frombuffer(data, dtype=np.uint8).reshape((IMAGE_RES, IMAGE_RES, 4))

    # Step 6: Display the result
    print("Displaying result...")
    plt.figure(figsize=(8, 8))
    plt.imshow(image, origin='lower')
    plt.title('Basic Volume Rendering - Sphere Dataset')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    print("Rendering complete!")


if __name__ == "__main__":
    main()
