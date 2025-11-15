# ABOUTME: Transfer function demonstration with different opacity patterns and colormaps
# ABOUTME: Shows 4 TF combinations in 2x2 grid with visualization plots

"""
Transfer Functions Demonstration

This example demonstrates different transfer function configurations:
1. Linear opacity with viridis colormap
2. Peak opacity with plasma colormap
3. Step opacity with coolwarm colormap
4. Wide peaks opacity with jet colormap

Transfer functions control how volume data values map to colors and opacity,
allowing you to emphasize different features in the data.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from pyvr.camera import Camera
from pyvr.config import RenderConfig
from pyvr.datasets import create_sample_volume, compute_normal_volume
from pyvr.lighting import Light
from pyvr.moderngl_renderer import VolumeRenderer
from pyvr.transferfunctions import ColorTransferFunction, OpacityTransferFunction
from pyvr.volume import Volume

# Rendering parameters
VOLUME_SIZE = 128  # Volume dimensions (128x128x128)
IMAGE_RES = 384    # Output resolution per view
LUT_SIZE = 256     # Transfer function lookup table size


def main():
    """Run the transfer functions demo."""
    print("Transfer Functions Demo")
    print("=" * 50)

    # Step 1: Create volume with normals
    # Using torus dataset as it shows transfer function effects well
    print("Creating volume...")
    volume_data = create_sample_volume(VOLUME_SIZE, 'torus')
    normals = compute_normal_volume(volume_data)
    volume = Volume(
        data=volume_data,
        normals=normals,
        min_bounds=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
        max_bounds=np.array([1.0, 1.0, 1.0], dtype=np.float32)
    )

    # Step 2: Create renderer with consistent settings
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

    # Use isometric view for all renders (good 3D perspective)
    camera = Camera.isometric_view(distance=3.0)
    renderer.set_camera(camera)

    # Step 3: Define different transfer function combinations
    print("Creating transfer functions...")

    # Configuration 1: Linear opacity with viridis
    # Simple linear ramp from transparent to semi-opaque
    tf_configs = [
        {
            'ctf': ColorTransferFunction.from_colormap('viridis'),
            'otf': OpacityTransferFunction.linear(0.0, 0.1),
            'title': 'Linear Opacity\nViridis Colors'
        },

        # Configuration 2: Peak opacity with plasma
        # Highlights specific density values with opacity peaks
        {
            'ctf': ColorTransferFunction.from_colormap('plasma'),
            'otf': OpacityTransferFunction.peaks(
                peaks=[0.3, 0.7],  # Peak at 30% and 70% density
                opacity=0.3,        # Peak opacity value
                eps=0.05,           # Peak width
                base=0.0            # Base opacity elsewhere
            ),
            'title': 'Peak Opacity (0.3, 0.7)\nPlasma Colors'
        },

        # Configuration 3: Step opacity with coolwarm
        # Step function creating sharp transition
        {
            'ctf': ColorTransferFunction.from_colormap('coolwarm'),
            'otf': OpacityTransferFunction.one_step(
                step=0.5,       # Step at 50% density
                low=0.0,        # Opacity below step
                high=0.15       # Opacity above step
            ),
            'title': 'Step Opacity (at 0.5)\nCoolwarm Colors'
        },

        # Configuration 4: Wide peaks opacity with jet
        # Multiple wide peaks at different positions
        {
            'ctf': ColorTransferFunction.from_colormap('jet'),
            'otf': OpacityTransferFunction.peaks(
                peaks=[0.25, 0.75],  # Different peak positions than config 2
                opacity=0.2,          # Different opacity value
                eps=0.1,              # Wider peaks
                base=0.02             # Small base opacity
            ),
            'title': 'Wide Peaks (0.25, 0.75)\nJet Colors'
        }
    ]

    # Step 4: Render all configurations
    print(f"Rendering {len(tf_configs)} configurations...")

    results = []
    for idx, config in enumerate(tf_configs):
        # Set transfer functions for this configuration
        renderer.set_transfer_functions(config['ctf'], config['otf'])

        # Render
        data = renderer.render()
        image = np.frombuffer(data, dtype=np.uint8).reshape(
            (IMAGE_RES, IMAGE_RES, 4)
        )

        # Store result with TF data for visualization
        results.append({
            'image': image,
            'title': config['title'],
            'ctf': config['ctf'],
            'otf': config['otf']
        })

        print(f"  Rendered: {config['title'].split(chr(10))[0]}")  # First line of title

    # Step 5: Display results in 2x2 grid with TF visualizations
    print("Creating visualization...")

    fig = plt.figure(figsize=(12, 10))
    # Grid: 2 rows of images, 2 rows of transfer function plots
    gs = GridSpec(4, 2, height_ratios=[2, 0.5, 2, 0.5], hspace=0.3, wspace=0.2)

    for idx, result in enumerate(results):
        row = (idx // 2) * 2  # 0 or 2 (skip TF rows)
        col = idx % 2

        # Render image
        ax_image = fig.add_subplot(gs[row, col])
        ax_image.imshow(result['image'], origin='lower')
        ax_image.set_title(result['title'], fontsize=10)
        ax_image.axis('off')

        # Transfer function plot
        ax_tf = fig.add_subplot(gs[row + 1, col])

        # Generate LUTs for visualization
        x = np.linspace(0, 1, LUT_SIZE)
        color_lut = result['ctf'].to_lut(LUT_SIZE)
        opacity_lut = result['otf'].to_lut(LUT_SIZE)

        # Show colormap as background
        ax_tf.imshow(
            color_lut[np.newaxis, :, :],
            aspect='auto',
            extent=[0, 1, 0, 1]
        )

        # Overlay opacity curve
        ax_tf.plot(x, opacity_lut, 'k-', linewidth=2, label='Opacity')
        ax_tf.set_xlim(0, 1)
        ax_tf.set_ylim(0, 1)
        ax_tf.set_xlabel('Data Value', fontsize=8)
        ax_tf.set_ylabel('Opacity', fontsize=8)
        ax_tf.tick_params(labelsize=7)

    plt.suptitle(
        'Transfer Function Configurations',
        fontsize=14,
        fontweight='bold'
    )
    plt.show()

    print("=" * 50)
    print("Demo complete!")


if __name__ == "__main__":
    main()
