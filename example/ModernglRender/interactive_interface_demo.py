"""
Interactive Volume Renderer Interface Demo

Demonstrates the interactive matplotlib-based interface for volume rendering
with real-time transfer function editing and camera controls.

Usage:
    python example/ModernglRender/interactive_interface_demo.py
"""

import numpy as np
from pyvr.interface import InteractiveVolumeRenderer
from pyvr.volume import Volume
from pyvr.datasets import create_sample_volume, compute_normal_volume


def main():
    """Run interactive interface demo."""
    print("PyVR Interactive Interface Demo")
    print("=" * 50)

    # Create a sample volume
    print("Creating sample volume (double sphere, 128³)...")
    volume_data = create_sample_volume(128, 'double_sphere')
    normals = compute_normal_volume(volume_data)

    # Create Volume object
    volume = Volume(
        data=volume_data,
        normals=normals,
        min_bounds=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
        max_bounds=np.array([1.0, 1.0, 1.0], dtype=np.float32)
    )

    print(f"Volume shape: {volume.shape}")
    print(f"Volume dimensions: {volume.dimensions}")
    print(f"Has normals: {volume.has_normals}")

    # Create interactive interface
    print("\nLaunching interactive interface...")
    interface = InteractiveVolumeRenderer(
        volume=volume,
        width=512,
        height=512
    )

    # Add some initial control points for interesting visualization
    interface.state.add_control_point(0.3, 0.1)
    interface.state.add_control_point(0.5, 0.8)
    interface.state.add_control_point(0.7, 0.2)

    print("\nControls:")
    print("  Mouse Controls:")
    print("    - Image: Drag to orbit camera, scroll to zoom")
    print("    - Opacity Plot: Left-click to add/select, right-click to remove, drag to move")
    print("  Keyboard Shortcuts:")
    print("    - r: Reset camera view")
    print("    - s: Save current rendering to file")
    print("    - Esc: Deselect control point")
    print("    - Delete: Remove selected control point")
    print("\nClose the window to exit.")

    # Show interface (blocking)
    interface.show()

    print("\nDemo completed.")


if __name__ == "__main__":
    main()
