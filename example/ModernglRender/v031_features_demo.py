"""
Demonstration of PyVR v0.3.1 interface refinements.

This example showcases all new features:
- FPS counter for performance monitoring
- Quality preset selector
- Camera-linked lighting
- Histogram background in opacity editor
- Automatic quality switching
- Status display
"""

import numpy as np
from pyvr.datasets import create_sample_volume, compute_normal_volume
from pyvr.volume import Volume
from pyvr.interface import InteractiveVolumeRenderer
from pyvr.lighting import Light
from pyvr.config import RenderConfig
from pyvr.camera import Camera


def main():
    """Run v0.3.1 features demo."""
    print("PyVR v0.3.1 Interface Refinements Demo")
    print("=" * 50)

    # Create volume with normals
    print("Creating volume...")
    volume_data = create_sample_volume(128, 'double_sphere')
    normals = compute_normal_volume(volume_data)
    volume = Volume(data=volume_data, normals=normals)

    # Create interface with balanced preset
    print("Initializing interface...")
    interface = InteractiveVolumeRenderer(
        volume=volume,
        width=512,
        height=512,
        config=RenderConfig.balanced()
    )

    # Enable all v0.3.1 features
    print("\nEnabling v0.3.1 features:")

    # 1. FPS Counter (enabled by default)
    interface.state.show_fps = True
    print("  ✓ FPS counter enabled")

    # 2. Histogram Background (enabled by default)
    interface.state.show_histogram = True
    print("  ✓ Histogram background enabled")

    # 3. Camera-Linked Lighting
    interface.set_camera_linked_lighting(
        azimuth_offset=np.pi/4,  # 45° horizontal offset
        elevation_offset=0.0
    )
    print("  ✓ Camera-linked lighting enabled (45° offset)")

    # 4. Automatic Quality Switching (enabled by default)
    interface.state.auto_quality_enabled = True
    print("  ✓ Automatic quality switching enabled")

    # Add some control points for demonstration
    interface.state.add_control_point(0.3, 0.2)
    interface.state.add_control_point(0.7, 0.9)
    print("\n  ✓ Added demonstration control points")

    # Print keyboard shortcuts
    print("\n" + "=" * 50)
    print("KEYBOARD SHORTCUTS:")
    print("  'r'   : Reset view to isometric")
    print("  's'   : Save current rendering")
    print("  'f'   : Toggle FPS counter")
    print("  'h'   : Toggle histogram")
    print("  'l'   : Toggle light linking")
    print("  'q'   : Toggle auto-quality")
    print("  'Esc' : Deselect control point")
    print("  'Del' : Delete selected control point")
    print("\nMOUSE CONTROLS:")
    print("  Image:")
    print("    - Drag: Orbit camera")
    print("    - Scroll: Zoom in/out")
    print("  Opacity Editor:")
    print("    - Left click: Add/select control point")
    print("    - Right click: Remove control point")
    print("    - Drag: Move control point")
    print("\nQUALITY PRESETS:")
    print("  Use radio buttons on right side to switch:")
    print("    - Preview (fastest)")
    print("    - Fast")
    print("    - Balanced")
    print("    - High Quality")
    print("    - Ultra (slowest)")
    print("\nFEATURE DEMONSTRATIONS:")
    print("  1. Watch FPS counter (top-left) during camera movement")
    print("  2. Notice histogram showing data distribution")
    print("  3. Light follows camera as you orbit")
    print("  4. Auto-quality makes interaction smooth (watch FPS)")
    print("=" * 50)

    # Launch interface
    print("\nLaunching interactive interface...")
    interface.show()

    # Demonstration: Capture high quality image after interface closes
    print("\nDemonstration: Capturing ultra-quality image...")
    try:
        # Set up nice view
        interface.camera_controller.params = Camera.isometric_view(distance=3.0)

        # Capture with ultra quality
        path = interface.capture_high_quality_image("v031_demo_render.png")
        print(f"Saved to: {path}")

    except Exception as e:
        print(f"Note: Image capture skipped (interface closed): {e}")

    print("\nDemo complete!")


if __name__ == "__main__":
    main()
