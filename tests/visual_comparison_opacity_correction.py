"""Visual comparison tool for opacity correction across quality presets.

This script renders a test volume at all quality presets and saves images
for visual comparison. Use this to verify that all presets produce similar
overall opacity/brightness.

Usage:
    python tests/visual_comparison_opacity_correction.py

Output:
    - opacity_comparison_preview.png
    - opacity_comparison_fast.png
    - opacity_comparison_balanced.png
    - opacity_comparison_high_quality.png
    - opacity_comparison_ultra_quality.png
"""

from pyvr.moderngl_renderer import VolumeRenderer
from pyvr.config import RenderConfig
from pyvr.volume import Volume
from pyvr.datasets import create_sample_volume
from pyvr.transferfunctions import ColorTransferFunction, OpacityTransferFunction
from PIL import Image
import numpy as np


def render_all_presets():
    """Render test volume at all presets for visual comparison."""
    print("Creating test volume...")
    volume_data = create_sample_volume(128, "double_sphere")
    volume = Volume(data=volume_data)

    print("Setting up renderer...")
    renderer = VolumeRenderer(width=512, height=512)
    renderer.load_volume(volume)

    # Use same transfer functions for all
    ctf = ColorTransferFunction.from_colormap("viridis")
    otf = OpacityTransferFunction.linear(0.0, 0.3)
    renderer.set_transfer_functions(ctf, otf)

    presets = ["preview", "fast", "balanced", "high_quality", "ultra_quality"]

    print("\nRendering at all presets...")
    stats = []

    for preset_name in presets:
        print(f"\n{preset_name}:")

        # Get and apply preset
        preset_method = getattr(RenderConfig, preset_name)
        config = preset_method()
        renderer.set_config(config)

        print(f"  step_size: {config.step_size}")
        print(f"  reference_step_size: {config.reference_step_size}")

        # Render
        image = renderer.render_to_pil()

        # Convert to numpy for stats
        img_array = np.array(image)

        # Compute statistics
        mean_alpha = img_array[:, :, 3].mean()
        max_alpha = img_array[:, :, 3].max()
        mean_rgb = img_array[:, :, :3].mean()

        print(f"  Mean alpha: {mean_alpha:.1f}")
        print(f"  Max alpha: {max_alpha}")
        print(f"  Mean RGB: {mean_rgb:.1f}")

        stats.append(
            {
                "preset": preset_name,
                "mean_alpha": mean_alpha,
                "max_alpha": max_alpha,
                "mean_rgb": mean_rgb,
            }
        )

        # Save image
        filename = f"opacity_comparison_{preset_name}.png"
        image.save(filename)
        print(f"  Saved: {filename}")

    # Print comparison summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Preset':<20} {'Mean Alpha':>12} {'Max Alpha':>12} {'Mean RGB':>12}")
    print("-" * 60)

    for stat in stats:
        print(
            f"{stat['preset']:<20} {stat['mean_alpha']:>12.1f} "
            f"{stat['max_alpha']:>12} {stat['mean_rgb']:>12.1f}"
        )

    print("\n✓ Visual comparison images saved")
    print("\nManual Verification:")
    print("- All images should have similar overall opacity/brightness")
    print("- Differences should only be in smoothness (sampling quality)")
    print("- No image should be significantly lighter or darker than others")


if __name__ == "__main__":
    render_all_presets()
