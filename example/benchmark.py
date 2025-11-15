# ABOUTME: Performance benchmark comparing all quality presets
# ABOUTME: Tests preview, fast, balanced, high_quality, and ultra_quality configs

"""
Quality Presets Performance Benchmark

This example benchmarks PyVR rendering performance across all quality presets.
It measures timing statistics and provides recommendations for each preset's
best use case.

Quality presets control the step_size and max_steps parameters, which determine
how many samples are taken along each ray. More samples = higher quality but slower.
"""

import time

import numpy as np

from pyvr.camera import Camera
from pyvr.config import RenderConfig
from pyvr.datasets import create_sample_volume, compute_normal_volume
from pyvr.lighting import Light
from pyvr.moderngl_renderer import VolumeRenderer
from pyvr.transferfunctions import ColorTransferFunction, OpacityTransferFunction
from pyvr.volume import Volume

# Benchmark parameters
VOLUME_SIZE = 256  # Larger volume for realistic workload
IMAGE_RES = 512    # Standard output resolution
N_RUNS = 10        # Number of timing runs per preset


def benchmark_preset(preset_name, config, volume, camera, light, ctf, otf):
    """
    Benchmark a single quality preset.

    Args:
        preset_name: Name of the preset for display
        config: RenderConfig instance to test
        volume: Volume to render
        camera: Camera configuration
        light: Light configuration
        ctf: Color transfer function
        otf: Opacity transfer function

    Returns:
        Dictionary with timing statistics and configuration details
    """
    # Create fresh renderer for this preset
    renderer = VolumeRenderer(IMAGE_RES, IMAGE_RES, config=config, light=light)
    renderer.load_volume(volume)
    renderer.set_camera(camera)
    renderer.set_transfer_functions(ctf, otf)

    # Warmup: First render is often slower due to GPU initialization
    _ = renderer.render()

    # Benchmark: Run N_RUNS times and collect timings
    times = []
    for _ in range(N_RUNS):
        start = time.perf_counter()
        _ = renderer.render()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to milliseconds

    # Calculate statistics
    times = np.array(times)
    return {
        'name': preset_name,
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
        'fps': 1000 / np.mean(times),
        'config': config
    }


def main():
    """Run the performance benchmark."""
    print("=" * 60)
    print("PyVR Performance Benchmark")
    print("Quality Presets Comparison")
    print("=" * 60)

    # Step 1: Create volume with normals
    # Using larger volume (256³) for realistic performance testing
    print(f"\nCreating {VOLUME_SIZE}³ volume...")
    volume_data = create_sample_volume(VOLUME_SIZE, 'double_sphere')
    normals = compute_normal_volume(volume_data)
    volume = Volume(
        data=volume_data,
        normals=normals,
        min_bounds=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
        max_bounds=np.array([1.0, 1.0, 1.0], dtype=np.float32)
    )

    # Step 2: Create camera and light (same for all tests)
    camera = Camera.from_spherical(
        target=np.array([0.0, 0.0, 0.0]),
        distance=3.0,
        azimuth=np.pi / 4,    # 45 degrees
        elevation=np.pi / 6,  # 30 degrees
        roll=0.0
    )
    light = Light.default()

    # Step 3: Create transfer functions (same for all tests)
    ctf = ColorTransferFunction.from_colormap('plasma')
    otf = OpacityTransferFunction.linear(0.0, 0.1)

    # Step 4: Define presets to benchmark
    # Testing all 5 quality levels from fastest to slowest
    presets = [
        ('Preview', RenderConfig.preview()),
        ('Fast', RenderConfig.fast()),
        ('Balanced', RenderConfig.balanced()),
        ('High Quality', RenderConfig.high_quality()),
        ('Ultra Quality', RenderConfig.ultra_quality())
    ]

    print(f"\nBenchmarking {len(presets)} presets ({N_RUNS} runs each)...")
    print(f"Resolution: {IMAGE_RES}x{IMAGE_RES} pixels\n")

    # Run benchmarks
    results = []
    for preset_name, config in presets:
        print(f"Testing {preset_name}... ", end='', flush=True)
        result = benchmark_preset(
            preset_name, config, volume, camera, light, ctf, otf
        )
        results.append(result)
        print(f"✓ {result['mean']:.2f} ms ({result['fps']:.1f} FPS)")

    # Step 5: Display detailed results
    print("\n" + "=" * 60)
    print("Timing Results")
    print("=" * 60)
    print(f"{'Preset':<15} {'Mean (ms)':<12} {'Std (ms)':<10} {'FPS':<8} {'Speedup':<10}")
    print("-" * 60)

    # Use Balanced as baseline for speedup comparison
    baseline_time = results[2]['mean']  # Balanced is index 2

    for result in results:
        speedup = baseline_time / result['mean']
        print(
            f"{result['name']:<15} "
            f"{result['mean']:>6.2f} ± {result['std']:<4.2f}  "
            f"{result['fps']:>6.1f}   "
            f"{speedup:>4.2f}x"
        )

    # Step 6: Display configuration details
    print("\n" + "=" * 60)
    print("Configuration Details")
    print("=" * 60)
    print(f"{'Preset':<15} {'Step Size':<12} {'Max Steps':<12} {'Samples/Ray':<12}")
    print("-" * 60)

    for result in results:
        config = result['config']
        samples = config.estimate_samples_per_ray()
        print(
            f"{result['name']:<15} "
            f"{config.step_size:<12.4f} "
            f"{config.max_steps:<12} "
            f"{samples:<12}"
        )

    # Step 7: Display recommendations
    print("\n" + "=" * 60)
    print("Recommendations")
    print("=" * 60)
    print("✓ Preview:      Fastest - use for rapid iteration and testing")
    print("✓ Fast:         Interactive - use for real-time exploration")
    print("✓ Balanced:     Default - good quality/performance balance")
    print("✓ High Quality: Production - use for final renders and screenshots")
    print("✓ Ultra Quality: Best - maximum quality for publication")

    print("\n" + "=" * 60)
    print("Benchmark Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
