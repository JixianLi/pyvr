#!/usr/bin/env python3
"""
PyVR Performance Benchmark

This script benchmarks PyVR volume rendering performance across different
quality presets introduced in v0.2.6.

Updated for PyVR v0.2.6:
- Compares RenderConfig presets (preview, fast, balanced, high_quality)
- Uses new Volume class (v0.2.5)
- Uses new Camera class (v0.2.3)
- Uses new Light class (v0.2.4)
"""

import time

import numpy as np

from pyvr.camera import Camera
from pyvr.config import RenderConfig
from pyvr.datasets import compute_normal_volume, create_sample_volume
from pyvr.lighting import Light
from pyvr.moderngl_renderer import VolumeRenderer
from pyvr.transferfunctions import ColorTransferFunction, OpacityTransferFunction
from pyvr.volume import Volume


def benchmark_config_preset(config_name, config, volume, camera, light, ctf, otf, n_runs=10):
    """Benchmark a specific RenderConfig preset."""
    IMAGE_RES = 512

    # Create renderer with config
    renderer = VolumeRenderer(IMAGE_RES, IMAGE_RES, config=config, light=light)

    # Load volume and set rendering parameters
    renderer.load_volume(volume)
    renderer.set_camera(camera)
    renderer.set_transfer_functions(ctf, otf)

    # Warm up
    _ = renderer.render()

    # Benchmark
    times = []
    for _ in range(n_runs):
        start_time = time.perf_counter()
        data = renderer.render()
        end_time = time.perf_counter()
        times.append((end_time - start_time) * 1000)  # Convert to milliseconds

    # Statistics
    mean_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    fps = 1000 / mean_time

    # Validate output
    data_array = np.frombuffer(data, dtype=np.uint8).reshape((IMAGE_RES, IMAGE_RES, 4))
    non_zero_pixels = np.count_nonzero(data_array.max(axis=2))
    coverage = non_zero_pixels / (IMAGE_RES * IMAGE_RES) * 100

    return {
        "name": config_name,
        "mean_time": mean_time,
        "std_time": std_time,
        "min_time": min_time,
        "max_time": max_time,
        "fps": fps,
        "coverage": coverage,
        "config": config,
    }


def benchmark_render_performance():
    print("=" * 60)
    print("PyVR v0.2.6 Performance Benchmark")
    print("Comparing RenderConfig Quality Presets")
    print("=" * 60)

    # Test parameters
    VOLUME_SIZE = 256
    N_RUNS = 10

    # Create volume (v0.2.5)
    print(f"\nLoading {VOLUME_SIZE}³ volume...")
    volume_data = create_sample_volume(VOLUME_SIZE, "double_sphere")
    normals = compute_normal_volume(volume_data)
    volume = Volume(
        data=volume_data,
        normals=normals,
        min_bounds=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
        max_bounds=np.array([1.0, 1.0, 1.0], dtype=np.float32),
    )

    # Create camera (v0.2.3)
    camera = Camera.from_spherical(
        target=np.array([0, 0, 0], dtype=np.float32),
        distance=3.0,
        azimuth=np.pi / 4,
        elevation=np.pi / 6,
        roll=0.0,
        init_up=np.array([0, 0, 1], dtype=np.float32),
    )

    # Create light (v0.2.4)
    light = Light.default()

    # Create transfer functions
    ctf = ColorTransferFunction.from_colormap("plasma")
    otf = OpacityTransferFunction.linear(0.0, 0.1)

    # Define configs to benchmark
    configs = [
        ("Preview", RenderConfig.preview()),
        ("Fast", RenderConfig.fast()),
        ("Balanced", RenderConfig.balanced()),
        ("High Quality", RenderConfig.high_quality()),
    ]

    print(f"\nBenchmarking {len(configs)} quality presets ({N_RUNS} runs each)...")
    print(f"Resolution: 512x512 pixels\n")

    results = []
    for config_name, config in configs:
        print(f"Testing {config_name}... ", end="", flush=True)
        result = benchmark_config_preset(
            config_name, config, volume, camera, light, ctf, otf, n_runs=N_RUNS
        )
        results.append(result)
        print(f"✓ {result['mean_time']:.2f} ms ({result['fps']:.1f} FPS)")

    # Print detailed results
    print("\n" + "=" * 60)
    print("Detailed Results")
    print("=" * 60)
    print(
        f"{'Preset':<15} {'Mean (ms)':<12} {'Std (ms)':<10} {'FPS':<8} {'Speedup':<10}"
    )
    print("-" * 60)

    baseline_time = results[2]["mean_time"]  # Balanced as baseline

    for result in results:
        speedup = baseline_time / result["mean_time"]
        print(
            f"{result['name']:<15} "
            f"{result['mean_time']:>6.2f} ± {result['std_time']:<4.2f}  "
            f"{result['fps']:>6.1f}   "
            f"{speedup:>4.2f}x"
        )

    # Print configuration details
    print("\n" + "=" * 60)
    print("Configuration Details")
    print("=" * 60)
    print(f"{'Preset':<15} {'Step Size':<12} {'Max Steps':<12} {'Samples/Ray':<12}")
    print("-" * 60)

    for result in results:
        config = result["config"]
        samples = config.estimate_samples_per_ray()
        print(
            f"{result['name']:<15} "
            f"{config.step_size:<12.4f} "
            f"{config.max_steps:<12} "
            f"{samples:<12}"
        )

    # Performance recommendations
    print("\n" + "=" * 60)
    print("Recommendations")
    print("=" * 60)
    print("✓ Preview:      Use for rapid iteration and testing")
    print("✓ Fast:         Use for interactive exploration")
    print("✓ Balanced:     Use for general visualization (default)")
    print("✓ High Quality: Use for final renders and screenshots")

    print("\n" + "=" * 60)
    print("Benchmark Complete")
    print("=" * 60)


if __name__ == "__main__":
    benchmark_render_performance()
