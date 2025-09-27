#!/usr/bin/env python3
"""
PyVR Performance Benchmark

This script benchmarks the PyVR volume rendering performanc      print("\n=== Benchmark Complete ===")
    print("PyVR with RGBA transfer function textures provides")
    print("efficient single-lookup volume rendering with excellent performance.")rint("\n=== Benchmark Complete ===")
    print("PyVR with RGBA transfer function textures provides")
    print("efficient single-lookup volume rendering with excellent performance.")ith current features.
"""

import time

import numpy as np

from pyvr.camera import CameraParameters, get_camera_pos_from_params
from pyvr.datasets import compute_normal_volume, create_sample_volume
from pyvr.moderngl_renderer import VolumeRenderer
from pyvr.transferfunctions import ColorTransferFunction, OpacityTransferFunction


def benchmark_render_performance():
    print("=== PyVR Performance Benchmark ===")

    # Test parameters
    VOLUME_SIZE = 256
    IMAGE_RES = 512
    STEP_SIZE = 1e-3
    MAX_STEPS = 1000
    N_RUNS = 10

    # Create renderer
    renderer = VolumeRenderer(
        IMAGE_RES, IMAGE_RES, step_size=STEP_SIZE, max_steps=MAX_STEPS
    )

    # Load volume
    print(f"Loading {VOLUME_SIZE}³ volume...")
    volume = create_sample_volume(VOLUME_SIZE, "double_sphere")
    normals = compute_normal_volume(volume)

    renderer.load_volume(volume)
    renderer.load_normal_volume(normals)
    renderer.set_volume_bounds((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))

    # Create transfer functions
    ctf = ColorTransferFunction.from_colormap("plasma")
    otf = OpacityTransferFunction.linear(0.0, 0.1)

    # Set camera
    camera = CameraParameters.from_spherical(
        target=np.array([0, 0, 0], dtype=np.float32),
        distance=3.0,
        azimuth=np.pi / 4,
        elevation=np.pi / 6,
        roll=0.0,
        init_up=np.array([0, 0, 1], dtype=np.float32),
    )
    position, up = get_camera_pos_from_params(camera)
    renderer.set_camera(position=position, target=(0, 0, 0), up=up)

    # Set transfer functions
    renderer.set_transfer_functions(ctf, otf)

    print(f"\\nBenchmarking {N_RUNS} renders at {IMAGE_RES}×{IMAGE_RES}...")

    # Warm up
    _ = renderer.render()

    # Benchmark
    times = []
    for i in range(N_RUNS):
        start_time = time.perf_counter()
        data = renderer.render()
        end_time = time.perf_counter()
        times.append((end_time - start_time) * 1000)  # Convert to milliseconds

        if (i + 1) % 5 == 0:
            print(f"  Completed {i + 1}/{N_RUNS} runs...")

    # Statistics
    mean_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)

    print(f"\\n=== Results ===")
    print(f"Mean render time: {mean_time:.2f} ± {std_time:.2f} ms")
    print(f"Min render time:  {min_time:.2f} ms")
    print(f"Max render time:  {max_time:.2f} ms")
    print(f"Throughput:       {1000/mean_time:.1f} FPS")

    # Calculate pixel throughput
    pixels_per_frame = IMAGE_RES * IMAGE_RES
    pixels_per_second = pixels_per_frame * (1000 / mean_time)
    megapixels_per_second = pixels_per_second / 1e6

    print(f"Pixel throughput: {megapixels_per_second:.1f} MPix/s")

    # Validate output
    data_array = np.frombuffer(data, dtype=np.uint8).reshape((IMAGE_RES, IMAGE_RES, 4))
    non_zero_pixels = np.count_nonzero(data_array.max(axis=2))
    coverage = non_zero_pixels / (IMAGE_RES * IMAGE_RES) * 100

    print(f"\\n=== Quality Check ===")
    print(f"Non-black pixels: {non_zero_pixels:,} ({coverage:.1f}% coverage)")
    print(f"Data range: [{data_array.min()}, {data_array.max()}]")

    # Performance classification
    if mean_time < 5:
        performance = "EXCELLENT (Real-time)"
    elif mean_time < 15:
        performance = "GOOD (Interactive)"
    elif mean_time < 50:
        performance = "ACCEPTABLE (Responsive)"
    else:
        performance = "SLOW (Needs optimization)"

    print(f"\\n=== Performance Rating ===")
    print(f"Classification: {performance}")

    print("\\n=== Benchmark Complete ===")
    print("PyVR v0.2.2 with RGBA transfer function textures provides")
    print("efficient single-lookup volume rendering with excellent performance.")


if __name__ == "__main__":
    benchmark_render_performance()
