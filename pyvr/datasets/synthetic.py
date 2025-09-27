"""Synthetic Volume Dataset Generation

This module provides functions for creating synthetic 3D volume datasets
for testing and development purposes using NumPy.
"""

import numpy as np


def create_sample_volume(size=64, shape="sphere"):
    """Create sample 3D volume data with various shapes.

    Args:
        size: Volume size (creates size x size x size volume)
        shape: Shape type ('sphere', 'torus', 'double_sphere', 'cube', 'helix', 'random_blob')

    Returns:
        Volume data as np.ndarray with shape (size, size, size) and dtype float32
    """
    x, y, z = np.meshgrid(
        np.linspace(-1, 1, size), np.linspace(-1, 1, size), np.linspace(-1, 1, size)
    )

    if shape == "sphere":
        # Simple sphere
        distance = np.sqrt(x * x + y * y + z * z)
        volume = np.exp(-((distance * 3) ** 2)).astype(np.float32)

    elif shape == "torus":
        # Torus shape
        R = 0.6  # Major radius
        r = 0.3  # Minor radius
        distance_to_center = np.sqrt(x * x + y * y)
        torus_distance = np.sqrt((distance_to_center - R) ** 2 + z * z)
        volume = np.exp(-((torus_distance / r * 4) ** 2)).astype(np.float32)

    elif shape == "double_sphere":
        # Two overlapping spheres
        distance1 = np.sqrt((x - 0.3) ** 2 + y * y + z * z)
        distance2 = np.sqrt((x + 0.3) ** 2 + y * y + z * z)
        sphere1 = np.exp(-((distance1 * 4) ** 2))
        sphere2 = np.exp(-((distance2 * 4) ** 2))
        volume = np.maximum(sphere1, sphere2).astype(np.float32)

    elif shape == "cube":
        # Rounded cube
        cube_dist = np.maximum(np.maximum(np.abs(x), np.abs(y)), np.abs(z))
        volume = np.exp(-(((cube_dist - 0.4) * 8) ** 2)).astype(np.float32)
        volume[cube_dist > 0.6] = 0

    elif shape == "helix":
        # Helical structure
        theta = np.arctan2(y, x)
        height = z
        radius = np.sqrt(x * x + y * y)

        # Parametric helix
        helix_radius = 0.5
        helix_thickness = 0.15
        turns = 3

        # Distance to helix centerline
        helix_x = helix_radius * np.cos(height * turns * 2 * np.pi)
        helix_y = helix_radius * np.sin(height * turns * 2 * np.pi)

        distance_to_helix = np.sqrt((x - helix_x) ** 2 + (y - helix_y) ** 2)
        volume = np.exp(-((distance_to_helix / helix_thickness * 3) ** 2)).astype(
            np.float32
        )

    elif shape == "random_blob":
        # Non-symmetric random blob using noise and spatial gradient
        from scipy.ndimage import gaussian_filter

        # Random offset for non-symmetry
        np.random.seed(42)  # For reproducible results
        offset = np.random.uniform(-1, 1, size=3)
        x_off = x + offset[0] * 0.5
        y_off = y + offset[1] * 0.5
        z_off = z + offset[2] * 0.5

        # Generate random noise
        noise = np.random.random((size, size, size)).astype(np.float32)
        noise = gaussian_filter(noise, sigma=size / 18)

        # Apply a spatial gradient for non-symmetry
        gradient = (x_off + 1.5) * (y_off + 1.2) * (z_off + 0.8)
        gradient = gradient / np.max(np.abs(gradient))

        # Combine noise and gradient, threshold for structure
        volume = noise * (0.7 + 0.3 * gradient)
        volume = np.maximum(0, volume - 0.25) * 2.5

        # Optional: add a few random "hotspots"
        for _ in range(3):
            cx, cy, cz = np.random.uniform(-0.7, 0.7, 3)
            dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2)
            volume += np.exp(-((dist * 6) ** 2)) * np.random.uniform(0.5, 1.2)

        volume = np.clip(volume, 0, 1).astype(np.float32)

    else:
        raise ValueError(
            f"Unknown shape: {shape}. Available shapes: sphere, torus, double_sphere, cube, helix, random_blob"
        )
    return volume


def compute_normal_volume(volume):
    """Compute normalized gradient (normal) for a 3D volume.

    Args:
        volume: 3D volume array with shape (D, H, W)

    Returns:
        Normal vectors array with shape (D, H, W, 3)
    """
    gx, gy, gz = np.gradient(volume)
    normals = np.stack((gx, gy, gz), axis=-1)
    norm = np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-8
    normals = normals / norm
    return normals.astype(np.float32)
