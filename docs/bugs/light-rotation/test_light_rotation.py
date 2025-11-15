#!/usr/bin/env python3
"""
Test script to verify light behavior during camera rotation.

This script tests whether the light actually updates when camera rotates,
and what the light position/direction looks like from different angles.
"""

import numpy as np
from pyvr.camera import Camera
from pyvr.lighting import Light


def test_light_rotation():
    """Test light position as camera rotates around scene."""
    print("=" * 80)
    print("TEST: Light Position During Camera Rotation")
    print("=" * 80)
    print()

    # Create camera looking at origin
    camera = Camera.from_spherical(
        target=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        distance=3.0,
        azimuth=np.pi/4,  # 45 degrees (front-right)
        elevation=np.pi/6,  # 30 degrees up
        roll=0.0
    )

    # Create linked light (no offsets - should match camera position)
    light = Light.directional([1, -1, 0])
    light.link_to_camera(
        azimuth_offset=0.0,
        elevation_offset=0.0,
        distance_offset=0.0
    )

    print("Initial Camera State:")
    print(f"  Target: {camera.target}")
    print(f"  Azimuth: {np.degrees(camera.azimuth):.1f}° (45° = front-right)")
    print(f"  Elevation: {np.degrees(camera.elevation):.1f}° (30° = up)")
    print(f"  Distance: {camera.distance}")
    camera_pos, camera_up = camera.get_camera_vectors()
    print(f"  Camera Position: {camera_pos}")
    print()

    # Update light from initial camera
    light.update_from_camera(camera)
    print("Initial Light State:")
    print(f"  Position: {light.position}")
    print(f"  Target: {light.target}")
    print(f"  Direction: {light.get_direction()}")
    print()

    # Rotate camera to back of scene (azimuth + 180°)
    camera.azimuth = camera.azimuth + np.pi  # Add 180 degrees
    print("After Rotating to Back (azimuth +180°):")
    print(f"  Azimuth: {np.degrees(camera.azimuth):.1f}° (225° = back-left)")
    camera_pos_rotated, _ = camera.get_camera_vectors()
    print(f"  Camera Position: {camera_pos_rotated}")
    print()

    # Update light from rotated camera
    light.update_from_camera(camera)
    print("Light State After Camera Rotation:")
    print(f"  Position: {light.position}")
    print(f"  Target: {light.target}")
    print(f"  Direction: {light.get_direction()}")
    print()

    # Analysis
    print("=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print()

    # Check if light position changed
    initial_light_pos = np.array([1.8371173, 1.5, 1.8371173], dtype=np.float32)  # Approx initial pos
    pos_changed = not np.allclose(light.position, initial_light_pos, atol=0.1)

    if pos_changed:
        print("✓ Light position CHANGED when camera rotated")
    else:
        print("✗ Light position DID NOT change when camera rotated")
    print()

    # Check if light and camera are at same position
    camera_and_light_same = np.allclose(camera_pos_rotated, light.position, atol=0.01)

    if camera_and_light_same:
        print("✓ Light position MATCHES camera position (headlight mode)")
        print("  This means light moves WITH camera")
    else:
        print("✗ Light position DIFFERENT from camera position")
        print(f"  Camera position: {camera_pos_rotated}")
        print(f"  Light position: {light.position}")
        print(f"  Difference: {camera_pos_rotated - light.position}")
    print()

    # Check what the light direction looks like from camera's perspective
    # Vector from camera to light
    cam_to_light = light.position - camera_pos_rotated
    print(f"Vector from Camera to Light: {cam_to_light}")
    print(f"  Magnitude: {np.linalg.norm(cam_to_light):.3f}")

    if np.linalg.norm(cam_to_light) < 0.1:
        print("  → Light is AT camera position (headlight/flashlight mode)")
    else:
        print("  → Light is OFFSET from camera")
    print()

    # Check light direction (where light points)
    light_dir = light.get_direction()
    print(f"Light Direction (normalized): {light_dir}")
    print(f"  → Light points toward {light.target}")
    print()

    return light, camera


def test_light_with_offset():
    """Test light with azimuth offset."""
    print()
    print("=" * 80)
    print("TEST: Light With Azimuth Offset (+45°)")
    print("=" * 80)
    print()

    # Create camera
    camera = Camera.from_spherical(
        target=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        distance=3.0,
        azimuth=0.0,  # Front view
        elevation=0.0,
        roll=0.0
    )

    # Create light with 45° azimuth offset
    light = Light.directional([1, -1, 0])
    light.link_to_camera(
        azimuth_offset=np.pi/4,  # +45 degrees
        elevation_offset=0.0,
        distance_offset=0.0
    )

    camera_pos, _ = camera.get_camera_vectors()
    light.update_from_camera(camera)

    print(f"Camera azimuth: {np.degrees(camera.azimuth):.1f}° (front)")
    print(f"Camera position: {camera_pos}")
    print()
    print(f"Light azimuth: {np.degrees(camera.azimuth + np.pi/4):.1f}° (front-right)")
    print(f"Light position: {light.position}")
    print()

    # Rotate camera
    camera.azimuth = np.pi  # Back view
    light.update_from_camera(camera)
    camera_pos_back, _ = camera.get_camera_vectors()

    print(f"After rotation:")
    print(f"Camera azimuth: {np.degrees(camera.azimuth):.1f}° (back)")
    print(f"Camera position: {camera_pos_back}")
    print()
    print(f"Light azimuth: {np.degrees(camera.azimuth + np.pi/4):.1f}° (back-right)")
    print(f"Light position: {light.position}")
    print()

    print("Observation:")
    print("  The light maintains a +45° offset from camera azimuth")
    print("  As camera orbits, light orbits with it (not a headlight)")
    print()


if __name__ == "__main__":
    test_light_rotation()
    test_light_with_offset()

    print("=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print()
    print("Current Implementation:")
    print("  - Light orbits around scene center (camera.target)")
    print("  - Light maintains constant angular offset from camera")
    print("  - Light position = target + spherical_coords(cam_angle + offset)")
    print()
    print("Expected Behavior (from user description):")
    print("  - Light should 'follow camera'")
    print("  - This could mean either:")
    print("    1. Headlight mode: Light at camera position, pointing forward")
    print("    2. Orbit mode: Light orbits with camera (current)")
    print()
    print("If user expects headlight mode:")
    print("  → Bug: Light should be at camera.position, not orbiting")
    print()
    print("If user expects orbit mode:")
    print("  → Code should work correctly")
    print("  → Need to check if _update_display() is actually being called")
    print()
