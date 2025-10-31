"""
Lighting system for PyVR volume rendering.

This module provides light classes and utilities for configuring
illumination in volume rendering scenes.
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional

import numpy as np


@dataclass
class Light:
    """
    Encapsulates lighting parameters for volume rendering.

    This class manages all lighting-related parameters including position,
    direction, and intensity values for ambient and diffuse lighting.

    Attributes:
        position: 3D position of the light source in world space
        target: 3D point the light is directed toward
        ambient_intensity: Ambient light intensity (0.0 to 1.0)
        diffuse_intensity: Diffuse light intensity (0.0 to 1.0)
    """

    position: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 1.0, 1.0], dtype=np.float32)
    )
    target: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=np.float32)
    )
    ambient_intensity: float = 0.2
    diffuse_intensity: float = 0.8

    # Camera linking (private attributes)
    _is_linked: bool = field(default=False, init=False, repr=False)
    _camera_offsets: Optional[dict] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        """Validate lighting parameters after initialization."""
        self.validate()

    def validate(self) -> None:
        """
        Validate lighting parameters.

        Raises:
            ValueError: If any parameter is invalid
        """
        # Validate position
        if not isinstance(self.position, np.ndarray) or self.position.shape != (3,):
            raise ValueError("position must be a 3D numpy array")

        # Validate target
        if not isinstance(self.target, np.ndarray) or self.target.shape != (3,):
            raise ValueError("target must be a 3D numpy array")

        # Validate intensities
        if not (0.0 <= self.ambient_intensity <= 1.0):
            raise ValueError("ambient_intensity must be between 0.0 and 1.0")

        if not (0.0 <= self.diffuse_intensity <= 1.0):
            raise ValueError("diffuse_intensity must be between 0.0 and 1.0")

    @classmethod
    def directional(
        cls,
        direction: np.ndarray,
        ambient: float = 0.2,
        diffuse: float = 0.8,
        distance: float = 10.0,
    ) -> "Light":
        """
        Create a directional light.

        A directional light illuminates from a specific direction, simulating
        distant light sources like the sun.

        Args:
            direction: Direction vector the light points toward (will be normalized)
            ambient: Ambient light intensity (default: 0.2)
            diffuse: Diffuse light intensity (default: 0.8)
            distance: Distance to place light source from origin (default: 10.0)

        Returns:
            Light instance configured as directional light

        Example:
            >>> light = Light.directional(
            ...     direction=np.array([1, -1, 0]),
            ...     ambient=0.3,
            ...     diffuse=0.9
            ... )
        """
        direction = np.array(direction, dtype=np.float32)
        direction = direction / np.linalg.norm(direction)

        # Position light far away in opposite direction
        position = -direction * distance
        target = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        return cls(
            position=position,
            target=target,
            ambient_intensity=ambient,
            diffuse_intensity=diffuse,
        )

    @classmethod
    def point_light(
        cls,
        position: np.ndarray,
        target: np.ndarray = None,
        ambient: float = 0.2,
        diffuse: float = 0.8,
    ) -> "Light":
        """
        Create a point light at a specific position.

        A point light radiates in all directions from a specific point in space.

        Args:
            position: 3D position of the light source
            target: Optional target point (default: origin)
            ambient: Ambient light intensity (default: 0.2)
            diffuse: Diffuse light intensity (default: 0.8)

        Returns:
            Light instance configured as point light

        Example:
            >>> light = Light.point_light(
            ...     position=np.array([5, 5, 5]),
            ...     ambient=0.1,
            ...     diffuse=0.7
            ... )
        """
        position = np.array(position, dtype=np.float32)
        if target is None:
            target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        else:
            target = np.array(target, dtype=np.float32)

        return cls(
            position=position,
            target=target,
            ambient_intensity=ambient,
            diffuse_intensity=diffuse,
        )

    @classmethod
    def default(cls) -> "Light":
        """
        Create default light configuration.

        Returns a standard light suitable for most volume rendering scenarios.

        Returns:
            Light instance with default parameters

        Example:
            >>> light = Light.default()
        """
        return cls()

    @classmethod
    def ambient_only(cls, intensity: float = 0.5) -> "Light":
        """
        Create ambient-only lighting (no directional component).

        Useful for examining internal structures without strong shadows.

        Args:
            intensity: Ambient light intensity (default: 0.5)

        Returns:
            Light instance with only ambient lighting

        Example:
            >>> light = Light.ambient_only(intensity=0.3)
        """
        return cls(
            position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            target=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            ambient_intensity=intensity,
            diffuse_intensity=0.0,
        )

    @classmethod
    def camera_linked(
        cls,
        azimuth_offset: float = 0.0,
        elevation_offset: float = 0.0,
        distance_offset: float = 0.0,
        ambient: float = 0.2,
        diffuse: float = 0.8,
    ) -> "Light":
        """
        Create a camera-linked directional light.

        The light will follow camera movement with specified offsets,
        providing consistent illumination from the camera's perspective.

        Args:
            azimuth_offset: Horizontal angle offset in radians (default: 0.0)
            elevation_offset: Vertical angle offset in radians (default: 0.0)
            distance_offset: Additional distance from target (default: 0.0)
            ambient: Ambient light intensity (default: 0.2)
            diffuse: Diffuse light intensity (default: 0.8)

        Returns:
            Light instance configured for camera linking

        Example:
            >>> # Light follows camera with 45° horizontal offset
            >>> light = Light.camera_linked(azimuth_offset=np.pi/4)
            >>> renderer = VolumeRenderer(light=light)
            >>>
            >>> # In render loop:
            >>> light.update_from_camera(camera)
            >>> renderer.set_light(light)
        """
        # Start with default light
        light = cls(
            position=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            target=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            ambient_intensity=ambient,
            diffuse_intensity=diffuse,
        )

        # Link to camera
        light.link_to_camera(
            azimuth_offset=azimuth_offset,
            elevation_offset=elevation_offset,
            distance_offset=distance_offset,
        )

        return light

    def get_direction(self) -> np.ndarray:
        """
        Calculate light direction vector (from position to target).

        Returns:
            Normalized direction vector as np.ndarray

        Example:
            >>> light = Light.directional(direction=[1, 0, 0])
            >>> direction = light.get_direction()
        """
        direction = self.target - self.position
        norm = np.linalg.norm(direction)

        if norm < 1e-9:
            # Position equals target, return default direction
            return np.array([0.0, 0.0, -1.0], dtype=np.float32)

        return direction / norm

    def copy(self) -> "Light":
        """
        Create a copy of this light.

        Returns:
            New Light instance with same parameters

        Example:
            >>> original = Light.default()
            >>> copy = original.copy()
            >>> copy.ambient_intensity = 0.5  # Doesn't affect original
        """
        return Light(
            position=self.position.copy(),
            target=self.target.copy(),
            ambient_intensity=self.ambient_intensity,
            diffuse_intensity=self.diffuse_intensity,
        )

    @property
    def is_linked(self) -> bool:
        """
        Check if light is linked to camera.

        Returns:
            True if light is currently linked to camera
        """
        return self._is_linked

    def link_to_camera(
        self,
        azimuth_offset: float = 0.0,
        elevation_offset: float = 0.0,
        distance_offset: float = 0.0,
    ) -> "Light":
        """
        Link this light to follow camera movement with offsets.

        The light position will be calculated relative to camera orientation
        using the provided offsets. This creates consistent illumination
        from the camera's perspective.

        Args:
            azimuth_offset: Horizontal angle offset in radians (default: 0.0)
            elevation_offset: Vertical angle offset in radians (default: 0.0)
            distance_offset: Additional distance from target (default: 0.0)

        Returns:
            Self for method chaining

        Example:
            >>> light = Light.directional([1, -1, 0])
            >>> light.link_to_camera(azimuth_offset=np.pi/4, elevation_offset=0.0)
            >>> # Now light will follow camera with 45° horizontal offset
        """
        self._is_linked = True
        self._camera_offsets = {
            'azimuth': azimuth_offset,
            'elevation': elevation_offset,
            'distance': distance_offset,
        }
        return self

    def unlink_from_camera(self) -> "Light":
        """
        Unlink this light from camera movement.

        Light position becomes fixed at current location.

        Returns:
            Self for method chaining

        Example:
            >>> light.unlink_from_camera()
            >>> # Light no longer follows camera
        """
        self._is_linked = False
        self._camera_offsets = None
        return self

    def update_from_camera(self, camera) -> None:
        """
        Update light position based on camera and offsets.

        This method should be called each frame if the light is linked.

        Args:
            camera: Camera instance to derive position from

        Raises:
            ValueError: If light is not linked or camera is invalid

        Example:
            >>> if light.is_linked:
            ...     light.update_from_camera(camera)
            ...     renderer.set_light(light)
        """
        if not self._is_linked:
            raise ValueError("Light is not linked to camera. Call link_to_camera() first.")

        if self._camera_offsets is None:
            raise ValueError("Camera offsets not set. Call link_to_camera() first.")

        # Import Camera here to avoid circular dependency
        from pyvr.camera import Camera
        if not isinstance(camera, Camera):
            raise ValueError("camera must be a Camera instance")

        # Calculate light position from camera orientation + offsets
        azimuth = camera.azimuth + self._camera_offsets['azimuth']
        elevation = camera.elevation + self._camera_offsets['elevation']
        distance = camera.distance + self._camera_offsets['distance']

        # Convert spherical coordinates to Cartesian position
        # This mirrors the Camera.get_camera_vectors() logic
        cos_elev = np.cos(elevation)
        x = distance * cos_elev * np.cos(azimuth)
        y = distance * np.sin(elevation)
        z = distance * cos_elev * np.sin(azimuth)

        # Position relative to camera target
        self.position = camera.target + np.array([x, y, z], dtype=np.float32)
        self.target = camera.target.copy()

    def get_offsets(self) -> Optional[dict]:
        """
        Get current camera offsets if linked.

        Returns:
            Dictionary with 'azimuth', 'elevation', 'distance' keys, or None if not linked

        Example:
            >>> offsets = light.get_offsets()
            >>> if offsets:
            ...     print(f"Azimuth offset: {offsets['azimuth']:.2f} rad")
        """
        return self._camera_offsets.copy() if self._camera_offsets else None

    def __repr__(self) -> str:
        """String representation of light configuration."""
        return (
            f"Light("
            f"position={self.position}, "
            f"target={self.target}, "
            f"ambient={self.ambient_intensity:.2f}, "
            f"diffuse={self.diffuse_intensity:.2f})"
        )


class LightError(Exception):
    """Exception raised for lighting configuration errors."""

    pass
