"""
Camera parameter management and validation for PyVR.

This module provides classes and utilities for managing camera parameters,
including validation, presets, and serialization support.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np


@dataclass
class Camera:
    """
    Camera with validation, matrix creation, and presets.

    This class encapsulates all camera-related parameters and provides
    validation, serialization, matrix generation, and preset management.

    Attributes:
        target: 3D point the camera is looking at
        azimuth: Horizontal rotation angle in radians
        elevation: Vertical rotation angle in radians
        roll: Roll rotation angle in radians
        distance: Distance from camera to target
        init_pos: Initial camera position (relative to target)
        init_up: Initial up vector
        fov: Field of view in radians
        near_plane: Near clipping plane distance
        far_plane: Far clipping plane distance
    """

    # Core positioning parameters
    target: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=np.float32)
    )
    azimuth: float = 0.0  # Horizontal rotation (radians)
    elevation: float = 0.0  # Vertical rotation (radians)
    roll: float = 0.0  # Roll rotation (radians)
    distance: float = 3.0  # Distance from target

    # Initial vectors
    init_pos: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=np.float32)
    )
    init_up: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 1.0], dtype=np.float32)
    )

    # Projection parameters (for future use)
    fov: float = np.pi / 4  # 45 degrees
    near_plane: float = 0.1
    far_plane: float = 100.0

    def __post_init__(self):
        """Validate parameters after initialization."""
        self.validate()

    def validate(self) -> None:
        """
        Validate all camera parameters.

        Raises:
            ValueError: If any parameter is invalid
        """
        # Validate target
        if not isinstance(self.target, np.ndarray) or self.target.shape != (3,):
            raise ValueError("target must be a 3D numpy array")

        # Validate angles (allow full range, but warn about extreme values)
        angle_params = ["azimuth", "elevation", "roll"]
        for param in angle_params:
            value = getattr(self, param)
            if not isinstance(value, (int, float)):
                raise ValueError(f"{param} must be numeric")
            if abs(value) > 4 * np.pi:  # More than 2 full rotations
                print(
                    f"Warning: {param} = {value:.3f} rad ({np.degrees(value):.1f}°) is unusually large"
                )

        # Validate distance
        if not isinstance(self.distance, (int, float)) or self.distance <= 0:
            raise ValueError("distance must be positive")

        # Validate initial vectors
        if not isinstance(self.init_pos, np.ndarray) or self.init_pos.shape != (3,):
            raise ValueError("init_pos must be a 3D numpy array")

        if not isinstance(self.init_up, np.ndarray) or self.init_up.shape != (3,):
            raise ValueError("init_up must be a 3D numpy array")

        # Check that init_pos is not zero (relative to target)
        rel_pos = self.init_pos - self.target
        if np.linalg.norm(rel_pos) < 1e-9:
            raise ValueError("init_pos must not be at the same location as target")

        # Check that init_up is not zero
        if np.linalg.norm(self.init_up) < 1e-9:
            raise ValueError("init_up must not be the zero vector")

        # Validate projection parameters
        if self.fov <= 0 or self.fov >= np.pi:
            raise ValueError("fov must be between 0 and π radians")

        if self.near_plane <= 0:
            raise ValueError("near_plane must be positive")

        if self.far_plane <= self.near_plane:
            raise ValueError("far_plane must be greater than near_plane")

    @classmethod
    def from_spherical(
        cls,
        target: np.ndarray,
        azimuth: float,
        elevation: float,
        roll: float,
        distance: float,
        **kwargs,
    ) -> "Camera":
        """
        Create camera from spherical coordinates (main interface).

        Args:
            target: 3D point to look at
            azimuth: Horizontal angle in radians
            elevation: Vertical angle in radians
            roll: Roll angle in radians
            distance: Distance from target
            **kwargs: Additional parameters (init_pos, init_up, etc.)

        Returns:
            Camera instance
        """
        return cls(
            target=target,
            azimuth=azimuth,
            elevation=elevation,
            roll=roll,
            distance=distance,
            **kwargs,
        )

    @classmethod
    def front_view(
        cls, target: Optional[np.ndarray] = None, distance: float = 3.0
    ) -> "Camera":
        """Create camera for front view."""
        if target is None:
            target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        return cls(
            target=target, azimuth=0.0, elevation=0.0, roll=0.0, distance=distance
        )

    @classmethod
    def side_view(
        cls, target: Optional[np.ndarray] = None, distance: float = 3.0
    ) -> "Camera":
        """Create camera for side view."""
        if target is None:
            target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        return cls(
            target=target, azimuth=np.pi / 2, elevation=0.0, roll=0.0, distance=distance
        )

    @classmethod
    def top_view(
        cls, target: Optional[np.ndarray] = None, distance: float = 3.0
    ) -> "Camera":
        """Create camera for top view."""
        if target is None:
            target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        return cls(
            target=target, azimuth=0.0, elevation=np.pi / 2, roll=0.0, distance=distance
        )

    @classmethod
    def isometric_view(
        cls, target: Optional[np.ndarray] = None, distance: float = 3.0
    ) -> "Camera":
        """Create camera for isometric view."""
        if target is None:
            target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        return cls(
            target=target,
            azimuth=np.pi / 4,  # 45 degrees
            elevation=np.pi / 6,  # 30 degrees
            roll=0.0,
            distance=distance,
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize camera parameters to dictionary.

        Returns:
            Dictionary representation suitable for JSON serialization
        """
        return {
            "target": self.target.tolist(),
            "azimuth": float(self.azimuth),
            "elevation": float(self.elevation),
            "roll": float(self.roll),
            "distance": float(self.distance),
            "init_pos": self.init_pos.tolist(),
            "init_up": self.init_up.tolist(),
            "fov": float(self.fov),
            "near_plane": float(self.near_plane),
            "far_plane": float(self.far_plane),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Camera":
        """
        Deserialize camera from dictionary.

        Args:
            data: Dictionary with camera parameter values

        Returns:
            Camera instance
        """
        return cls(
            target=np.array(data["target"], dtype=np.float32),
            azimuth=data["azimuth"],
            elevation=data["elevation"],
            roll=data["roll"],
            distance=data["distance"],
            init_pos=np.array(data["init_pos"], dtype=np.float32),
            init_up=np.array(data["init_up"], dtype=np.float32),
            fov=data.get("fov", np.pi / 4),
            near_plane=data.get("near_plane", 0.1),
            far_plane=data.get("far_plane", 100.0),
        )

    def save_to_file(self, filepath: Union[str, Path]) -> None:
        """
        Save camera parameters to JSON file.

        Args:
            filepath: Path to save the JSON file
        """
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_file(cls, filepath: Union[str, Path]) -> "Camera":
        """
        Load camera from JSON file.

        Args:
            filepath: Path to the JSON file

        Returns:
            Camera instance
        """
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def copy(self) -> "Camera":
        """Create a copy of the camera."""
        return Camera.from_dict(self.to_dict())

    def get_camera_vectors(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get camera position and up vector from spherical parameters.

        Delegates to get_camera_pos_from_params() for quaternion-based calculation.

        Returns:
            Tuple of (position, up) vectors

        Example:
            >>> camera = Camera.from_spherical(
            ...     target=np.array([0, 0, 0]),
            ...     azimuth=np.pi/4,
            ...     elevation=np.pi/6,
            ...     roll=0.0,
            ...     distance=3.0
            ... )
            >>> position, up = camera.get_camera_vectors()
        """
        from .control import get_camera_pos_from_params
        return get_camera_pos_from_params(self)

    def get_view_matrix(self) -> np.ndarray:
        """
        Create view matrix from camera parameters.

        Transforms from world space to camera space (Geometry Stage).

        Returns:
            4x4 view matrix as np.ndarray (float32)

        Example:
            >>> camera = Camera.isometric_view(distance=5.0)
            >>> view_matrix = camera.get_view_matrix()
            >>> view_matrix.shape
            (4, 4)
        """
        position, up = self.get_camera_vectors()

        # Look-at algorithm
        forward = self.target - position
        forward = forward / np.linalg.norm(forward)

        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)

        up_corrected = np.cross(right, forward)

        # View matrix (column-major for OpenGL)
        view_matrix = np.array(
            [
                [right[0], up_corrected[0], -forward[0], 0],
                [right[1], up_corrected[1], -forward[1], 0],
                [right[2], up_corrected[2], -forward[2], 0],
                [
                    -np.dot(right, position),
                    -np.dot(up_corrected, position),
                    np.dot(forward, position),
                    1,
                ],
            ],
            dtype=np.float32,
        )

        return view_matrix

    def get_projection_matrix(self, aspect_ratio: float) -> np.ndarray:
        """
        Create perspective projection matrix.

        Transforms from camera space to clip space.
        Uses camera's FOV, near plane, and far plane settings.

        Args:
            aspect_ratio: Width / height ratio of viewport

        Returns:
            4x4 projection matrix as np.ndarray (float32)

        Example:
            >>> camera = Camera.front_view(distance=3.0)
            >>> camera.fov = np.radians(60)
            >>> proj = camera.get_projection_matrix(aspect_ratio=16/9)
        """
        fov = self.fov
        near = self.near_plane
        far = self.far_plane

        f = 1.0 / np.tan(fov / 2.0)
        projection_matrix = np.array(
            [
                [f / aspect_ratio, 0, 0, 0],
                [0, f, 0, 0],
                [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
                [0, 0, -1, 0],
            ],
            dtype=np.float32,
        )

        return projection_matrix

    def __repr__(self) -> str:
        """String representation of camera."""
        return (
            f"Camera("
            f"target={self.target}, "
            f"azimuth={self.azimuth:.3f} rad ({np.degrees(self.azimuth):.1f}°), "
            f"elevation={self.elevation:.3f} rad ({np.degrees(self.elevation):.1f}°), "
            f"roll={self.roll:.3f} rad ({np.degrees(self.roll):.1f}°), "
            f"distance={self.distance:.2f})"
        )


class CameraError(Exception):
    """Exception raised for camera errors."""

    pass


def validate_camera_angles(azimuth: float, elevation: float, roll: float) -> None:
    """
    Validate camera angle parameters.

    Args:
        azimuth: Horizontal rotation angle in radians
        elevation: Vertical rotation angle in radians
        roll: Roll rotation angle in radians

    Raises:
        CameraError: If any angle is invalid
    """
    angles = {"azimuth": azimuth, "elevation": elevation, "roll": roll}

    for name, value in angles.items():
        if not isinstance(value, (int, float)):
            raise CameraError(f"{name} must be numeric, got {type(value)}")
        if not np.isfinite(value):
            raise CameraError(f"{name} must be finite, got {value}")


def degrees_to_radians(**kwargs) -> Dict[str, float]:
    """
    Convert angle parameters from degrees to radians.

    Args:
        **kwargs: Angle parameters in degrees

    Returns:
        Dictionary with angles converted to radians
    """
    return {key: np.radians(value) for key, value in kwargs.items()}


def radians_to_degrees(**kwargs) -> Dict[str, float]:
    """
    Convert angle parameters from radians to degrees.

    Args:
        **kwargs: Angle parameters in radians

    Returns:
        Dictionary with angles converted to degrees
    """
    return {key: np.degrees(value) for key, value in kwargs.items()}
