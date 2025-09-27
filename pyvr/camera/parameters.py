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
class CameraParameters:
    """
    Comprehensive camera parameter container with validation and presets.

    This class encapsulates all camera-related parameters and provides
    validation, serialization, and preset management functionality.

    Attributes:
        target: 3D point the camera is looking at
        azimuth: Horizontal rotation angle in radians
        elevation: Vertical rotation angle in radians
        roll: Roll rotation angle in radians
        distance: Distance from camera to target
        init_pos: Initial camera position (relative to target)
        init_up: Initial up vector
        fov: Field of view in radians (for future use)
        near_plane: Near clipping plane distance (for future use)
        far_plane: Far clipping plane distance (for future use)
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
    ) -> "CameraParameters":
        """
        Create camera parameters from spherical coordinates (main interface).

        Args:
            target: 3D point to look at
            azimuth: Horizontal angle in radians
            elevation: Vertical angle in radians
            roll: Roll angle in radians
            distance: Distance from target
            **kwargs: Additional parameters (init_pos, init_up, etc.)

        Returns:
            CameraParameters instance
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
    ) -> "CameraParameters":
        """Create parameters for front view."""
        if target is None:
            target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        return cls(
            target=target, azimuth=0.0, elevation=0.0, roll=0.0, distance=distance
        )

    @classmethod
    def side_view(
        cls, target: Optional[np.ndarray] = None, distance: float = 3.0
    ) -> "CameraParameters":
        """Create parameters for side view."""
        if target is None:
            target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        return cls(
            target=target, azimuth=np.pi / 2, elevation=0.0, roll=0.0, distance=distance
        )

    @classmethod
    def top_view(
        cls, target: Optional[np.ndarray] = None, distance: float = 3.0
    ) -> "CameraParameters":
        """Create parameters for top view."""
        if target is None:
            target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        return cls(
            target=target, azimuth=0.0, elevation=np.pi / 2, roll=0.0, distance=distance
        )

    @classmethod
    def isometric_view(
        cls, target: Optional[np.ndarray] = None, distance: float = 3.0
    ) -> "CameraParameters":
        """Create parameters for isometric view."""
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
    def from_dict(cls, data: Dict[str, Any]) -> "CameraParameters":
        """
        Deserialize camera parameters from dictionary.

        Args:
            data: Dictionary with camera parameter values

        Returns:
            CameraParameters instance
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
    def load_from_file(cls, filepath: Union[str, Path]) -> "CameraParameters":
        """
        Load camera parameters from JSON file.

        Args:
            filepath: Path to the JSON file

        Returns:
            CameraParameters instance
        """
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def copy(self) -> "CameraParameters":
        """Create a copy of the camera parameters."""
        return CameraParameters.from_dict(self.to_dict())

    def __repr__(self) -> str:
        """String representation of camera parameters."""
        return (
            f"CameraParameters("
            f"target={self.target}, "
            f"azimuth={self.azimuth:.3f} rad ({np.degrees(self.azimuth):.1f}°), "
            f"elevation={self.elevation:.3f} rad ({np.degrees(self.elevation):.1f}°), "
            f"roll={self.roll:.3f} rad ({np.degrees(self.roll):.1f}°), "
            f"distance={self.distance:.2f})"
        )


class CameraParameterError(Exception):
    """Exception raised for camera parameter errors."""

    pass


def validate_camera_angles(azimuth: float, elevation: float, roll: float) -> None:
    """
    Validate camera angle parameters.

    Args:
        azimuth: Horizontal rotation angle in radians
        elevation: Vertical rotation angle in radians
        roll: Roll rotation angle in radians

    Raises:
        CameraParameterError: If any angle is invalid
    """
    angles = {"azimuth": azimuth, "elevation": elevation, "roll": roll}

    for name, value in angles.items():
        if not isinstance(value, (int, float)):
            raise CameraParameterError(f"{name} must be numeric, got {type(value)}")
        if not np.isfinite(value):
            raise CameraParameterError(f"{name} must be finite, got {value}")


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
