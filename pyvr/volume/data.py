"""
Volume data management for PyVR.

This module provides the Volume class for encapsulating 3D volume data,
normal volumes, and spatial metadata. Volume is backend-agnostic and can
be used with any renderer implementation.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


@dataclass
class Volume:
    """
    Encapsulates 3D volume data and associated metadata.

    Volume is backend-agnostic - it stores data that can be rendered
    by any backend (OpenGL, Vulkan, CPU, etc.).

    Attributes:
        data: 3D numpy array (D, H, W) containing volume data
        normals: Optional 3D normal vectors (D, H, W, 3)
        min_bounds: Minimum corner of bounding box in world space
        max_bounds: Maximum corner of bounding box in world space
        name: Optional descriptive name
    """

    data: np.ndarray
    normals: Optional[np.ndarray] = None
    min_bounds: np.ndarray = field(
        default_factory=lambda: np.array([-0.5, -0.5, -0.5], dtype=np.float32)
    )
    max_bounds: np.ndarray = field(
        default_factory=lambda: np.array([0.5, 0.5, 0.5], dtype=np.float32)
    )
    name: Optional[str] = None

    def __post_init__(self):
        """Validate volume data after initialization."""
        self.validate()

    def validate(self) -> None:
        """
        Validate all volume data and metadata.

        Raises:
            ValueError: If volume data or parameters are invalid
        """
        if not isinstance(self.data, np.ndarray):
            raise ValueError("Volume data must be a numpy array")

        if len(self.data.shape) != 3:
            raise ValueError(f"Volume data must be 3D, got shape {self.data.shape}")

        if self.normals is not None:
            if not isinstance(self.normals, np.ndarray):
                raise ValueError("Normal volume must be a numpy array")

            expected_shape = self.data.shape + (3,)
            if self.normals.shape != expected_shape:
                raise ValueError(
                    f"Normal volume must have shape {expected_shape}, "
                    f"got {self.normals.shape}"
                )

        if not isinstance(self.min_bounds, np.ndarray) or self.min_bounds.shape != (3,):
            raise ValueError("min_bounds must be a 3D numpy array")

        if not isinstance(self.max_bounds, np.ndarray) or self.max_bounds.shape != (3,):
            raise ValueError("max_bounds must be a 3D numpy array")

        if np.any(self.max_bounds <= self.min_bounds):
            raise ValueError("max_bounds must be greater than min_bounds")

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Get volume dimensions (D, H, W)."""
        return self.data.shape

    @property
    def dimensions(self) -> np.ndarray:
        """Get physical dimensions of bounding box."""
        return self.max_bounds - self.min_bounds

    @property
    def center(self) -> np.ndarray:
        """Get center point of bounding box."""
        return (self.min_bounds + self.max_bounds) / 2.0

    @property
    def has_normals(self) -> bool:
        """Check if volume has normal data."""
        return self.normals is not None

    @property
    def voxel_spacing(self) -> np.ndarray:
        """Get spacing between voxels in world space."""
        return self.dimensions / np.array(self.shape, dtype=np.float32)

    def compute_normals(self, method: str = "gradient") -> None:
        """
        Compute normal vectors from volume data.

        Args:
            method: Computation method (default: "gradient")

        Raises:
            ValueError: If method is not supported
        """
        if method != "gradient":
            raise ValueError(f"Unsupported method: {method}")

        from ..datasets.synthetic import compute_normal_volume

        self.normals = compute_normal_volume(self.data)

    def normalize(self, method: str = "minmax") -> "Volume":
        """
        Create normalized volume.

        Args:
            method: Normalization method ("minmax" or "zscore")

        Returns:
            New Volume with normalized data

        Raises:
            ValueError: If method is not supported
        """
        if method == "minmax":
            data_min, data_max = self.data.min(), self.data.max()
            if data_max - data_min < 1e-9:
                normalized_data = np.zeros_like(self.data)
            else:
                normalized_data = (self.data - data_min) / (data_max - data_min)

        elif method == "zscore":
            mean, std = self.data.mean(), self.data.std()
            if std < 1e-9:
                normalized_data = np.zeros_like(self.data)
            else:
                normalized_data = (self.data - mean) / std
        else:
            raise ValueError(f"Unsupported method: {method}")

        return Volume(
            data=normalized_data.astype(np.float32),
            normals=self.normals.copy() if self.normals is not None else None,
            min_bounds=self.min_bounds.copy(),
            max_bounds=self.max_bounds.copy(),
            name=f"{self.name}_normalized" if self.name else None,
        )

    def copy(self) -> "Volume":
        """Create deep copy of volume."""
        return Volume(
            data=self.data.copy(),
            normals=self.normals.copy() if self.normals is not None else None,
            min_bounds=self.min_bounds.copy(),
            max_bounds=self.max_bounds.copy(),
            name=self.name,
        )

    def __repr__(self) -> str:
        """String representation."""
        name_str = f"'{self.name}'" if self.name else "unnamed"
        normals_str = "with normals" if self.has_normals else "no normals"
        return (
            f"Volume({name_str}, shape={self.shape}, "
            f"bounds=[{self.min_bounds}, {self.max_bounds}], {normals_str})"
        )


class VolumeError(Exception):
    """Exception raised for volume data errors."""

    pass
