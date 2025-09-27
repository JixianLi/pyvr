"""PyVR Datasets Module

This module provides functions for creating synthetic 3D volume datasets
for testing and demonstration purposes.
"""

from .synthetic import compute_normal_volume, create_sample_volume

__all__ = ["create_sample_volume", "compute_normal_volume"]
