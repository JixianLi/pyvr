"""PyVR Datasets Module

This module provides functions for creating synthetic 3D volume datasets
for testing and demonstration purposes.
"""

from .synthetic import (
    create_test_volume,
    create_medical_phantom,
    create_sample_volume,
    compute_normal_volume
)

__all__ = [
    'create_test_volume',
    'create_medical_phantom', 
    'create_sample_volume',
    'compute_normal_volume'
]