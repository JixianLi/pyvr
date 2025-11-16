"""ABOUTME: PyVR Data Loaders Module
ABOUTME: Provides functions for loading volume data from various file formats.
"""

from .vtk_loader import load_vtk_volume

__all__ = ["load_vtk_volume"]
