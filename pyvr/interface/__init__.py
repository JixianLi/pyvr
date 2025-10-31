"""
Interactive matplotlib-based interface for PyVR volume rendering.

This module provides testing/development interfaces for real-time volume
visualization with transfer function editing and camera controls.

Example:
    >>> from pyvr.interface import InteractiveVolumeRenderer
    >>> from pyvr.datasets import create_sample_volume
    >>> from pyvr.volume import Volume
    >>>
    >>> volume_data = create_sample_volume(128, 'sphere')
    >>> volume = Volume(data=volume_data)
    >>>
    >>> interface = InteractiveVolumeRenderer(volume=volume)
    >>> interface.show()  # Launch interactive GUI
"""

from pyvr.interface.matplotlib import InteractiveVolumeRenderer
from pyvr.interface.widgets import ImageDisplay, OpacityEditor, ColorSelector
from pyvr.interface.state import InterfaceState

__all__ = [
    "InteractiveVolumeRenderer",
    "ImageDisplay",
    "OpacityEditor",
    "ColorSelector",
    "InterfaceState",
]
