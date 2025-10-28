"""
Lighting system for PyVR volume rendering.

Provides light classes and utilities for configuring illumination.
"""

from .light import Light, LightError

__all__ = [
    "Light",
    "LightError",
]
