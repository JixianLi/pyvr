"""
Abstract volume renderer interface for PyVR.

This module defines the VolumeRenderer base class that all backend
implementations must inherit from. This allows PyVR to support multiple
rendering backends (OpenGL, Vulkan, CPU, etc.) with a consistent API.
"""

from abc import ABC, abstractmethod
from typing import Optional

from ..camera import Camera
from ..lighting import Light
from ..transferfunctions import ColorTransferFunction, OpacityTransferFunction
from ..volume import Volume


class VolumeRenderer(ABC):
    """
    Abstract base class for volume renderers.

    All backend implementations (OpenGL, Vulkan, CPU, etc.) must inherit
    from this class and implement the abstract methods.

    This design enables:
    - Multiple rendering backends with consistent API
    - Backend-agnostic application code
    - Easy testing with mock renderers
    - Future extensibility
    """

    def __init__(self, width: int = 512, height: int = 512):
        """
        Initialize base renderer.

        Args:
            width: Rendering viewport width
            height: Rendering viewport height
        """
        self.width = width
        self.height = height
        self.volume: Optional[Volume] = None
        self.camera: Optional[Camera] = None
        self.light: Optional[Light] = None

    @abstractmethod
    def load_volume(self, volume: Volume) -> None:
        """
        Load volume data into the renderer.

        Args:
            volume: Volume instance with data and metadata

        Raises:
            TypeError: If volume is not a Volume instance
        """
        pass

    @abstractmethod
    def set_camera(self, camera: Camera) -> None:
        """
        Set camera configuration.

        Args:
            camera: Camera instance with position and projection parameters

        Raises:
            TypeError: If camera is not a Camera instance
        """
        pass

    @abstractmethod
    def set_light(self, light: Light) -> None:
        """
        Set lighting configuration.

        Args:
            light: Light instance with lighting parameters

        Raises:
            TypeError: If light is not a Light instance
        """
        pass

    @abstractmethod
    def set_transfer_functions(
        self,
        color_transfer_function: ColorTransferFunction,
        opacity_transfer_function: OpacityTransferFunction,
    ) -> None:
        """
        Set transfer functions for volume rendering.

        Args:
            color_transfer_function: Color transfer function
            opacity_transfer_function: Opacity transfer function
        """
        pass

    @abstractmethod
    def render(self) -> bytes:
        """
        Render the volume and return raw pixel data.

        Returns:
            Raw RGBA pixel data as bytes

        Raises:
            RuntimeError: If renderer is not properly configured
        """
        pass

    def get_volume(self) -> Optional[Volume]:
        """Get current volume."""
        return self.volume

    def get_camera(self) -> Optional[Camera]:
        """Get current camera."""
        return self.camera

    def get_light(self) -> Optional[Light]:
        """Get current light."""
        return self.light


class RendererError(Exception):
    """Exception raised for renderer errors."""

    pass
