"""
PyVR Camera Module

This module provides camera control and positioning utilities for volume rendering.
It handles camera positioning, orientation, parameter management, and animations.

Core Functions:
    get_camera_pos: Calculate camera position from spherical coordinates
    get_camera_pos_from_params: Calculate position from CameraParameters object

Classes:
    CameraParameters: Camera parameter validation and management
    CameraPath: Camera path animation utility
    CameraController: High-level camera controller for interactive manipulation

Exceptions:
    CameraParameterError: Raised when camera parameters are invalid

Utility Functions:
    validate_camera_angles: Validate camera angle parameters
    degrees_to_radians: Convert angles from degrees to radians
    radians_to_degrees: Convert angles from radians to degrees

Examples:
    # Using camera parameters (recommended interface)
    params = CameraParameters.from_spherical(
        target=np.array([0, 0, 0]), distance=5.0,
        azimuth=np.pi/4, elevation=np.pi/6, roll=0
    )
    pos, up = params.get_camera_vectors()
    
    # Camera presets
    params = CameraParameters.preset_isometric_view(distance=3.0)
    pos, up = params.get_camera_vectors()
    
    # Camera animation
    path = CameraPath([start_params, end_params])
    frames = path.generate_frames(30)
    
    # Interactive camera control
    controller = CameraController()
    controller.orbit(np.pi/8, np.pi/12)  # Orbit camera
    controller.zoom(0.8)  # Zoom in
    pos, up = controller.get_position_and_up()
"""

__version__ = "0.2.1"

from .control import (
    get_camera_pos,
    get_camera_pos_from_params,
    CameraPath,
    CameraController
)

from .parameters import (
    CameraParameters,
    CameraParameterError,
    validate_camera_angles,
    degrees_to_radians,
    radians_to_degrees
)

__all__ = [
    # Core functions
    'get_camera_pos',
    'get_camera_pos_from_params',
    
    # Classes
    'CameraParameters',
    'CameraPath', 
    'CameraController',
    
    # Exceptions
    'CameraParameterError',
    
    # Utilities
    'validate_camera_angles',
    'degrees_to_radians',
    'radians_to_degrees',
]