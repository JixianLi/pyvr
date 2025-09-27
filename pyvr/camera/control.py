"""
Enhanced camera control and positioning utilities for PyVR.

This module provides advanced camera positioning, animation, and control
functionality, including the core get_camera_pos function and enhanced
camera manipulation utilities.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Tuple, List, Optional, Union, Callable
from .parameters import CameraParameters, CameraParameterError, validate_camera_angles


def get_camera_pos(
    target: np.ndarray,
    azimuth: float,
    elevation: float,
    roll: float,
    distance: float,
    init_pos: Optional[np.ndarray] = None,
    init_up: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate camera position and up vector using quaternion rotations.
    
    This is the core camera positioning function, providing smooth camera
    movement using spherical coordinates with quaternion-based rotations.
    
    Args:
        target: 3D point the camera should look at
        azimuth: Horizontal rotation angle in radians
        elevation: Vertical rotation angle in radians
        roll: Roll rotation angle in radians
        distance: Distance from camera to target (must be positive)
        init_pos: Initial camera position (relative to target). 
                 Defaults to [0, 0, distance]
        init_up: Initial up vector. Defaults to [0, 1, 0]
    
    Returns:
        Tuple of (position, up) where:
        - position: 3D camera position as np.ndarray
        - up: 3D up vector as np.ndarray
    
    Raises:
        ValueError: If parameters are invalid
        CameraParameterError: If angles are invalid
    
    Examples:
        # Basic usage - camera in front of origin
        pos, up = get_camera_pos(
            target=np.array([0, 0, 0]),
            azimuth=0, elevation=0, roll=0,
            distance=5.0
        )
        
        # Angled view with custom initial vectors
        pos, up = get_camera_pos(
            target=np.array([0, 0, 0]),
            azimuth=np.pi/4, elevation=np.pi/6, roll=0,
            distance=3.0,
            init_pos=np.array([1, 0, 0]),
            init_up=np.array([0, 0, 1])
        )
    """
    # Validate inputs
    validate_camera_angles(azimuth, elevation, roll)
    
    if not isinstance(target, np.ndarray) or target.shape != (3,):
        raise ValueError("target must be a 3D numpy array")
    
    if not isinstance(distance, (int, float)) or distance <= 0:
        raise ValueError("distance must be positive")
    
    # Set defaults
    if init_pos is None:
        init_pos = np.array([0, 0, distance], dtype=np.float32)
    if init_up is None:
        init_up = np.array([0, 1, 0], dtype=np.float32)
    
    # Validate initial vectors
    if not isinstance(init_pos, np.ndarray) or init_pos.shape != (3,):
        raise ValueError("init_pos must be a 3D numpy array")
    if not isinstance(init_up, np.ndarray) or init_up.shape != (3,):
        raise ValueError("init_up must be a 3D numpy array")
    
    # Ensure target and initial vectors are float32 for consistency
    target = target.astype(np.float32)
    init_pos = init_pos.astype(np.float32)
    init_up = init_up.astype(np.float32)
    
    # Make sure init_pos is relative to target and normalized to correct distance
    rel_init_pos = init_pos - target
    norm = np.linalg.norm(rel_init_pos)
    if norm == 0:
        raise ValueError("init_pos must not be the zero vector (relative to target)")
    rel_init_pos = rel_init_pos / norm * distance
    
    # Azimuth: rotate around init_up axis
    rot_azimuth = R.from_rotvec(azimuth * init_up)
    
    # Elevation: rotate around axis perpendicular to init_up and view direction
    view_dir = -rel_init_pos / np.linalg.norm(rel_init_pos)
    elev_axis = np.cross(init_up, view_dir)
    elev_norm = np.linalg.norm(elev_axis)
    
    if elev_norm < 1e-6:  # Handle gimbal lock
        # Use a fallback axis when init_up and view_dir are parallel
        fallback_axis = (
            np.array([1, 0, 0]) if abs(init_up[0]) < 0.9 else np.array([0, 0, 1])
        )
        elev_axis = np.cross(init_up, fallback_axis)
        elev_norm = np.linalg.norm(elev_axis)
    
    elev_axis = elev_axis / elev_norm
    rot_elevation = R.from_rotvec(elevation * elev_axis)
    
    # Roll: rotate around view direction
    rot_roll = R.from_rotvec(roll * view_dir)
    
    # Combined rotation
    rot = rot_azimuth * rot_elevation * rot_roll
    
    # Apply rotation and translate back to world coordinates
    position = rot.apply(rel_init_pos) + target
    up = rot.apply(init_up)
    
    return position, up


def get_camera_pos_from_params(params: CameraParameters) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate camera position and up vector from CameraParameters object.
    
    Args:
        params: CameraParameters instance with all camera settings
        
    Returns:
        Tuple of (position, up) vectors
    """
    return get_camera_pos(
        target=params.target,
        azimuth=params.azimuth,
        elevation=params.elevation,
        roll=params.roll,
        distance=params.distance,
        init_pos=params.init_pos,
        init_up=params.init_up
    )


class CameraPath:
    """
    Camera path animation utility for smooth camera movements.
    
    This class enables smooth interpolation between multiple camera positions,
    useful for creating camera animations and transitions.
    """
    
    def __init__(self, keyframes: List[CameraParameters]):
        """
        Initialize camera path with keyframe positions.
        
        Args:
            keyframes: List of CameraParameters defining the path
        """
        if len(keyframes) < 2:
            raise ValueError("At least 2 keyframes are required for a camera path")
        
        self.keyframes = keyframes
        self.validate_keyframes()
    
    def validate_keyframes(self) -> None:
        """Validate that all keyframes are consistent."""
        for i, keyframe in enumerate(self.keyframes):
            if not isinstance(keyframe, CameraParameters):
                raise ValueError(f"Keyframe {i} must be a CameraParameters instance")
            keyframe.validate()
    
    def interpolate(self, t: float) -> CameraParameters:
        """
        Interpolate camera parameters at time t.
        
        Args:
            t: Time parameter (0.0 = first keyframe, 1.0 = last keyframe)
            
        Returns:
            Interpolated CameraParameters
        """
        if not (0.0 <= t <= 1.0):
            raise ValueError("t must be between 0.0 and 1.0")
        
        if t == 0.0:
            return self.keyframes[0].copy()
        if t == 1.0:
            return self.keyframes[-1].copy()
        
        # Find surrounding keyframes
        n_segments = len(self.keyframes) - 1
        segment = min(int(t * n_segments), n_segments - 1)
        local_t = (t * n_segments) - segment
        
        kf1 = self.keyframes[segment]
        kf2 = self.keyframes[segment + 1]
        
        # Linear interpolation for all parameters
        return CameraParameters(
            target=self._lerp_vector(kf1.target, kf2.target, local_t),
            azimuth=self._lerp_angle(kf1.azimuth, kf2.azimuth, local_t),
            elevation=self._lerp_angle(kf1.elevation, kf2.elevation, local_t),
            roll=self._lerp_angle(kf1.roll, kf2.roll, local_t),
            distance=kf1.distance + (kf2.distance - kf1.distance) * local_t,
            init_pos=self._lerp_vector(kf1.init_pos, kf2.init_pos, local_t),
            init_up=self._lerp_vector(kf1.init_up, kf2.init_up, local_t),
            fov=kf1.fov + (kf2.fov - kf1.fov) * local_t,
            near_plane=kf1.near_plane + (kf2.near_plane - kf1.near_plane) * local_t,
            far_plane=kf1.far_plane + (kf2.far_plane - kf1.far_plane) * local_t
        )
    
    def _lerp_vector(self, v1: np.ndarray, v2: np.ndarray, t: float) -> np.ndarray:
        """Linear interpolation for vectors."""
        return v1 + (v2 - v1) * t
    
    def _lerp_angle(self, a1: float, a2: float, t: float) -> float:
        """
        Linear interpolation for angles, handling wraparound.
        
        Uses the shortest path between angles.
        """
        # Normalize angles to [-π, π]
        a1 = ((a1 + np.pi) % (2 * np.pi)) - np.pi
        a2 = ((a2 + np.pi) % (2 * np.pi)) - np.pi
        
        # Find shortest path
        diff = a2 - a1
        if diff > np.pi:
            diff -= 2 * np.pi
        elif diff < -np.pi:
            diff += 2 * np.pi
        
        return a1 + diff * t
    
    def generate_frames(self, n_frames: int) -> List[CameraParameters]:
        """
        Generate a sequence of camera parameters for animation.
        
        Args:
            n_frames: Number of frames to generate
            
        Returns:
            List of CameraParameters for each frame
        """
        if n_frames < 1:
            raise ValueError("n_frames must be at least 1")
        
        if n_frames == 1:
            return [self.keyframes[0].copy()]
        
        frames = []
        for i in range(n_frames):
            t = i / (n_frames - 1)
            frames.append(self.interpolate(t))
        
        return frames


class CameraController:
    """
    High-level camera controller for interactive camera manipulation.
    
    This class provides methods for common camera operations like orbiting,
    panning, zooming, and smooth transitions.
    """
    
    def __init__(self, initial_params: Optional[CameraParameters] = None):
        """
        Initialize camera controller.
        
        Args:
            initial_params: Initial camera parameters. 
                          Defaults to front view at origin.
        """
        if initial_params is None:
            initial_params = CameraParameters.front_view()
        
        self.params = initial_params.copy()
    
    def orbit(self, delta_azimuth: float, delta_elevation: float) -> None:
        """
        Orbit camera around target.
        
        Args:
            delta_azimuth: Change in azimuth angle (radians)
            delta_elevation: Change in elevation angle (radians)
        """
        self.params.azimuth += delta_azimuth
        self.params.elevation += delta_elevation
        
        # Clamp elevation to avoid flipping
        self.params.elevation = np.clip(self.params.elevation, -np.pi/2, np.pi/2)
    
    def zoom(self, factor: float) -> None:
        """
        Zoom camera (change distance to target).
        
        Args:
            factor: Zoom factor (< 1.0 zooms in, > 1.0 zooms out)
        """
        if factor <= 0:
            raise ValueError("Zoom factor must be positive")
        
        self.params.distance *= factor
        
        # Prevent getting too close or too far
        self.params.distance = np.clip(self.params.distance, 0.01, 1000.0)
    
    def pan(self, delta_target: np.ndarray) -> None:
        """
        Pan camera (move target position).
        
        Args:
            delta_target: Change in target position
        """
        if not isinstance(delta_target, np.ndarray) or delta_target.shape != (3,):
            raise ValueError("delta_target must be a 3D numpy array")
        
        self.params.target += delta_target
    
    def roll_camera(self, delta_roll: float) -> None:
        """
        Roll camera around view direction.
        
        Args:
            delta_roll: Change in roll angle (radians)
        """
        self.params.roll += delta_roll
    
    def set_distance(self, distance: float) -> None:
        """
        Set camera distance to target.
        
        Args:
            distance: New distance (must be positive)
        """
        if distance <= 0:
            raise ValueError("Distance must be positive")
        self.params.distance = distance
    
    def look_at(self, target: np.ndarray) -> None:
        """
        Point camera at new target.
        
        Args:
            target: New target position
        """
        if not isinstance(target, np.ndarray) or target.shape != (3,):
            raise ValueError("target must be a 3D numpy array")
        
        self.params.target = target.astype(np.float32)
    
    def reset_to_preset(self, preset: str, distance: Optional[float] = None) -> None:
        """
        Reset camera to a preset view.
        
        Args:
            preset: Preset name ('front', 'side', 'top', 'isometric')
            distance: Optional distance override
        """
        target = self.params.target.copy()
        dist = distance if distance is not None else self.params.distance
        
        preset_methods = {
            'front': CameraParameters.front_view,
            'side': CameraParameters.side_view,
            'top': CameraParameters.top_view,
            'isometric': CameraParameters.isometric_view
        }
        
        if preset not in preset_methods:
            raise ValueError(f"Unknown preset: {preset}. Available: {list(preset_methods.keys())}")
        
        self.params = preset_methods[preset](target=target, distance=dist)
    
    def get_position_and_up(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get current camera position and up vector.
        
        Returns:
            Tuple of (position, up) vectors
        """
        return get_camera_pos_from_params(self.params)
    
    def animate_to(self, target_params: CameraParameters, n_frames: int = 30) -> List[CameraParameters]:
        """
        Create animation frames to transition to target parameters.
        
        Args:
            target_params: Target camera parameters
            n_frames: Number of transition frames
            
        Returns:
            List of CameraParameters for smooth transition
        """
        path = CameraPath([self.params.copy(), target_params.copy()])
        return path.generate_frames(n_frames)