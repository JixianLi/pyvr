"""
Enhanced camera control and positioning utilities for PyVR.

This module provides advanced camera positioning, animation, and control
functionality, including the core get_camera_pos function and enhanced
camera manipulation utilities.
"""

from typing import Callable, List, Optional, Tuple, Union

import numpy as np
from scipy.spatial.transform import Rotation as R

from .camera import Camera, CameraError, validate_camera_angles


def _map_to_sphere(x: float, y: float, radius: float = 1.0) -> np.ndarray:
    """
    Map 2D point to 3D point on virtual sphere (arcball algorithm).

    This implements the standard arcball sphere mapping:
    - Points inside the sphere radius are projected onto the sphere surface
    - Points outside use a hyperbolic sheet for smooth behavior at edges

    Args:
        x: Normalized x coordinate in range [-1, 1]
        y: Normalized y coordinate in range [-1, 1]
        radius: Virtual sphere radius (default: 1.0)

    Returns:
        3D point as np.ndarray of shape (3,), normalized to unit length

    Notes:
        The hyperbolic sheet at the sphere edge prevents discontinuities
        when the mouse moves outside the virtual trackball region.

    Example:
        >>> point = _map_to_sphere(0.0, 0.0)  # Center
        >>> np.allclose(point, [0, 0, 1])
        True
    """
    # Compute distance from origin in 2D
    d_squared = x * x + y * y
    r_squared = radius * radius

    if d_squared <= r_squared / 2.0:
        # Inside sphere - project directly to sphere surface
        # z² = r² - x² - y²
        z = np.sqrt(r_squared - d_squared)
    else:
        # Outside sphere - use hyperbolic sheet
        # z = (r²/2) / √(x² + y²)
        z = (r_squared / 2.0) / np.sqrt(d_squared)

    # Return normalized 3D point
    point = np.array([x, y, z], dtype=np.float32)
    norm = np.linalg.norm(point)
    return point / norm


def _camera_to_quaternion(camera: Camera) -> R:
    """
    Convert camera spherical coordinates to orientation quaternion.

    Builds a rotation matrix from the camera's position and up vectors,
    then converts to a quaternion representation using scipy.

    Args:
        camera: Camera instance with spherical coordinates

    Returns:
        scipy.spatial.transform.Rotation representing camera orientation

    Example:
        >>> camera = Camera.front_view(distance=3.0)
        >>> quat = _camera_to_quaternion(camera)
        >>> isinstance(quat, R)
        True
    """
    # Get camera position and up vector from spherical coordinates
    position, up = get_camera_pos_from_params(camera)

    # Build camera coordinate system (right-handed)
    # Forward: from camera to target
    forward = camera.target - position
    forward = forward / np.linalg.norm(forward)

    # Right: perpendicular to forward and up
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)

    # Up: corrected to be perpendicular to forward and right
    up_corrected = np.cross(right, forward)

    # Build rotation matrix (3x3, column-major)
    # Each column represents a basis vector in world space
    rotation_matrix = np.column_stack([right, up_corrected, -forward])

    # Convert rotation matrix to quaternion
    return R.from_matrix(rotation_matrix)


def _quaternion_to_camera_angles(
    rotation: R,
    target: np.ndarray,
    distance: float,
    init_pos: np.ndarray,
    init_up: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Decompose quaternion rotation to azimuth/elevation/roll angles.

    This is the inverse of the camera orientation calculation. It finds
    the spherical angles that would produce the given rotation when
    applied via get_camera_pos().

    Args:
        rotation: scipy Rotation (quaternion)
        target: Camera target point (3D)
        distance: Camera distance from target
        init_pos: Initial camera position for reference frame
        init_up: Initial up vector for reference frame

    Returns:
        Tuple of (azimuth, elevation, roll) in radians

    Notes:
        - Handles gimbal lock at elevation = ±π/2 by convention (azimuth=0)
        - Normalizes angles to [-π, π] range
        - Uses numerical clamping to avoid domain errors in arcsin/arccos

    Example:
        >>> camera = Camera.isometric_view(distance=3.0)
        >>> quat = _camera_to_quaternion(camera)
        >>> az, el, r = _quaternion_to_camera_angles(
        ...     quat, camera.target, camera.distance,
        ...     camera.init_pos, camera.init_up
        ... )
        >>> np.allclose(az, camera.azimuth, atol=1e-6)
        True
    """
    # Get rotation matrix from quaternion
    rot_matrix = rotation.as_matrix()

    # Extract camera basis vectors from rotation matrix
    right = rot_matrix[:, 0]
    up = rot_matrix[:, 1]
    forward = -rot_matrix[:, 2]  # Negative because camera looks down -Z

    # Compute camera position from forward vector and distance
    position = target - forward * distance

    # Now we need to find (azimuth, elevation, roll) that produces this orientation
    # Starting from init_pos and init_up

    # Normalize init_pos relative to target
    rel_init_pos = init_pos - target
    rel_init_pos = rel_init_pos / np.linalg.norm(rel_init_pos) * distance

    # Compute azimuth: rotation around init_up axis
    # Project position onto plane perpendicular to init_up
    pos_rel = position - target
    init_pos_proj = rel_init_pos - np.dot(rel_init_pos, init_up) * init_up
    pos_proj = pos_rel - np.dot(pos_rel, init_up) * init_up

    # Normalize projections
    init_pos_proj_norm = np.linalg.norm(init_pos_proj)
    pos_proj_norm = np.linalg.norm(pos_proj)

    if init_pos_proj_norm < 1e-9 or pos_proj_norm < 1e-9:
        # Gimbal lock: camera at pole (elevation = ±π/2)
        # Use convention: azimuth = 0
        azimuth = 0.0
    else:
        init_pos_proj = init_pos_proj / init_pos_proj_norm
        pos_proj = pos_proj / pos_proj_norm

        # Angle between projections
        cos_az = np.clip(np.dot(init_pos_proj, pos_proj), -1.0, 1.0)
        azimuth = np.arccos(cos_az)

        # Determine sign using cross product
        cross = np.cross(init_pos_proj, pos_proj)
        if np.dot(cross, init_up) < 0:
            azimuth = -azimuth

    # Compute elevation: angle from horizontal plane
    forward_norm = forward / np.linalg.norm(forward)
    # Project forward onto init_up to get vertical component
    # Note: forward points from camera TO target, so negate for elevation
    sin_el = np.clip(-np.dot(forward_norm, init_up), -1.0, 1.0)
    elevation = np.arcsin(sin_el)

    # Compute roll: rotation around view direction
    # Apply azimuth and elevation to get expected up vector (without roll)
    # Then compare with actual up vector to extract roll

    # Simulate rotation by azimuth and elevation only (roll=0)
    temp_camera = Camera(
        target=target,
        azimuth=azimuth,
        elevation=elevation,
        roll=0.0,
        distance=distance,
        init_pos=init_pos,
        init_up=init_up,
    )
    _, expected_up = get_camera_pos_from_params(temp_camera)

    # Actual up vector from rotation
    actual_up = up

    # Project both onto plane perpendicular to forward
    expected_up_proj = expected_up - np.dot(expected_up, forward_norm) * forward_norm
    actual_up_proj = actual_up - np.dot(actual_up, forward_norm) * forward_norm

    expected_up_norm = np.linalg.norm(expected_up_proj)
    actual_up_norm = np.linalg.norm(actual_up_proj)

    if expected_up_norm < 1e-9 or actual_up_norm < 1e-9:
        # Edge case: up vector parallel to forward (shouldn't happen normally)
        roll = 0.0
    else:
        expected_up_proj = expected_up_proj / expected_up_norm
        actual_up_proj = actual_up_proj / actual_up_norm

        cos_roll = np.clip(np.dot(expected_up_proj, actual_up_proj), -1.0, 1.0)
        roll = np.arccos(cos_roll)

        # Determine sign
        cross = np.cross(expected_up_proj, actual_up_proj)
        if np.dot(cross, forward_norm) < 0:
            roll = -roll

    return azimuth, elevation, roll


def get_camera_pos(
    target: np.ndarray,
    azimuth: float,
    elevation: float,
    roll: float,
    distance: float,
    init_pos: Optional[np.ndarray] = None,
    init_up: Optional[np.ndarray] = None,
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
        CameraError: If angles are invalid

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


def get_camera_pos_from_params(
    params: Camera,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate camera position and up vector from Camera object.

    Args:
        params: Camera instance with all camera settings

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
        init_up=params.init_up,
    )


class CameraPath:
    """
    Camera path animation utility for smooth camera movements.

    This class enables smooth interpolation between multiple camera positions,
    useful for creating camera animations and transitions.
    """

    def __init__(self, keyframes: List[Camera]):
        """
        Initialize camera path with keyframe positions.

        Args:
            keyframes: List of Camera defining the path
        """
        if len(keyframes) < 2:
            raise ValueError("At least 2 keyframes are required for a camera path")

        self.keyframes = keyframes
        self.validate_keyframes()

    def validate_keyframes(self) -> None:
        """Validate that all keyframes are consistent."""
        for i, keyframe in enumerate(self.keyframes):
            if not isinstance(keyframe, Camera):
                raise ValueError(f"Keyframe {i} must be a Camera instance")
            keyframe.validate()

    def interpolate(self, t: float) -> Camera:
        """
        Interpolate camera parameters at time t.

        Args:
            t: Time parameter (0.0 = first keyframe, 1.0 = last keyframe)

        Returns:
            Interpolated Camera
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
        return Camera(
            target=self._lerp_vector(kf1.target, kf2.target, local_t),
            azimuth=self._lerp_angle(kf1.azimuth, kf2.azimuth, local_t),
            elevation=self._lerp_angle(kf1.elevation, kf2.elevation, local_t),
            roll=self._lerp_angle(kf1.roll, kf2.roll, local_t),
            distance=kf1.distance + (kf2.distance - kf1.distance) * local_t,
            init_pos=self._lerp_vector(kf1.init_pos, kf2.init_pos, local_t),
            init_up=self._lerp_vector(kf1.init_up, kf2.init_up, local_t),
            fov=kf1.fov + (kf2.fov - kf1.fov) * local_t,
            near_plane=kf1.near_plane + (kf2.near_plane - kf1.near_plane) * local_t,
            far_plane=kf1.far_plane + (kf2.far_plane - kf1.far_plane) * local_t,
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

    def generate_frames(self, n_frames: int) -> List[Camera]:
        """
        Generate a sequence of camera parameters for animation.

        Args:
            n_frames: Number of frames to generate

        Returns:
            List of Camera for each frame
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

    def __init__(self, initial_params: Optional[Camera] = None):
        """
        Initialize camera controller.

        Args:
            initial_params: Initial camera parameters.
                          Defaults to front view at origin.
        """
        if initial_params is None:
            initial_params = Camera.front_view()

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
        self.params.elevation = np.clip(self.params.elevation, -np.pi / 2, np.pi / 2)

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

    def trackball(
        self,
        dx: float,
        dy: float,
        viewport_width: int,
        viewport_height: int,
        sensitivity: float = 1.0,
    ) -> None:
        """
        Rotate camera using trackball/arcball control.

        Provides intuitive 3D rotation following mouse movement,
        like rotating a physical ball. Mouse movement is mapped to
        rotation on a virtual sphere centered on the target.

        Args:
            dx: Mouse delta in pixels (horizontal, right is positive)
            dy: Mouse delta in pixels (vertical, down is positive)
            viewport_width: Viewport width in pixels
            viewport_height: Viewport height in pixels
            sensitivity: Rotation sensitivity multiplier (default: 1.0)
                        Higher values = more rotation per pixel

        Raises:
            ValueError: If viewport dimensions are invalid (<= 0)

        Example:
            >>> controller = CameraController()
            >>> # User dragged mouse 50 pixels right, 30 pixels down
            >>> controller.trackball(
            ...     dx=50, dy=-30,
            ...     viewport_width=800, viewport_height=600
            ... )
            >>> # Camera has rotated smoothly

        Notes:
            - Uses quaternion-based arcball algorithm for smooth rotation
            - No gimbal lock artifacts
            - Movement is relative to current camera orientation
            - Small movements (< 0.001 normalized) are ignored for stability
        """
        # Validate viewport dimensions
        if viewport_width <= 0 or viewport_height <= 0:
            raise ValueError(
                f"Viewport dimensions must be positive, got width={viewport_width}, height={viewport_height}"
            )

        # Early exit for zero movement
        if dx == 0 and dy == 0:
            return

        # Normalize pixel deltas to [-1, 1] range
        # Use the smaller dimension for uniform scaling
        scale = min(viewport_width, viewport_height)
        dx_norm = (dx / scale) * sensitivity
        dy_norm = (dy / scale) * sensitivity

        # Invert dx and dy for intuitive movement
        # (drag right = rotate left, drag up = rotate up)
        dx_norm = -dx_norm
        dy_norm = -dy_norm

        # Early exit for very small movements (avoid numerical instability)
        if abs(dx_norm) < 0.001 and abs(dy_norm) < 0.001:
            return

        # Map start and end points to sphere
        start_point = _map_to_sphere(0.0, 0.0)
        end_point = _map_to_sphere(dx_norm, dy_norm)

        # Compute rotation axis and angle
        axis = np.cross(start_point, end_point)
        axis_length = np.linalg.norm(axis)

        # Check for parallel vectors (no rotation needed)
        if axis_length < 1e-8:
            return

        axis = axis / axis_length

        # Compute rotation angle
        dot_product = np.clip(np.dot(start_point, end_point), -1.0, 1.0)
        angle = np.arccos(dot_product)

        # Create trackball rotation quaternion
        trackball_rotation = R.from_rotvec(angle * axis)

        # Get current camera orientation as quaternion
        current_rotation = _camera_to_quaternion(self.params)

        # Apply trackball rotation to current orientation
        # Important: trackball rotation is in camera-local space
        new_rotation = current_rotation * trackball_rotation

        # Decompose back to spherical angles
        new_azimuth, new_elevation, new_roll = _quaternion_to_camera_angles(
            new_rotation,
            self.params.target,
            self.params.distance,
            self.params.init_pos,
            self.params.init_up,
        )

        # Update camera parameters
        self.params.azimuth = new_azimuth
        self.params.elevation = new_elevation
        self.params.roll = new_roll

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
            "front": Camera.front_view,
            "side": Camera.side_view,
            "top": Camera.top_view,
            "isometric": Camera.isometric_view,
        }

        if preset not in preset_methods:
            raise ValueError(
                f"Unknown preset: {preset}. Available: {list(preset_methods.keys())}"
            )

        self.params = preset_methods[preset](target=target, distance=dist)

    def get_position_and_up(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get current camera position and up vector.

        Returns:
            Tuple of (position, up) vectors
        """
        return get_camera_pos_from_params(self.params)

    def animate_to(self, target_params: Camera, n_frames: int = 30) -> List[Camera]:
        """
        Create animation frames to transition to target parameters.

        Args:
            target_params: Target camera parameters
            n_frames: Number of transition frames

        Returns:
            List of Camera for smooth transition
        """
        path = CameraPath([self.params.copy(), target_params.copy()])
        return path.generate_frames(n_frames)
