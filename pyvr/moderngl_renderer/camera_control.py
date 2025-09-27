import numpy as np
from scipy.spatial.transform import Rotation as R


def get_camera_pos(
    target, azimuth, elevation, roll, distance, init_pos=None, init_up=None
):
    """
    Returns (position, up) for the camera using quaternion rotations.
    All angles in radians.
    Ensures the camera is exactly `distance` away from the target.
    """
    if init_pos is None:
        init_pos = np.array([0, 0, distance], dtype=np.float32)
    if init_up is None:
        init_up = np.array([0, 1, 0], dtype=np.float32)

    # Make sure init_pos is relative to target
    rel_init_pos = init_pos - target
    norm = np.linalg.norm(rel_init_pos)
    if norm == 0:
        raise ValueError("init_pos must not be the zero vector (relative to target).")
    rel_init_pos = rel_init_pos / norm * distance

    # Continue with rel_init_pos as your orbit vector
    # Azimuth: rotate around init_up
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

    position = rot.apply(rel_init_pos) + target
    up = rot.apply(init_up)

    return position, up
