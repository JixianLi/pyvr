"""ABOUTME: VTK volume data loader implementation.
ABOUTME: Loads VTK ImageData (.vti) files as PyVR Volume objects.
"""

from pathlib import Path
from typing import Union

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy  # type: ignore -- VTK is a large C++ library with Python bindings and often has incomplete type annotations.

from pyvr.volume import Volume


def load_vtk_volume(
    file_path: Union[str, Path], scalars_name: str = "Scalars_"
) -> Volume:
    """
    Load VTK ImageData (.vti) file as PyVR Volume.

    This function loads VTK ImageData files and converts them to PyVR's
    backend-agnostic Volume format. The data is automatically normalized
    to [0, 1] range, and bounds are calculated to preserve physical aspect
    ratio while centering the volume at [0, 0, 0].

    Args:
        file_path: Path to .vti file (str or pathlib.Path)
        scalars_name: Name of scalar array to load (default: 'Scalars_')

    Returns:
        Volume object with:
        - Data normalized to [0, 1] as float32
        - Bounds rescaled to [-1, 1] space (longest dimension)
        - Aspect ratio preserved from physical dimensions (spacing × dimensions)
        - Centered at [0, 0, 0]
        - Name set to "filename.vti(ScalarName)"
        - Normals computed automatically

    Raises:
        FileNotFoundError: File doesn't exist
        ValueError: Invalid VTK data (multi-component scalars, missing array,
                    invalid dimensions/spacing)

    Example:
        >>> from pyvr.dataloaders import load_vtk_volume
        >>> volume = load_vtk_volume("example_data/hydrogen.vti")
        >>> print(volume.name)
        hydrogen.vti(Scalars_)
        >>> print(volume.shape)
        (128, 128, 128)
    """
    # Convert to Path object (handles both str and Path)
    file_path = Path(file_path)

    # Validate file exists
    if not file_path.exists():
        raise FileNotFoundError(f"VTK file not found: {file_path}")

    # Load VTK file
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(str(file_path))
    reader.Update()

    # Get image data
    image_data = reader.GetOutput()

    # Extract and validate metadata
    dims = image_data.GetDimensions()  # (nx, ny, nz) in X,Y,Z order
    spacing = image_data.GetSpacing()  # (sx, sy, sz) in X,Y,Z order

    # Validate dimensions
    if any(d <= 0 for d in dims):
        raise ValueError(f"Invalid dimensions: {dims}")

    # Validate spacing
    if any(s <= 0 for s in spacing):
        raise ValueError(f"Invalid spacing (must be positive): {spacing}")

    # Extract and validate scalar array
    point_data = image_data.GetPointData()
    scalar_array = point_data.GetArray(scalars_name)

    if scalar_array is None:
        available = [
            point_data.GetArrayName(i) for i in range(point_data.GetNumberOfArrays())
        ]
        raise ValueError(
            f"Scalar array '{scalars_name}' not found in {file_path}. "
            f"Available arrays: {available}"
        )

    # Validate single-component
    num_components = scalar_array.GetNumberOfComponents()
    if num_components != 1:
        raise ValueError(
            f"Multi-component scalars not supported. "
            f"Array '{scalars_name}' has {num_components} components, expected 1."
        )

    # Convert to numpy and reshape
    # VTK internal data is in Z,Y,X order
    numpy_data = vtk_to_numpy(scalar_array)
    data_3d = numpy_data.reshape((dims[2], dims[1], dims[0]))  # (nz, ny, nx)

    # Normalize data to [0, 1]
    data_float = data_3d.astype(np.float32)
    data_min, data_max = data_float.min(), data_float.max()

    if data_max - data_min < 1e-9:
        # Handle constant volume
        normalized = np.zeros_like(data_float)
    else:
        normalized = (data_float - data_min) / (data_max - data_min)

    # Calculate bounds preserving physical aspect ratio
    # Physical dimensions = voxel dimensions × spacing (in X,Y,Z order)
    physical_dims = np.array(
        [dims[0] * spacing[0], dims[1] * spacing[1], dims[2] * spacing[2]],
        dtype=np.float32,
    )

    # Scale longest dimension to [-1, 1]
    longest = np.max(physical_dims)
    scale = 2.0 / longest

    # Calculate half-extents centered at [0, 0, 0]
    half_extents = physical_dims * scale / 2.0
    min_bounds = -half_extents
    max_bounds = +half_extents

    # Create volume name
    volume_name = f"{file_path.name}({scalars_name})"

    # Create Volume object
    volume = Volume(
        data=normalized,
        normals=None,
        min_bounds=min_bounds,
        max_bounds=max_bounds,
        name=volume_name,
    )

    # Compute normals
    volume.compute_normals()

    return volume
