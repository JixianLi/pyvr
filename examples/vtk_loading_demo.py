"""Demonstrate loading and rendering VTK volume data.

This example shows how to:
1. Load VTK .vti files using pyvr.dataloaders
2. Render with interactive interface
3. Inspect loaded volume properties
"""

from pyvr.config import RenderConfig
from pyvr.dataloaders import load_vtk_volume
from pyvr.interface import InteractiveVolumeRenderer

# Load hydrogen dataset
print("Loading hydrogen.vti...")
volume = load_vtk_volume("example_data/hydrogen.vti")

# Display volume properties
print(f"\nLoaded: {volume.name}")
print(f"Shape: {volume.shape}")
print(f"Data type: {volume.data.dtype}")
print(f"Data range: [{volume.data.min():.3f}, {volume.data.max():.3f}]")
print(f"Bounds: {volume.min_bounds} to {volume.max_bounds}")
print(f"Center: {volume.center}")
print(f"Dimensions: {volume.dimensions}")
print(f"Voxel spacing: {volume.voxel_spacing}")
print(f"Has normals: {volume.has_normals}")

# Create interactive renderer
print("\nCreating interactive renderer...")
interface = InteractiveVolumeRenderer(
    volume=volume, width=800, height=800, config=RenderConfig.balanced()
)

print("Launching interactive interface...")
print("\nControls:")
print("  - Drag to rotate camera")
print("  - Scroll to zoom")
print("  - Edit transfer functions using widgets")
print("  - Press 'r' to reset view")
print("  - Press 's' to save screenshot")
print("  - Press 'q' to quit")

interface.show()
