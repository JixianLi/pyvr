# PyVR ModernGL Renderer

GPU-accelerated, real-time OpenGL volume rendering using ModernGL.

## Overview

The ModernGL renderer provides interactive 3D volume visualization with real-time performance. It uses OpenGL shaders for GPU-accelerated ray casting and supports interactive camera controls, lighting, and transfer functions.

## Features

- **Real-time Performance**: GPU-accelerated ray casting using OpenGL shaders
- **Interactive Camera**: Quaternion-based orbit, roll, and zoom controls
- **Flexible Transfer Functions**: Support for opacity and color mapping with matplotlib colormaps
- **Lighting Model**: Ambient and diffuse lighting with gradient-based normals
- **Multi-view Rendering**: Efficient rendering from multiple camera positions

## Quick Start

```python
import numpy as np
from pyvr.moderngl_renderer import VolumeRenderer, ColorTransferFunction, OpacityTransferFunction
from pyvr.datasets import create_sample_volume, compute_normal_volume

# Create volume data
volume = create_sample_volume(size=128, shape='torus')
normals = compute_normal_volume(volume)

# Initialize renderer
renderer = VolumeRenderer(width=512, height=512)
renderer.load_volume(volume)
renderer.load_normal_volume(normals)
renderer.set_volume_bounds((-1, -1, -1), (1, 1, 1))

# Set up transfer functions
color_tf = ColorTransferFunction.from_matplotlib_colormap('viridis')
opacity_tf = OpacityTransferFunction.linear(0.0, 0.1)

# Render
renderer.set_camera(position=(3, 3, 3), target=(0, 0, 0), up=(0, 0, 1))
image_data = renderer.render()
```

## Key Components

### VolumeRenderer

Main rendering class that handles:
- Volume data loading and texture management
- Camera positioning and projection matrices
- Shader program management
- Render loop execution

### Transfer Functions

- **ColorTransferFunction**: Maps intensity values to RGB colors
- **OpacityTransferFunction**: Maps intensity values to opacity/transparency

### Camera Control

Interactive camera system with:
- Orbital rotation (azimuth, elevation, roll)
- Distance-based zoom
- Programmable positioning

## Shader Pipeline

The renderer uses OpenGL shaders located in `shaders/`:
- `volume.vert.glsl`: Vertex shader for ray setup
- `volume.frag.glsl`: Fragment shader for ray casting and volume sampling

## Performance Notes

- Optimized for real-time interaction
- GPU memory usage scales with volume size
- Ray casting step size affects quality vs. performance trade-off
- Best performance with power-of-2 volume dimensions

## Dependencies

See `requirements.txt`:
- moderngl: OpenGL rendering
- numpy: Numerical operations  
- matplotlib: Colormap support
- pillow: Image processing

## Examples

See `example/ModernglRender/multiview_example.py` for a complete multi-view rendering demonstration.