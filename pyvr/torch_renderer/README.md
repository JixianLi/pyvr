# PyVR PyTorch Renderer

Fully vectorized, differentiable volume rendering in PyTorch.

## Overview

The PyTorch renderer provides a pure Python implementation of volume ray casting that is fully differentiable and vectorized. It's designed for research applications, machine learning workflows, and scenarios where gradient computation through the rendering process is needed.

## Features

- **Differentiable**: Full gradient support through the rendering pipeline
- **Vectorized**: Efficient batch processing of rays and samples
- **Flexible**: Pure Python implementation easily modified for research
- **GPU Accelerated**: CUDA support for performance
- **Multiple Compositing**: Various alpha compositing strategies
- **Advanced Lighting**: Gradient-based normal computation and shading

## Quick Start

```python
import torch
from pyvr.torch_renderer import VolumeRenderer, Camera, ColorTransferFunction, OpacityTransferFunction
from pyvr.datasets import create_test_volume

# Create volume data
volume = create_test_volume(shape=(128, 128, 128), volume_type='torus')

# Initialize renderer
device = 'cuda' if torch.cuda.is_available() else 'cpu'
renderer = VolumeRenderer(volume_shape=volume.shape, device=device)

# Set up camera
camera = Camera(
    position=(200, 100, 200),
    target=(64, 64, 64),  # Volume center
    fov=45,
    device=device
)

# Configure transfer functions
renderer.set_color_transfer_function(
    ColorTransferFunction('hot', device=device)
)
renderer.set_opacity_transfer_function(
    OpacityTransferFunction.linear(0.0, 0.1, device=device)
)

# Render
rendered_image = renderer.render(
    volume=volume,
    camera=camera,
    image_size=(512, 512),
    n_samples=1000,
    use_lighting=True
)
```

## Key Components

### VolumeRenderer

Core rendering engine featuring:
- Vectorized ray casting
- Multiple alpha compositing methods
- Configurable sampling strategies
- Memory-efficient processing

### Camera

Flexible camera system with:
- Programmable positioning and orientation
- Perspective projection with configurable FOV
- View and projection matrix computation

### Transfer Functions

- **ColorTransferFunction**: Supports matplotlib colormaps and custom mappings
- **OpacityTransferFunction**: Linear, exponential, and custom opacity curves

## Rendering Pipeline

1. **Ray Generation**: Compute ray origins and directions for each pixel
2. **Volume Sampling**: Sample volume data along rays at regular intervals  
3. **Transfer Function Application**: Map intensity values to colors and opacity
4. **Lighting Computation**: Calculate normals from gradients and apply lighting
5. **Alpha Compositing**: Composite samples using various strategies

## Compositing Methods

- **vectorized**: Fully parallel compositing (default)
- **early_termination**: Stop compositing when opacity reaches threshold
- **sequential**: Original iterative method for comparison

## Advanced Features

### Differentiable Rendering

The renderer supports gradient computation through the entire pipeline:

```python
volume.requires_grad_(True)
rendered = renderer.render(volume, camera, ...)
loss = compute_loss(rendered, target)
loss.backward()
# volume.grad now contains gradients
```

### Lighting Models

- Ambient lighting for base illumination
- Diffuse lighting based on surface normals
- Configurable light direction and intensities

## Performance Optimization

- **Batch Processing**: Process multiple rays simultaneously
- **Memory Management**: Efficient tensor operations and memory reuse
- **Early Termination**: Skip unnecessary computation for fully opaque pixels
- **CUDA Support**: GPU acceleration when available

## Dependencies

See `requirements.txt`:
- torch: Core tensor operations and GPU support
- numpy: Numerical utilities
- matplotlib: Visualization and colormaps

## Examples

See `example/TorchRenderer/simple_demo.py` for a complete rendering demonstration with performance metrics.