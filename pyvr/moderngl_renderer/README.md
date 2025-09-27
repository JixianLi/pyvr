# PyVR ModernGL Renderer

GPU-accelerated, real-time OpenGL volume rendering using ModernGL. This renderer is designed for interactive visualization and provides excellent performance for real-time exploration of 3D volume data.

## 🎯 Overview

The ModernGL renderer leverages OpenGL shaders to perform volume ray marching directly on the GPU, providing:

- **Real-time performance**: 5-30ms render times for typical volumes
- **Interactive visualization**: Smooth camera movement and parameter adjustment
- **High-quality rendering**: Advanced lighting, transfer functions, and alpha blending
- **Modern architecture**: Clean separation between high-level operations and OpenGL management

## 🏗️ Architecture

The ModernGL renderer is built with a clean separation of concerns:

```
VolumeRenderer (High-level API)
    ↓
ModernGLManager (OpenGL Resource Management)
    ↓
OpenGL Shaders (GPU Ray Marching)
```

### Key Components

- **`VolumeRenderer`**: Main interface for volume rendering operations
- **`ModernGLManager`**: Handles OpenGL contexts, textures, shaders, and framebuffers
- **Transfer Functions**: Color and opacity mapping utilities
- **Camera Control**: Utilities for camera positioning and movement
- **GLSL Shaders**: GPU-optimized volume ray marching implementation

## 🚀 Quick Start

```python
import numpy as np
import matplotlib.pyplot as plt
from pyvr.moderngl_renderer import VolumeRenderer, create_sample_volume
from pyvr.moderngl_renderer import ColorTransferFunction, OpacityTransferFunction

# Create renderer
renderer = VolumeRenderer(width=512, height=512, step_size=0.01, max_steps=500)

# Create and load volume data
volume = create_sample_volume(256, 'double_sphere')
renderer.load_volume(volume)

# Set up transfer functions
ctf = ColorTransferFunction.from_matplotlib_colormap(plt.get_cmap('plasma'))
otf = OpacityTransferFunction.linear(0.0, 0.1)

# Configure camera
renderer.set_camera(position=(2, 2, 2), target=(0, 0, 0), up=(0, 0, 1))

# Upload transfer functions (new ModernGLManager architecture)
color_unit = ctf.to_texture(moderngl_manager=renderer.gl_manager)
opacity_unit = otf.to_texture(moderngl_manager=renderer.gl_manager)
renderer.gl_manager.set_uniform_int('color_lut', color_unit)
renderer.gl_manager.set_uniform_int('opacity_lut', opacity_unit)

# Render
data = renderer.render()
image = np.frombuffer(data, dtype=np.uint8).reshape((512, 512, 4))

# Display
plt.imshow(image, origin='lower')
plt.axis('off')
plt.show()
```

## 🎨 Transfer Functions

### Color Transfer Functions

```python
# From matplotlib colormaps
ctf = ColorTransferFunction.from_matplotlib_colormap(
    plt.get_cmap('viridis')
)

# Custom color points
control_points = [
    (0.0, (0.0, 0.0, 1.0)),  # Blue
    (0.3, (0.0, 1.0, 1.0)),  # Cyan
    (0.7, (1.0, 1.0, 0.0)),  # Yellow
    (1.0, (1.0, 0.0, 0.0))   # Red
]
ctf = ColorTransferFunction(control_points)

# Generate LUT for inspection
lut = ctf.to_lut(256)  # Returns (256, 3) RGB array
```

### Opacity Transfer Functions

```python
# Linear ramp
otf = OpacityTransferFunction.linear(low=0.0, high=0.5)

# Step function
otf = OpacityTransferFunction.one_step(step=0.3, low=0.0, high=1.0)

# Multiple peaks
otf = OpacityTransferFunction.peaks([0.2, 0.5, 0.8], opacity=0.8, eps=0.05)

# Custom control points
control_points = [(0.0, 0.0), (0.2, 0.1), (0.8, 0.9), (1.0, 1.0)]
otf = OpacityTransferFunction(control_points)
```

## 📷 Camera Control

### Manual Camera Setup

```python
# Direct camera positioning
renderer.set_camera(
    position=(3, 2, 1),
    target=(0, 0, 0),
    up=(0, 0, 1)
)
```

### Spherical Camera Control

```python
from pyvr.moderngl_renderer import get_camera_pos
import numpy as np

# Spherical coordinates
azimuth = np.pi / 4      # 45 degrees
elevation = np.pi / 6    # 30 degrees  
roll = 0.0               # No roll
distance = 3.0           # Distance from target

position, up = get_camera_pos(
    target=np.array([0, 0, 0]),
    azimuth=azimuth,
    elevation=elevation,
    roll=roll,
    distance=distance
)

renderer.set_camera(position=position, target=(0, 0, 0), up=up)
```

### Camera Animation

```python
import numpy as np
import time

# Orbit animation
n_frames = 60
for i in range(n_frames):
    azimuth = 2 * np.pi * i / n_frames
    position, up = get_camera_pos(
        target=np.array([0, 0, 0]),
        azimuth=azimuth,
        elevation=np.pi/6,
        roll=0,
        distance=3
    )
    
    renderer.set_camera(position=position, target=(0, 0, 0), up=up)
    data = renderer.render()
    
    # Process frame...
    time.sleep(0.016)  # ~60 FPS
```

## 💡 Lighting

```python
# Ambient and diffuse lighting
renderer.set_ambient_light(0.3)    # Ambient intensity
renderer.set_diffuse_light(0.7)    # Diffuse intensity

# Light positioning
renderer.set_light_position((1.0, 1.0, 1.0))
renderer.set_light_target((0.0, 0.0, 0.0))
```

## 🔧 Rendering Parameters

### Quality vs Performance Trade-offs

```python
# High Quality (Slower)
renderer = VolumeRenderer(
    width=1024, 
    height=1024,
    step_size=0.001,    # Smaller steps = higher quality
    max_steps=1000      # More steps = better depth
)

# High Performance (Faster)  
renderer = VolumeRenderer(
    width=256,
    height=256, 
    step_size=0.02,     # Larger steps = faster
    max_steps=100       # Fewer steps = faster
)

# Balanced
renderer = VolumeRenderer(
    width=512,
    height=512,
    step_size=0.01,
    max_steps=500
)
```

### Runtime Parameter Updates

```python
# Adjust parameters during runtime
renderer.set_step_size(0.005)      # Finer sampling
renderer.set_max_steps(800)        # More depth resolution

# Update volume bounds
renderer.set_volume_bounds(
    min_bounds=(-1.0, -1.0, -1.0),
    max_bounds=(1.0, 1.0, 1.0)
)
```

## 📊 Volume Data

### Loading Volume Data

```python
# Volume data should be 3D numpy array with shape (D, H, W)
volume = np.random.random((128, 128, 128)).astype(np.float32)
renderer.load_volume(volume)

# With normal vectors for lighting (shape: D, H, W, 3)  
from pyvr.datasets import compute_normal_volume
normals = compute_normal_volume(volume)
renderer.load_normal_volume(normals)
```

### Synthetic Volumes

```python
from pyvr.moderngl_renderer import create_sample_volume

# Available shapes
shapes = ['sphere', 'torus', 'double_sphere', 'cube', 'helix', 'random_blob']

for shape in shapes:
    volume = create_sample_volume(size=256, shape=shape)
    renderer.load_volume(volume)
    # ... render and save
```

## ⚡ Performance Optimization

### GPU Memory Management

The ModernGLManager handles OpenGL resources efficiently:

```python
# Texture management is automatic
renderer.load_volume(volume1)  # Uploads to GPU
renderer.load_volume(volume2)  # Replaces previous volume

# Manual cleanup (usually not needed)
renderer.gl_manager.cleanup()
```

### Rendering Pipeline

```python
# Optimal rendering loop
for frame in range(num_frames):
    # Update camera/parameters (fast)
    renderer.set_camera(position, target, up)
    
    # Update transfer functions only when needed
    if transfer_functions_changed:
        color_unit = ctf.to_texture(moderngl_manager=renderer.gl_manager)
        opacity_unit = otf.to_texture(moderngl_manager=renderer.gl_manager)
        renderer.gl_manager.set_uniform_int('color_lut', color_unit)
        renderer.gl_manager.set_uniform_int('opacity_lut', opacity_unit)
    
    # Render (GPU operation)
    data = renderer.render()
    
    # Process/display data...
```

### Batch Operations

```python
# Multiple viewpoints
camera_params = [
    {'position': (2, 2, 2), 'target': (0, 0, 0)},
    {'position': (-2, 2, 2), 'target': (0, 0, 0)}, 
    {'position': (2, -2, 2), 'target': (0, 0, 0)},
    {'position': (2, 2, -2), 'target': (0, 0, 0)}
]

images = []
for params in camera_params:
    renderer.set_camera(**params)
    data = renderer.render()
    images.append(data)
```

## 🎛️ Advanced Usage

### Custom Shaders

The renderer uses GLSL shaders located in `pyvr/moderngl_renderer/shaders/`:

- `volume.vert.glsl`: Vertex shader (fullscreen quad)
- `volume.frag.glsl`: Fragment shader (volume ray marching)

You can modify these shaders for custom effects or create new shader variants.

### Multi-Volume Rendering

```python
# Load different volumes
volume1 = create_sample_volume(256, 'sphere')
volume2 = create_sample_volume(256, 'torus') 

# Render sequence
for i, volume in enumerate([volume1, volume2]):
    renderer.load_volume(volume)
    data = renderer.render()
    # Save frame...
```

### Integration with Other Libraries

```python
# Convert to PIL Image
image_pil = renderer.render_to_pil()
image_pil.save('volume_render.png')

# Convert to numpy array
data = renderer.render()
image_array = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 4))

# Integration with matplotlib
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.imshow(image_array, origin='lower')
ax.axis('off')
plt.show()
```

## 🛠️ Troubleshooting

### Common Issues

**"OpenGL context creation failed"**
- Ensure you have proper OpenGL drivers installed
- Try running with software rendering: `export MESA_GL_VERSION_OVERRIDE=3.3`

**"Texture size too large"**
- Reduce volume size or check GPU memory limits
- Use smaller step sizes instead of higher resolution

**"Rendering appears black"**
- Check transfer function ranges
- Verify volume data is normalized [0, 1]
- Adjust camera position/distance

### Performance Issues

**Slow rendering**
- Increase `step_size` parameter
- Reduce `max_steps` parameter  
- Lower rendering resolution
- Check if volume fits in GPU memory

**Memory issues**
- Use smaller volume sizes
- Check available GPU memory
- Call `renderer.gl_manager.cleanup()` between renders

## 📋 API Reference

### VolumeRenderer

```python
VolumeRenderer(width, height, step_size, max_steps, ambient_light, diffuse_light, light_position, light_target)
```

**Methods:**
- `load_volume(volume_data)`: Load 3D volume data
- `load_normal_volume(normal_data)`: Load normal vectors for lighting
- `set_camera(position, target, up)`: Set camera parameters
- `render()`: Render volume and return raw RGBA bytes
- `render_to_pil()`: Render and return PIL Image
- `set_volume_bounds(min_bounds, max_bounds)`: Set world space bounds
- `set_step_size(step_size)`: Update ray marching step size
- `set_max_steps(max_steps)`: Update maximum ray marching steps
- `set_ambient_light(intensity)`: Set ambient lighting
- `set_diffuse_light(intensity)`: Set diffuse lighting

### Transfer Functions

**ColorTransferFunction:**
- `ColorTransferFunction(control_points, lut_size)`
- `from_matplotlib_colormap(cmap, lut_size)`: Create from matplotlib colormap
- `to_lut(size)`: Generate color lookup table
- `to_texture(moderngl_manager)`: Create OpenGL texture

**OpacityTransferFunction:**
- `OpacityTransferFunction(control_points, lut_size)`
- `linear(low, high, lut_size)`: Linear opacity ramp
- `one_step(step, low, high, lut_size)`: Step function
- `peaks(peaks, opacity, eps, lut_size)`: Multiple peaks
- `to_lut(size)`: Generate opacity lookup table
- `to_texture(moderngl_manager)`: Create OpenGL texture

### Camera Control

```python
get_camera_pos(target, azimuth, elevation, roll, distance, init_pos, init_up)
```

Returns camera position and up vector from spherical coordinates.

## 🏆 Best Practices

1. **Volume Data Format**: Always use `np.float32` for volume data
2. **Normalization**: Normalize volume data to [0, 1] range
3. **Transfer Function Design**: Start with simple linear functions, then customize
4. **Performance Tuning**: Profile with different step sizes and max steps
5. **Memory Management**: Monitor GPU memory usage for large volumes
6. **Camera Control**: Use spherical coordinates for smooth camera animations

---

The ModernGL renderer provides a powerful foundation for real-time volume visualization with excellent performance and flexibility! 🎉