# PyVR: Python Volume Rendering Toolkit

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: WTFPL](https://img.shields.io/badge/License-WTFPL-brightgreen.svg)](http://www.wtfpl.net/about/)
![Version](https://img.shields.io/badge/version-0.2.0-blue.svg)

PyVR is a GPU-accelerated 3D volume rendering toolkit focused on real-time interactive visualization using OpenGL. Built with ModernGL, it provides high-performance volume rendering with a modern, modular architecture designed for flexibility and maintainability.

> **🚨 Breaking Changes in v0.2.0**: PyVR has been completely refactored with a new modular architecture. See the [Migration Guide](#-migration-from-v010) for updating existing code.

## 🎯 Key Features

- **GPU-Accelerated Rendering**: Real-time OpenGL volume rendering via ModernGL
- **Interactive Visualization**: Advanced camera controls with quaternion rotations and animation paths
- **Modular Architecture**: Clean separation of concerns with dedicated modules for transfer functions, camera, and rendering
- **Flexible Transfer Functions**: Sophisticated color and opacity mappings with matplotlib integration and peak detection
- **Advanced Camera System**: Spherical coordinates, camera paths, presets, and smooth animations
- **Synthetic Datasets**: Built-in generators for testing and development
- **Modern OpenGL**: Efficient shader-based ray marching with optimized resource management
- **Easy Integration**: Simple, clean API for embedding in visualization applications

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/JixianLi/pyvr.git
cd pyvr

# Install with Poetry (recommended)
poetry install

# Or install dependencies manually
pip install moderngl numpy matplotlib pillow scipy
```

### Basic Usage

```python
import numpy as np
import matplotlib.pyplot as plt
from pyvr.moderngl_renderer import VolumeRenderer
from pyvr.transferfunctions import ColorTransferFunction, OpacityTransferFunction
from pyvr.camera import CameraParameters
from pyvr.datasets import create_sample_volume

# Create renderer and volume
renderer = VolumeRenderer(width=512, height=512, step_size=0.01, max_steps=500)
volume = create_sample_volume(256, 'double_sphere')
renderer.load_volume(volume)

# Set up advanced transfer functions
ctf = ColorTransferFunction.from_colormap('viridis', value_range=(0.2, 0.8))
otf = OpacityTransferFunction.with_peaks([0.3, 0.7], widths=[0.1, 0.1], opacities=[0.5, 0.8])

# Configure camera with spherical coordinates
camera_params = CameraParameters.from_spherical(
    target=np.array([0.0, 0.0, 0.0]),
    distance=3.0,
    azimuth=np.pi/4,      # 45 degrees 
    elevation=np.pi/6,    # 30 degrees
    roll=0.0
)
position, up = camera_params.get_camera_vectors()
renderer.set_camera(position=position, target=camera_params.target, up=up)

# Upload transfer functions and render
color_unit = ctf.to_texture(moderngl_manager=renderer.gl_manager)
opacity_unit = otf.to_texture(moderngl_manager=renderer.gl_manager)
renderer.gl_manager.set_uniform_int('color_lut', color_unit)
renderer.gl_manager.set_uniform_int('opacity_lut', opacity_unit)

# Render and display
data = renderer.render()
image = np.frombuffer(data, dtype=np.uint8).reshape((512, 512, 4))
plt.imshow(image, origin='lower')
plt.show()
```

## 🏗️ Architecture

PyVR v0.2.0 features a completely redesigned modular architecture:

```
pyvr/
├── transferfunctions/     # Color and opacity mapping (Phase 2)
│   ├── base.py           # Abstract base class with common functionality  
│   ├── color.py          # Color transfer functions with matplotlib integration
│   └── opacity.py        # Opacity transfer functions with peak detection
├── camera/               # Advanced camera system (Phase 3)  
│   ├── parameters.py     # Camera parameter management with presets
│   └── control.py        # Camera controllers and animation paths
├── shaders/              # Shared OpenGL shaders (Phase 1)
│   ├── volume.vert.glsl  # Vertex shader for volume rendering
│   └── volume.frag.glsl  # Fragment shader with ray marching
├── datasets/             # Synthetic volume generators
│   └── synthetic.py      # Various 3D shapes and patterns
└── moderngl_renderer/    # OpenGL rendering backend (Phase 4)
    ├── renderer.py       # High-level volume renderer API
    └── manager.py        # Low-level OpenGL resource management
```

**Key Components:**
- **Transfer Functions**: Modular color and opacity mapping with advanced features
- **Camera System**: Sophisticated parameter management with spherical coordinates and animation
- **Renderer**: Clean separation between high-level API and OpenGL implementation  
- **Shared Resources**: Common shaders and utilities available to all components

## 📊 Datasets

PyVR includes synthetic dataset generators for testing and development:

```python
from pyvr.datasets import create_sample_volume, compute_normal_volume

# Create various synthetic volumes
sphere_vol = create_sample_volume(256, 'sphere')
torus_vol = create_sample_volume(256, 'torus')
double_sphere_vol = create_sample_volume(256, 'double_sphere')
helix_vol = create_sample_volume(256, 'helix')

# Compute normal vectors for lighting
normals = compute_normal_volume(sphere_vol)
```

**Available volume types:**
- `sphere`: Simple sphere geometry
- `torus`: Torus (donut) shape
- `double_sphere`: Two overlapping spheres
- `cube`: Rounded cube geometry
- `helix`: Helical structure
- `random_blob`: Asymmetric random blob with noise

## 🎨 Transfer Functions

PyVR v0.2.0 provides a completely redesigned transfer function system with advanced features:

### Color Transfer Functions
```python
from pyvr.transferfunctions import ColorTransferFunction

# From matplotlib colormaps with value range
ctf = ColorTransferFunction.from_colormap('viridis', value_range=(0.2, 0.8))

# Custom control points  
ctf = ColorTransferFunction(control_points=[
    (0.0, [0.0, 0.0, 1.0]),  # Blue at low values
    (0.5, [0.0, 1.0, 0.0]),  # Green at mid values  
    (1.0, [1.0, 0.0, 0.0])   # Red at high values
])

# Grayscale with custom intensity
ctf = ColorTransferFunction.grayscale(intensity=0.8)

# Single color with opacity variation
ctf = ColorTransferFunction.single_color([1.0, 0.5, 0.0])  # Orange
```

### Opacity Transfer Functions
```python
from pyvr.transferfunctions import OpacityTransferFunction

# Linear opacity ramp
otf = OpacityTransferFunction.linear(min_opacity=0.0, max_opacity=0.5)

# Step function
otf = OpacityTransferFunction.step(threshold=0.3, low_opacity=0.0, high_opacity=1.0)

# Multiple peaks with custom widths
otf = OpacityTransferFunction.with_peaks(
    positions=[0.3, 0.7], 
    widths=[0.1, 0.15], 
    opacities=[0.6, 0.9]
)

# Custom control points for complex shapes
otf = OpacityTransferFunction(control_points=[
    (0.0, 0.0),
    (0.2, 0.1), 
    (0.5, 0.8),    # Peak at middle
    (0.8, 0.2),
    (1.0, 0.0)
])
```

## � Advanced Camera System

PyVR v0.2.0 introduces a sophisticated camera system with spherical coordinates and animation support:

### Camera Parameters
```python
from pyvr.camera import CameraParameters
import numpy as np

# Create camera with spherical coordinates
camera = CameraParameters.from_spherical(
    target=np.array([0.0, 0.0, 0.0]),
    distance=5.0,
    azimuth=np.pi/4,      # 45° rotation around target
    elevation=np.pi/6,    # 30° elevation angle  
    roll=0.0              # No roll rotation
)

# Use camera presets
camera = CameraParameters.preset_front_view(target=np.array([0, 0, 0]), distance=3.0)
camera = CameraParameters.preset_diagonal_view(target=np.array([0, 0, 0]), distance=4.0)

# Get camera vectors for renderer
position, up = camera.get_camera_vectors()
renderer.set_camera(position=position, target=camera.target, up=up)
```

### Camera Animation and Paths
```python
from pyvr.camera import CameraController, CameraPath

# Create smooth camera paths
path = CameraPath()
path.add_keyframe(0.0, CameraParameters.preset_front_view(target, distance=3.0))
path.add_keyframe(1.0, CameraParameters.preset_diagonal_view(target, distance=3.0))
path.add_keyframe(2.0, CameraParameters.preset_side_view(target, distance=3.0))

# Interpolate camera positions
t = 0.5  # Halfway between first and second keyframe  
interpolated_camera = path.interpolate(t)
position, up = interpolated_camera.get_camera_vectors()

# Advanced camera controller
controller = CameraController(initial_params=camera)
controller.orbit(delta_azimuth=0.1, delta_elevation=0.05)  # Smooth orbiting
controller.zoom(factor=1.1)  # Zoom in/out
controller.move_target(delta=np.array([0.1, 0, 0]))  # Pan target
```

## �📸 Examples

Check out the `example/` directory for complete working examples:

- **`ModernglRender/multiview_example_v0_2_0.py`**: Multi-view rendering with new v0.2.0 API
- **`ModernglRender/enhanced_camera_demo_v0_2_0.py`**: Advanced camera system demonstration
- **`ModernglRender/multiview_example.py`**: Legacy example (v0.1.0 compatibility)

### Multi-view Rendering (v0.2.0)
The new example demonstrates the modular architecture:

```bash
python example/ModernglRender/multiview_example_v0_2_0.py
```

### Camera Animation Demo
See the advanced camera system in action:

```bash
python example/ModernglRender/enhanced_camera_demo_v0_2_0.py
```

## ⚡ Performance

### ModernGL Renderer
- **Typical performance**: 5-30ms per frame (depending on volume size and quality settings)
- **Optimizations**: GPU ray marching, efficient texture sampling, hardware alpha blending
- **Scalability**: Real-time for volumes up to 512³ voxels on modern GPUs
- **Memory efficiency**: Automatic OpenGL resource management

## 🛠️ API Reference

### V0.2.0 Modular API

```python
# Volume rendering
from pyvr.moderngl_renderer import VolumeRenderer
renderer = VolumeRenderer(width, height, step_size, max_steps)
renderer.load_volume(volume_data)
renderer.set_camera(position, target, up) 
renderer.render()  # Returns raw RGBA bytes

# Transfer functions (NEW modular design)
from pyvr.transferfunctions import ColorTransferFunction, OpacityTransferFunction
ctf = ColorTransferFunction.from_colormap('viridis', value_range=(0.2, 0.8))
otf = OpacityTransferFunction.with_peaks([0.3, 0.7], widths=[0.1, 0.1])
color_unit = ctf.to_texture(moderngl_manager=renderer.gl_manager)
opacity_unit = otf.to_texture(moderngl_manager=renderer.gl_manager)

# Advanced camera system (NEW)
from pyvr.camera import CameraParameters, CameraController, CameraPath
camera = CameraParameters.from_spherical(target, distance, azimuth, elevation, roll)
controller = CameraController(camera)
path = CameraPath()

# Datasets
from pyvr.datasets import create_sample_volume, compute_normal_volume
volume = create_sample_volume(256, 'double_sphere')
normals = compute_normal_volume(volume)
```

### Legacy API (for backward compatibility)

```python
# Still available through moderngl_renderer module
from pyvr.moderngl_renderer import ColorTransferFunction, OpacityTransferFunction, get_camera_pos
position, up = get_camera_pos(target, azimuth, elevation, roll, distance)
```

## 🔄 Migration from v0.1.0

PyVR v0.2.0 introduces breaking changes with a new modular architecture. Here's how to update your code:

### Import Changes
```python
# OLD v0.1.0 imports ❌
from pyvr.moderngl_renderer import ColorTransferFunction, OpacityTransferFunction

# NEW v0.2.0 imports ✅  
from pyvr.transferfunctions import ColorTransferFunction, OpacityTransferFunction
```

### Camera System Changes
```python
# OLD v0.1.0 camera ❌
from pyvr.moderngl_renderer import get_camera_pos
position, up = get_camera_pos(target, azimuth, elevation, roll, distance)

# NEW v0.2.0 camera ✅
from pyvr.camera import CameraParameters  
camera = CameraParameters.from_spherical(target, distance, azimuth, elevation, roll)
position, up = camera.get_camera_vectors()

# Legacy function still available for backward compatibility
from pyvr.moderngl_renderer import get_camera_pos  # Still works
```

### Transfer Function API Changes
```python
# OLD v0.1.0 transfer functions ❌
ctf = ColorTransferFunction.from_matplotlib_colormap(plt.get_cmap('viridis'))

# NEW v0.2.0 transfer functions ✅
ctf = ColorTransferFunction.from_colormap('viridis', value_range=(0.0, 1.0))
```

### What's Compatible
- `VolumeRenderer` API remains the same
- `create_sample_volume()` and dataset functions unchanged  
- Basic rendering workflow is identical
- Legacy imports still work through `moderngl_renderer` module

## 🔧 Configuration

### Rendering Quality vs Performance

**High Quality (Slower)**:
```python
renderer = VolumeRenderer(1024, 1024, step_size=0.001, max_steps=1000)
```

**High Performance (Faster)**:
```python
renderer = VolumeRenderer(256, 256, step_size=0.02, max_steps=100)
```

**Balanced**:
```python
renderer = VolumeRenderer(512, 512, step_size=0.01, max_steps=500)
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest tests/

# Run ModernGL renderer tests  
pytest tests/test_moderngl_renderer/

# Run with coverage
pytest --cov=pyvr tests/
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Development Setup

```bash
# Clone and install in development mode
git clone https://github.com/JixianLi/pyvr.git
cd pyvr
poetry install --with dev

# Run tests
poetry run pytest

# Format code  
poetry run black pyvr/
poetry run isort pyvr/
```

## 📋 Requirements

- **Python**: 3.11+
- **Core Dependencies**: 
  - NumPy >= 2.3
  - Matplotlib >= 3.10
  - Pillow >= 11.0
  - SciPy >= 1.16
- **OpenGL Backend**:
  - ModernGL >= 5.0

## 📄 License

This project is licensed under the WTFPL (Do What The F*ck You Want To Public License) - see the [LICENSE](LICENSE) file for details.

## 🙋 Support & Questions

- **Issues**: Report bugs and request features on [GitHub Issues](https://github.com/JixianLi/pyvr/issues)
- **Email**: jixianli@sci.utah.edu

## 🏆 Acknowledgments

- ModernGL community for excellent OpenGL bindings
- Contributors and testers who helped improve PyVR
- The broader volume rendering and scientific visualization community

---

**PyVR** - High-performance OpenGL volume rendering made simple! 🎉

