# PyVR: Python Volume Rendering Toolkit

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: WTFPL](https://img.shields.io/badge/License-WTFPL-brightgreen.svg)](http://www.wtfpl.net/about/)

PyVR is a GPU-accelerated 3D volume rendering toolkit focused on real-time interactive visualization using OpenGL. Built with ModernGL, it provides high-performance volume rendering with modern OpenGL features.

## 🎯 Key Features

- **GPU-Accelerated Rendering**: Real-time OpenGL volume rendering via ModernGL
- **Interactive Visualization**: Smooth camera controls and parameter adjustment
- **Flexible Transfer Functions**: Custom color and opacity mappings with matplotlib integration
- **Synthetic Datasets**: Built-in generators for testing and development
- **Modern Architecture**: Clean separation between high-level API and OpenGL management
- **Easy Integration**: Simple API for embedding in visualization applications

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
from pyvr.moderngl_renderer import VolumeRenderer, create_sample_volume
from pyvr.moderngl_renderer import ColorTransferFunction, OpacityTransferFunction

# Create renderer
renderer = VolumeRenderer(512, 512, step_size=0.01, max_steps=500)

# Create and load volume
volume = create_sample_volume(256, 'double_sphere')
renderer.load_volume(volume)

# Set up transfer functions
ctf = ColorTransferFunction.from_matplotlib_colormap(plt.get_cmap('plasma'))
otf = OpacityTransferFunction.linear(0.0, 0.1)

# Configure camera
renderer.set_camera(position=(2, 2, 2), target=(0, 0, 0), up=(0, 0, 1))

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

PyVR is built around a clean, modern OpenGL rendering architecture:

```
VolumeRenderer (High-level API)
    ↓
ModernGLManager (OpenGL Resource Management)
    ↓
OpenGL Shaders (GPU Ray Marching)
```

**Key Components:**
- `VolumeRenderer`: Main rendering interface with high-level volume operations
- `ModernGLManager`: Low-level OpenGL resource management (textures, shaders, framebuffers)
- `TransferFunctions`: Color and opacity mapping utilities with matplotlib integration
- `CameraControl`: Camera positioning and orientation utilities for smooth animations
- `GLSL Shaders`: GPU-optimized volume ray marching implementation

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

PyVR supports flexible transfer function customization with matplotlib integration:

### Color Transfer Functions
```python
# From matplotlib colormaps
ctf = ColorTransferFunction.from_matplotlib_colormap(plt.get_cmap('viridis'))

# Custom color points
ctf = ColorTransferFunction()
ctf.set_custom_colors([
    (0.0, (0.0, 0.0, 1.0)),  # Blue at low values
    (0.5, (0.0, 1.0, 0.0)),  # Green at mid values  
    (1.0, (1.0, 0.0, 0.0))   # Red at high values
])
```

### Opacity Transfer Functions
```python
# Linear opacity ramp
otf = OpacityTransferFunction.linear(min_opacity=0.0, max_opacity=0.5)

# Step function
otf = OpacityTransferFunction.step(threshold=0.3, low_opacity=0.0, high_opacity=1.0)

# Bell curve
otf = OpacityTransferFunction.bell_curve(center=0.5, width=0.2, max_opacity=0.8)

# Custom control points
otf = OpacityTransferFunction(control_points=[
    (0.0, 0.0),
    (0.2, 0.1), 
    (0.8, 0.9),
    (1.0, 1.0)
])
```

## 📸 Examples

Check out the `example/` directory for complete working examples:

- **`ModernglRender/multiview_example.py`**: Multi-view rendering with 2x2 layout

### Multi-view Rendering
The example demonstrates rendering the same volume from multiple camera angles:

```bash
python example/ModernglRender/multiview_example.py
```

## ⚡ Performance

### ModernGL Renderer
- **Typical performance**: 5-30ms per frame (depending on volume size and quality settings)
- **Optimizations**: GPU ray marching, efficient texture sampling, hardware alpha blending
- **Scalability**: Real-time for volumes up to 512³ voxels on modern GPUs
- **Memory efficiency**: Automatic OpenGL resource management

## 🛠️ API Reference

### ModernGL Renderer API

```python
# Core renderer
renderer = VolumeRenderer(width, height, step_size, max_steps)
renderer.load_volume(volume_data)
renderer.set_camera(position, target, up)
renderer.render()  # Returns raw RGBA bytes

# Transfer functions  
ctf = ColorTransferFunction.from_matplotlib_colormap(cmap)
otf = OpacityTransferFunction.linear(low, high)
color_unit = ctf.to_texture(moderngl_manager=renderer.gl_manager)
opacity_unit = otf.to_texture(moderngl_manager=renderer.gl_manager)

# Camera utilities
from pyvr.moderngl_renderer import get_camera_pos
position, up = get_camera_pos(target, azimuth, elevation, roll, distance)
```

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

