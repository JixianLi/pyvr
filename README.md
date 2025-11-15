# PyVR: Python Volume Rendering Toolkit

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: WTFPL](https://img.shields.io/badge/License-WTFPL-brightgreen.svg)](http://www.wtfpl.net/about/)
![Version](https://img.shields.io/badge/version-0.3.2-blue.svg)
[![Tests](https://img.shields.io/badge/tests-398%20passing-brightgreen.svg)](#-testing)

PyVR is a GPU-accelerated 3D volume rendering toolkit for real-time interactive visualization using OpenGL. Built with ModernGL, it provides high-performance volume rendering with a modern, modular architecture.

> ⚠️ **Pre-1.0 Development**: Breaking changes are expected. API stability comes at v1.0.0.

## 🎯 Key Features

- **⚡ GPU-Accelerated Rendering**: Real-time OpenGL volume rendering via ModernGL at 64+ FPS
- **🚀 High-Performance RGBA Textures**: Single-texture transfer function lookups for optimal performance
- **⚙️ Quality Presets**: Easy performance/quality tradeoff with RenderConfig presets (fast, balanced, high_quality)
- **🧩 Pipeline-Aligned Architecture**: Clean separation of Application, Geometry, and Fragment stages
- **📦 Backend-Agnostic Volume Data**: Unified Volume class for data, normals, and bounds management
- **📹 Advanced Camera System**: Matrix generation, spherical coordinates, animation paths, and presets
- **💡 Flexible Lighting System**: Directional, point, and ambient light presets with easy configuration
- **🎨 Sophisticated Transfer Functions**: Color and opacity mappings with matplotlib integration
- **🎮 Interactive Interface**: Real-time volume visualization with transfer function editing (v0.3.0+)
  - **v0.3.1**: FPS counter, quality presets, camera-linked lighting, histogram background
  - **NEW in v0.3.2**: 3-column layout for better display of all interface information
- **📊 Synthetic Datasets**: Built-in generators for testing and development
- **✅ Comprehensive Testing**: 398 tests with 86%+ coverage

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
from pyvr.config import RenderConfig
from pyvr.transferfunctions import ColorTransferFunction, OpacityTransferFunction
from pyvr.camera import Camera
from pyvr.lighting import Light
from pyvr.volume import Volume
from pyvr.datasets import create_sample_volume, compute_normal_volume

# Create volume data with Volume class
volume_data = create_sample_volume(256, 'double_sphere')
normals = compute_normal_volume(volume_data)
volume = Volume(
    data=volume_data,
    normals=normals,
    min_bounds=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
    max_bounds=np.array([1.0, 1.0, 1.0], dtype=np.float32)
)

# Create camera
camera = Camera.from_spherical(
    target=np.array([0.0, 0.0, 0.0]),
    distance=3.0,
    azimuth=np.pi/4,      # 45 degrees
    elevation=np.pi/6,    # 30 degrees
    roll=0.0
)

# Create light
light = Light.directional(direction=[1, -1, 0], ambient=0.2, diffuse=0.8)

# Create renderer with high quality preset
config = RenderConfig.high_quality()  # Or: fast(), balanced(), preview()
renderer = VolumeRenderer(width=512, height=512, config=config, light=light)
renderer.load_volume(volume)
renderer.set_camera(camera)

# Set up transfer functions
ctf = ColorTransferFunction.from_colormap('viridis')
otf = OpacityTransferFunction.linear(0.0, 0.1)
renderer.set_transfer_functions(ctf, otf)

# Render
data = renderer.render()
image = np.frombuffer(data, dtype=np.uint8).reshape((512, 512, 4))

plt.imshow(image, origin='lower')
plt.show()
```

## 🎮 Interactive Interface

PyVR includes an interactive matplotlib-based interface for real-time volume visualization and transfer function editing (v0.3.0+):

```python
from pyvr.interface import InteractiveVolumeRenderer
from pyvr.datasets import create_sample_volume
from pyvr.volume import Volume
import numpy as np

# Create volume
volume_data = create_sample_volume(128, 'sphere')
volume = Volume(data=volume_data)

# Launch interactive interface
interface = InteractiveVolumeRenderer(volume=volume)

# NEW in v0.3.1: Enable camera-linked lighting
interface.set_camera_linked_lighting(azimuth_offset=np.pi/4)

# Launch (FPS counter and histogram enabled by default)
interface.show()

# After closing: Capture ultra-quality image
# path = interface.capture_high_quality_image("render.png")
```

**Features:**
- 🎥 **Real-time camera controls**: Mouse drag to orbit, scroll to zoom
- 🎨 **Interactive opacity transfer function editor**: Add, remove, and drag control points
- 🌈 **Colormap selection**: Choose from 12+ matplotlib colormaps
- ⚡ **Performance optimized**: Render throttling, caching, and smart updates
- **NEW in v0.3.1:**
  - 📊 **FPS Counter**: Real-time performance monitoring with rolling average
  - ⚙️ **Quality Presets**: 5 rendering quality levels (preview → ultra)
  - 💡 **Camera-Linked Lighting**: Light follows camera for consistent illumination
  - 📈 **Histogram Background**: Log-scale data distribution in opacity editor

**Mouse Controls:**
- **Image Display**: Drag to orbit camera, scroll to zoom
- **Opacity Editor**: Left-click to add/select points, right-click to remove, drag to move

**Keyboard Shortcuts:**
- `r`: Reset camera to isometric view
- `s`: Save current rendering to PNG file
- `f`: Toggle FPS counter ✨ *NEW in v0.3.1*
- `h`: Toggle histogram background ✨ *NEW in v0.3.1*
- `l`: Toggle light linking to camera ✨ *NEW in v0.3.1*
- `q`: Toggle automatic quality switching ✨ *NEW in v0.3.1*
- `Esc`: Deselect control point
- `Delete`/`Backspace`: Remove selected control point

**Quality Presets (v0.3.1):**
- **Preview**: Extremely fast (~50 samples/ray)
- **Fast**: Interactive quality (~86 samples/ray)
- **Balanced**: Default quality (~173 samples/ray)
- **High Quality**: Publication quality (~346 samples/ray)
- **Ultra**: Maximum quality (~1732 samples/ray)

**Performance Features (v0.3.1):**
- Auto-quality switching: Automatically uses "fast" preset during camera interaction
- Histogram caching: >5x speedup with persistent cache
- All monitoring features: <1% overhead

**Note:** This is a testing/development interface. For production use, consider implementing a custom backend.

See `example/ModernglRender/v031_features_demo.py` for a complete v0.3.1 example.

## 🏗️ Architecture

PyVR follows a pipeline-aligned architecture based on traditional rendering pipeline stages:

```
pyvr/
├── volume/               # Application Stage - Volume data management
│   └── data.py           # Volume class with properties and operations
├── camera/               # Geometry Stage - Camera transformations
│   ├── camera.py         # Camera class with matrix generation
│   └── control.py        # Camera controllers and animation
├── lighting/             # Application Stage - Light configuration
│   └── light.py          # Light class with presets and camera linking
├── config.py             # Rasterization Stage - Rendering configuration
├── transferfunctions/    # Application Stage - Material properties
│   ├── color.py          # Color transfer functions
│   └── opacity.py        # Opacity transfer functions
├── moderngl_renderer/    # OpenGL Volume Renderer
│   ├── renderer.py       # ModernGLVolumeRenderer (main renderer)
│   └── manager.py        # Low-level OpenGL resource management
├── interface/            # Interactive Interface (v0.3.0+)
│   ├── matplotlib_interface.py  # InteractiveVolumeRenderer with all features
│   ├── widgets.py        # UI components (ImageDisplay, OpacityEditor, etc.)
│   ├── state.py          # InterfaceState for state management
│   └── cache.py          # Histogram caching (v0.3.1)
├── shaders/              # Fragment Stage - Shading operations
│   ├── volume.vert.glsl  # Vertex shader
│   └── volume.frag.glsl  # Fragment shader with RGBA lookups
└── datasets/             # Application Stage - Volume data utilities
    └── synthetic.py      # Synthetic volume generators
```

## 📦 Volume System

PyVR provides a unified Volume class for backend-agnostic data management:

### Volume Creation

```python
from pyvr.volume import Volume
import numpy as np

# Create Volume with all attributes
volume = Volume(
    data=volume_data,                                     # 3D numpy array
    normals=normal_data,                                  # Optional 4D array (D,H,W,3)
    min_bounds=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
    max_bounds=np.array([1.0, 1.0, 1.0], dtype=np.float32),
    name="my_volume"                                      # Optional name
)

# Simple volume (default bounds: [-0.5, -0.5, -0.5] to [0.5, 0.5, 0.5])
volume = Volume(data=volume_data)
```

### Volume Properties

```python
# Access volume metadata
print(volume.shape)          # (256, 256, 256) - voxel dimensions
print(volume.dimensions)     # [2.0, 2.0, 2.0] - physical size (max - min)
print(volume.center)         # [0.0, 0.0, 0.0] - bounding box center
print(volume.has_normals)    # True/False - check if normals present
print(volume.voxel_spacing)  # [0.0078, 0.0078, 0.0078] - spacing between voxels
```

### Volume Operations

```python
# Compute normals from volume data
volume.compute_normals()  # Generates normals using gradient
assert volume.has_normals  # True after computation

# Normalize volume data
normalized_vol = volume.normalize(method="minmax")  # Scale to [0, 1]
normalized_vol = volume.normalize(method="zscore")  # Z-score normalization

# Create independent copy
vol_copy = volume.copy()
vol_copy.data[0, 0, 0] = 1.0  # Doesn't affect original
```

### Using Volume with Renderer

```python
from pyvr.moderngl_renderer import VolumeRenderer

# Create and load volume (single call)
renderer = VolumeRenderer()
renderer.load_volume(volume)

# Get current volume
current_volume = renderer.get_volume()
```

## 📊 Synthetic Datasets

PyVR includes built-in synthetic volume generators:

```python
from pyvr.datasets import create_sample_volume, compute_normal_volume

# Create various synthetic volumes
sphere = create_sample_volume(256, 'sphere')
torus = create_sample_volume(256, 'torus')
double_sphere = create_sample_volume(256, 'double_sphere')
helix = create_sample_volume(256, 'helix')
cube = create_sample_volume(256, 'cube')
blob = create_sample_volume(256, 'random_blob')

# Compute normal vectors for lighting
normals = compute_normal_volume(sphere)
```

## ⚙️ Rendering Configuration

PyVR provides quality presets for easy performance/quality tradeoffs via the `RenderConfig` class:

### Quality Presets

```python
from pyvr.config import RenderConfig
from pyvr.moderngl_renderer import VolumeRenderer

# Use a preset
renderer = VolumeRenderer(config=RenderConfig.fast())          # Fast, interactive
renderer = VolumeRenderer(config=RenderConfig.balanced())      # Default, good balance
renderer = VolumeRenderer(config=RenderConfig.high_quality())  # High quality, slower
renderer = VolumeRenderer(config=RenderConfig.preview())       # Very fast, low quality
renderer = VolumeRenderer(config=RenderConfig.ultra_quality()) # Maximum quality, very slow
```

### Preset Comparison

| Preset | Step Size | Max Steps | Est. Speed | Use Case |
|--------|-----------|-----------|------------|----------|
| **preview** | 0.05 | 50 | ~15x faster | Quick iteration |
| **fast** | 0.02 | 100 | ~5x faster | Interactive exploration |
| **balanced** | 0.01 | 500 | **1x (baseline)** | **General use (default)** |
| **high_quality** | 0.005 | 1000 | ~5x slower | Final renders |
| **ultra_quality** | 0.001 | 2000 | ~20x slower | Publication quality |

### Custom Configuration

```python
# Create custom config
config = RenderConfig(
    step_size=0.015,
    max_steps=300,
    early_ray_termination=True,
    opacity_threshold=0.95
)
renderer = VolumeRenderer(config=config)

# Or modify a preset
config = RenderConfig.balanced().with_step_size(0.008)
config = RenderConfig.fast().with_max_steps(200)
```

### Runtime Configuration Changes

```python
# Change quality on the fly
renderer.set_config(RenderConfig.fast())       # Switch to fast rendering
renderer.set_config(RenderConfig.high_quality()) # Switch to high quality

# Get current config
current_config = renderer.get_config()
print(current_config)  # Shows current settings
```

### Performance Estimation

```python
# Estimate rendering performance
config = RenderConfig.high_quality()
samples = config.estimate_samples_per_ray()      # ~346 samples
relative_time = config.estimate_render_time_relative()  # ~5.0x slower than balanced
```

### Opacity Correction

PyVR implements Beer-Lambert law for physically correct opacity accumulation (v0.3.3+). This ensures all quality presets produce consistent visual appearance.

**How It Works:**

Transfer functions define opacity at a reference step size. When rendering at different step sizes, opacity is automatically corrected:

```python
# Formula: alpha_corrected = 1.0 - exp(-alpha_tf * step_size / reference_step_size)
```

**Default Behavior:**

```python
# All presets use reference_step_size=0.01 by default
# This means transfer functions are designed for "balanced" quality
config = RenderConfig.high_quality()  # Works correctly, looks same as balanced
```

**Customizing for Your Data:**

```python
# Feature-dense volumes (medical, turbulence): use smaller reference
config = RenderConfig(
    step_size=0.01,
    max_steps=500,
    reference_step_size=0.005  # Denser sampling reference
)

# Simple volumes (synthetic, smooth): use larger reference
config = RenderConfig(
    step_size=0.01,
    max_steps=500,
    reference_step_size=0.02  # Sparser sampling reference
)
```

**Benefits:**
- All presets produce same overall appearance
- Switch quality without changing how it looks
- Physically accurate (Beer-Lambert law)
- Industry standard (matches VTK, ParaView)

## 🎨 Transfer Functions

### Color Transfer Functions

```python
from pyvr.transferfunctions import ColorTransferFunction

# From matplotlib colormaps
ctf = ColorTransferFunction.from_colormap('viridis')
ctf = ColorTransferFunction.from_colormap('plasma', value_range=(0.2, 0.8))

# Custom control points
ctf = ColorTransferFunction(control_points=[
    (0.0, [0.0, 0.0, 1.0]),  # Blue at low values
    (0.5, [0.0, 1.0, 0.0]),  # Green at mid values
    (1.0, [1.0, 0.0, 0.0])   # Red at high values
])

# Convenience methods
ctf = ColorTransferFunction.grayscale(intensity=0.8)
ctf = ColorTransferFunction.single_color([1.0, 0.5, 0.0])
```

### Opacity Transfer Functions

```python
from pyvr.transferfunctions import OpacityTransferFunction

# Linear opacity ramp
otf = OpacityTransferFunction.linear(min_opacity=0.0, max_opacity=0.5)

# Step function
otf = OpacityTransferFunction.step(threshold=0.3, low_opacity=0.0, high_opacity=1.0)

# Multiple peaks
otf = OpacityTransferFunction.peaks(
    peaks=[0.3, 0.7],
    opacity=0.8,
    eps=0.1
)

# Custom control points
otf = OpacityTransferFunction(control_points=[
    (0.0, 0.0),
    (0.2, 0.1),
    (0.5, 0.8),  # Peak
    (0.8, 0.2),
    (1.0, 0.0)
])
```

## 📹 Camera System

### Camera Creation and Control

```python
from pyvr.camera import Camera, CameraController, CameraPath
import numpy as np

# Create camera with spherical coordinates
camera = Camera.from_spherical(
    target=np.array([0.0, 0.0, 0.0]),
    distance=5.0,
    azimuth=np.pi/4,      # 45° horizontal
    elevation=np.pi/6,    # 30° vertical
    roll=0.0
)

# Use preset views
camera = Camera.front_view(distance=3.0)
camera = Camera.side_view(distance=3.0)
camera = Camera.top_view(distance=3.0)
camera = Camera.isometric_view(distance=4.0)

# Camera generates transformation matrices
view_matrix = camera.get_view_matrix()
projection_matrix = camera.get_projection_matrix(aspect_ratio=16/9)
position, up = camera.get_camera_vectors()

# Set camera in renderer
renderer.set_camera(camera)
```

### Camera Animation

```python
# Create smooth camera paths
cameras = [
    Camera.front_view(distance=3.0),
    Camera.isometric_view(distance=3.0),
    Camera.side_view(distance=3.0)
]
path = CameraPath(keyframes=cameras)

# Interpolate between keyframes
interpolated_camera = path.interpolate(t=0.5)  # Halfway

# Generate animation frames
frames = path.generate_frames(n_frames=30)

# Interactive camera controller
controller = CameraController(initial_params=camera)
controller.orbit(delta_azimuth=0.1, delta_elevation=0.05)
controller.zoom(factor=1.1)
controller.pan(delta=np.array([0.1, 0, 0]))
controller.roll_camera(delta_roll=0.1)
```

## 💡 Lighting System

### Light Creation

```python
from pyvr.lighting import Light

# Directional light (like sunlight)
light = Light.directional(
    direction=[1, -1, 0],  # Normalized automatically
    ambient=0.2,           # Ambient intensity (0.0-1.0)
    diffuse=0.8            # Diffuse intensity (0.0-1.0)
)

# Point light at specific position
light = Light.point_light(
    position=[5, 5, 5],
    target=[0, 0, 0],
    ambient=0.1,
    diffuse=0.7
)

# Ambient-only lighting (no directional component)
light = Light.ambient_only(intensity=0.5)

# Default light
light = Light.default()
```

### Using Lights with Renderer

```python
# Pass light to renderer constructor
light = Light.directional(direction=[1, -1, 0], ambient=0.3)
renderer = VolumeRenderer(width=512, height=512, light=light)

# Or set/update later
light = Light.point_light(position=[5, 5, 5])
renderer.set_light(light)

# Get current light
current_light = renderer.get_light()

# Access light properties
direction = light.get_direction()
print(f"Position: {light.position}")
print(f"Ambient: {light.ambient_intensity}")
print(f"Diffuse: {light.diffuse_intensity}")

# Copy and modify
new_light = light.copy()
new_light.ambient_intensity = 0.5
```

## 📸 Examples

The `example/ModernglRender/` directory contains complete working examples:

- **`rgba_demo.py`**: RGBA transfer function demonstration
- **`enhanced_camera_demo.py`**: Advanced camera system with presets and paths
- **`multiview_example.py`**: Multi-view rendering example

Run examples:
```bash
python example/ModernglRender/rgba_demo.py
python example/ModernglRender/enhanced_camera_demo.py
python example/ModernglRender/multiview_example.py
```

## ⚡ Performance

- **Rendering Performance**: 64+ FPS at 512×512 resolution (15.6ms avg render time)
- **Pixel Throughput**: 16.8+ MPix/s on modern GPUs
- **Memory Efficiency**: Single RGBA texture lookup vs dual RGB+Alpha operations
- **Scalability**: Real-time for volumes up to 512³ voxels on modern hardware

### Performance Tuning

**High Quality** (slower):
```python
renderer = VolumeRenderer(1024, 1024, step_size=0.001, max_steps=1000)
```

**Balanced**:
```python
renderer = VolumeRenderer(512, 512, step_size=0.01, max_steps=500)
```

**High Performance** (faster):
```python
renderer = VolumeRenderer(256, 256, step_size=0.02, max_steps=100)
```

## 🧪 Testing

PyVR has comprehensive test coverage for reliability:

```bash
# Run all tests (398 tests)
pytest tests/

# Run with coverage report
pytest --cov=pyvr --cov-report=term-missing tests/

# Run specific test modules
pytest tests/test_camera/              # Camera system tests (42 tests)
pytest tests/test_config.py            # RenderConfig tests (33 tests)
pytest tests/test_lighting/            # Lighting tests (22 tests)
pytest tests/test_moderngl_renderer/   # ModernGL renderer tests (71 tests)
pytest tests/test_transferfunctions/   # Transfer function tests (36 tests)
```

**Test Coverage Breakdown:**
```
Module                        Tests    Coverage
----------------------------------------------
📷 Camera System              42       95-97%
⚙️  RenderConfig              63       100%
💡 Lighting System            22       100%
🎨 Transfer Functions         36       88-100%
🖥️  ModernGL Renderer         101      93-98%
🎮 Interactive Interface      80       >90%
📊 Volume & Datasets          54       56-93%
----------------------------------------------
📈 Total                     398       ~86%
```

**Key Testing Features:**
- Zero abstract base tests (removed in v0.2.7)
- Comprehensive RenderConfig preset testing
- Full integration test coverage
- Type checking and validation tests
- Edge case and error handling tests

## 🛠️ API Reference

### Volume Rendering Pipeline

```python
from pyvr.moderngl_renderer import VolumeRenderer
from pyvr.config import RenderConfig
from pyvr.volume import Volume
from pyvr.camera import Camera
from pyvr.lighting import Light
from pyvr.transferfunctions import ColorTransferFunction, OpacityTransferFunction
from pyvr.datasets import create_sample_volume, compute_normal_volume
import numpy as np

# 1. Create RenderConfig (v0.2.6)
config = RenderConfig.high_quality()  # Or: fast(), balanced(), preview()

# 2. Create Volume (v0.2.5)
volume_data = create_sample_volume(256, 'double_sphere')
normals = compute_normal_volume(volume_data)
volume = Volume(
    data=volume_data,
    normals=normals,
    min_bounds=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
    max_bounds=np.array([1.0, 1.0, 1.0], dtype=np.float32)
)

# 3. Create Camera (v0.2.3)
camera = Camera.from_spherical(
    target=np.array([0, 0, 0]),
    distance=5.0,
    azimuth=np.pi/4,
    elevation=np.pi/6,
    roll=0.0
)

# 4. Create Light (v0.2.4)
light = Light.directional(direction=[1, -1, 0], ambient=0.2, diffuse=0.8)

# 5. Create Renderer (v0.2.7 - no abstract base)
renderer = VolumeRenderer(width=512, height=512, config=config, light=light)
renderer.set_camera(camera)
renderer.load_volume(volume)

# 6. Configure Transfer Functions
ctf = ColorTransferFunction.from_colormap('viridis')
otf = OpacityTransferFunction.linear(0.0, 0.1)
renderer.set_transfer_functions(ctf, otf)

# 7. Render
data = renderer.render()

# RenderConfig operations (v0.2.6)
renderer.set_config(RenderConfig.fast())  # Change quality at runtime
current_config = renderer.get_config()
samples = config.estimate_samples_per_ray()  # ~346 samples

# Volume operations (v0.2.5)
print(f"Volume shape: {volume.shape}")
print(f"Volume dimensions: {volume.dimensions}")
normalized_vol = volume.normalize(method="minmax")
current_volume = renderer.get_volume()

# Camera operations (v0.2.3)
view_matrix = camera.get_view_matrix()
projection_matrix = camera.get_projection_matrix(aspect_ratio=16/9)
current_camera = renderer.get_camera()

# Camera animation (v0.2.3)
from pyvr.camera import CameraController, CameraPath
controller = CameraController(camera)
controller.orbit(delta_azimuth=0.1, delta_elevation=0.05)
path = CameraPath(keyframes=[camera1, camera2])
frames = path.generate_frames(30)
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

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and add tests
4. Ensure all tests pass (`poetry run pytest`)
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

## 📚 Version History

### v0.3.1 (2025-MM-DD) - Interface Refinements ✨

**New Features:**
- 📊 **FPS Counter**: Real-time performance monitoring with rolling 30-frame average
- ⚙️ **Quality Preset Selector**: 5 rendering quality levels (preview/fast/balanced/high_quality/ultra_quality)
- 💡 **Camera-Linked Lighting**: Directional lights follow camera with configurable offsets
- 📈 **Histogram Background**: Log-scale data distribution visualization in opacity editor
- 🎯 **Automatic Quality Switching**: Auto-switches to "fast" during camera interaction
- 📊 **Status Display**: Shows current preset, FPS, histogram, light linking states

**Performance:**
- All new features add <1% overhead
- Histogram caching provides >5x speedup (100ms → <10ms)
- Auto-quality makes interaction feel smoother
- Persistent cache in `tmp_dev/histogram_cache/`

**Convenience Methods:**
- `interface.set_high_quality_mode()` - Quick HQ switch
- `interface.set_camera_linked_lighting(offsets)` - Easy light setup
- `interface.capture_high_quality_image(filename)` - Ultra quality screenshots

**Keyboard Shortcuts (new):**
- `f`: Toggle FPS counter
- `h`: Toggle histogram
- `l`: Toggle light linking
- `q`: Toggle auto-quality

**Tests:** +77 new tests (284 → 361)
**Breaking Changes:** None - fully backward compatible with v0.3.0
**See:** `version_notes/v0.3.1_interface_refinements.md` for complete release notes

### v0.3.0 (2025-10-31) - Interactive Interface

- Interactive matplotlib-based interface for real-time visualization
- Mouse-based camera controls and opacity transfer function editing
- Real-time rendering with throttling and caching
- Colormap selection from matplotlib
- Keyboard shortcuts for common operations

### v0.2.7 (2025-10-28) - Architecture Simplification

- Removed abstract base renderer class for simpler design
- ModernGLVolumeRenderer now standalone (no inheritance)
- Backward compatibility maintained via alias

### v0.2.6 (2025-10-28) - RenderConfig System

- Quality presets (preview, fast, balanced, high_quality, ultra_quality)
- Performance estimation methods
- Runtime quality switching

### v0.2.5 (2025-10-28) - Volume Refactoring

- Unified Volume class for backend-agnostic data management
- Volume properties and operations (compute_normals, normalize, copy)
- Simpler renderer API

### v0.2.4 (2025-10-27) - Light System

- Light class with presets (directional, point_light, ambient_only)
- Easy light configuration and switching

### v0.2.3 (2025-10-27) - Camera System

- Camera class with matrix generation
- Spherical coordinates and camera presets
- Camera controller and animation paths

### v0.2.2 - RGBA Texture Optimization

- Combined RGBA transfer function textures
- Single texture lookup (previously dual)
- 64+ FPS performance improvement

### v0.2.0 - Major Refactoring

- Separated transfer functions into dedicated module
- Advanced camera system
- Modular architecture improvements

### v0.1.0 - Initial Release

- Basic ModernGL volume rendering
- Core ray marching implementation

## 📄 License

This project is licensed under the WTFPL (Do What The F*ck You Want To Public License) - see the [LICENSE](LICENSE) file for details.

## 🙋 Support

- **Issues**: [GitHub Issues](https://github.com/JixianLi/pyvr/issues)
- **Email**: jixianli@sci.utah.edu
- **Version Notes**: See `version_notes/` directory for detailed release information

## 🏆 Acknowledgments

- Claude Sonnet 4.5 model from Claude Code is responsible for code after v0.2.3
- Claude Sonnet 4 model from GitHub Copilot for the creation of almost all code/documentation/test (before v0.2.3) in this repository (some code was created by Claude Sonnet 3.5)
- ModernGL community for excellent OpenGL bindings
- The scientific visualization community

---

**PyVR** - High-performance OpenGL volume rendering for real-time interactive visualization! 🚀
