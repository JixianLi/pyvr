# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PyVR is a GPU-accelerated 3D volume rendering toolkit built with ModernGL for real-time interactive visualization using OpenGL. The project emphasizes clean, modular architecture with comprehensive testing.

**Status**: Active development (v0.x - pre-1.0)
**Python**: 3.11+
**License**: WTFPL

### Development Philosophy (Pre-1.0)

**PyVR is in active development (v0.x) - API stability is NOT a priority before v1.0.0.**

Key principles:
- ✅ **Feature correctness over API stability** - Breaking changes are acceptable
- ✅ **Comprehensive testing required** - Every feature must have tests
- ✅ **Interface evolution** - The right API emerges through iteration
- ✅ **No backward compatibility burden** - Clean implementations without legacy code
- ⚠️ **Expect breaking changes** - Pin your version if using in production

**Stability timeline**:
- v0.x: Active development, breaking changes expected, comprehensive tests required
- v1.0.0: Stable API, semantic versioning, backward compatibility guarantees

## Development Commands

### Package Management
```bash
# Install dependencies (Poetry recommended)
poetry install

# Install with dev dependencies
poetry install --with dev
```

### Testing
```bash
# Run all tests
pytest tests/

# Run with coverage report
pytest --cov=pyvr --cov-report=term-missing tests/

# Run specific test modules
pytest tests/test_moderngl_renderer/  # OpenGL rendering tests
pytest tests/test_camera/             # Camera system tests
pytest tests/test_config.py           # RenderConfig tests
pytest tests/test_lighting/           # Lighting tests
pytest tests/test_transferfunctions/  # Transfer function tests

# Run single test file
pytest tests/test_camera/test_parameters.py

# Run specific test function
pytest tests/test_camera/test_parameters.py::test_function_name
```

### Code Formatting
```bash
# Format code with black
poetry run black pyvr/

# Sort imports with isort
poetry run isort pyvr/

# Format both
poetry run black pyvr/ && poetry run isort pyvr/
```

### Running Examples
```bash
# Run basic rendering example
python example/basic_rendering.py

# Run camera demo
python example/camera_demo.py

# Run transfer functions demo
python example/transfer_functions_demo.py

# Run benchmark (compares quality presets)
python example/benchmark.py
```

## Architecture & Design

### Core Module Structure

The codebase follows a pipeline-aligned architecture with clean separation of concerns:

1. **`pyvr/volume/`** - Backend-agnostic volume data management
   - `data.py`: `Volume` class with data, normals, bounds, and operations
   - Properties: shape, dimensions, center, voxel_spacing, has_normals
   - Operations: compute_normals(), normalize(), copy()
   - No renderer dependencies - pure data management

2. **`pyvr/camera/`** - Advanced camera system with matrix generation
   - `camera.py`: `Camera` class with spherical coordinates and matrix generation
   - `control.py`: `CameraController`, `CameraPath` for animation
   - Uses quaternion rotations (scipy) for smooth movement without gimbal lock
   - Methods: get_view_matrix(), get_projection_matrix(), get_camera_vectors()
   - Presets: front_view(), side_view(), top_view(), isometric_view()

3. **`pyvr/lighting/`** - Light configuration and presets
   - `light.py`: `Light` class managing position, target, ambient, and diffuse intensities
   - Presets: default(), directional(), point_light(), ambient_only(), camera_linked()
   - Methods: get_direction(), copy(), link_to_camera(), update_from_camera()

4. **`pyvr/config.py`** - Rendering configuration
   - `RenderConfig` class with quality presets
   - Presets: preview(), fast(), balanced(), high_quality(), ultra_quality()
   - Parameters: step_size, max_steps, early_ray_termination, opacity_threshold, reference_step_size
   - Methods: with_step_size(), with_max_steps(), estimate_samples_per_ray()

5. **`pyvr/transferfunctions/`** - Color and opacity mapping
   - `base.py`: Abstract base classes (`BaseTransferFunction`)
   - `color.py`: RGB color mapping with matplotlib colormap integration
   - `opacity.py`: Alpha/opacity mapping with peak detection
   - RGBA texture optimization: Combined RGBA texture (single texture lookup)

6. **`pyvr/moderngl_renderer/`** - OpenGL rendering backend
   - `renderer.py`: `ModernGLVolumeRenderer` - standalone GPU-accelerated renderer
   - `manager.py`: `ModernGLManager` - low-level OpenGL resource management
   - No abstract base class for simpler architecture
   - Backward compatibility alias: `VolumeRenderer = ModernGLVolumeRenderer`

7. **`pyvr/datasets/`** - Synthetic volume generators
   - `synthetic.py`: Testing datasets (sphere, torus, double_sphere, cube, helix, random_blob)
   - `compute_normal_volume()`: Generate normal vectors for lighting

8. **`pyvr/interface/`** - Interactive matplotlib-based interface
   - `matplotlib_interface.py`: `InteractiveVolumeRenderer` - main interface class
   - Layout: 3-column design (18"×8" figure, image | opacity+color | info+preset)
   - `widgets.py`: UI components (`ImageDisplay`, `OpacityEditor`, `ColorSelector`, `PresetSelector`)
   - `state.py`: `InterfaceState` - centralized state management
   - `cache.py`: Histogram caching utilities
   - Features:
     - FPS counter with rolling average
     - Quality preset selector (5 quality levels)
     - Camera-linked lighting (headlight mode)
     - Log-scale histogram background in opacity editor
     - Automatic quality switching during interaction
     - Status display with current settings
   - Mouse controls: orbit (drag), zoom (scroll), control point editing
   - Keyboard shortcuts: r, s, f, h, l, q, Esc, Del
   - Performance: Render throttling, caching, smart updates

### Key Architectural Patterns

**Shader Management**: Shaders live in `pyvr/shaders/` (shared across renderers):
- `volume.vert.glsl`: Vertex shader
- `volume.frag.glsl`: Fragment shader with ray marching, RGBA texture lookups, and Beer-Lambert opacity correction

**RGBA Transfer Function Optimization**:
- Combined RGBA texture (1 lookup) instead of separate RGB + Alpha textures (2 lookups)
- Better cache locality and performance
- API: `renderer.set_transfer_functions(ctf, otf)` handles texture creation automatically
- Implementation: `ModernGLManager.create_rgba_transfer_function_texture()`

**Volume System Design**:
- `Volume`: Encapsulates data, normals, bounds management
- Backend-agnostic: No renderer dependencies, works with any backend
- Unified interface: `renderer.load_volume(volume)` instead of separate data/normals/bounds calls
- Operations: compute_normals(), normalize(), copy() provide data manipulation

**Camera System Design**:
- `Camera`: Immutable-style dataclass with spherical coordinates (target, azimuth, elevation, roll, distance)
- Matrix generation: get_view_matrix(), get_projection_matrix(aspect) generate transformation matrices
- Core function: get_camera_vectors() converts spherical coords to position/up vectors using quaternion math
- `CameraController`: Stateful controller with methods like orbit(), zoom(), pan(), roll_camera()
- `CameraPath`: Keyframe-based animation with interpolation

**Lighting System Design**:
- `Light`: Manages position, target, ambient_intensity, diffuse_intensity
- Directional lights: Positioned far from target with normalized direction
- Point lights: At specific position, direction computed from position-target vector
- Camera-linked lights: Headlight mode (light at camera position, pointing at target)
- Integration: `renderer.set_light(light)` updates shader uniforms

**RenderConfig System**:
- Quality presets: preview (fast), fast (interactive), balanced (default), high_quality, ultra_quality
- Performance estimation: estimate_samples_per_ray(), estimate_render_time_relative()
- Runtime changes: `renderer.set_config(RenderConfig.fast())` for dynamic quality switching
- Parameters: step_size, max_steps, early_ray_termination, opacity_threshold, reference_step_size

**Simplified Architecture**:
- ModernGLVolumeRenderer is standalone (no abstract base class inheritance)
- One less abstraction layer for simpler, more maintainable code
- Backward compatibility maintained via alias

### Testing Infrastructure

**Mock-based OpenGL testing**: Tests use mocking to run without OpenGL/GPU:
- `tests/test_moderngl_renderer/conftest.py`: Central mock fixtures
- Allows CI/CD testing without display server
- Mock `moderngl.create_context()`, texture creation, shader compilation

**Test organization**:
- Each module has corresponding test directory
- Tests validate edge cases, error conditions, boundary values
- Integration tests verify module interactions
- Comprehensive test coverage (>85%)

## API Usage

### Complete Rendering Pipeline

```python
import numpy as np
from pyvr.moderngl_renderer import VolumeRenderer
from pyvr.config import RenderConfig
from pyvr.volume import Volume
from pyvr.camera import Camera
from pyvr.lighting import Light
from pyvr.transferfunctions import ColorTransferFunction, OpacityTransferFunction
from pyvr.datasets import create_sample_volume, compute_normal_volume

# 1. Create Volume
volume_data = create_sample_volume(256, 'double_sphere')
normals = compute_normal_volume(volume_data)
volume = Volume(
    data=volume_data,
    normals=normals,
    min_bounds=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
    max_bounds=np.array([1.0, 1.0, 1.0], dtype=np.float32)
)

# 2. Create Camera
camera = Camera.from_spherical(
    target=np.array([0.0, 0.0, 0.0]),
    distance=3.0,
    azimuth=np.pi/4,      # 45 degrees
    elevation=np.pi/6,    # 30 degrees
    roll=0.0
)

# 3. Create Light
light = Light.directional(direction=[1, -1, 0], ambient=0.2, diffuse=0.8)

# 4. Create RenderConfig
config = RenderConfig.high_quality()  # Or: fast(), balanced(), preview(), ultra_quality()

# 5. Create Renderer
renderer = VolumeRenderer(width=512, height=512, config=config, light=light)
renderer.set_camera(camera)
renderer.load_volume(volume)

# 6. Configure Transfer Functions
ctf = ColorTransferFunction.from_colormap('viridis')
otf = OpacityTransferFunction.linear(0.0, 0.1)
renderer.set_transfer_functions(ctf, otf)

# 7. Render
data = renderer.render()
image = renderer.render_to_pil()

# Runtime changes
renderer.set_config(RenderConfig.fast())  # Switch to fast rendering
renderer.set_light(Light.point_light([5, 5, 5]))  # Change lighting
current_volume = renderer.get_volume()  # Access current volume
current_camera = renderer.get_camera()  # Access current camera
```

### Interactive Interface

```python
from pyvr.interface import InteractiveVolumeRenderer
from pyvr.volume import Volume
from pyvr.datasets import create_sample_volume
from pyvr.config import RenderConfig
import numpy as np

# Create volume
volume_data = create_sample_volume(128, 'sphere')
volume = Volume(data=volume_data)

# Create interface (histogram loaded automatically)
interface = InteractiveVolumeRenderer(
    volume=volume,
    width=512,
    height=512,
    config=RenderConfig.balanced()  # Initial preset
)

# Configure features
interface.state.show_fps = True  # FPS counter (default: True)
interface.state.show_histogram = True  # Histogram (default: True)
interface.state.auto_quality_enabled = True  # Auto-quality (default: True)

# Set camera-linked lighting (headlight mode)
interface.set_camera_linked_lighting()

# Launch interface
interface.show()

# Programmatic control
interface.set_high_quality_mode()  # Switch to high quality
path = interface.capture_high_quality_image()  # Ultra quality screenshot

# Keyboard shortcuts in interface:
# 'f': Toggle FPS counter
# 'h': Toggle histogram
# 'l': Toggle light linking
# 'q': Toggle auto-quality
# 'r': Reset view
# 's': Save image
# 'Esc': Deselect
# 'Del': Delete selected
```

### Volume Operations

```python
from pyvr.volume import Volume

# Create volume
volume = Volume(data=volume_data)  # Uses default bounds

# Properties
print(volume.shape)          # (256, 256, 256)
print(volume.dimensions)     # [1.0, 1.0, 1.0]
print(volume.center)         # [0.0, 0.0, 0.0]
print(volume.has_normals)    # False
print(volume.voxel_spacing)  # [0.0039, 0.0039, 0.0039]

# Operations
volume.compute_normals()             # Generate normals from gradient
normalized = volume.normalize("minmax")  # Scale to [0, 1]
volume_copy = volume.copy()          # Independent copy
```

### Camera Presets and Animation

```python
from pyvr.camera import Camera, CameraController, CameraPath

# Use presets
camera = Camera.front_view(distance=3.0)
camera = Camera.isometric_view(distance=4.0)

# Camera controller
controller = CameraController(camera)
controller.orbit(delta_azimuth=0.1, delta_elevation=0.05)
controller.zoom(factor=1.1)
controller.pan(delta=np.array([0.1, 0, 0]))

# Camera animation
cameras = [
    Camera.front_view(distance=3.0),
    Camera.side_view(distance=3.0),
    Camera.top_view(distance=3.0)
]
path = CameraPath(keyframes=cameras)
frames = path.generate_frames(n_frames=30)
```

### RenderConfig Quality Presets

```python
from pyvr.config import RenderConfig

# Use preset
config = RenderConfig.preview()       # Very fast, low quality
config = RenderConfig.fast()          # Fast, interactive
config = RenderConfig.balanced()      # Default, good balance
config = RenderConfig.high_quality()  # High quality, slower
config = RenderConfig.ultra_quality() # Maximum quality, very slow

# Custom config
config = RenderConfig(
    step_size=0.015,
    max_steps=300,
    early_ray_termination=True,
    opacity_threshold=0.95
)

# Modify preset
config = RenderConfig.balanced().with_step_size(0.008)

# Performance estimation
samples = config.estimate_samples_per_ray()      # ~346 samples
relative_time = config.estimate_render_time_relative()  # ~5.0x slower
```

## Important Implementation Details

**Volume Data Format**:
- 3D volumes: (D, H, W) shape
- Normal volumes: (D, H, W, 3) shape
- Always float32 for GPU compatibility

**Coordinate System**:
- Camera: Right-handed coordinate system
- Default: X-right, Y-up, Z-forward
- Spherical coords: azimuth (horizontal), elevation (vertical), roll

**OpenGL Resource Management**:
- `ModernGLManager` owns all OpenGL contexts, textures, shaders
- Textures use sequential texture units (0: volume, 1: normals, 2+: transfer functions)
- Manual cleanup not required (Python GC handles it)

**Transfer Function Texture Creation**:
- ColorTransferFunction and OpacityTransferFunction generate lookup tables (LUTs)
- LUTs converted to OpenGL textures by ModernGLManager
- Default LUT size: 256 samples (configurable)

**Ray Marching Parameters** (controlled by RenderConfig):
- `step_size`: Controls quality vs performance (smaller = higher quality)
- `max_steps`: Maximum ray samples (prevents infinite loops)
- Typical values: step_size=0.01, max_steps=500 for balanced quality

**Renderer Architecture**:
- No abstract base class - ModernGLVolumeRenderer is standalone
- Backward compatibility: `VolumeRenderer = ModernGLVolumeRenderer` alias
- Future backends can be added as separate implementations (e.g., PyTorchVolumeRenderer)

## Dependencies

**Core Runtime**:
- moderngl >= 5.0 (OpenGL bindings)
- numpy >= 2.3 (array operations)
- matplotlib >= 3.10 (colormaps)
- pillow >= 11.0 (image I/O)
- scipy >= 1.16 (quaternion rotations)

**Dev/Test**:
- pytest >= 7.0
- pytest-cov >= 7.0
- black >= 24.9 (code formatting)
- isort >= 6.0 (import sorting)

## Known Patterns & Conventions

**Error Handling**:
- Validation in `__post_init__()` for dataclasses
- Descriptive error messages with parameter values
- Type checking in setters (e.g., `if not isinstance(volume, Volume)`)

**Type Hints**:
- Extensive use throughout codebase
- `Optional[T]` for nullable parameters
- `Tuple[...]` for return types
- numpy arrays typed as `np.ndarray`

**Naming Conventions**:
- Classes: PascalCase (`VolumeRenderer`, `Camera`, `Volume`)
- Functions/methods: snake_case (`get_camera_pos`, `set_transfer_functions`)
- Constants: UPPER_SNAKE_CASE (rare in this codebase)
- Private methods: `_prefixed_with_underscore`

**Documentation Style**:
- Google-style docstrings
- Args/Returns/Raises sections
- Examples in docstrings where helpful

## Development Notes

**Testing Requirements**:
- All new features must have comprehensive tests
- Aim for >85% coverage
- Include edge cases, error conditions, and integration tests
- Use mocks for OpenGL testing

**Breaking Changes Policy**:
- Breaking changes are acceptable in v0.x
- Provide detailed migration notes in version_notes/
- Update README version history with link to version notes
- Consider backward compatibility aliases where reasonable

**Refactoring Approach**:
- Plan first: Create detailed plan in plan/ directory
- Implement: Make changes systematically
- Test: Ensure all tests pass
- Document: Update README, version notes, examples
- Clean up: Remove plan files after completion