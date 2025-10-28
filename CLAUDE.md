# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PyVR is a GPU-accelerated 3D volume rendering toolkit built with ModernGL for real-time interactive visualization using OpenGL. The project emphasizes clean, modular architecture with comprehensive testing (124 tests, 88% coverage).

**Current Version**: 0.2.2 (branch: 0.2.3-CameraRefactoring)
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
# Run all tests (124 tests)
pytest tests/

# Run with coverage report
pytest --cov=pyvr --cov-report=term-missing tests/

# Run specific test modules
pytest tests/test_moderngl_renderer/  # OpenGL rendering tests
pytest tests/test_camera/            # Camera system tests
pytest tests/test_transferfunctions/ # Transfer function tests

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
# Run multiview example with v0.2.2 API
python example/ModernglRender/multiview_example.py

# Run camera demo
python example/ModernglRender/enhanced_camera_demo.py

# Run RGBA transfer function demo
python example/ModernglRender/rgba_demo.py

# Run benchmark
python example/benchmark.py
```

## Architecture & Design

### Core Module Structure

The codebase follows a clean separation of concerns with four main modules:

1. **`pyvr/transferfunctions/`** - Color and opacity mapping
   - `base.py`: Abstract base classes (`BaseTransferFunction`)
   - `color.py`: RGB color mapping with matplotlib colormap integration
   - `opacity.py`: Alpha/opacity mapping with peak detection
   - Key feature: v0.2.2 introduced RGBA texture optimization (single texture lookup)

2. **`pyvr/camera/`** - Advanced camera system
   - `parameters.py`: `CameraParameters` class managing position, spherical coordinates, presets
   - `control.py`: `CameraController`, `CameraPath` for animation, core `get_camera_pos()` function
   - Uses quaternion rotations (scipy) for smooth movement without gimbal lock

3. **`pyvr/moderngl_renderer/`** - OpenGL rendering backend
   - `renderer.py`: `VolumeRenderer` class - high-level rendering API
   - `manager.py`: `ModernGLManager` - low-level OpenGL resource management (contexts, textures, shaders, framebuffers)
   - Clear separation: renderer focuses on volume rendering logic, manager handles OpenGL details

4. **`pyvr/datasets/`** - Synthetic volume generators
   - `synthetic.py`: Testing datasets (sphere, torus, double_sphere, cube, helix, random_blob)

### Key Architectural Patterns

**Shader Management**: Shaders live in `pyvr/shaders/` (shared across renderers):
- `volume.vert.glsl`: Vertex shader
- `volume.frag.glsl`: Fragment shader with ray marching and RGBA texture lookups

**RGBA Transfer Function Optimization (v0.2.2)**:
- Previous versions: Separate RGB texture + Alpha texture (2 lookups)
- Current: Combined RGBA texture (1 lookup, better cache locality)
- API: `renderer.set_transfer_functions(ctf, otf)` handles texture creation automatically
- Implementation: `ModernGLManager.create_rgba_transfer_function_texture()`

**Camera System Design**:
- `CameraParameters`: Immutable-style dataclass storing camera state (target, azimuth, elevation, roll, distance)
- `get_camera_pos()`: Core function converting spherical coords to position/up vectors using quaternion math
- `CameraController`: Stateful controller with methods like `orbit()`, `zoom()`, `pan()`
- `CameraPath`: Keyframe-based animation with interpolation

### Testing Infrastructure

**Mock-based OpenGL testing**: Tests use mocking to run without OpenGL/GPU:
- `tests/test_moderngl_renderer/conftest.py`: Central mock fixtures
- Allows CI/CD testing without display server
- Mock `moderngl.create_context()`, texture creation, shader compilation

**Test organization**:
- Each module has corresponding test directory
- Tests validate edge cases, error conditions, boundary values
- Integration tests verify module interactions

## Active Development: Camera Refactoring (v0.2.3)

See `plan.md` for detailed refactoring plan:

**Goal**: Rename `CameraParameters` → `Camera`, move camera matrix logic from `VolumeRenderer` into `Camera` class

**Key changes planned**:
1. Rename class, maintain `CameraParameters` as backward-compatible alias
2. Add `camera: Camera` attribute to `VolumeRenderer`
3. Move matrix creation from `VolumeRenderer.set_camera()` to new `Camera` methods:
   - `Camera.create_view_matrix()`
   - `Camera.create_projection_matrix(width, height)`
   - `Camera.apply_to_renderer(renderer)`

**Files involved**:
- `pyvr/camera/parameters.py` - rename class
- `pyvr/camera/control.py` - update type hints
- `pyvr/moderngl_renderer/renderer.py` - add camera attribute, simplify `set_camera()`
- All test files and examples need updates

## Version History & Migration Notes

**v0.2.2** (Current): RGBA texture optimization
- Breaking: Transfer function API simplified to `set_transfer_functions(ctf, otf)`
- Performance: 64+ FPS at 512×512 (single texture lookup vs dual)

**v0.2.0**: Major refactoring
- Separated transfer functions into dedicated module
- Introduced advanced camera system with spherical coordinates
- Breaking: Import changes (`from pyvr.transferfunctions import ...` instead of `from pyvr.moderngl_renderer import ...`)

**v0.1.0**: Initial release
- Basic volume rendering with ModernGL
- Monolithic design (transfer functions in renderer module)

## Important Implementation Details

**Volume Data Format**: Always (D, H, W) shape for 3D volumes, (D, H, W, 3) for normal volumes

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

**Ray Marching Parameters**:
- `step_size`: Controls quality vs performance (smaller = higher quality)
- `max_steps`: Maximum ray samples (prevents infinite loops)
- Typical values: step_size=0.01, max_steps=500 for balanced quality

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
- Custom exceptions: `CameraParameterError`, `TransferFunctionError`, `InvalidControlPointError`
- Validation in `__post_init__()` for dataclasses
- Descriptive error messages with parameter values

**Type Hints**:
- Extensive use throughout codebase
- `Optional[T]` for nullable parameters
- `Tuple[...]` for return types
- numpy arrays typed as `np.ndarray`

**Naming Conventions**:
- Classes: PascalCase (`VolumeRenderer`, `CameraParameters`)
- Functions/methods: snake_case (`get_camera_pos`, `set_transfer_functions`)
- Constants: UPPER_SNAKE_CASE (rare in this codebase)
- Private methods: `_prefixed_with_underscore`

**Documentation Style**:
- Google-style docstrings
- Args/Returns/Raises sections
- Examples in docstrings where helpful
