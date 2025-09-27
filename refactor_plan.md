# PyVR Project Refactor Plan

## Executive Summary

This document outlines a comprehensive refactoring plan to restructure the PyVR project based on the user's vision for improved modularity and maintainability. The refactor focuses on creating cleaner separation of concerns while maintaining backward compatibility during the transition.

## Current vs. Proposed Structure

### Current Structure
```
pyvr/
├── __init__.py
├── datasets/
│   ├── __init__.py
│   └── synthetic.py
└── moderngl_renderer/
    ├── __init__.py
    ├── camera_control.py
    ├── moderngl_manager.py
    ├── requirements.txt
    ├── transfer_functions.py
    ├── volume_renderer.py
    └── shaders/
        ├── volume.frag.glsl
        └── volume.vert.glsl
```

### Proposed Structure
```
pyvr/
├── __init__.py
├── datasets/
│   ├── __init__.py
│   └── synthetic.py
├── transferfunctions/
│   ├── __init__.py
│   ├── color.py
│   ├── opacity.py
│   └── base.py
├── camera/
│   ├── __init__.py
│   ├── control.py
│   └── parameters.py
├── shaders/
│   ├── volume.frag.glsl
│   └── volume.vert.glsl
└── moderngl_renderer/
    ├── __init__.py
    ├── manager.py
    ├── renderer.py
    └── requirements.txt
```

## Detailed Analysis

### 1. Transfer Functions Module (`pyvr/transferfunctions/`)

**Current State:**
- All transfer function logic is in `pyvr/moderngl_renderer/transfer_functions.py`
- Contains both `ColorTransferFunction` and `OpacityTransferFunction` classes
- Tightly coupled with ModernGL texture creation

**Proposed Changes:**
- **`pyvr/transferfunctions/base.py`**: Common base class and utilities
- **`pyvr/transferfunctions/color.py`**: `ColorTransferFunction` implementation
- **`pyvr/transferfunctions/opacity.py`**: `OpacityTransferFunction` implementation
- **`pyvr/transferfunctions/__init__.py`**: Clean imports and public API

**Benefits:**
- ✅ **Separation of Concerns**: Transfer functions become independent of rendering backend
- ✅ **Extensibility**: Easy to add new transfer function types (e.g., multi-dimensional)
- ✅ **Testing**: Individual components can be tested in isolation
- ✅ **Reusability**: Transfer functions could be used with other renderers in future

**Migration Strategy:**
- Create new module structure with identical API
- Add deprecation warnings to old imports
- Update all internal imports gradually
- Maintain backward compatibility for 2-3 minor versions

### 2. Camera Module (`pyvr/camera/`)

**Current State:**
- Camera functionality in `pyvr/moderngl_renderer/camera_control.py`
- Single `get_camera_pos()` function with basic spherical coordinate support

**Proposed Changes:**
- **`pyvr/camera/control.py`**: Camera positioning and orientation utilities
- **`pyvr/camera/parameters.py`**: Camera parameter validation and management
- **`pyvr/camera/__init__.py`**: Clean API exports

**Benefits:**
- ✅ **Enhanced Functionality**: Room for camera path interpolation, smooth transitions
- ✅ **Parameter Validation**: Centralized camera parameter checking
- ✅ **Animation Support**: Better structure for camera animations and keyframing
- ✅ **Independence**: Camera logic separated from renderer-specific code

**Potential Extensions:**
- Camera path interpolation for smooth animations
- Camera parameter presets (common viewpoints)
- Camera state serialization/deserialization
- Multi-camera support for stereo rendering

### 3. Shared Shaders Module (`pyvr/shaders/`)

**Current State:**
- Shaders are located in `pyvr/moderngl_renderer/shaders/`
- Tightly coupled with the moderngl_renderer module

**Proposed Changes:**
- **`pyvr/shaders/`**: Move all GLSL shaders to top-level shared location
- **Future-ready**: All OpenGL-based renderers can share shaders
- **Better organization**: Shaders as first-class citizens in project structure

**Benefits:**
- ✅ **Shared Resources**: Multiple OpenGL renderers can use same shaders
- ✅ **Centralized Management**: Easy to maintain and version shaders
- ✅ **Clear Architecture**: Shaders are renderer-agnostic resources
- ✅ **Extensibility**: Easy to add specialized shaders for different techniques

### 4. ModernGL Renderer Refactor (`pyvr/moderngl_renderer/`)

**Current State:**
- `volume_renderer.py`: High-level volume rendering API
- `moderngl_manager.py`: Low-level OpenGL resource management
- Good separation already exists

**Proposed Changes:**
- **`pyvr/moderngl_renderer/renderer.py`**: Rename from `volume_renderer.py` for clarity
- **`pyvr/moderngl_renderer/manager.py`**: Rename from `moderngl_manager.py` for consistency
- Update imports to use new transfer function and camera modules

**Benefits:**
- ✅ **Consistency**: Cleaner, more predictable naming
- ✅ **Modularity**: Uses external transfer function and camera modules
- ✅ **Maintainability**: Reduced coupling between components

## Implementation Roadmap

### Phase 1: Foundation Setup (Week 1)
1. **Create new module directories**
   ```bash
   mkdir -p pyvr/transferfunctions pyvr/camera pyvr/shaders
   mv pyvr/moderngl_renderer/shaders/* pyvr/shaders/
   ```

2. **Create base infrastructure**
   - `pyvr/transferfunctions/__init__.py`
   - `pyvr/camera/__init__.py`
   - Update shader loading paths in renderer
   - **Version bump to 0.2.0** in pyproject.toml

3. **Set up testing framework**
   - Create test directories for new modules
   - Establish CI/CD compatibility

### Phase 2: Transfer Functions Migration (Week 1-2)
1. **Extract and refactor transfer functions**
   - Move `ColorTransferFunction` to `pyvr/transferfunctions/color.py`
   - Move `OpacityTransferFunction` to `pyvr/transferfunctions/opacity.py`
   - Create base class in `pyvr/transferfunctions/base.py`

2. **Create clean APIs**
   - Design consistent interface across both classes
   - Remove ModernGL dependencies from core logic
   - Add comprehensive docstrings and type hints

3. **Update imports throughout codebase**
   - Update renderer to import from new location
   - Update example code to use new imports
   - Remove old transfer_functions.py file

### Phase 3: Camera System Enhancement (Week 2)
1. **Migrate camera functionality**
   - Move `get_camera_pos` to `pyvr/camera/control.py`
   - Enhanced error handling and input validation
   - Add camera parameter classes

2. **Add new camera features**
   - Camera animation utilities
   - Parameter presets and validation
   - Enhanced spherical coordinate handling

### Phase 4: ModernGL Renderer Updates (Week 2-3) ✅ **COMPLETED**
1. **Update renderer to use new modules** ✅
   - ✅ Modified imports to use `pyvr.transferfunctions`
   - ✅ Updated camera imports to use `pyvr.camera`
   - ✅ Confirmed shader loading uses `pyvr.shaders`

2. **Rename files for consistency** ✅
   - ✅ `volume_renderer.py` → `renderer.py`
   - ✅ `moderngl_manager.py` → `manager.py`
   - ✅ Updated all internal imports
   - ✅ Updated example imports
   - ✅ Removed old transfer_functions.py
   - ✅ Maintained backward compatibility

### Phase 5: Documentation and Testing (Week 3)
1. **Update documentation**
   - Revise README.md with new structure
   - Update API documentation
   - Document breaking changes for v0.2.0

2. **Comprehensive testing**
   - Test all new modules independently
   - Integration testing with examples
   - Performance regression testing

3. **Example updates**
   - Update `multiview_example.py` to use new imports
   - Create examples showcasing new functionality
   - Update all import statements

## API Compatibility Strategy

### Clean Break Approach (v0.2.0)
With the version bump from 0.1.0 to 0.2.0, we're implementing a clean break approach:

```python
# OLD (v0.1.0) - NO LONGER SUPPORTED
from pyvr.moderngl_renderer import ColorTransferFunction, get_camera_pos

# NEW (v0.2.0) - Clean, modular structure
from pyvr.transferfunctions import ColorTransferFunction
from pyvr.camera import get_camera_pos
```

### Version Strategy
- **v0.1.0**: Current structure (legacy)
- **v0.2.0**: New modular structure with breaking changes
- Clear documentation of breaking changes in release notes
- Focus on clean, testable, maintainable code

## Expected Benefits

### 1. **Improved Modularity**
- Each module has a single, well-defined responsibility
- Reduced coupling between components
- Easier to test and maintain individual components

### 2. **Enhanced Extensibility** 
- Easy to add new transfer function types
- Camera system can be extended with advanced features
- Clear interfaces for future renderer backends

### 3. **Better Developer Experience**
- More intuitive import structure
- Cleaner API documentation
- Easier to find relevant functionality

### 4. **Future-Proofing**
- Structure supports multiple OpenGL renderer types
- Transfer functions and camera control become renderer-agnostic
- Shared shaders enable consistent rendering across different OpenGL techniques
- Foundation for advanced features like multi-view rendering

## Risk Assessment

### Low Risk ✅
- **File reorganization**: Straightforward with clear benefits
- **Clean API design**: Simplified without backward compatibility concerns
- **Documentation updates**: Improves user experience
- **Version bump**: Clear signal of breaking changes

### Medium Risk ⚠️
- **Breaking changes**: Requires users to update their code
- **Testing coverage**: Need comprehensive tests for new structure
- **Shader path updates**: Need to update all shader loading logic

### Mitigation Strategies
- **Comprehensive testing**: Unit tests for each module, integration tests for examples
- **Clear communication**: Documentation, breaking changes guide, and version release notes
- **Gradual rollout**: Test thoroughly before release

## Success Metrics

1. **Code Quality**
   - Reduced coupling between modules (measured by import dependencies)
   - Improved test coverage (target: >90% for new modules)
   - Clear separation of concerns

2. **Developer Experience**
   - More intuitive API (validated through documentation review)
   - Easier to extend (measured by effort to add new features)
   - Better error messages and validation

3. **Clean Architecture**
   - All existing examples work with updated imports
   - Performance regression < 5%
   - Clear breaking changes documentation for v0.2.0

## Conclusion

This refactoring plan provides a clear path to improve the PyVR project structure with a clean, modern architecture. The proposed changes enhance modularity, extensibility, and maintainability while establishing a solid foundation for future development.

The v0.2.0 release represents a significant architectural improvement, focusing on clean separation of concerns and testable code. The shared shader approach future-proofs the project for multiple OpenGL rendering techniques.

**Recommendation**: Proceed with Phase 1 to establish the foundation, implementing the breaking changes as a clean slate for improved long-term maintainability.