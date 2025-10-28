# PyVR Refactoring Plan (v0.2.3 → v0.2.6)

> ⚠️ **BREAKING CHANGES EXPECTED**: See [REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md) for migration guide.

## Overview

This directory contains detailed plans for a 4-phase refactoring to align PyVR with traditional rendering pipeline architecture. The refactoring transforms the codebase from a monolithic renderer into a modular, well-architected system with clear separation of concerns.

**Pre-1.0 Development**: PyVR prioritizes feature correctness and clean architecture over API stability. Breaking changes are acceptable and expected in v0.x releases.

## Goals

Transform PyVR architecture from:
```
VolumeRenderer (Monolithic)
├── Camera logic ❌
├── Lighting logic ❌
├── Volume data handling ❌
├── Ray marching config ❌
└── OpenGL rendering
```

To:
```
Application Stage Components:
├── Camera ✅ (Phase 1)
├── Light ✅ (Phase 2)
├── Volume ✅ (Phase 3)
├── RenderConfig ✅ (Phase 4)
└── TransferFunctions ✅ (already done in v0.2.0)

Renderer (orchestrates pipeline):
└── VolumeRenderer (clean, focused)
    └── ModernGLManager (OpenGL abstraction)
```

## Phase Overview

| Phase | Version | Component | Duration | Status |
|-------|---------|-----------|----------|--------|
| [Phase 1](phase1_camera_v0.2.3.md) | v0.2.3 | **Camera** | 2-3 days | 🟡 Current |
| [Phase 2](phase2_light_v0.2.4.md) | v0.2.4 | **Light** | 2 days | ⚪ Planned |
| [Phase 3](phase3_volume_v0.2.5.md) | v0.2.5 | **Volume** | 3-4 days | ⚪ Planned |
| [Phase 4](phase4_renderconfig_v0.2.6.md) | v0.2.6 | **RenderConfig** | 2 days | ⚪ Planned |
| **Total** | **v0.2.3→v0.2.6** | **Full Pipeline** | **~10 days** | **In Progress** |

## Phase Summaries

### Phase 1: Camera Refactoring (v0.2.3) 📷

**Goal**: Move camera matrix logic from `VolumeRenderer` to `Camera` class

**Key Changes**:
- Rename `CameraParameters` → `Camera` (with backward compatibility)
- Add `get_view_matrix()` and `get_projection_matrix()` methods to Camera
- Add `camera` attribute to `VolumeRenderer`
- Camera returns matrices, doesn't apply them (maintains encapsulation)

**Benefits**:
- Camera owns its transformations (Geometry Stage isolation)
- Improved testability (matrix creation testable without OpenGL)
- Foundation for multiple camera types (orthographic, etc.)

**Files**: See [phase1_camera_v0.2.3.md](phase1_camera_v0.2.3.md)

---

### Phase 2: Light Refactoring (v0.2.4) 💡

**Goal**: Extract lighting parameters into dedicated `Light` class

**Key Changes**:
- Create `Light` class with factory methods (directional, point_light, ambient_only)
- Add `light` attribute to `VolumeRenderer`
- Remove 4 lighting setter methods from `VolumeRenderer`
- Deprecate old lighting API

**Benefits**:
- Lighting logic centralized and testable
- Easy to extend (specular, multiple lights, shadows)
- Follows Camera refactoring pattern
- Removes setter methods, cleaner API

**Files**: See [phase2_light_v0.2.4.md](phase2_light_v0.2.4.md)

---

### Phase 3: Volume Refactoring (v0.2.5) 📦

**Goal**: Encapsulate volume data and metadata in `Volume` class

**Key Changes**:
- Create `Volume` class with data, normals, bounds as single unit
- Add volume properties (shape, dimensions, center, voxel_spacing)
- Add methods: `compute_normals()`, `normalize()`, `set_bounds()`
- Simplify `VolumeRenderer.load_volume()` to single method

**Benefits**:
- Volume data + metadata travel together
- Rich API for volume manipulation
- Automatic validation
- Simplified renderer interface (1 method vs 3)

**Files**: See [phase3_volume_v0.2.5.md](phase3_volume_v0.2.5.md)

---

### Phase 4: RenderConfig Refactoring (v0.2.6) ⚙️

**Goal**: Isolate rendering quality parameters into `RenderConfig` class

**Key Changes**:
- Create `RenderConfig` with quality presets (fast, balanced, high_quality, ultra_quality, preview)
- Add `config` attribute to `VolumeRenderer`
- Simplify constructor (3 params instead of 8)
- Remove ray marching setter methods

**Benefits**:
- User-friendly quality presets
- Clear performance tradeoffs
- Extensible (adaptive step size, progressive rendering)
- Completes pipeline alignment (100%)

**Files**: See [phase4_renderconfig_v0.2.6.md](phase4_renderconfig_v0.2.6.md)

---

## Architecture Evolution

### Current (v0.2.2)
```python
# Monolithic constructor
renderer = VolumeRenderer(
    width=512,
    height=512,
    step_size=0.01,
    max_steps=500,
    ambient_light=0.2,
    diffuse_light=0.8,
    light_position=(1, 1, 1),
    light_target=(0, 0, 0)
)

# Multiple API calls to configure
renderer.load_volume(volume_data)
renderer.load_normal_volume(normals)
renderer.set_volume_bounds((-1, -1, -1), (1, 1, 1))
renderer.set_camera([3, 3, 3], [0, 0, 0], [0, 1, 0])
renderer.set_ambient_light(0.3)
renderer.set_transfer_functions(ctf, otf)
```

### After Phase 4 (v0.2.6)
```python
# Clean, modular construction
from pyvr.camera import Camera
from pyvr.lighting import Light
from pyvr.volume import Volume
from pyvr.config import RenderConfig

# Configure components
camera = Camera.isometric_view(distance=5.0)
light = Light.directional(direction=[1, -1, 0], ambient=0.3)
volume = Volume(data=volume_data, normals=normals)
volume.set_bounds((-1, -1, -1), (1, 1, 1))
config = RenderConfig.high_quality()

# Simple renderer setup
renderer = VolumeRenderer(
    width=512,
    height=512,
    camera=camera,
    light=light,
    config=config
)

# Load and render
renderer.load_volume(volume)
renderer.set_transfer_functions(ctf, otf)
image = renderer.render()
```

**Improvements**:
- ✅ Clear component separation
- ✅ Each class has single responsibility
- ✅ Easy to test each component independently
- ✅ Intuitive for users familiar with 3D graphics
- ✅ Simple to extend (new camera types, lights, etc.)

## Pipeline Alignment Score

| Stage | v0.2.2 (Current) | After Phase 4 |
|-------|------------------|---------------|
| **Application Stage** | | |
| Volume | ⚠️ Mixed | ✅ Separate |
| Camera | ⚠️ Mixed | ✅ Separate |
| Lighting | ⚠️ Mixed | ✅ Separate |
| Materials | ✅ Separate | ✅ Separate |
| **Geometry Stage** | | |
| View/Projection | ⚠️ In renderer | ✅ In Camera |
| **Rasterization Stage** | | |
| Ray config | ⚠️ Constructor | ✅ RenderConfig |
| **Fragment Stage** | ✅ Shaders | ✅ Shaders |
| **Backend** | ✅ ModernGLManager | ✅ ModernGLManager |
| | | |
| **Score** | **3/9 (33%)** | **9/9 (100%)** ✅ |

## Testing Strategy

Each phase includes:
- **Unit tests** for new classes (+10-15 tests per phase)
- **Integration tests** with VolumeRenderer
- **Backward compatibility tests** for deprecated APIs
- **Example updates** demonstrating new API

**Total test growth**: 124 tests (v0.2.2) → ~176 tests (v0.2.6)

## Development Approach (Pre-1.0)

**Breaking changes are acceptable and expected in v0.x releases.**

Key principles:
- ✅ **Clean implementations** - No backward compatibility burden
- ✅ **Test coverage required** - Every feature must have comprehensive tests
- ✅ **Interface iteration** - Find the right API through usage
- ⚠️ **Breaking changes expected** - API will evolve based on practical usage

**Why this approach?**
- PyVR is pre-1.0, still finding the optimal architecture
- Backward compatibility code adds complexity and technical debt
- Clean refactoring produces better long-term code quality
- Comprehensive tests ensure correctness despite API changes

**Stability milestone**: v1.0.0 will mark API stability and semantic versioning

## Dependencies Between Phases

```
Phase 1 (Camera) ──┐
                   ├──> Can run in parallel
Phase 2 (Light) ───┘

Phase 3 (Volume) ─────> Requires Phase 1, 2 complete

Phase 4 (RenderConfig) ──> Requires Phase 1, 2, 3 complete
```

**Recommended order**: Sequential (1→2→3→4) for simplicity

## Timeline

| Phase | Duration | Cumulative |
|-------|----------|------------|
| Phase 1: Camera | 2-3 days | 3 days |
| Phase 2: Light | 2 days | 5 days |
| Phase 3: Volume | 3-4 days | 9 days |
| Phase 4: RenderConfig | 2 days | **11 days** |

**Total**: ~2 weeks of focused development

## Success Criteria

### Technical Metrics
- ✅ All 176+ tests pass
- ✅ No performance regression (benchmark.py)
- ✅ 100% pipeline alignment score
- ✅ Test coverage maintained at 85%+
- ✅ All examples run successfully

### Code Quality Metrics
- ✅ VolumeRenderer complexity reduced by 50%
- ✅ Each component independently testable
- ✅ Clear separation of concerns
- ✅ Consistent API patterns across components

### User Experience
- ✅ Intuitive quality presets (RenderConfig)
- ✅ Clear component organization
- ✅ Easy to extend and customize
- ✅ Comprehensive documentation and examples

## Post-Refactoring (v0.3.0+)

After completing all phases, PyVR will be ready for:

### v0.3.0: Advanced Features
- Scene abstraction (aggregate Camera, Light, Volume)
- Multiple light sources
- Light types (spotlight, area lights)
- Advanced camera modes (orthographic, fisheye)

### v0.3.x: Performance
- Adaptive step size based on gradient
- Progressive rendering (multi-pass refinement)
- Octree-based empty space skipping
- GPU-accelerated transfer function updates

### v0.4.x: Advanced Rendering
- Ambient occlusion
- Isosurface rendering
- Volume clipping planes
- Annotation overlays

## References

### Key Documents
- **[REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md)** - Breaking changes overview and migration guide
- **[../plan_eval.md](../plan_eval.md)** - Detailed architectural analysis and rationale
- **[../CLAUDE.md](../CLAUDE.md)** - Current architecture and development philosophy
- **[original_camera_plan.md](original_camera_plan.md)** - Original camera refactoring plan (archived)

### Phase Plans
- **[phase1_camera_v0.2.3.md](phase1_camera_v0.2.3.md)** - Camera refactoring (breaking)
- **[phase2_light_v0.2.4.md](phase2_light_v0.2.4.md)** - Light refactoring (breaking)
- **[phase3_volume_v0.2.5.md](phase3_volume_v0.2.5.md)** - Volume refactoring (breaking)
- **[phase4_renderconfig_v0.2.6.md](phase4_renderconfig_v0.2.6.md)** - RenderConfig refactoring (breaking)

## Important Notes

⚠️ **All phases introduce breaking changes**
- No backward compatibility before v1.0.0
- Comprehensive tests required for each phase
- Examples must be updated
- Documentation must reflect changes

✅ **Focus on correctness**
- Clean implementations without legacy code
- Comprehensive test coverage
- Interface evolution through practical usage

## Contact

For questions about the refactoring:
- Create an issue on GitHub
- Review phase-specific markdown files for implementation details
- Check REFACTORING_SUMMARY.md for migration examples

---

**Status**: Phase 1 (v0.2.3) in progress
**Philosophy**: Breaking changes acceptable pre-1.0
**Last Updated**: 2025-10-27
