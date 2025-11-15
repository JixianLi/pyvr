# PyVR Roadmap: v0.4.0 → v0.6.0

**Last Updated:** 2025-11-15
**Status:** Planning phase

## Overview

This roadmap outlines the planned evolution of PyVR from the current v0.3.x state through v0.6.0, focusing on:
1. Real scientific data support (VTK)
2. Advanced interactive GUI (DearPyGUI)
3. Progressive rendering infrastructure
4. Complex light transport simulation

Each major version introduces ONE primary feature, with minor versions for refinement and stabilization.

---

## v0.4.0 - VTK Data Loader

**Primary Goal:** Support loading real scientific volume data via VTK

**Status:** Planning - research phase

### Scope

**Core Feature:**
- VTK data loader supporting `vtkImageData` format
- Integration with existing `Volume` class
- Preserve backend-agnostic design

**Implementation:**
- New module: `pyvr/loaders/` or `pyvr/io/`
- Function/class to load VTK files → `Volume` objects
- Handle metadata: spacing, origin, orientation
- Support common VTK scalar types

**Out of Scope:**
- VTK structured/unstructured grids (focus on images only)
- VTK rendering integration (just data loading)
- File format conversion (users use VTK tools for that)

### Design Considerations

**Questions to resolve:**
- Module naming: `pyvr.loaders` vs `pyvr.io` vs `pyvr.data`?
- VTK as core dependency or optional (poetry group)?
- Normal computation from VTK gradient data vs our compute_normals()?
- Multi-component volumes (RGB volumes, vector fields)?

**Integration points:**
```python
# Desired API (tentative)
from pyvr.loaders import load_vtk_volume
volume = load_vtk_volume("path/to/data.vti")
# Returns Volume object, ready for renderer
```

### Success Criteria

- Load standard VTK image data formats (.vti, .vtk)
- Correctly map VTK spacing/bounds to Volume
- Existing Volume operations work (compute_normals, normalize, etc.)
- Examples demonstrating real data loading
- Tests for various VTK data types

---

## v0.4.x - Stabilization

**Primary Goal:** Debug and polish VTK integration

**Status:** Future

### Scope

- Fix issues discovered during v0.4.0 usage
- Performance optimization for large VTK volumes
- Documentation improvements
- User feedback integration
- Test real-world VTK datasets from scientific domains

### Timeline

Allocate 1-2 minor releases (v0.4.1, v0.4.2) before starting v0.5.0 work.

---

## v0.5.0 - DearPyGUI Interface

**Primary Goal:** Advanced GUI interface with direct framebuffer rendering

**Status:** Planned

### Scope

**Core Features:**
- DearPyGUI-based interface as alternative to matplotlib
- Direct ModernGL framebuffer display (no PIL conversion)
- Real-time continuous rendering loop
- Backend-agnostic state management system

**Implementation:**
- Optional poetry dependency group: `[gui]`
- New class: `AdvancedVolumeRenderer` or similar (naming TBD)
- Keep matplotlib `InteractiveVolumeRenderer` fully functional
- Design state system to support both GUIs

**Architecture:**
```
pyvr/interface/
├── matplotlib_interface.py    # Existing, unchanged
├── dearpygui_interface.py     # New GUI
├── state.py                   # Refactor to be backend-agnostic
└── widgets/                   # Shared widget logic (if applicable)
```

### Design Considerations

**Backend-agnostic state system:**
- Needs to handle current params: camera, transfer functions, render config, lighting
- Must be extensible for future params: scattering coefficients, phase functions, etc.
- Consider pub/sub or observer pattern for GUI updates
- State categories to design for:
  - **Geometry state:** Camera, volume bounds
  - **Material state:** Transfer functions, absorption, scattering (v0.6.0+)
  - **Rendering state:** Quality presets, ray tracing settings
  - **Lighting state:** Light properties, phase functions (v0.6.0+)

**User entry points:**
```python
# Matplotlib interface (existing)
from pyvr.interface import InteractiveVolumeRenderer
interface = InteractiveVolumeRenderer(volume)
interface.show()

# DearPyGUI interface (new)
from pyvr.interface import AdvancedVolumeRenderer  # Name TBD
interface = AdvancedVolumeRenderer(volume)
interface.run()
```

**Performance benefits:**
- Direct GPU texture display (no framebuffer → numpy → PIL → matplotlib)
- Continuous render loop instead of event-driven updates
- Better responsiveness during camera manipulation

### Out of Scope for v0.5.0

- Progressive rendering (that's v0.5.1)
- Complex widgets beyond basics
- Multi-window/docking layouts (can add in v0.5.x if needed)

### Success Criteria

- DearPyGUI interface displays volume with camera controls
- Transfer function editing works
- Performance measurably better than matplotlib for interactive use
- Matplotlib interface still works (backward compatibility)
- State system designed for extensibility

---

## v0.5.1 - Progressive Rendering

**Primary Goal:** Implement temporal accumulation for Monte Carlo rendering

**Status:** Planned

### Scope

**Core Features:**
- Frame accumulation buffer
- Sample count tracking
- Reset logic (camera/parameter changes trigger restart)
- Display sample count to user

**Implementation:**
- Accumulation buffer: Start with CPU-based (GPU optimization in v0.5.x)
- Accumulation formula: `accumulated = (accumulated * n + newFrame) / (n + 1)`
- Reset triggers:
  - Camera movement
  - Transfer function changes
  - Render config changes
  - Any parameter affecting output

**Integration with DearPyGUI:**
- Continuous render loop already in place (from v0.5.0)
- Display sample count in GUI
- Optional: Show convergence indicator

**Initial implementation strategy:**
```python
# Pseudo-code for accumulation loop
sample_count = 0
accumulated_buffer = None

while gui_running:
    if state_changed():
        sample_count = 0
        accumulated_buffer = None

    new_frame = renderer.render()  # With jittered rays

    if accumulated_buffer is None:
        accumulated_buffer = new_frame
    else:
        accumulated_buffer = (accumulated_buffer * sample_count + new_frame) / (sample_count + 1)

    sample_count += 1
    display(accumulated_buffer)
```

### Design Considerations

**Questions to resolve:**
- Jittering strategy: Ray origin jitter? Sample point jitter along ray?
- Accumulation location: CPU or GPU? (Start CPU, optimize to GPU in v0.5.x)
- Convergence detection: Track variance? Simple sample count threshold?
- User controls: Manual reset button? Auto vs manual accumulation mode?

**State system integration:**
- Add accumulation state tracking
- Dirty flags for efficient reset detection
- Sample count as part of rendering state

### Out of Scope for v0.5.1

- Sophisticated jittering strategies (refine in v0.5.x)
- GPU-based accumulation (optimize in v0.5.x)
- Convergence metrics beyond sample count
- Adaptive sampling

### Success Criteria

- Progressive rendering converges from noisy → clean
- Resets correctly when parameters change
- Sample count displayed to user
- Basic jittering reduces aliasing
- Lays foundation for v0.6.0 Monte Carlo light tracing

---

## v0.5.x - Refinement

**Primary Goal:** Polish GUI and progressive rendering based on usage

**Status:** Planned

### Scope

**GUI refinement:**
- User feedback integration
- Additional widgets as needed
- Layout improvements
- Performance optimization

**Progressive rendering refinement:**
- Improved jittering strategies
- GPU-based accumulation (performance)
- Convergence metrics (variance tracking)
- Adaptive sampling strategies

**State system:**
- Refinements based on v0.5.0/v0.5.1 experience
- Ensure extensibility for v0.6.0 light tracing parameters

### Timeline

Allocate 2-3 minor releases (v0.5.2, v0.5.3, possibly v0.5.4) to ensure solid foundation before v0.6.0.

---

## v0.6.0 - Light Transport Simulation

**Primary Goal:** Complex light scattering and Monte Carlo path tracing

**Status:** Research phase - parameters and encoding TBD

### Scope (Tentative)

**Physics simulation:**
- Light scattering in participating media
- Absorption and emission
- Phase functions (Henyey-Greenstein? Rayleigh? Mie?)
- Multiple scattering (Monte Carlo sampling)

**Volume encoding:**
- Multi-channel volumes for material properties
- Possible channels:
  - Absorption coefficient
  - Scattering coefficient
  - Anisotropy factor (g)
  - Emission (optional)
  - Density/albedo

**Integration with v0.5.x:**
- Progressive rendering handles noisy Monte Carlo samples
- DearPyGUI provides real-time parameter adjustment
- State system already supports complex material properties

### Research Needed

**Physics questions:**
- Which phase function models to support?
- Single vs multiple scattering (computational cost)?
- How many ray bounces practical for real-time?
- Importance sampling strategies?

**Implementation questions:**
- Shader complexity: Can we do this in fragment shader, or need compute shaders?
- Volume encoding: Separate textures vs packed multi-channel?
- Parameter space: What needs to be user-adjustable vs precomputed?

**Performance questions:**
- Target frame rate with progressive rendering?
- GPU memory requirements for accumulation + multi-channel volumes?
- Fallback for less capable hardware?

### Example Use Cases (to guide design)

- **Medical imaging:** Tissue scattering, X-ray attenuation
- **Atmospheric rendering:** Clouds, fog, volumetric lighting
- **Scientific visualization:** Turbulence, combustion, plasma

### Success Criteria (TBD)

Will be defined after research phase. Tentatively:
- Physically plausible light transport
- Real-time interaction with progressive rendering
- Parameter control through DearPyGUI
- Examples demonstrating different phase functions
- Validation against ground truth (research code/commercial tools)

---

## Timeline (Rough Estimates)

| Version | Focus | Estimated Effort | Status |
|---------|-------|------------------|--------|
| v0.4.0 | VTK data loader | 2-3 weeks | Planning |
| v0.4.x | Stabilization | 1-2 weeks | Future |
| v0.5.0 | DearPyGUI interface | 3-4 weeks | Planned |
| v0.5.1 | Progressive rendering | 2-3 weeks | Planned |
| v0.5.x | Refinement | 2-4 weeks | Planned |
| v0.6.0 | Light transport | 4-6 weeks (research + impl) | Research |

**Total estimated timeline:** 4-6 months for v0.4.0 → v0.6.0

---

## Design Principles

Throughout this roadmap, maintain PyVR's core philosophy:

1. **Pre-1.0 flexibility:** Breaking changes acceptable for better design
2. **Comprehensive testing:** All features must have tests
3. **Clean architecture:** One feature per major version
4. **No premature optimization:** Get it working, then make it fast
5. **User-focused:** Real use cases drive feature design

---

## Open Questions

### v0.4.0 (VTK)
- [ ] VTK as core dependency or optional poetry group?
- [ ] Which VTK file formats to prioritize (.vti, .vtk, .vtu)?
- [ ] How to handle VTK multi-component data?

### v0.5.0 (GUI)
- [ ] Class naming for DearPyGUI interface?
- [ ] State system architecture (pub/sub vs direct updates)?
- [ ] How much matplotlib/DearPyGUI code can be shared?

### v0.5.1 (Progressive)
- [ ] CPU vs GPU accumulation for initial release?
- [ ] Jittering strategy (ray origin, sample points, both)?
- [ ] User control interface (auto vs manual modes)?

### v0.6.0 (Light Transport)
- [ ] Which phase function models?
- [ ] Volume encoding strategy?
- [ ] Target performance (samples/second)?
- [ ] Validation methodology?

---

## Next Steps

**Immediate (pre-v0.4.0):**
1. Research VTK data loading best practices
2. Explore existing VTK datasets for testing
3. Prototype VTK → Volume conversion
4. Decide on VTK dependency strategy

**Before v0.5.0:**
1. Research DearPyGUI + ModernGL integration
2. Design backend-agnostic state system
3. Prototype framebuffer → DearPyGUI texture display
4. Plan state categories for extensibility

**Before v0.6.0:**
1. Literature review: Light transport in participating media
2. Survey phase function models
3. Prototype Monte Carlo sampling in shader
4. Identify validation datasets/references

---

## Notes

- This roadmap is a living document - update as research progresses
- Each version should have detailed design doc before implementation
- Use feature-design skill for complex features
- Maintain version notes for all releases
- Pin this conversation for continuity across sessions
