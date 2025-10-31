# Interactive Volume Renderer Interface

## Feature Description

Create an interactive matplotlib-based interface for PyVR volume rendering with real-time transfer function editing and camera controls. This is a testing/development interface (not production) that allows researchers and developers to:

- Visualize volumes with real-time rendering updates
- Interactively edit opacity transfer functions using control points
- Switch between different matplotlib colormaps for color transfer functions
- Control camera position through mouse interactions (orbit, zoom)
- Experiment with rendering parameters in an intuitive GUI

The interface will serve as both a testing tool during development and a demonstration of PyVR's capabilities for new users.

## Target Version

**v0.3.0**

This is a minor version bump (v0.2.7 → v0.3.0) because:
- New major feature addition (interactive interface module)
- New public API surface (`pyvr.interface` module)
- No breaking changes to existing APIs
- Follows pre-1.0 development philosophy where new features warrant minor version increments

## Phase Progress Tracker

- [ ] **Phase 1**: Project structure and module scaffolding
- [ ] **Phase 2**: Core matplotlib interface layout
- [ ] **Phase 3**: Image display widget with rendering integration
- [ ] **Phase 4**: Camera controls (mouse orbit and zoom)
- [ ] **Phase 5**: Color transfer function selector
- [ ] **Phase 6**: Interactive opacity transfer function editor
- [ ] **Phase 7**: Event handling and state management
- [ ] **Phase 8**: Testing, documentation, and version increment

## Success Criteria

### Functional Requirements
- [ ] Interface displays rendered volume in real-time
- [ ] Mouse scroll zooms camera in/out smoothly
- [ ] Mouse drag orbits camera around scene center
- [ ] Dropdown menu switches between matplotlib colormaps
- [ ] Opacity transfer function displays as editable line plot
- [ ] Can add control points by left-clicking empty space
- [ ] Can select control points by left-clicking them (visual highlight)
- [ ] Can remove control points by right-clicking them (except first/last)
- [ ] Can drag control points to modify position and opacity
- [ ] First/last control points locked to x=0.0 and x=1.0 (only opacity changes)
- [ ] Rendering updates in real-time after transfer function or camera changes

### Quality Requirements
- [ ] Test coverage maintained at >85%
- [ ] All 204+ existing tests pass
- [ ] New interface has comprehensive unit and integration tests
- [ ] Documentation updated (CLAUDE.md, README.md, examples)
- [ ] Version notes created for v0.3.0

### Performance Requirements
- [ ] Interface responsive (<100ms lag for user interactions)
- [ ] Rendering uses appropriate quality preset for interactive performance
- [ ] Only re-renders when needed (not on every mouse move)

## Dependencies

### Existing PyVR Modules
- `pyvr.moderngl_renderer.VolumeRenderer` - Core rendering engine
- `pyvr.camera.Camera` - Camera positioning and matrix generation
- `pyvr.camera.CameraController` - Camera manipulation (orbit, zoom, pan)
- `pyvr.config.RenderConfig` - Quality presets (will use `fast()` for interactivity)
- `pyvr.volume.Volume` - Volume data management
- `pyvr.lighting.Light` - Lighting configuration
- `pyvr.transferfunctions.ColorTransferFunction` - Color mapping
- `pyvr.transferfunctions.OpacityTransferFunction` - Opacity mapping

### External Libraries (Already in Dependencies)
- `matplotlib >= 3.10` - GUI framework and plotting
- `numpy >= 2.3` - Array operations
- `pillow >= 11.0` - Image conversion

### New Test Dependencies
- `pytest >= 7.0` - Testing framework (already present)
- `pytest-cov >= 7.0` - Coverage reporting (already present)

## Breaking Changes

**None.** This feature is purely additive:
- New module: `pyvr.interface`
- No changes to existing public APIs
- Existing code continues to work unchanged
- Optional feature that doesn't affect core rendering pipeline

## Architecture Overview

### Module Structure
```
pyvr/
├── interface/
│   ├── __init__.py          # Public API exports
│   ├── matplotlib.py         # Main interface class (InteractiveVolumeRenderer)
│   ├── widgets.py           # Reusable widget components
│   │   ├── ImageDisplay     # Volume rendering display with camera controls
│   │   ├── OpacityEditor    # Interactive opacity transfer function editor
│   │   └── ColorSelector    # Colormap dropdown selector
│   └── state.py             # Interface state management
```

### Design Patterns

**Widget-Based Architecture**: Each UI component is encapsulated as a widget class:
- `ImageDisplay`: Manages matplotlib axes for volume rendering and camera interaction
- `OpacityEditor`: Manages matplotlib axes for transfer function editing
- `ColorSelector`: Manages dropdown widget for colormap selection

**Event-Driven Updates**: Interface uses matplotlib's event system:
- `button_press_event`: Handle clicks (select/add/remove control points)
- `button_release_event`: Handle drag end
- `motion_notify_event`: Handle dragging (camera orbit, control point movement)
- `scroll_event`: Handle zoom

**State Management**: Centralized state in `InterfaceState` class:
- Current camera parameters
- Selected control point index
- Transfer function control points
- Dragging state (camera vs control point)

**Efficient Rendering**: Only trigger re-render when needed:
- Debounce camera movements (render on drag end, not every mouse move)
- Immediate transfer function updates (small overhead)
- Use `RenderConfig.fast()` for interactive performance

### Integration with PyVR Core

```python
# Interface wraps existing PyVR components
renderer = VolumeRenderer(config=RenderConfig.fast())
camera_controller = CameraController(Camera.from_spherical(...))
ctf = ColorTransferFunction.from_colormap('viridis')
otf = OpacityTransferFunction(control_points=[(0.0, 0.0), (1.0, 1.0)])

# Interface provides interactive wrapper
interface = InteractiveVolumeRenderer(
    volume=volume,
    renderer=renderer,
    camera_controller=camera_controller,
    initial_ctf=ctf,
    initial_otf=otf
)
interface.show()  # Launch matplotlib GUI
```

## Estimated Complexity

**Overall Effort**: 2-3 days

**Phase Breakdown**:
- Phase 1 (Scaffolding): 1-2 hours
- Phase 2 (Layout): 2-3 hours
- Phase 3 (Image Display): 3-4 hours
- Phase 4 (Camera Controls): 3-4 hours
- Phase 5 (Color Selector): 2-3 hours
- Phase 6 (Opacity Editor): 4-5 hours (most complex - control point interaction)
- Phase 7 (Event Integration): 3-4 hours
- Phase 8 (Testing & Docs): 4-6 hours

**Risk Factors**:
- **Medium Risk**: Matplotlib event handling can be tricky (event coordinate transforms, event conflicts)
- **Medium Risk**: Efficient rendering updates (need to balance responsiveness vs performance)
- **Low Risk**: Integration with existing PyVR components (APIs are stable and well-tested)

## Future Enhancements (Out of Scope for v0.3.0)

- Multi-dimensional transfer functions (2D histograms)
- Animation timeline for camera paths
- Volume clipping plane controls
- Real-time performance metrics display
- Export rendered frames to video
- Multiple viewport support
- Preset library (save/load transfer functions)

These can be added in future minor versions (v0.3.1, v0.3.2, etc.) without breaking changes.
