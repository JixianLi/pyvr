# PyVR v0.3.1 Interface Refinements

## Feature Description

This release enhances PyVR's interactive matplotlib-based interface (introduced in v0.3.0) with four key usability and performance monitoring features:

1. **Directional Light Camera Linking**: Allow directional lights to automatically follow camera movement with configurable offsets
2. **FPS Counter**: Real-time frames-per-second display for performance monitoring
3. **Rendering Preset Selector**: UI widget for switching between RenderConfig quality presets
4. **Log-Scale Histogram Background**: Add histogram visualization to opacity editor showing volume data distribution

These features improve the interactive experience by providing better lighting control, performance visibility, quality switching, and visual feedback during transfer function editing.

## Summary

The v0.3.1 refinements build incrementally on the existing v0.3.0 architecture without breaking changes. The implementation follows PyVR's established patterns:

**Architecture approach**: Each feature integrates naturally with existing components:
- Light linking extends the `Light` class with new methods and integrates with `CameraController`
- FPS counter is a lightweight text overlay in the `ImageDisplay` widget with minimal performance impact
- Preset selector is a new widget following the pattern of `ColorSelector` (RadioButtons-based)
- Histogram caching uses the `tmp_dev/` directory for persistent storage between sessions

**Implementation strategy**: The plan is organized into 6 focused phases that can be implemented sequentially. Each phase delivers working, tested functionality:
1. FPS counter (simplest, validates performance monitoring approach)
2. Rendering preset selector (builds on widget patterns)
3. Light camera linking (extends lighting system)
4. Histogram background (most complex, requires caching)
5. Integration and polish (connects all features)
6. Documentation update (finalizes release)

**Testing approach**: Each phase includes comprehensive tests (>85% coverage target) using the established mock-based testing infrastructure. Integration tests validate feature interactions.

The total implementation is estimated at ~800-1000 lines of new/modified code across 6 phases, keeping each phase focused and achievable.

## Progress Tracker

- [ ] Phase 1: FPS Counter Implementation
- [ ] Phase 2: Rendering Preset Selector Widget
- [ ] Phase 3: Directional Light Camera Linking
- [ ] Phase 4: Log-Scale Histogram Background
- [ ] Phase 5: Integration and Polish
- [ ] Phase 6: Documentation Update

## Final Acceptance Criteria

### Functional Requirements
1. **FPS Counter**:
   - Displays current FPS in corner of rendering view
   - Updates in real-time during camera movement
   - Minimal performance overhead (<1% impact)
   - Can be toggled on/off programmatically

2. **Rendering Preset Selector**:
   - Provides access to all 5 RenderConfig presets (preview, fast, balanced, high_quality, ultra_quality)
   - Updates rendering immediately when preset changes
   - Visual indication of current preset
   - Integrates cleanly with existing layout

3. **Light Camera Linking**:
   - `Light.link_to_camera(offsets)` method creates linked light
   - Offsets allow azimuth/elevation/distance adjustment relative to camera
   - Link can be toggled on/off via `Light.unlink_from_camera()`
   - Light follows camera movement smoothly during orbit/zoom
   - Works with existing `Light` presets (directional, point_light, etc.)

4. **Histogram Background**:
   - Log-scale histogram overlaid on opacity editor background
   - Computed once and cached in `tmp_dev/histogram_cache/`
   - Cache uses volume shape + data hash for invalidation
   - Subtle coloring that doesn't interfere with control points
   - Optional toggle to show/hide histogram

### Quality Requirements
1. **Test Coverage**: >85% coverage for all new code
2. **Test Suite**: All 284+ tests pass (including new tests)
3. **Performance**: No degradation to existing rendering speed
4. **Documentation**: All public APIs documented with Google-style docstrings
5. **Examples**: Working examples demonstrate all new features

### Non-Breaking Changes
- All existing APIs remain unchanged
- New features are additive only
- Backward compatibility maintained for v0.3.0 code
- No changes to existing widget behavior (unless enhanced)

## Architecture Changes

### New Classes/Methods

**`pyvr/interface/widgets.py`**:
- New `PresetSelector` widget class (follows `ColorSelector` pattern)
- New `FPSCounter` helper class for FPS calculation and display
- Modified `OpacityEditor` to support histogram background

**`pyvr/lighting/light.py`**:
- New `link_to_camera(azimuth_offset, elevation_offset, distance_offset)` method
- New `unlink_from_camera()` method
- New `is_linked` property
- New `_camera_offsets` private attribute
- New `update_from_camera(camera)` method to compute linked position

**`pyvr/interface/state.py`**:
- New `current_preset_name` attribute (tracks RenderConfig preset)
- New `show_fps` boolean flag
- New `show_histogram` boolean flag
- New `set_preset(preset_name)` method

**`pyvr/interface/matplotlib_interface.py`**:
- Modified `_update_display()` to update light from camera if linked
- Modified layout to accommodate new preset selector
- New FPS counter integration in `ImageDisplay`
- New preset selector event handlers

**New module: `pyvr/interface/cache.py`**:
- `compute_volume_hash(volume)` - generates hash for cache key
- `get_cached_histogram(volume)` - retrieves cached histogram
- `cache_histogram(volume, histogram)` - saves histogram to cache
- Uses `tmp_dev/histogram_cache/` directory

### Interaction Patterns

```python
# Light camera linking pattern
light = Light.directional([1, -1, 0])
light.link_to_camera(azimuth_offset=0.0, elevation_offset=0.2, distance_offset=0.0)

# In interface loop:
if light.is_linked:
    light.update_from_camera(camera)
    renderer.set_light(light)

# Histogram caching pattern
histogram = get_cached_histogram(volume)
if histogram is None:
    histogram = compute_histogram_with_log_scale(volume.data)
    cache_histogram(volume, histogram)
```

### Modified Components

1. **`InterfaceState`**: Expanded to track preset selection and display flags
2. **`OpacityEditor`**: Enhanced to render histogram background
3. **`ImageDisplay`**: Enhanced with FPS counter overlay
4. **`InteractiveVolumeRenderer`**: Modified layout and event handling

## Breaking Changes

**None** - This is an additive release. All v0.3.0 code continues to work unchanged.

The following changes are API extensions (not breaking):
- `Light` class gains new optional methods (existing behavior unchanged)
- `InterfaceState` gains new optional attributes (defaults maintain v0.3.0 behavior)
- Widget classes gain new optional parameters (existing APIs unchanged)

## Dependencies

### Runtime Dependencies (No Changes)
All existing dependencies remain:
- moderngl >= 5.0
- numpy >= 2.3
- matplotlib >= 3.10
- pillow >= 11.0
- scipy >= 1.16

### Development Dependencies (No Changes)
- pytest >= 7.0
- pytest-cov >= 7.0
- black >= 24.9
- isort >= 6.0

### System Requirements
- `tmp_dev/` directory for histogram caching (already exists, in .gitignore)
- No new OpenGL requirements
- No new system dependencies

## Implementation Notes

### Phase Ordering Rationale

1. **Phase 1 (FPS Counter)**: Start with simplest feature to validate performance monitoring approach and establish FPS calculation patterns
2. **Phase 2 (Preset Selector)**: Build on widget patterns before more complex features; provides immediate user value
3. **Phase 3 (Light Linking)**: Extends lighting system in focused way; can reference FPS counter for performance validation
4. **Phase 4 (Histogram)**: Most complex feature requiring caching infrastructure; benefits from earlier learning
5. **Phase 5 (Integration)**: Connect features and add polish now that all components exist
6. **Phase 6 (Documentation)**: Final phase updates all docs and examples

### Key Design Decisions

**Light Linking Design**:
- Chose method-based API (`light.link_to_camera()`) over constructor parameter for clarity
- Offsets are relative angles/distance (not absolute positions) for intuitive control
- Update is manual via `update_from_camera()` rather than automatic to give interface full control

**FPS Counter Design**:
- Text overlay on image display (not separate widget) minimizes layout changes
- Rolling average over last N frames for stable display
- Can be toggled without rebuilding layout

**Preset Selector Design**:
- RadioButtons widget (like ColorSelector) for consistency
- Positioned in right column with other controls
- Shows current preset prominently

**Histogram Caching Design**:
- Cache key: `volume.shape` + `hash(volume.data)` for reliable invalidation
- Log scale essential for typical volume data distribution
- Stored in `tmp_dev/histogram_cache/` (already in .gitignore)
- Cache files use pickle format for simplicity

### Performance Considerations

- FPS calculation uses `time.perf_counter()` for microsecond precision
- Histogram computed once per volume, cached indefinitely
- Preset switching updates config but doesn't rebuild entire interface
- Light linking adds negligible overhead (one matrix multiplication per frame)
- All features designed for <1% performance impact on rendering

### Testing Strategy

Each phase includes:
1. **Unit tests**: Test individual methods/classes in isolation with mocks
2. **Integration tests**: Test feature interaction with existing components
3. **Edge cases**: Boundary values, error conditions, invalid inputs
4. **Performance tests**: Verify <1% overhead for new features

Total expected test count: +40-50 new tests across all phases.
