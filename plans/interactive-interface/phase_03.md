# Phase 3: Image Display Widget with Rendering Integration

**Status**: Not Started
**Estimated Effort**: 3-4 hours
**Dependencies**: Phase 2 (layout implementation)

## Overview

Ensure the image display widget properly integrates with the ModernGL renderer, handles rendering efficiently, and displays the volume correctly. Add error handling and performance optimizations.

## Implementation Plan

### Key Changes

1. **Verify renderer integration** - Ensure `_render_volume()` works correctly
2. **Add render caching** - Only re-render when necessary (not on every frame)
3. **Add error handling** - Handle OpenGL context issues gracefully
4. **Add loading indicator** - Show feedback during slow renders

### Modify: `pyvr/interface/matplotlib_interface.py`

Add render optimization:

```python
def __init__(self, ...):
    # ... existing code ...
    self._cached_image: Optional[np.ndarray] = None
    self._render_pending = False

def _render_volume(self) -> np.ndarray:
    """Render volume with caching."""
    if not self.state.needs_render and self._cached_image is not None:
        return self._cached_image

    try:
        # Update camera
        self.renderer.set_camera(self.camera_controller.camera)

        # Render
        image = self.renderer.render_to_pil()
        self._cached_image = np.array(image)
        return self._cached_image

    except Exception as e:
        # Handle OpenGL errors gracefully
        print(f"Rendering error: {e}")
        # Return a placeholder image
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)
```

## Testing Plan

```python
def test_render_caching():
    """Test that rendering is cached when state doesn't change."""
    interface = InteractiveVolumeRenderer(volume=small_volume)
    img1 = interface._render_volume()
    interface.state.needs_render = False
    img2 = interface._render_volume()
    assert img1 is img2  # Same object, not re-rendered

def test_render_error_handling():
    """Test rendering handles errors gracefully."""
    interface = InteractiveVolumeRenderer(volume=small_volume)
    interface.renderer.render_to_pil = Mock(side_effect=Exception("OpenGL error"))
    img = interface._render_volume()
    assert img.shape == (512, 512, 3)  # Returns placeholder
```

## Acceptance Criteria

- [x] Rendering works correctly with real ModernGL backend
- [x] Render caching prevents unnecessary re-renders
- [x] Error handling prevents crashes on OpenGL issues
- [x] Test coverage maintained >85%

## Git Commit

```bash
pytest tests/
git add pyvr/interface/ tests/test_interface/
git commit -m "Phase 3: Add rendering integration and optimization

- Implement render caching to prevent unnecessary re-renders
- Add error handling for OpenGL context issues
- Add tests for render caching and error handling
- Verify integration with ModernGL backend

Part of v0.3.0 interactive interface feature"
```
