# Bug Analysis: Silent Crash on Quality Preset Change + Zoom

**Date**: 2025-11-14
**Severity**: HIGH (causes silent crash, data loss)
**Status**: Identified, not fixed

## Summary

The interactive interface crashes silently when the user:
1. Selects a different quality preset
2. Scrolls to zoom in/out

The crash occurs approximately 0.5 seconds after scrolling due to a **thread-safety violation** in matplotlib widget manipulation.

## Root Cause

**Threading.Timer executes matplotlib operations from a background thread.**

Matplotlib is **NOT thread-safe**. GUI operations (widget modifications, drawing) MUST occur on the main thread. The current implementation uses `threading.Timer` to delay quality restoration after zoom, but the timer callback executes in a background thread.

### Code Location

**Primary bug location**: `pyvr/interface/matplotlib_interface.py:502`

```python
def _on_scroll(self, event) -> None:
    # ... zoom handling ...

    # Restore quality after short delay
    if self.state.auto_quality_enabled:
        import threading
        # Cancel any existing timer
        if hasattr(self, '_scroll_restore_timer') and self._scroll_restore_timer is not None:
            self._scroll_restore_timer.cancel()
        # Start new timer
        self._scroll_restore_timer = threading.Timer(0.5, self._restore_quality_after_interaction)
        self._scroll_restore_timer.start()  # ❌ Callback runs in BACKGROUND THREAD
```

**Secondary violation location**: `pyvr/interface/matplotlib_interface.py:346`

```python
def _restore_quality_after_interaction(self) -> None:
    # ...

    # Update preset selector UI
    if self.preset_selector:
        self.preset_selector.set_preset(self.state.saved_preset_name)  # ❌ Called from Timer thread
```

**Tertiary violation location**: `pyvr/interface/widgets.py:509`

```python
def set_preset(self, preset_name: str) -> None:
    # ...

    # Update radio button selection
    preset_index = self.AVAILABLE_PRESETS.index(preset_name)
    self.radio.set_active(preset_index)  # ❌ Modifies matplotlib widget from non-main thread
```

## Detailed Event Sequence

### Step 1: User selects quality preset

User clicks "High Quality" in the Rendering Quality panel.

```
Thread: MainThread
├─ _on_preset_change('high_quality')
├─ state.set_preset('high_quality')
├─ state.current_preset_name = 'high_quality'
└─ renderer.set_config(RenderConfig.high_quality())
```

State after:
- `state.current_preset_name = 'high_quality'`
- `state.saved_preset_name = None`

### Step 2: User scrolls to zoom

User moves mouse over volume rendering and scrolls wheel.

```
Thread: MainThread
├─ _on_scroll(event)
├─ _switch_to_interaction_quality()
│   ├─ state.saved_preset_name = 'high_quality'  # Save current
│   ├─ state.current_preset_name = 'fast'        # Switch to fast
│   └─ renderer.set_config(RenderConfig.fast())
├─ camera_controller.zoom(factor)
├─ _update_display()
└─ threading.Timer(0.5, _restore_quality_after_interaction).start()  # ⚠️ Creates timer
```

State after:
- `state.current_preset_name = 'fast'`
- `state.saved_preset_name = 'high_quality'`
- Timer scheduled to fire in 0.5s

### Step 3: Timer fires (0.5s later)

**⚠️ CRITICAL: This executes in a BACKGROUND THREAD, not MainThread!**

```
Thread: Thread-1 (Timer background thread) ❌
├─ _restore_quality_after_interaction()
├─ restored_config = RenderConfig.high_quality()
├─ renderer.set_config(restored_config)
├─ preset_selector.set_preset('high_quality')  # ❌ Modifies GUI from wrong thread
│   └─ radio.set_active(3)                     # ❌ Modifies RadioButtons
├─ _update_display(force_render=True)          # ❌ Renders from wrong thread
│   ├─ renderer.render()
│   └─ image_display.update_image()
│       └─ fig.canvas.draw_idle()              # ❌ matplotlib draw from wrong thread
└─ 💥 CRASH (silent or exception)
```

## Why It Crashes

### Thread-Safety Violation

Matplotlib's GUI backends (like TkAgg, Qt5Agg) use native GUI toolkits that are **not thread-safe**:

1. **Widget modification**: `radio.set_active()` modifies the RadioButtons widget
   - This calls into the native GUI toolkit (Tk, Qt, etc.)
   - Native GUI toolkits require all operations on the main/UI thread
   - Calling from background thread causes undefined behavior

2. **Canvas drawing**: `fig.canvas.draw_idle()` schedules a redraw
   - May trigger immediate rendering operations
   - Rendering uses OpenGL context which may be bound to main thread
   - Background thread access → crash or corruption

3. **Race conditions**: Multiple threads accessing matplotlib objects
   - No locks protecting internal state
   - Concurrent access → data corruption → crash

### Platform-Specific Behavior

The crash may be "silent" (window just closes) or raise exceptions depending on:
- **macOS (TkAgg)**: Often silent crash or SystemError
- **Linux (Qt5Agg)**: May raise QThread errors
- **Windows**: May raise Win32 exceptions or freeze

## Evidence

### Code Flow Trace

1. `_on_scroll()` → MainThread ✓
2. `threading.Timer.start()` → MainThread ✓
3. Timer callback fires → **Thread-1 (background)** ❌
4. `_restore_quality_after_interaction()` → **Thread-1** ❌
5. `preset_selector.set_preset()` → **Thread-1** ❌
6. `radio.set_active()` → **Thread-1** ❌ CRASH

### Matplotlib Documentation

From matplotlib docs:
> Matplotlib is not thread-safe: in fact, there are known race conditions that affect certain artists. Hence, if you work with threads, it is your responsibility to set up the proper locks to serialize access to Matplotlib artists.

> GUI backends (like TkAgg, Qt5Agg) are not thread-safe. All GUI operations must happen on the main thread.

## Reproduction Steps

### Minimal Steps

1. Launch `InteractiveVolumeRenderer`
2. Click any quality preset OTHER than the current one (e.g., "High Quality")
3. Scroll mouse wheel over the volume rendering
4. Wait 0.5 seconds
5. **CRASH** (window closes or freezes)

### Conditions Required

- `state.auto_quality_enabled = True` (default)
- User changes quality preset before scrolling
- User scrolls (triggers timer)
- Timer callback executes (0.5s delay)

### Reproduction Scripts

- **Simple**: `tmp_dev/reproduce_zoom_crash.py`
- **Instrumented**: `tmp_dev/trace_threading_bug.py` (shows exact thread violations)

## Impact

### User Experience

- **Data loss**: Work lost when window closes unexpectedly
- **Confusion**: Silent crash provides no error message
- **Unreliable**: Basic workflow (change quality + zoom) broken

### Affected Workflows

Any workflow involving:
1. Changing quality presets during interaction
2. Using zoom with auto-quality enabled (default)

## Solution Approaches

### Option 1: Use matplotlib's timer (RECOMMENDED)

Replace `threading.Timer` with `matplotlib.animation.Timer` or `fig.canvas.new_timer()`:

```python
# Instead of threading.Timer
self._scroll_restore_timer = self.fig.canvas.new_timer(interval=500)  # 500ms
self._scroll_restore_timer.add_callback(self._restore_quality_after_interaction)
self._scroll_restore_timer.single_shot = True
self._scroll_restore_timer.start()
```

**Pros**:
- Callback executes on main thread (thread-safe)
- Native matplotlib integration
- Works across all backends

**Cons**:
- Requires figure to be created first
- Slightly different API than threading.Timer

### Option 2: Queue callback for main thread

Use matplotlib's `canvas.call_after()` or similar:

```python
# Queue for execution on main thread after delay
self.fig.canvas.flush_events()  # Process pending events
# Use after_idle or similar backend-specific method
```

**Pros**:
- Thread-safe
- Can work with existing threading.Timer

**Cons**:
- Backend-specific APIs
- More complex implementation

### Option 3: Disable auto-quality by default

Simple workaround:

```python
# In InterfaceState
auto_quality_enabled: bool = False  # Changed from True
```

**Pros**:
- Immediate fix
- No threading issues

**Cons**:
- Removes useful feature
- Doesn't fix the underlying bug

### Option 4: Remove delayed restoration

Remove the timer entirely, restore quality immediately on mouse release:

```python
def _on_scroll(self, event) -> None:
    # ... zoom handling ...

    # No timer - restore immediately
    if self.state.auto_quality_enabled:
        self._restore_quality_after_interaction()
```

**Pros**:
- No threading issues
- Simpler code

**Cons**:
- Less responsive (quality switches during zoom instead of after)
- Defeats purpose of auto-quality

## Recommended Fix

**Use matplotlib's native timer** (Option 1)

1. Replace `threading.Timer` with `fig.canvas.new_timer()`
2. Update `_on_scroll()` method
3. Ensure timer cancellation works correctly
4. Add tests for thread-safety

### Implementation Notes

- Check if timer exists before creating new one
- Cancel previous timer if exists
- Use single_shot mode for one-time callback
- Test across different matplotlib backends

## Testing Recommendations

1. **Manual testing**: Follow reproduction steps with fix applied
2. **Automated testing**: Create test that simulates preset change + scroll
3. **Thread safety test**: Verify all callbacks execute on main thread
4. **Backend testing**: Test with TkAgg, Qt5Agg, and other backends

## Related Issues

- Similar threading issues may exist with camera drag (uses same auto-quality pattern)
- Check `_on_mouse_release()` for similar Timer usage patterns

## References

- Matplotlib thread-safety: https://matplotlib.org/stable/users/explain/performance.html#threading
- Timer API: https://matplotlib.org/stable/api/backend_bases_api.html#matplotlib.backend_bases.TimerBase
- RadioButtons widget: https://matplotlib.org/stable/api/widgets_api.html#matplotlib.widgets.RadioButtons
