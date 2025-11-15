#!/usr/bin/env python3
"""
ABOUTME: Diagnostic script to trace the threading bug in quality preset + zoom
ABOUTME: Uses monkey patching to intercept and log all relevant function calls with thread info

This script instruments the code to show EXACTLY what's happening during the crash:
- Which thread each function is called from
- The sequence of operations
- When matplotlib widget modifications occur from wrong thread
"""

import sys
import threading
import functools
import numpy as np
from pyvr.datasets import create_sample_volume
from pyvr.volume import Volume
from pyvr.interface import InteractiveVolumeRenderer
from pyvr.config import RenderConfig


def trace_call(prefix):
    """Decorator to trace function calls with thread information."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_thread = threading.current_thread()
            is_main = current_thread == threading.main_thread()
            thread_name = current_thread.name
            thread_id = current_thread.ident

            print(f"[{prefix}] {func.__name__}() called")
            print(f"  Thread: {thread_name} (id={thread_id})")
            print(f"  Main thread: {is_main}")

            if not is_main and 'matplotlib' in func.__module__.lower():
                print(f"  ⚠️  WARNING: Matplotlib function called from NON-MAIN thread!")

            try:
                result = func(*args, **kwargs)
                print(f"  ✓ {func.__name__}() completed successfully")
                return result
            except Exception as e:
                print(f"  ✗ {func.__name__}() FAILED: {e}")
                import traceback
                traceback.print_exc()
                raise

        return wrapper
    return decorator


def instrument_interface():
    """Monkey-patch InteractiveVolumeRenderer to trace the bug."""
    print("=" * 80)
    print("INSTRUMENTING InteractiveVolumeRenderer FOR DEBUGGING")
    print("=" * 80)
    print()

    # Patch the critical methods
    original_on_scroll = InteractiveVolumeRenderer._on_scroll
    original_switch_to_interaction = InteractiveVolumeRenderer._switch_to_interaction_quality
    original_restore_quality = InteractiveVolumeRenderer._restore_quality_after_interaction

    InteractiveVolumeRenderer._on_scroll = trace_call("SCROLL")(original_on_scroll)
    InteractiveVolumeRenderer._switch_to_interaction_quality = trace_call("SWITCH")(original_switch_to_interaction)
    InteractiveVolumeRenderer._restore_quality_after_interaction = trace_call("RESTORE")(original_restore_quality)

    # Also patch the preset selector set_preset method
    from pyvr.interface.widgets import PresetSelector
    original_set_preset = PresetSelector.set_preset

    @functools.wraps(original_set_preset)
    def traced_set_preset(self, preset_name):
        current_thread = threading.current_thread()
        is_main = current_thread == threading.main_thread()
        thread_name = current_thread.name

        print(f"[PRESET] PresetSelector.set_preset('{preset_name}') called")
        print(f"  Thread: {thread_name}")
        print(f"  Main thread: {is_main}")

        if not is_main:
            print(f"  🔥 BUG FOUND! PresetSelector.set_preset() called from non-main thread!")
            print(f"  🔥 This will modify matplotlib RadioButtons from background thread!")
            print(f"  🔥 Matplotlib is NOT thread-safe - this causes the crash!")
            print()
            print(f"  Call stack:")
            import traceback
            traceback.print_stack()

        return original_set_preset(self, preset_name)

    PresetSelector.set_preset = traced_set_preset

    # Patch threading.Timer to show when timers are created
    original_timer_init = threading.Timer.__init__

    @functools.wraps(original_timer_init)
    def traced_timer_init(self, interval, function, args=None, kwargs=None):
        current_thread = threading.current_thread()
        print(f"[TIMER] Creating Timer with {interval}s delay")
        print(f"  Will call: {function.__name__}()")
        print(f"  Created from thread: {current_thread.name}")
        print(f"  ⚠️  Timer callback will execute in BACKGROUND THREAD!")
        return original_timer_init(self, interval, function, args, kwargs)

    threading.Timer.__init__ = traced_timer_init

    print("✓ Instrumentation complete")
    print()


def main():
    print("=" * 80)
    print("THREADING BUG DIAGNOSTIC TRACE")
    print("=" * 80)
    print()
    print("This script will show you EXACTLY when the thread-safety violation occurs.")
    print()
    print("Watch for these key events:")
    print("  1. [SCROLL] _on_scroll() - when you scroll to zoom")
    print("  2. [SWITCH] _switch_to_interaction_quality() - temporary fast preset")
    print("  3. [TIMER] Creating Timer - sets up 0.5s delayed callback")
    print("  4. [RESTORE] _restore_quality_after_interaction() - FROM BACKGROUND THREAD")
    print("  5. [PRESET] set_preset() - MODIFIES MATPLOTLIB FROM BACKGROUND THREAD")
    print("  6. 🔥 BUG FOUND! - The exact moment of the crash")
    print()
    print("=" * 80)
    input("Press Enter to start instrumented interface...")
    print()

    # Instrument the code
    instrument_interface()

    # Create volume
    print("Creating sample volume...")
    volume_data = create_sample_volume(128, 'sphere')
    volume = Volume(data=volume_data)

    # Create interface
    print("Initializing interface...")
    interface = InteractiveVolumeRenderer(
        volume=volume,
        width=512,
        height=512,
        config=RenderConfig.fast()
    )

    print()
    print("=" * 80)
    print("READY TO TRACE")
    print("=" * 80)
    print()
    print("Instructions:")
    print("1. Click on 'High Quality' or 'Ultra (slowest)' preset")
    print("2. Scroll mouse wheel to zoom")
    print("3. Watch the console output below")
    print("4. You'll see the exact sequence of events leading to the crash")
    print()
    print("=" * 80)
    print()

    try:
        interface.show()
    except Exception as e:
        print()
        print("=" * 80)
        print("EXCEPTION CAUGHT")
        print("=" * 80)
        print(f"Exception: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
