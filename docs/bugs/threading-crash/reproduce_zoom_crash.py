#!/usr/bin/env python3
"""
ABOUTME: Minimal script to reproduce the quality preset + zoom crash bug
ABOUTME: Demonstrates thread-safety violation when using Timer for quality restoration

Bug reproduction steps:
1. Launch interface
2. Click on a different quality preset (e.g., "High Quality")
3. Scroll mouse wheel to zoom in/out
4. Program crashes silently after ~0.5 seconds

Expected behavior: Smooth zoom with auto-quality switching
Actual behavior: Silent crash due to matplotlib thread-safety violation

Root cause: threading.Timer calls matplotlib functions from background thread
Location: pyvr/interface/matplotlib_interface.py:502
"""

import numpy as np
from pyvr.datasets import create_sample_volume
from pyvr.volume import Volume
from pyvr.interface import InteractiveVolumeRenderer
from pyvr.config import RenderConfig

def main():
    print("=" * 80)
    print("BUG REPRODUCTION SCRIPT - Quality Preset + Zoom Crash")
    print("=" * 80)
    print()
    print("Instructions to reproduce the bug:")
    print("1. Wait for the interface to load")
    print("2. Click on 'High Quality' or 'Ultra (slowest)' in the Rendering Quality panel")
    print("3. Move mouse over the volume rendering (left panel)")
    print("4. Scroll mouse wheel up or down to zoom")
    print("5. Wait ~0.5 seconds")
    print("6. Program should crash silently (window closes or freezes)")
    print()
    print("Note: The crash happens because threading.Timer executes the quality")
    print("      restoration callback in a background thread, but matplotlib")
    print("      widgets are NOT thread-safe.")
    print()
    print("Root cause location: pyvr/interface/matplotlib_interface.py:502")
    print("   threading.Timer(0.5, self._restore_quality_after_interaction)")
    print()
    print("=" * 80)
    input("Press Enter to launch the interface...")
    print()

    # Create a simple volume
    print("Creating sample volume...")
    volume_data = create_sample_volume(128, 'sphere')
    volume = Volume(data=volume_data)

    # Create interface with auto-quality enabled (default)
    print("Initializing interface with auto_quality_enabled=True...")
    interface = InteractiveVolumeRenderer(
        volume=volume,
        width=512,
        height=512,
        config=RenderConfig.fast()  # Start with fast preset
    )

    # Verify auto-quality is enabled
    print(f"Auto-quality enabled: {interface.state.auto_quality_enabled}")
    print()
    print("Launching interface...")
    print("Now follow the instructions above to reproduce the crash.")
    print()

    # Show the interface
    try:
        interface.show()
    except Exception as e:
        print()
        print("=" * 80)
        print("CRASH DETECTED!")
        print("=" * 80)
        print(f"Exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
