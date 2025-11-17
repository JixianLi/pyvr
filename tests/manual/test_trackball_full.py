"""Manual integration test for trackball control.

Run this script and manually verify all functionality works correctly.
"""

from pyvr.interface import InteractiveVolumeRenderer
from pyvr.volume import Volume
from pyvr.datasets import create_sample_volume
from pyvr.camera import Camera, CameraController
import numpy as np


def test_trackball_api():
    """Test trackball API directly."""
    print("Testing trackball API...")

    controller = CameraController(Camera.isometric_view(distance=3.0))
    initial_az = controller.params.azimuth

    # Test horizontal rotation
    controller.trackball(dx=100, dy=0, viewport_width=800, viewport_height=600)
    assert controller.params.azimuth != initial_az
    print("✓ Horizontal rotation works")

    # Test vertical rotation
    initial_el = controller.params.elevation
    controller.trackball(dx=0, dy=100, viewport_width=800, viewport_height=600)
    assert controller.params.elevation != initial_el
    print("✓ Vertical rotation works")

    # Test zero movement
    before = controller.params.azimuth
    controller.trackball(dx=0, dy=0, viewport_width=800, viewport_height=600)
    assert controller.params.azimuth == before
    print("✓ Zero movement handled")

    print("\nAPI tests passed!\n")


def test_interface():
    """Launch interface for manual testing."""
    print("Launching interface for manual testing...")
    print("\nTest checklist:")
    print("[ ] Interface launches successfully")
    print("[ ] Default mode is trackball (check info panel)")
    print("[ ] Drag rotates camera smoothly")
    print("[ ] Press 't' to switch to orbit mode")
    print("[ ] Orbit mode feels different (azimuth/elevation)")
    print("[ ] Press 't' again to switch back")
    print("[ ] Zoom works in both modes")
    print("[ ] All other keys work (r, s, f, h, l, q)")
    print("[ ] No errors or crashes")
    print("\nClose window when done testing.\n")

    volume_data = create_sample_volume(128, "double_sphere")
    volume = Volume(data=volume_data)

    interface = InteractiveVolumeRenderer(volume=volume, width=512, height=512)

    interface.show()

    print("\nInterface test complete!")


if __name__ == "__main__":
    test_trackball_api()
    test_interface()
