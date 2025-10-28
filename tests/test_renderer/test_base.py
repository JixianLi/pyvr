"""Tests for abstract VolumeRenderer base class."""
import numpy as np
import pytest

from pyvr.renderer.base import VolumeRenderer
from pyvr.volume import Volume
from pyvr.camera import Camera
from pyvr.lighting import Light
from pyvr.transferfunctions import ColorTransferFunction, OpacityTransferFunction


class TestAbstractRenderer:
    """Test abstract VolumeRenderer base class."""

    def test_cannot_instantiate_abstract_class(self):
        """Abstract VolumeRenderer cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            VolumeRenderer()

    def test_must_implement_load_volume(self):
        """Subclass must implement load_volume method."""

        class IncompleteRenderer(VolumeRenderer):
            def set_camera(self, camera):
                pass

            def set_light(self, light):
                pass

            def set_transfer_functions(self, ctf, otf, size=None):
                pass

            def render(self):
                pass

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteRenderer()

    def test_must_implement_set_camera(self):
        """Subclass must implement set_camera method."""

        class IncompleteRenderer(VolumeRenderer):
            def load_volume(self, volume):
                pass

            def set_light(self, light):
                pass

            def set_transfer_functions(self, ctf, otf, size=None):
                pass

            def render(self):
                pass

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteRenderer()

    def test_must_implement_set_light(self):
        """Subclass must implement set_light method."""

        class IncompleteRenderer(VolumeRenderer):
            def load_volume(self, volume):
                pass

            def set_camera(self, camera):
                pass

            def set_transfer_functions(self, ctf, otf, size=None):
                pass

            def render(self):
                pass

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteRenderer()

    def test_must_implement_set_transfer_functions(self):
        """Subclass must implement set_transfer_functions method."""

        class IncompleteRenderer(VolumeRenderer):
            def load_volume(self, volume):
                pass

            def set_camera(self, camera):
                pass

            def set_light(self, light):
                pass

            def render(self):
                pass

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteRenderer()

    def test_must_implement_render(self):
        """Subclass must implement render method."""

        class IncompleteRenderer(VolumeRenderer):
            def load_volume(self, volume):
                pass

            def set_camera(self, camera):
                pass

            def set_light(self, light):
                pass

            def set_transfer_functions(self, ctf, otf, size=None):
                pass

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteRenderer()


class TestConcreteRenderer:
    """Test concrete implementation of VolumeRenderer."""

    def test_complete_implementation(self):
        """Subclass with all methods can be instantiated."""

        class CompleteRenderer(VolumeRenderer):
            def load_volume(self, volume):
                self.volume = volume

            def set_camera(self, camera):
                self.camera = camera

            def set_light(self, light):
                self.light = light

            def set_transfer_functions(self, ctf, otf, size=None):
                pass

            def render(self):
                return b""

        renderer = CompleteRenderer(width=512, height=512)
        assert renderer.width == 512
        assert renderer.height == 512
        assert renderer.volume is None
        assert renderer.camera is None
        assert renderer.light is None

    def test_concrete_getters(self):
        """Base class provides concrete getter methods."""

        class CompleteRenderer(VolumeRenderer):
            def load_volume(self, volume):
                self.volume = volume

            def set_camera(self, camera):
                self.camera = camera

            def set_light(self, light):
                self.light = light

            def set_transfer_functions(self, ctf, otf, size=None):
                pass

            def render(self):
                return b""

        renderer = CompleteRenderer()

        # Create test objects
        data = np.zeros((32, 32, 32), dtype=np.float32)
        volume = Volume(data=data)
        camera = Camera()
        light = Light.default()

        # Test getters return None initially
        assert renderer.get_volume() is None
        assert renderer.get_camera() is None
        assert renderer.get_light() is None

        # Set objects
        renderer.load_volume(volume)
        renderer.set_camera(camera)
        renderer.set_light(light)

        # Test getters return set objects
        assert renderer.get_volume() is volume
        assert renderer.get_camera() is camera
        assert renderer.get_light() is light

    def test_initialization_with_custom_size(self):
        """Renderer can be initialized with custom width and height."""

        class CompleteRenderer(VolumeRenderer):
            def load_volume(self, volume):
                pass

            def set_camera(self, camera):
                pass

            def set_light(self, light):
                pass

            def set_transfer_functions(self, ctf, otf, size=None):
                pass

            def render(self):
                return b""

        renderer = CompleteRenderer(width=1024, height=768)
        assert renderer.width == 1024
        assert renderer.height == 768


if __name__ == "__main__":
    pytest.main([__file__])
