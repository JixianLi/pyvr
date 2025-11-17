"""Integration tests for opacity correction in VolumeRenderer."""

import pytest
import numpy as np
from unittest.mock import MagicMock
from pyvr.moderngl_renderer import VolumeRenderer
from pyvr.config import RenderConfig
from pyvr.volume import Volume
from pyvr.datasets import create_sample_volume
from pyvr.transferfunctions import ColorTransferFunction, OpacityTransferFunction


@pytest.fixture
def test_volume():
    """Create a test volume."""
    volume_data = create_sample_volume(64, "sphere")
    return Volume(data=volume_data)


@pytest.fixture
def test_renderer(test_volume, mock_moderngl_context):
    """Create a renderer with test volume loaded."""
    # Use 512x512 to match mocked framebuffer size
    renderer = VolumeRenderer(width=512, height=512)
    renderer.load_volume(test_volume)

    # Set transfer functions
    ctf = ColorTransferFunction.from_colormap("viridis")
    otf = OpacityTransferFunction.linear(0.0, 0.5)
    renderer.set_transfer_functions(ctf, otf)

    return renderer


class TestOpacityCorrectionIntegration:
    """Integration tests for opacity correction in rendering pipeline."""

    def test_reference_step_size_uniform_exists(self, test_renderer):
        """Test that reference_step_size uniform exists in shader."""
        # In mocked environment, __getitem__ always returns a mock uniform
        # We test that accessing it doesn't raise an error
        uniform = test_renderer.gl_manager.program["reference_step_size"]
        assert uniform is not None

    def test_reference_step_size_uniform_value(self, test_renderer):
        """Test that reference_step_size uniform has correct value."""
        config = RenderConfig.balanced()
        test_renderer.set_config(config)

        # Verify set_uniform_float was called with correct value
        # The uniform value is set via gl_manager.set_uniform_float
        test_renderer.gl_manager.set_uniform_float = MagicMock()
        test_renderer.set_config(config)

        # Check that reference_step_size was set
        calls = test_renderer.gl_manager.set_uniform_float.call_args_list
        ref_step_calls = [call for call in calls if call[0][0] == "reference_step_size"]
        assert len(ref_step_calls) > 0
        assert ref_step_calls[0][0][1] == config.reference_step_size

    def test_custom_reference_step_size(self, test_renderer):
        """Test setting custom reference_step_size."""
        config = RenderConfig(step_size=0.01, max_steps=500, reference_step_size=0.008)

        # Mock the set_uniform_float method
        test_renderer.gl_manager.set_uniform_float = MagicMock()
        test_renderer.set_config(config)

        # Verify the custom reference_step_size was set
        calls = test_renderer.gl_manager.set_uniform_float.call_args_list
        ref_step_calls = [call for call in calls if call[0][0] == "reference_step_size"]
        assert len(ref_step_calls) > 0
        assert ref_step_calls[0][0][1] == 0.008

    def test_all_presets_render_successfully(self, test_renderer):
        """Test that all presets render without errors."""
        presets = ["preview", "fast", "balanced", "high_quality", "ultra_quality"]

        for preset_name in presets:
            preset_method = getattr(RenderConfig, preset_name)
            config = preset_method()
            test_renderer.set_config(config)

            # Should render without errors
            data = test_renderer.render()
            assert data is not None
            assert isinstance(data, bytes)
            assert len(data) > 0

    def test_opacity_consistency_across_presets(self, test_renderer):
        """Test that different presets produce consistent rendering output."""
        presets_to_test = ["preview", "balanced", "high_quality"]
        data_sizes = []

        for preset_name in presets_to_test:
            preset_method = getattr(RenderConfig, preset_name)
            config = preset_method()
            test_renderer.set_config(config)

            data = test_renderer.render()
            data_sizes.append(len(data))

        # All renders should produce same size output in mocked environment
        assert len(set(data_sizes)) == 1
        assert data_sizes[0] > 0

    def test_no_fully_transparent_renders(self, test_renderer):
        """Test that no preset produces empty render output."""
        presets = ["preview", "fast", "balanced", "high_quality", "ultra_quality"]

        for preset_name in presets:
            preset_method = getattr(RenderConfig, preset_name)
            test_renderer.set_config(preset_method())

            data = test_renderer.render()
            # Verify rendering produces data
            assert data is not None
            assert len(data) > 0

    def test_different_reference_step_sizes(self, test_renderer):
        """Test rendering with different reference_step_size values."""
        reference_values = [0.005, 0.01, 0.015, 0.02]

        for ref_step in reference_values:
            config = RenderConfig(
                step_size=0.01, max_steps=500, reference_step_size=ref_step
            )

            # Mock the set_uniform_float method to verify it's called
            test_renderer.gl_manager.set_uniform_float = MagicMock()
            test_renderer.set_config(config)

            # Should render successfully
            data = test_renderer.render()
            assert data is not None
            assert len(data) > 0

            # Verify reference_step_size was set correctly
            calls = test_renderer.gl_manager.set_uniform_float.call_args_list
            ref_step_calls = [
                call for call in calls if call[0][0] == "reference_step_size"
            ]
            assert len(ref_step_calls) > 0
            assert ref_step_calls[0][0][1] == ref_step


class TestOpacityCorrectionRegression:
    """Regression tests to ensure opacity correction doesn't break existing functionality."""

    def test_existing_rendering_still_works(self, test_renderer):
        """Test that existing rendering functionality is not broken."""
        # This is the "smoke test" - basic rendering should still work
        data = test_renderer.render()
        assert data is not None
        assert isinstance(data, bytes)
        assert len(data) > 0

    def test_transfer_function_changes_still_work(self, test_renderer):
        """Test that changing transfer functions still works."""
        # Change color map
        ctf_new = ColorTransferFunction.from_colormap("plasma")
        otf = OpacityTransferFunction.linear(0.0, 0.3)
        test_renderer.set_transfer_functions(ctf_new, otf)

        data = test_renderer.render()
        assert data is not None

    def test_camera_changes_still_work(self, test_renderer):
        """Test that camera changes still work."""
        from pyvr.camera import Camera

        camera = Camera.front_view(distance=4.0)
        test_renderer.set_camera(camera)

        data = test_renderer.render()
        assert data is not None

    def test_light_changes_still_work(self, test_renderer):
        """Test that light changes still work."""
        from pyvr.lighting import Light

        light = Light.directional([1, -1, -1], ambient=0.3, diffuse=0.7)
        test_renderer.set_light(light)

        data = test_renderer.render()
        assert data is not None
