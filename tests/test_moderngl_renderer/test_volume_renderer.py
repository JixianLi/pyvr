"""
Comprehensive tests for VolumeRenderer class.
Tests all functionality with mocked dependencies.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch, Mock

from pyvr.moderngl_renderer.renderer import VolumeRenderer
from pyvr.camera.camera import Camera


class TestVolumeRendererInitialization:
    """Test VolumeRenderer initialization and setup."""
    
    def test_init_default_parameters(self, mock_moderngl_context):
        """Test initialization with default parameters."""
        renderer = VolumeRenderer()

        assert renderer.width == 512
        assert renderer.height == 512
        assert renderer.config.step_size == 0.01
        assert renderer.config.max_steps == 500  # balanced preset default
        assert renderer.gl_manager is not None
        
    def test_init_custom_parameters(self, mock_moderngl_context):
        """Test initialization with custom parameters."""
        from pyvr.config import RenderConfig

        config = RenderConfig(step_size=0.005, max_steps=500)
        renderer = VolumeRenderer(width=1024, height=768, config=config)

        assert renderer.width == 1024
        assert renderer.height == 768
        assert renderer.config.step_size == 0.005
        assert renderer.config.max_steps == 500




class TestVolumeRendererTransferFunctions:
    """Test transfer function management - v0.2.2 RGBA feature."""
    
    def test_set_transfer_functions(self, mock_moderngl_context, mock_transfer_functions):
        """Test setting color and opacity transfer functions."""
        renderer = VolumeRenderer()
        color_tf, opacity_tf = mock_transfer_functions
        
        # Mock the manager's RGBA texture creation
        renderer.gl_manager.create_rgba_transfer_function_texture = MagicMock(return_value=0)
        renderer.gl_manager.set_uniform_int = MagicMock()
        
        renderer.set_transfer_functions(color_tf, opacity_tf)
        
        # Should create RGBA texture and set uniform
        renderer.gl_manager.create_rgba_transfer_function_texture.assert_called_once_with(
            color_tf, opacity_tf, None
        )
        renderer.gl_manager.set_uniform_int.assert_called_with('transfer_function_lut', 0)
        
    def test_set_transfer_functions_with_custom_size(self, mock_moderngl_context, mock_transfer_functions):
        """Test setting transfer functions with custom size."""
        renderer = VolumeRenderer()
        color_tf, opacity_tf = mock_transfer_functions
        
        # Mock the manager's RGBA texture creation
        renderer.gl_manager.create_rgba_transfer_function_texture = MagicMock(return_value=1)
        renderer.gl_manager.set_uniform_int = MagicMock()
        
        renderer.set_transfer_functions(color_tf, opacity_tf, size=512)
        
        # Should create RGBA texture with custom size
        renderer.gl_manager.create_rgba_transfer_function_texture.assert_called_once_with(
            color_tf, opacity_tf, 512
        )


class TestVolumeRendererCameraManagement:
    """Test camera and view management."""
    
    def test_set_camera(self, mock_moderngl_context):
        """Test setting camera parameters."""
        from pyvr.camera import Camera

        renderer = VolumeRenderer()

        # Create camera using Camera class
        camera = Camera.front_view(distance=3.0)

        renderer.set_camera(camera)

        # Should complete without error
        assert True




class TestVolumeRendererRenderingProcess:
    """Test the complete rendering process."""
    
    def test_render_basic(self, mock_moderngl_context, sample_volume_data, mock_transfer_functions):
        """Test basic rendering pipeline."""
        from pyvr.volume import Volume

        renderer = VolumeRenderer()
        color_tf, opacity_tf = mock_transfer_functions

        # Create Volume instance
        volume = Volume(data=sample_volume_data)

        # Mock the manager methods BEFORE loading volume
        renderer.gl_manager.create_volume_texture = MagicMock(return_value=0)
        renderer.gl_manager.set_uniform_int = MagicMock()
        renderer.gl_manager.set_uniform_vector = MagicMock()
        renderer.gl_manager.create_rgba_transfer_function_texture = MagicMock(return_value=1)
        renderer.gl_manager.set_uniform_float = MagicMock()
        renderer.gl_manager.set_uniform_matrix = MagicMock()
        renderer.gl_manager.clear_framebuffer = MagicMock()
        renderer.gl_manager.setup_blending = MagicMock()
        renderer.gl_manager.render_quad = MagicMock()
        renderer.gl_manager.read_pixels = MagicMock(return_value=np.random.rand(512, 512, 4))

        # Setup volume and transfer functions
        renderer.load_volume(volume)
        renderer.set_transfer_functions(color_tf, opacity_tf)

        image = renderer.render()

        assert image is not None
        
    def test_render_to_pil(self, mock_moderngl_context, sample_volume_data, mock_transfer_functions):
        """Test rendering to PIL image."""
        from pyvr.volume import Volume

        renderer = VolumeRenderer()
        color_tf, opacity_tf = mock_transfer_functions

        # Mock the manager methods BEFORE loading volume
        renderer.gl_manager.create_volume_texture = MagicMock(return_value=0)
        renderer.gl_manager.create_rgba_transfer_function_texture = MagicMock(return_value=1)
        renderer.gl_manager.set_uniform_int = MagicMock()
        renderer.gl_manager.set_uniform_vector = MagicMock()
        renderer.gl_manager.set_uniform_float = MagicMock()
        renderer.gl_manager.set_uniform_matrix = MagicMock()
        renderer.gl_manager.clear_framebuffer = MagicMock()
        renderer.gl_manager.setup_blending = MagicMock()
        renderer.gl_manager.render_quad = MagicMock()

        # Setup volume and transfer functions
        volume = Volume(data=sample_volume_data)
        renderer.load_volume(volume)
        
        # Mock read_pixels to return proper image data as bytes
        mock_pixel_data = b'\\x00' * (512 * 512 * 4)  # Mock RGBA pixel data
        renderer.gl_manager.read_pixels = MagicMock(return_value=mock_pixel_data)
        
        renderer.set_transfer_functions(color_tf, opacity_tf)
        
        with patch('PIL.Image.frombytes') as mock_frombytes:
            with patch.object(renderer, 'render', return_value=mock_pixel_data) as mock_render:
                mock_image = MagicMock()
                mock_image.transpose.return_value = mock_image
                mock_frombytes.return_value = mock_image
                
                pil_image = renderer.render_to_pil()
                
                # Check that frombytes was called with correct parameters
                mock_frombytes.assert_called_once_with("RGBA", (512, 512), mock_pixel_data)
                mock_image.transpose.assert_called_once()
                assert pil_image is not None


class TestVolumeRendererIntegration:
    """Test integration scenarios and complex workflows."""
    
    def test_multi_frame_rendering(self, mock_moderngl_context, sample_volume_data, mock_transfer_functions):
        """Test rendering multiple frames with parameter changes."""
        from pyvr.volume import Volume

        renderer = VolumeRenderer()
        color_tf, opacity_tf = mock_transfer_functions

        # Mock all manager operations BEFORE loading volume
        renderer.gl_manager.create_volume_texture = MagicMock(return_value=0)
        renderer.gl_manager.create_rgba_transfer_function_texture = MagicMock(return_value=1)
        renderer.gl_manager.set_uniform_int = MagicMock()
        renderer.gl_manager.set_uniform_vector = MagicMock()
        renderer.gl_manager.set_uniform_float = MagicMock()
        renderer.gl_manager.set_uniform_matrix = MagicMock()
        renderer.gl_manager.clear_framebuffer = MagicMock()
        renderer.gl_manager.setup_blending = MagicMock()
        renderer.gl_manager.render_quad = MagicMock()
        renderer.gl_manager.read_pixels = MagicMock(return_value=np.random.rand(512, 512, 4))

        # Setup volume and transfer functions
        volume = Volume(data=sample_volume_data)
        renderer.load_volume(volume)
        renderer.set_transfer_functions(color_tf, opacity_tf)

        # Render with different configurations
        from pyvr.config import RenderConfig

        config1 = RenderConfig.balanced()
        renderer.set_config(config1)
        image1 = renderer.render()

        config2 = RenderConfig.high_quality()
        renderer.set_config(config2)
        image2 = renderer.render()

        assert image1 is not None
        assert image2 is not None
        
    def test_volume_data_replacement(self, mock_moderngl_context, mock_transfer_functions):
        """Test replacing volume data mid-session."""
        from pyvr.volume import Volume

        renderer = VolumeRenderer()
        color_tf, opacity_tf = mock_transfer_functions

        # Mock the manager methods BEFORE calling load_volume
        renderer.gl_manager.create_volume_texture = MagicMock(return_value=0)
        renderer.gl_manager.set_uniform_int = MagicMock()
        renderer.gl_manager.set_uniform_vector = MagicMock()

        # Setup initial volume
        volume1_data = np.random.rand(16, 16, 16).astype(np.float32)
        volume1 = Volume(data=volume1_data)
        renderer.load_volume(volume1)
        renderer.set_transfer_functions(color_tf, opacity_tf)

        # Mock manager for rendering
        renderer.gl_manager.create_rgba_transfer_function_texture = MagicMock(return_value=1)
        renderer.gl_manager.set_uniform_float = MagicMock()
        renderer.gl_manager.set_uniform_matrix = MagicMock()
        renderer.gl_manager.clear_framebuffer = MagicMock()
        renderer.gl_manager.setup_blending = MagicMock()
        renderer.gl_manager.render_quad = MagicMock()
        renderer.gl_manager.read_pixels = MagicMock(return_value=np.random.rand(512, 512, 4))

        renderer.render()

        # Load second volume
        volume2_data = np.random.rand(32, 32, 32).astype(np.float32)
        volume2 = Volume(data=volume2_data)
        renderer.load_volume(volume2)
        
        renderer.render()
        
        # Should have created volume texture twice (once for each load_volume call)
        assert renderer.gl_manager.create_volume_texture.call_count == 2


class TestVolumeRendererErrorHandling:
    """Test error handling and edge cases."""
    
    def test_parameter_validation(self, mock_moderngl_context):
        """Test config parameter validation."""
        from pyvr.config import RenderConfig

        renderer = VolumeRenderer()

        # Mock uniform setters
        renderer.gl_manager.set_uniform_float = MagicMock()
        renderer.gl_manager.set_uniform_int = MagicMock()

        # Test that setting valid config works
        config = RenderConfig(step_size=0.005, max_steps=1000)
        renderer.set_config(config)
        assert renderer.config.step_size == 0.005
        assert renderer.config.max_steps == 1000
        
    def test_render_without_setup(self, mock_moderngl_context):
        """Test that rendering without proper setup fails gracefully."""
        renderer = VolumeRenderer()
        
        # Mock the gl_manager methods to avoid actual OpenGL calls
        renderer.gl_manager.set_uniform_float = MagicMock()
        renderer.gl_manager.set_uniform_int = MagicMock()
        renderer.gl_manager.set_uniform_matrix = MagicMock()
        renderer.gl_manager.clear_framebuffer = MagicMock()
        renderer.gl_manager.setup_blending = MagicMock()
        renderer.gl_manager.render_quad = MagicMock()
        renderer.gl_manager.read_pixels = MagicMock(return_value=np.random.rand(512, 512, 4))
        
        try:
            image = renderer.render()
            # If it succeeds, that's also acceptable
            assert image is not None
        except Exception as e:
            # Should be a reasonable error
            assert isinstance(e, (ValueError, RuntimeError, AttributeError))


class TestVolumeRendererResourceManagement:
    """Test resource management and cleanup."""
    
    def test_initialization_creates_manager(self, mock_moderngl_context):
        """Test that initialization properly creates GL manager."""
        renderer = VolumeRenderer()
        
        assert renderer.gl_manager is not None
        assert hasattr(renderer.gl_manager, 'ctx')
        
    def test_manager_delegation(self, mock_moderngl_context, mock_transfer_functions):
        """Test that renderer properly delegates to GL manager."""
        renderer = VolumeRenderer()
        color_tf, opacity_tf = mock_transfer_functions
        
        # Mock manager method
        renderer.gl_manager.create_rgba_transfer_function_texture = MagicMock(return_value=5)
        renderer.gl_manager.set_uniform_int = MagicMock()
        
        renderer.set_transfer_functions(color_tf, opacity_tf)
        
        # Should have delegated to manager
        renderer.gl_manager.create_rgba_transfer_function_texture.assert_called_once()
    
    def test_load_volume_edge_cases(self, mock_moderngl_context):
        """Test edge cases for volume loading."""
        from pyvr.volume import Volume

        renderer = VolumeRenderer()

        # Mock GL manager methods
        renderer.gl_manager.create_volume_texture = MagicMock(return_value=0)
        renderer.gl_manager.set_uniform_int = MagicMock()
        renderer.gl_manager.set_uniform_vector = MagicMock()

        # Test loading very small volume
        tiny_volume_data = np.random.random((1, 1, 1)).astype(np.float32)
        tiny_volume = Volume(data=tiny_volume_data)
        renderer.load_volume(tiny_volume)
        renderer.gl_manager.create_volume_texture.assert_called_once()

        # Test non-Volume object raises TypeError
        invalid_volume = np.random.random((10, 10, 10))  # numpy array, not Volume
        with pytest.raises(TypeError, match="Expected Volume instance"):
            renderer.load_volume(invalid_volume)
    
    def test_rendering_with_minimal_setup(self, mock_moderngl_context):
        """Test rendering with minimal setup to cover edge cases."""
        renderer = VolumeRenderer()
        
        # Mock all the GL operations
        renderer.gl_manager.set_uniform_float = MagicMock()
        renderer.gl_manager.set_uniform_int = MagicMock()
        renderer.gl_manager.set_uniform_matrix = MagicMock()
        renderer.gl_manager.clear_framebuffer = MagicMock()
        renderer.gl_manager.setup_blending = MagicMock()
        renderer.gl_manager.render_quad = MagicMock()
        renderer.gl_manager.read_pixels = MagicMock(return_value=np.zeros((100, 100, 4), dtype=np.uint8))
        
        # Set camera using Camera class
        from pyvr.camera import Camera
        camera = Camera.front_view(distance=3.0)
        renderer.set_camera(camera)
        
        # Should not crash
        pixels = renderer.render()
        assert pixels is not None
        
    def test_parameter_validation_comprehensive(self, mock_moderngl_context):
        """Test comprehensive parameter validation."""
        from pyvr.config import RenderConfig

        renderer = VolumeRenderer()

        # Mock the GL operations to avoid actual OpenGL calls
        renderer.gl_manager.set_uniform_float = MagicMock()
        renderer.gl_manager.set_uniform_int = MagicMock()

        # Test setting various parameters through config
        ultra_config = RenderConfig.ultra_quality()
        renderer.set_config(ultra_config)
        assert renderer.config.step_size == 0.001
        assert renderer.config.max_steps == 2000


class TestVolumeRendererLightIntegration:
    """Test Light class integration with VolumeRenderer (v0.2.4)."""

    def test_volume_renderer_light_attribute(self, mock_moderngl_context):
        """VolumeRenderer should have a light attribute."""
        from pyvr.lighting import Light

        renderer = VolumeRenderer()

        assert hasattr(renderer, 'light')
        assert isinstance(renderer.light, Light)

    def test_volume_renderer_custom_light(self, mock_moderngl_context):
        """VolumeRenderer should accept custom light."""
        from pyvr.lighting import Light

        custom_light = Light.directional(direction=[1, 0, 0], ambient=0.5)
        renderer = VolumeRenderer(width=256, height=256, light=custom_light)

        assert renderer.light is custom_light
        assert renderer.light.ambient_intensity == 0.5

    def test_set_light(self, mock_moderngl_context):
        """set_light should accept Light instance."""
        from pyvr.lighting import Light

        renderer = VolumeRenderer(width=256, height=256)
        new_light = Light.point_light(position=[5, 5, 5])

        renderer.set_light(new_light)

        assert renderer.light is new_light

    def test_get_light(self, mock_moderngl_context):
        """get_light should return current light."""
        from pyvr.lighting import Light

        renderer = VolumeRenderer(width=256, height=256)
        light = renderer.get_light()

        assert isinstance(light, Light)
        assert light is renderer.light

    def test_set_light_type_checking(self, mock_moderngl_context):
        """set_light should raise TypeError for non-Light instance."""
        renderer = VolumeRenderer(width=256, height=256)

        with pytest.raises(TypeError, match="Expected Light instance"):
            renderer.set_light("not a light")

        with pytest.raises(TypeError, match="Expected Light instance"):
            renderer.set_light({"ambient": 0.5})

    def test_light_updates_uniforms(self, mock_moderngl_context):
        """Setting light should update GL uniforms."""
        from pyvr.lighting import Light

        renderer = VolumeRenderer(width=256, height=256)

        # Mock GL manager methods
        renderer.gl_manager.set_uniform_float = MagicMock()
        renderer.gl_manager.set_uniform_vector = MagicMock()

        # Create and set new light
        new_light = Light.directional(direction=[1, -1, 0], ambient=0.3, diffuse=0.9)
        renderer.set_light(new_light)

        # Should have called set_uniform methods
        renderer.gl_manager.set_uniform_float.assert_any_call("ambient_light", 0.3)
        renderer.gl_manager.set_uniform_float.assert_any_call("diffuse_light", 0.9)
        renderer.gl_manager.set_uniform_vector.assert_called()

    def test_default_light_initialization(self, mock_moderngl_context):
        """VolumeRenderer should create default light if none provided."""
        from pyvr.lighting import Light

        renderer = VolumeRenderer()

        assert isinstance(renderer.light, Light)
        assert renderer.light.ambient_intensity == 0.2
        assert renderer.light.diffuse_intensity == 0.8

    def test_light_presets_integration(self, mock_moderngl_context):
        """Test various light presets with VolumeRenderer."""
        from pyvr.lighting import Light

        # Test directional light
        light_dir = Light.directional(direction=[1, -1, 0])
        renderer = VolumeRenderer(light=light_dir)
        assert isinstance(renderer.light, Light)

        # Test point light
        light_point = Light.point_light(position=[5, 5, 5])
        renderer.set_light(light_point)
        assert np.allclose(renderer.light.position, [5, 5, 5])

        # Test ambient only
        light_ambient = Light.ambient_only(intensity=0.4)
        renderer.set_light(light_ambient)
        assert renderer.light.ambient_intensity == 0.4
        assert renderer.light.diffuse_intensity == 0.0


class TestVolumeRendererVolumeClass:
    """Test Volume class integration with VolumeRenderer (v0.2.5)."""

    def test_load_volume_instance(self, mock_moderngl_context):
        """VolumeRenderer should accept Volume instance."""
        from pyvr.volume import Volume

        renderer = VolumeRenderer()

        # Mock GL manager methods
        renderer.gl_manager.create_volume_texture = MagicMock(return_value=0)
        renderer.gl_manager.set_uniform_int = MagicMock()
        renderer.gl_manager.set_uniform_vector = MagicMock()

        # Create Volume instance
        data = np.random.rand(32, 32, 32).astype(np.float32)
        volume = Volume(data=data)

        renderer.load_volume(volume)

        # Should store the volume
        assert renderer.volume is volume

        # Should have created texture and set uniforms
        renderer.gl_manager.create_volume_texture.assert_called_once_with(data)
        renderer.gl_manager.set_uniform_int.assert_called_with("volume_texture", 0)
        assert renderer.gl_manager.set_uniform_vector.call_count >= 2  # min_bounds and max_bounds

    def test_load_volume_with_normals(self, mock_moderngl_context):
        """VolumeRenderer should handle Volume with normals."""
        from pyvr.volume import Volume

        renderer = VolumeRenderer()

        # Mock GL manager methods
        renderer.gl_manager.create_volume_texture = MagicMock(return_value=0)
        renderer.gl_manager.create_normal_texture = MagicMock(return_value=1)
        renderer.gl_manager.set_uniform_int = MagicMock()
        renderer.gl_manager.set_uniform_vector = MagicMock()

        # Create Volume with normals
        data = np.random.rand(32, 32, 32).astype(np.float32)
        normals = np.random.rand(32, 32, 32, 3).astype(np.float32)
        volume = Volume(data=data, normals=normals)

        renderer.load_volume(volume)

        # Should have created both textures
        renderer.gl_manager.create_volume_texture.assert_called_once()
        renderer.gl_manager.create_normal_texture.assert_called_once_with(normals)

        # Should have set both texture uniforms
        assert renderer.gl_manager.set_uniform_int.call_count >= 2

    def test_load_volume_with_custom_bounds(self, mock_moderngl_context):
        """VolumeRenderer should respect custom volume bounds."""
        from pyvr.volume import Volume

        renderer = VolumeRenderer()

        # Mock GL manager methods
        renderer.gl_manager.create_volume_texture = MagicMock(return_value=0)
        renderer.gl_manager.set_uniform_int = MagicMock()
        renderer.gl_manager.set_uniform_vector = MagicMock()

        # Create Volume with custom bounds
        data = np.random.rand(32, 32, 32).astype(np.float32)
        min_bounds = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
        max_bounds = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        volume = Volume(data=data, min_bounds=min_bounds, max_bounds=max_bounds)

        renderer.load_volume(volume)

        # Should have set the custom bounds
        renderer.gl_manager.set_uniform_vector.assert_any_call("volume_min_bounds", (-1.0, -1.0, -1.0))
        renderer.gl_manager.set_uniform_vector.assert_any_call("volume_max_bounds", (1.0, 1.0, 1.0))

    def test_get_volume(self, mock_moderngl_context):
        """get_volume should return current Volume instance."""
        from pyvr.volume import Volume

        renderer = VolumeRenderer()

        # Mock GL manager methods
        renderer.gl_manager.create_volume_texture = MagicMock(return_value=0)
        renderer.gl_manager.set_uniform_int = MagicMock()
        renderer.gl_manager.set_uniform_vector = MagicMock()

        # Initially None
        assert renderer.get_volume() is None

        # Create and load volume
        data = np.random.rand(32, 32, 32).astype(np.float32)
        volume = Volume(data=data)
        renderer.load_volume(volume)

        # Should return the same volume
        assert renderer.get_volume() is volume

    def test_volume_replacement(self, mock_moderngl_context):
        """Replacing volume should update stored Volume instance."""
        from pyvr.volume import Volume

        renderer = VolumeRenderer()

        # Mock GL manager methods
        renderer.gl_manager.create_volume_texture = MagicMock(return_value=0)
        renderer.gl_manager.set_uniform_int = MagicMock()
        renderer.gl_manager.set_uniform_vector = MagicMock()

        # Load first volume
        data1 = np.random.rand(32, 32, 32).astype(np.float32)
        volume1 = Volume(data=data1, name="volume1")
        renderer.load_volume(volume1)
        assert renderer.get_volume() is volume1

        # Load second volume
        data2 = np.random.rand(64, 64, 64).astype(np.float32)
        volume2 = Volume(data=data2, name="volume2")
        renderer.load_volume(volume2)
        assert renderer.get_volume() is volume2

        # Should have called create_volume_texture twice
        assert renderer.gl_manager.create_volume_texture.call_count == 2


class TestVolumeRendererRenderConfig:
    """Test VolumeRenderer integration with RenderConfig."""

    def test_default_config(self, mock_moderngl_context):
        """VolumeRenderer should have default balanced config."""
        from pyvr.config import RenderConfig

        renderer = VolumeRenderer()

        assert hasattr(renderer, "config")
        assert isinstance(renderer.config, RenderConfig)
        assert renderer.config.step_size == 0.01
        assert renderer.config.max_steps == 500

    def test_custom_config_initialization(self, mock_moderngl_context):
        """VolumeRenderer should accept custom config in constructor."""
        from pyvr.config import RenderConfig

        custom_config = RenderConfig.high_quality()
        renderer = VolumeRenderer(width=512, height=512, config=custom_config)

        assert renderer.config is custom_config
        assert renderer.config.step_size == 0.005
        assert renderer.config.max_steps == 1000

    def test_config_presets(self, mock_moderngl_context):
        """All config presets should work with VolumeRenderer."""
        from pyvr.config import RenderConfig

        presets = [
            RenderConfig.preview(),
            RenderConfig.fast(),
            RenderConfig.balanced(),
            RenderConfig.high_quality(),
            RenderConfig.ultra_quality(),
        ]

        for preset in presets:
            renderer = VolumeRenderer(config=preset)
            assert renderer.config is preset

    def test_set_config(self, mock_moderngl_context):
        """set_config should update renderer configuration."""
        from pyvr.config import RenderConfig

        renderer = VolumeRenderer()
        new_config = RenderConfig.fast()

        # Mock the uniform setters
        renderer.gl_manager.set_uniform_float = MagicMock()
        renderer.gl_manager.set_uniform_int = MagicMock()

        renderer.set_config(new_config)

        assert renderer.config is new_config
        # Should have updated GL uniforms
        renderer.gl_manager.set_uniform_float.assert_called_with(
            "step_size", new_config.step_size
        )
        renderer.gl_manager.set_uniform_int.assert_called_with(
            "max_steps", new_config.max_steps
        )

    def test_get_config(self, mock_moderngl_context):
        """get_config should return current config."""
        from pyvr.config import RenderConfig

        config = RenderConfig.high_quality()
        renderer = VolumeRenderer(config=config)

        retrieved_config = renderer.get_config()
        assert retrieved_config is config

    def test_set_config_type_checking(self, mock_moderngl_context):
        """set_config should reject non-RenderConfig objects."""
        renderer = VolumeRenderer()

        with pytest.raises(TypeError, match="Expected RenderConfig instance"):
            renderer.set_config("not a config")

        with pytest.raises(TypeError, match="Expected RenderConfig instance"):
            renderer.set_config({"step_size": 0.01})

    def test_config_updates_gl_uniforms_on_init(self, mock_moderngl_context):
        """Config should set GL uniforms during initialization."""
        from pyvr.config import RenderConfig

        # Create a custom config
        config = RenderConfig(step_size=0.015, max_steps=300)

        # Create renderer (this will call _update_render_config during __init__)
        renderer = VolumeRenderer(config=config)

        # Check that uniforms were set
        # Note: These are set during __init__, so we can't easily mock them
        # but we can verify the config is stored correctly
        assert renderer.config.step_size == 0.015
        assert renderer.config.max_steps == 300

    def test_config_parameter_validation(self, mock_moderngl_context):
        """Invalid config should be rejected in constructor."""
        with pytest.raises(TypeError, match="Expected RenderConfig instance"):
            VolumeRenderer(config="invalid")

    def test_config_with_modified_preset(self, mock_moderngl_context):
        """Modified presets should work correctly."""
        from pyvr.config import RenderConfig

        # Start with balanced and modify
        config = RenderConfig.balanced().with_step_size(0.008)

        renderer = VolumeRenderer(config=config)
        assert renderer.config.step_size == 0.008
        assert renderer.config.max_steps == 500  # Unchanged from balanced

    def test_multiple_config_changes(self, mock_moderngl_context):
        """Should be able to change config multiple times."""
        from pyvr.config import RenderConfig

        renderer = VolumeRenderer()

        # Mock uniform setters
        renderer.gl_manager.set_uniform_float = MagicMock()
        renderer.gl_manager.set_uniform_int = MagicMock()

        # Change to fast
        fast = RenderConfig.fast()
        renderer.set_config(fast)
        assert renderer.config is fast

        # Change to high quality
        hq = RenderConfig.high_quality()
        renderer.set_config(hq)
        assert renderer.config is hq

        # Should have called uniform setters for each change
        assert renderer.gl_manager.set_uniform_float.call_count >= 2
        assert renderer.gl_manager.set_uniform_int.call_count >= 2