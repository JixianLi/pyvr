"""
Comprehensive tests for VolumeRenderer class.
Tests all functionality with mocked dependencies.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch, Mock

from pyvr.moderngl_renderer.renderer import VolumeRenderer
from pyvr.camera.parameters import CameraParameters


class TestVolumeRendererInitialization:
    """Test VolumeRenderer initialization and setup."""
    
    def test_init_default_parameters(self, mock_moderngl_context):
        """Test initialization with default parameters."""
        renderer = VolumeRenderer()
        
        assert renderer.width == 512
        assert renderer.height == 512
        assert renderer.step_size == 0.01
        assert renderer.max_steps == 200
        assert renderer.gl_manager is not None
        
    def test_init_custom_parameters(self, mock_moderngl_context):
        """Test initialization with custom parameters."""
        renderer = VolumeRenderer(width=1024, height=768, step_size=0.005, max_steps=500)
        
        assert renderer.width == 1024
        assert renderer.height == 768
        assert renderer.step_size == 0.005
        assert renderer.max_steps == 500


class TestVolumeRendererDataLoading:
    """Test volume data loading and texture creation."""
    
    def test_load_volume_3d_data(self, mock_moderngl_context, sample_volume_data):
        """Test loading 3D volume data."""
        renderer = VolumeRenderer()
        
        renderer.load_volume(sample_volume_data)
        
        # Should complete without error
        assert True
        
    def test_load_normal_volume(self, mock_moderngl_context, sample_normal_data):
        """Test loading normal volume data."""
        renderer = VolumeRenderer()
        
        renderer.load_normal_volume(sample_normal_data)
        
        # Should complete without error
        assert True


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
        renderer = VolumeRenderer()
        
        position = (1.0, 2.0, 3.0)
        target = (0.5, 0.5, 0.5)
        up = (0.0, 1.0, 0.0)
        
        renderer.set_camera(position, target, up)
        
        # Should complete without error
        assert True


class TestVolumeRendererRenderingParameters:
    """Test rendering parameter management."""
    
    def test_set_step_size(self, mock_moderngl_context):
        """Test setting ray marching step size."""
        renderer = VolumeRenderer()
        
        renderer.set_step_size(0.005)
        
        assert renderer.step_size == 0.005
        
    def test_set_max_steps(self, mock_moderngl_context):
        """Test setting maximum ray marching steps."""
        renderer = VolumeRenderer()
        
        renderer.set_max_steps(2000)
        
        assert renderer.max_steps == 2000
        
    def test_set_ambient_light(self, mock_moderngl_context):
        """Test setting ambient light parameter."""
        renderer = VolumeRenderer()
        
        renderer.set_ambient_light(0.5)
        
        assert renderer.ambient_light == 0.5
        
    def test_set_diffuse_light(self, mock_moderngl_context):
        """Test setting diffuse light parameter."""
        renderer = VolumeRenderer()
        
        renderer.set_diffuse_light(1.2)
        
        assert renderer.diffuse_light == 1.2
        
    def test_set_light_position(self, mock_moderngl_context):
        """Test setting light position."""
        renderer = VolumeRenderer()
        
        position = (2.0, 3.0, 4.0)
        renderer.set_light_position(position)
        
        assert np.allclose(renderer.light_position, position)
        
    def test_set_light_target(self, mock_moderngl_context):
        """Test setting light target."""
        renderer = VolumeRenderer()
        
        target = (0.5, 0.5, 0.5)
        renderer.set_light_target(target)
        
        assert np.allclose(renderer.light_target, target)


class TestVolumeRendererVolumeManagement:
    """Test volume bounds and positioning."""
    
    def test_set_volume_bounds(self, mock_moderngl_context):
        """Test setting volume bounds."""
        renderer = VolumeRenderer()
        
        min_bounds = (0.0, 0.0, 0.0)
        max_bounds = (1.0, 1.0, 1.0)
        
        renderer.set_volume_bounds(min_bounds, max_bounds)
        
        # Should complete without error
        assert True


class TestVolumeRendererRenderingProcess:
    """Test the complete rendering process."""
    
    def test_render_basic(self, mock_moderngl_context, sample_volume_data, mock_transfer_functions):
        """Test basic rendering pipeline."""
        renderer = VolumeRenderer()
        color_tf, opacity_tf = mock_transfer_functions
        
        # Setup volume and transfer functions
        renderer.load_volume(sample_volume_data)
        
        # Mock the manager methods that would be called during rendering
        renderer.gl_manager.create_volume_texture = MagicMock(return_value=0)
        renderer.gl_manager.create_rgba_transfer_function_texture = MagicMock(return_value=1)
        renderer.gl_manager.set_uniform_int = MagicMock()
        renderer.gl_manager.set_uniform_float = MagicMock()
        renderer.gl_manager.set_uniform_matrix = MagicMock()
        renderer.gl_manager.clear_framebuffer = MagicMock()
        renderer.gl_manager.setup_blending = MagicMock()
        renderer.gl_manager.render_quad = MagicMock()
        renderer.gl_manager.read_pixels = MagicMock(return_value=np.random.rand(512, 512, 4))
        
        renderer.set_transfer_functions(color_tf, opacity_tf)
        
        image = renderer.render()
        
        assert image is not None
        
    def test_render_to_pil(self, mock_moderngl_context, sample_volume_data, mock_transfer_functions):
        """Test rendering to PIL image."""
        renderer = VolumeRenderer()
        color_tf, opacity_tf = mock_transfer_functions
        
        # Setup volume and transfer functions
        renderer.load_volume(sample_volume_data)
        
        # Mock the manager methods
        renderer.gl_manager.create_volume_texture = MagicMock(return_value=0)
        renderer.gl_manager.create_rgba_transfer_function_texture = MagicMock(return_value=1)
        renderer.gl_manager.set_uniform_int = MagicMock()
        renderer.gl_manager.set_uniform_float = MagicMock()
        renderer.gl_manager.set_uniform_matrix = MagicMock()
        renderer.gl_manager.clear_framebuffer = MagicMock()
        renderer.gl_manager.setup_blending = MagicMock()
        renderer.gl_manager.render_quad = MagicMock()
        
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
        renderer = VolumeRenderer()
        color_tf, opacity_tf = mock_transfer_functions
        
        # Setup volume and transfer functions
        renderer.load_volume(sample_volume_data)
        renderer.set_transfer_functions(color_tf, opacity_tf)
        
        # Mock all manager operations
        renderer.gl_manager.create_volume_texture = MagicMock(return_value=0)
        renderer.gl_manager.create_rgba_transfer_function_texture = MagicMock(return_value=1)
        renderer.gl_manager.set_uniform_int = MagicMock()
        renderer.gl_manager.set_uniform_float = MagicMock()
        renderer.gl_manager.set_uniform_matrix = MagicMock()
        renderer.gl_manager.clear_framebuffer = MagicMock()
        renderer.gl_manager.setup_blending = MagicMock()
        renderer.gl_manager.render_quad = MagicMock()
        renderer.gl_manager.read_pixels = MagicMock(return_value=np.random.rand(512, 512, 4))
        
        # Render with different parameters
        renderer.set_step_size(0.01)
        image1 = renderer.render()
        
        renderer.set_step_size(0.005)
        image2 = renderer.render()
        
        assert image1 is not None
        assert image2 is not None
        
    def test_volume_data_replacement(self, mock_moderngl_context, mock_transfer_functions):
        """Test replacing volume data mid-session."""
        renderer = VolumeRenderer()
        color_tf, opacity_tf = mock_transfer_functions
        
        # Mock the manager create_volume_texture method BEFORE calling load_volume
        renderer.gl_manager.create_volume_texture = MagicMock(return_value=0)
        renderer.gl_manager.set_uniform_int = MagicMock()
        
        # Setup initial volume  
        volume1 = np.random.rand(16, 16, 16).astype(np.float32)
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
        volume2 = np.random.rand(32, 32, 32).astype(np.float32)
        renderer.load_volume(volume2)
        
        renderer.render()
        
        # Should have created volume texture twice (once for each load_volume call)
        assert renderer.gl_manager.create_volume_texture.call_count == 2


class TestVolumeRendererErrorHandling:
    """Test error handling and edge cases."""
    
    def test_parameter_validation(self, mock_moderngl_context):
        """Test parameter validation where it exists."""
        renderer = VolumeRenderer()
        
        # Test that setting parameters works without validation errors
        renderer.set_step_size(0.005)
        assert renderer.step_size == 0.005
        
        renderer.set_max_steps(1000)
        assert renderer.max_steps == 1000
        
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
        renderer = VolumeRenderer()
        
        # Test loading very small volume
        tiny_volume = np.random.random((1, 1, 1)).astype(np.float32)
        renderer.gl_manager.create_volume_texture = MagicMock(return_value=0)
        renderer.gl_manager.set_uniform_int = MagicMock()
        
        renderer.load_volume(tiny_volume)
        renderer.gl_manager.create_volume_texture.assert_called_once()
        
        # Test invalid volume dimensions
        invalid_volume = np.random.random((10, 10))  # 2D instead of 3D
        with pytest.raises(ValueError, match="Volume data must be 3D"):
            renderer.load_volume(invalid_volume)
    
    def test_load_normal_volume_edge_cases(self, mock_moderngl_context):
        """Test edge cases for normal volume loading.""" 
        renderer = VolumeRenderer()
        
        # Test valid normal volume
        normal_volume = np.random.random((5, 5, 5, 3)).astype(np.float32)
        renderer.gl_manager.create_normal_texture = MagicMock(return_value=1)
        renderer.gl_manager.set_uniform_int = MagicMock()
        
        renderer.load_normal_volume(normal_volume)
        renderer.gl_manager.create_normal_texture.assert_called_once()
        
        # Test invalid normal volume (wrong channel count)
        invalid_normal = np.random.random((5, 5, 5, 2))  # 2 channels instead of 3
        with pytest.raises(ValueError, match="Normal volume must have 3 channels"):
            renderer.load_normal_volume(invalid_normal)
    
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
        
        # Set camera using position, target, up vectors
        renderer.set_camera(position=[0, 0, 3], target=[0, 0, 0], up=[0, 1, 0])
        
        # Should not crash
        pixels = renderer.render()
        assert pixels is not None
        
    def test_parameter_validation_comprehensive(self, mock_moderngl_context):
        """Test comprehensive parameter validation."""
        renderer = VolumeRenderer()
        
        # Mock the GL operations to avoid actual OpenGL calls
        renderer.gl_manager.set_uniform_float = MagicMock()
        renderer.gl_manager.set_uniform_int = MagicMock()
        renderer.gl_manager.set_uniform_vector = MagicMock()
        
        # Test setting various parameters
        renderer.set_step_size(0.001)
        assert renderer.step_size == 0.001
        
        renderer.set_max_steps(2000)
        assert renderer.max_steps == 2000
        
        renderer.set_ambient_light(0.3)
        assert renderer.ambient_light == 0.3
        
        renderer.set_diffuse_light(0.8)
        assert renderer.diffuse_light == 0.8
        
        renderer.set_light_position(np.array([2.0, 2.0, 2.0]))
        assert np.allclose(renderer.light_position, [2.0, 2.0, 2.0])
        
        renderer.set_light_target(np.array([1.0, 1.0, 1.0]))
        assert np.allclose(renderer.light_target, [1.0, 1.0, 1.0])
        
        # Test volume bounds (doesn't store as attributes but should call GL manager)
        renderer.set_volume_bounds(np.array([0.0, 0.0, 0.0]), np.array([2.0, 2.0, 2.0]))
        renderer.gl_manager.set_uniform_vector.assert_called()
        renderer.gl_manager.set_uniform_int.assert_called_once()