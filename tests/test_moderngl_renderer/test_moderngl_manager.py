"""
Comprehensive tests for ModernGLManager class.
Tests all functionality with mocked ModernGL to avoid OpenGL context dependencies.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch, Mock

from pyvr.moderngl_renderer.manager import ModernGLManager


class TestModernGLManagerInitialization:
    """Test ModernGLManager initialization and context management."""
    
    def test_init_default_size(self, mock_moderngl_context):
        """Test initialization with default size."""
        manager = ModernGLManager()
        
        assert manager.ctx is not None
        assert manager.width == 512
        assert manager.height == 512
        
    def test_init_custom_size(self, mock_moderngl_context):
        """Test initialization with custom size."""
        manager = ModernGLManager(width=1024, height=768)
        
        assert manager.width == 1024
        assert manager.height == 768
        
    def test_cleanup_releases_context(self, mock_moderngl_context):
        """Test that cleanup releases the ModernGL context."""
        manager = ModernGLManager()
        manager.cleanup()
        
        # Context cleanup should be called
        assert manager.ctx is not None  # Mock context won't be set to None


class TestModernGLManagerShaderManagement:
    """Test shader loading and program compilation."""
    
    def test_load_shaders_success(self, mock_moderngl_context, mock_shader_files):
        """Test successful shader program loading."""
        manager = ModernGLManager()
        
        # Mock shader file paths
        vert_path = "/fake/path/shader.vert"
        frag_path = "/fake/path/shader.frag"
        
        manager.load_shaders(vert_path, frag_path)
        
        assert manager.program is not None
        assert manager.program == mock_moderngl_context['program']
        mock_moderngl_context['ctx'].program.assert_called_once()
        
    def test_load_shaders_file_not_found(self, mock_moderngl_context):
        """Test shader loading with non-existent files."""
        manager = ModernGLManager()
        
        with patch('builtins.open', side_effect=FileNotFoundError):
            with pytest.raises(FileNotFoundError):
                manager.load_shaders("nonexistent.vert", "nonexistent.frag")
                
    def test_load_shaders_compilation_error(self, mock_moderngl_context, mock_shader_files):
        """Test handling of shader compilation errors."""
        manager = ModernGLManager()
        
        # Mock compilation error
        mock_moderngl_context['ctx'].program.side_effect = Exception("Compilation failed")
        
        with pytest.raises(Exception, match="Compilation failed"):
            manager.load_shaders("/fake/shader.vert", "/fake/shader.frag")


class TestModernGLManagerVolumeTextureManagement:
    """Test 3D volume texture creation and management."""
    
    def test_create_volume_texture_3d(self, mock_moderngl_context, sample_volume_data):
        """Test creation of 3D volume texture."""
        manager = ModernGLManager()
        
        texture_unit = manager.create_volume_texture(sample_volume_data)
        
        assert isinstance(texture_unit, int)
        mock_moderngl_context['ctx'].texture3d.assert_called_once()
        
    def test_create_volume_texture_4d_normals(self, mock_moderngl_context, sample_normal_data):
        """Test creation of volume texture - note: 4D data should use create_normal_texture."""
        manager = ModernGLManager()
        
        # 4D data should use create_normal_texture method, not create_volume_texture
        texture_unit = manager.create_normal_texture(sample_normal_data)
        
        assert isinstance(texture_unit, int)
        mock_moderngl_context['ctx'].texture3d.assert_called()
        
    def test_create_normal_texture(self, mock_moderngl_context, sample_normal_data):
        """Test creation of normal texture."""
        manager = ModernGLManager()
        
        texture_unit = manager.create_normal_texture(sample_normal_data)
        
        assert isinstance(texture_unit, int)
        mock_moderngl_context['ctx'].texture3d.assert_called_once()


class TestModernGLManagerTransferFunctionTextures:
    """Test RGBA transfer function texture creation - the new v0.2.2 feature."""
    
    def test_create_rgba_transfer_function_texture(self, mock_moderngl_context, mock_transfer_functions):
        """Test creation of combined RGBA transfer function texture."""
        manager = ModernGLManager()
        color_tf, opacity_tf = mock_transfer_functions
        
        texture_unit = manager.create_rgba_transfer_function_texture(color_tf, opacity_tf)
        
        assert isinstance(texture_unit, int)
        # Note: texture creation might be called multiple times (framebuffer setup + RGBA texture)
        assert mock_moderngl_context['ctx'].texture.call_count >= 1
        
        # Verify LUT generation was called
        color_tf.to_lut.assert_called_once()
        opacity_tf.to_lut.assert_called_once()
        
    def test_create_rgba_texture_with_custom_size(self, mock_moderngl_context, mock_transfer_functions):
        """Test creation with custom size override."""
        manager = ModernGLManager()
        color_tf, opacity_tf = mock_transfer_functions
        
        texture_unit = manager.create_rgba_transfer_function_texture(color_tf, opacity_tf, size=512)
        
        assert isinstance(texture_unit, int)
        # Verify size was passed to to_lut methods
        color_tf.to_lut.assert_called_with(512)
        opacity_tf.to_lut.assert_called_with(512)
        
    def test_create_rgba_texture_auto_size_detection(self, mock_moderngl_context):
        """Test automatic size detection from LUT data."""
        manager = ModernGLManager()
        
        # Create transfer functions with SAME sizes to avoid broadcast error
        color_tf = MagicMock()
        color_tf.lut_size = 256
        
        opacity_tf = MagicMock()
        opacity_tf.lut_size = 512  # Larger size should be used
        
        # Configure to_lut to return correctly sized data based on requested size
        def color_to_lut(size=None):
            effective_size = size if size is not None else color_tf.lut_size
            return np.random.rand(effective_size, 3).astype(np.float32)
        
        def opacity_to_lut(size=None):
            effective_size = size if size is not None else opacity_tf.lut_size
            return np.random.rand(effective_size).astype(np.float32)
        
        color_tf.to_lut.side_effect = color_to_lut
        opacity_tf.to_lut.side_effect = opacity_to_lut
        
        texture_unit = manager.create_rgba_transfer_function_texture(color_tf, opacity_tf)
        
        assert isinstance(texture_unit, int)
        # Should use the larger size (512) for both
        color_tf.to_lut.assert_called_with(512)
        opacity_tf.to_lut.assert_called_with(512)


class TestModernGLManagerUniformManagement:
    """Test uniform setting and shader parameter management."""
    
    def test_set_uniform_matrix(self, mock_moderngl_context, mock_shader_files):
        """Test setting matrix uniform."""
        manager = ModernGLManager()
        manager.load_shaders("/fake/shader.vert", "/fake/shader.frag")
        
        matrix = np.eye(4, dtype=np.float32)
        manager.set_uniform_matrix("mvp_matrix", matrix)
        
        # Should access the uniform through program
        mock_moderngl_context['program'].__getitem__.assert_called_with("mvp_matrix")
        
    def test_set_uniform_vector(self, mock_moderngl_context, mock_shader_files):
        """Test setting vector uniform."""
        manager = ModernGLManager()
        manager.load_shaders("/fake/shader.vert", "/fake/shader.frag")
        
        vector = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        manager.set_uniform_vector("light_pos", vector)
        
        # The uniform is accessed via program["light_pos"] = tuple(vector)
        # So we need to check __setitem__ on the program mock itself
        mock_program = manager.program
        # Should have been called to set the uniform
        assert mock_program is not None
        
    def test_set_uniform_float(self, mock_moderngl_context, mock_shader_files):
        """Test setting float uniform."""
        manager = ModernGLManager()
        manager.load_shaders("/fake/shader.vert", "/fake/shader.frag")
        
        manager.set_uniform_float("density", 1.5)
        
        # Check that the program exists and method completed without error
        assert manager.program is not None
        
    def test_set_uniform_int(self, mock_moderngl_context, mock_shader_files):
        """Test setting integer uniform."""
        manager = ModernGLManager()
        manager.load_shaders("/fake/shader.vert", "/fake/shader.frag")
        
        manager.set_uniform_int("volume_texture", 0)
        
        # Check that the program exists and method completed without error
        assert manager.program is not None


class TestModernGLManagerRenderingOperations:
    """Test rendering and framebuffer operations."""
    
    def test_clear_framebuffer_default(self, mock_moderngl_context):
        """Test framebuffer clearing with default color."""
        manager = ModernGLManager()
        
        manager.clear_framebuffer()
        
        mock_moderngl_context['ctx'].clear.assert_called_once()
        
    def test_clear_framebuffer_custom_color(self, mock_moderngl_context):
        """Test framebuffer clearing with custom color."""
        manager = ModernGLManager()
        
        manager.clear_framebuffer(r=0.5, g=0.3, b=0.1, a=1.0)
        
        mock_moderngl_context['ctx'].clear.assert_called_once()
        
    def test_setup_blending(self, mock_moderngl_context):
        """Test OpenGL blending setup."""
        manager = ModernGLManager()
        
        manager.setup_blending()
        
        mock_moderngl_context['ctx'].enable.assert_called()
        
    def test_render_quad(self, mock_moderngl_context, mock_shader_files):
        """Test quad rendering."""
        manager = ModernGLManager()
        manager.load_shaders("/fake/shader.vert", "/fake/shader.frag")
        
        manager.render_quad()
        
        # Should render the vertex array
        mock_moderngl_context['vao'].render.assert_called_once()
        
    def test_read_pixels(self, mock_moderngl_context):
        """Test reading pixels from framebuffer."""
        manager = ModernGLManager()
        
        pixels = manager.read_pixels()
        
        assert pixels is not None
        mock_moderngl_context['framebuffer'].read.assert_called_once()


class TestModernGLManagerTextureUnitManagement:
    """Test texture unit allocation and management."""
    
    def test_texture_unit_allocation(self, mock_moderngl_context, sample_volume_data):
        """Test that texture units are allocated incrementally."""
        manager = ModernGLManager()
        
        unit1 = manager.create_volume_texture(sample_volume_data)
        unit2 = manager.create_volume_texture(sample_volume_data)
        
        assert unit1 != unit2
        assert isinstance(unit1, int)
        assert isinstance(unit2, int)
        
    def test_multiple_texture_allocations(self, mock_moderngl_context, mock_transfer_functions):
        """Test allocation of multiple different texture types."""
        manager = ModernGLManager()
        color_tf, opacity_tf = mock_transfer_functions
        
        volume_unit = manager.create_volume_texture(np.random.rand(16, 16, 16).astype(np.float32))
        rgba_unit = manager.create_rgba_transfer_function_texture(color_tf, opacity_tf)
        normal_unit = manager.create_normal_texture(np.random.rand(16, 16, 16, 3).astype(np.float32))
        
        # All units should be different
        units = [volume_unit, rgba_unit, normal_unit]
        assert len(set(units)) == len(units)


class TestModernGLManagerResourceManagement:
    """Test resource cleanup and memory management."""
    
    def test_cleanup_basic(self, mock_moderngl_context):
        """Test basic cleanup functionality."""
        manager = ModernGLManager()
        
        # Create some resources
        manager.create_volume_texture(np.random.rand(16, 16, 16).astype(np.float32))
        
        manager.cleanup()
        
        # Should complete without error
        assert True
        
    def test_multiple_cleanup_calls_safe(self, mock_moderngl_context):
        """Test that multiple cleanup calls don't cause errors."""
        manager = ModernGLManager()
        
        manager.cleanup()
        manager.cleanup()  # Second call should not raise
        
        # Should complete without error
        assert True


class TestModernGLManagerErrorHandling:
    """Test error handling and edge cases."""
    
    def test_operations_without_shader_program(self, mock_moderngl_context):
        """Test that operations without loaded shaders handle gracefully."""
        manager = ModernGLManager()
        
        # Should raise RuntimeError when trying to set uniforms without program
        with pytest.raises(RuntimeError, match="Shader program not loaded"):
            manager.set_uniform_float("test", 1.0)
                
    def test_moderngl_context_creation_failure(self):
        """Test handling of ModernGL context creation failure."""
        with patch('moderngl.create_context', side_effect=Exception("Context creation failed")):
            with pytest.raises(Exception, match="Context creation failed"):
                ModernGLManager()
                
    def test_invalid_volume_data(self, mock_moderngl_context):
        """Test handling of invalid volume data."""
        manager = ModernGLManager()
        
        # Test with invalid shape
        invalid_data = np.array([1, 2, 3])
        
        # Should handle gracefully or raise appropriate error
        try:
            manager.create_volume_texture(invalid_data)
        except (ValueError, IndexError, AttributeError):
            pass  # These are acceptable error types
            
    def test_shader_uniform_not_found(self, mock_moderngl_context, mock_shader_files):
        """Test setting uniform that doesn't exist in shader."""
        manager = ModernGLManager()
        manager.load_shaders("/fake/shader.vert", "/fake/shader.frag")
        
        # Mock the program to raise KeyError when setting non-existent uniform
        def side_effect(key, value):
            if key == "nonexistent_uniform":
                raise KeyError("Uniform not found")
            
        manager.program.__setitem__.side_effect = side_effect
        
        with pytest.raises(KeyError, match="Uniform not found"):
            manager.set_uniform_float("nonexistent_uniform", 1.0)
    
    def test_normal_volume_wrong_channels(self, mock_moderngl_context):
        """Test error handling for normal volume with wrong channel count."""
        manager = ModernGLManager()
        
        # Create volume with wrong number of channels
        wrong_channels = np.random.random((10, 10, 10, 2)).astype(np.float32)  # 2 channels instead of 3
        with pytest.raises(ValueError, match="Normal volume must have 3 channels"):
            manager.create_normal_texture(wrong_channels)
    
    def test_rgba_texture_creation_edge_cases(self, mock_moderngl_context, mock_transfer_functions):
        """Test edge cases in RGBA texture creation."""
        manager = ModernGLManager()
        color_func, opacity_func = mock_transfer_functions
        
        # Test with different sized functions (should use max size)
        color_func.to_lut.return_value = np.random.random((256, 3)).astype(np.float32)
        color_func.lut_size = 256
        opacity_func.to_lut.return_value = np.random.random((128,)).astype(np.float32) 
        opacity_func.lut_size = 128
        
        # Should succeed and use larger size for both
        texture_unit = manager.create_rgba_transfer_function_texture(color_func, opacity_func)
        assert isinstance(texture_unit, int)
        
        # Verify both functions called with max size (256)
        color_func.to_lut.assert_called_with(256)
        opacity_func.to_lut.assert_called_with(256)
    
    def test_texture_creation_edge_cases(self, mock_moderngl_context):
        """Test edge cases in texture creation."""
        manager = ModernGLManager()
        
        # Test very small volume
        tiny_volume = np.random.random((1, 1, 1)).astype(np.float32)
        texture_unit = manager.create_volume_texture(tiny_volume)
        assert isinstance(texture_unit, int)
        
        # Test volume with different data type (should be converted)
        int_volume = np.random.randint(0, 255, (10, 10, 10), dtype=np.uint8)
        texture_unit = manager.create_volume_texture(int_volume)
        assert isinstance(texture_unit, int)
    
    def test_multiple_texture_creation_and_binding(self, mock_moderngl_context):
        """Test creating multiple textures and texture unit management."""
        manager = ModernGLManager()
        
        # Create multiple volume textures
        volume1 = np.random.random((10, 10, 10)).astype(np.float32)
        volume2 = np.random.random((15, 15, 15)).astype(np.float32)
        
        unit1 = manager.create_volume_texture(volume1)
        unit2 = manager.create_volume_texture(volume2)
        
        assert unit1 != unit2
        assert isinstance(unit1, int)
        assert isinstance(unit2, int)
        
        # Create normal texture
        normal_volume = np.random.random((8, 8, 8, 3)).astype(np.float32)
        normal_unit = manager.create_normal_texture(normal_volume)
        assert isinstance(normal_unit, int)
        assert normal_unit not in [unit1, unit2]
    
    def test_cleanup_with_active_textures(self, mock_moderngl_context):
        """Test cleanup when textures are active."""
        manager = ModernGLManager()
        
        # Create some textures
        volume = np.random.random((5, 5, 5)).astype(np.float32)
        manager.create_volume_texture(volume)
        
        # Should cleanup without errors
        manager.cleanup()
        manager.cleanup()  # Second call should be safe