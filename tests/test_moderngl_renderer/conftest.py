"""
Test configuration and fixtures for moderngl_renderer tests.
Provides mocking capabilities for OpenGL-dependent tests.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch, Mock

@pytest.fixture
def mock_moderngl_context():
    """
    Create a comprehensive mock of ModernGL context for testing.
    
    This fixture mocks all ModernGL operations to allow testing without
    actual OpenGL context creation, making tests runnable in CI/CD environments.
    """
    with patch('moderngl.create_context') as mock_create_context:
        # Create mock context
        mock_ctx = MagicMock()
        mock_create_context.return_value = mock_ctx
        
        # Mock texture creation
        mock_texture = MagicMock()
        mock_texture.filter = None
        mock_texture.repeat_x = False
        mock_texture.repeat_y = False
        mock_texture.repeat_z = False
        mock_texture.use = MagicMock()
        mock_texture.release = MagicMock()
        
        # Mock texture3d creation
        mock_texture3d = MagicMock()
        mock_texture3d.filter = None
        mock_texture3d.repeat_x = False
        mock_texture3d.repeat_y = False
        mock_texture3d.repeat_z = False
        mock_texture3d.use = MagicMock()
        mock_texture3d.release = MagicMock()
        
        # Mock depth texture
        mock_depth_texture = MagicMock()
        mock_depth_texture.release = MagicMock()
        
        # Mock framebuffer
        mock_framebuffer = MagicMock()
        mock_framebuffer.use = MagicMock()
        mock_framebuffer.read = MagicMock(return_value=b'\\x00' * (512 * 512 * 4))  # Mock pixel data
        mock_framebuffer.release = MagicMock()
        
        # Mock program
        mock_program = MagicMock()
        mock_program.release = MagicMock()
        
        # Create a more sophisticated uniform mock
        mock_uniform = MagicMock()
        mock_uniform.write = MagicMock()
        mock_program.__getitem__ = MagicMock(return_value=mock_uniform)
        mock_program.__setitem__ = MagicMock()
        
        # Mock buffer and vertex array
        mock_buffer = MagicMock()
        mock_buffer.release = MagicMock()
        
        mock_vao = MagicMock()
        mock_vao.render = MagicMock()
        mock_vao.release = MagicMock()
        
        # Configure context methods
        mock_ctx.texture.return_value = mock_texture
        mock_ctx.texture3d.return_value = mock_texture3d
        mock_ctx.depth_texture.return_value = mock_depth_texture
        mock_ctx.framebuffer.return_value = mock_framebuffer
        mock_ctx.program.return_value = mock_program
        mock_ctx.buffer.return_value = mock_buffer
        mock_ctx.vertex_array.return_value = mock_vao
        mock_ctx.clear = MagicMock()
        mock_ctx.enable = MagicMock()
        mock_ctx.release = MagicMock()
        
        yield {
            'ctx': mock_ctx,
            'texture': mock_texture,
            'texture3d': mock_texture3d,
            'depth_texture': mock_depth_texture,
            'framebuffer': mock_framebuffer,
            'program': mock_program,
            'buffer': mock_buffer,
            'vao': mock_vao
        }

@pytest.fixture
def sample_volume_data():
    """Create sample 3D volume data for testing."""
    return np.random.rand(32, 32, 32).astype(np.float32)

@pytest.fixture
def sample_normal_data():
    """Create sample 3D normal data for testing."""
    return np.random.rand(32, 32, 32, 3).astype(np.float32)

@pytest.fixture
def sample_color_lut():
    """Create sample color LUT data for testing."""
    return np.random.rand(256, 3).astype(np.float32)

@pytest.fixture
def sample_opacity_lut():
    """Create sample opacity LUT data for testing."""
    return np.random.rand(256).astype(np.float32)

@pytest.fixture
def mock_shader_files():
    """Mock shader file reading for tests that don't need actual shader content."""
    vertex_shader_content = """
#version 330 core
layout(location = 0) in vec2 position;
out vec2 uv;
void main() {
    uv = position * 0.5 + 0.5;
    gl_Position = vec4(position, 0.0, 1.0);
}
"""
    
    fragment_shader_content = """
#version 330 core
uniform sampler3D volume_texture;
uniform sampler2D transfer_function_lut;
in vec2 uv;
out vec4 color;
void main() {
    color = vec4(1.0, 0.0, 0.0, 1.0);
}
"""
    
    def mock_open(filename, mode='r'):
        mock_file = MagicMock()
        if 'vert' in filename:
            mock_file.read.return_value = vertex_shader_content
        elif 'frag' in filename:
            mock_file.read.return_value = fragment_shader_content
        else:
            mock_file.read.return_value = ""
        return mock_file
    
    with patch('builtins.open', mock_open):
        yield {
            'vertex_shader': vertex_shader_content,
            'fragment_shader': fragment_shader_content
        }

@pytest.fixture 
def mock_transfer_functions():
    """Create mock transfer function objects for testing."""
    # Mock ColorTransferFunction
    mock_ctf = MagicMock()
    mock_ctf.lut_size = 256
    
    # Mock OpacityTransferFunction
    mock_otf = MagicMock()
    mock_otf.lut_size = 256
    
    # Configure return values to be consistent
    def color_to_lut(size=None):
        effective_size = size if size is not None else mock_ctf.lut_size
        return np.random.rand(effective_size, 3).astype(np.float32)
    
    def opacity_to_lut(size=None):
        effective_size = size if size is not None else mock_otf.lut_size
        return np.random.rand(effective_size).astype(np.float32)
    
    mock_ctf.to_lut.side_effect = color_to_lut
    mock_otf.to_lut.side_effect = opacity_to_lut
    
    return mock_ctf, mock_otf