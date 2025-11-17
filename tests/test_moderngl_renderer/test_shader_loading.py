"""
Test shader loading with new shared shader directory.
This test ensures Phase 1 refactor didn't break shader loading.
"""

import pytest
import os
import sys

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def test_shader_files_exist():
    """Test that shader files exist in the new shared location."""
    project_root = os.path.join(os.path.dirname(__file__), "..", "..")
    shader_dir = os.path.join(project_root, "pyvr", "shaders")

    vertex_shader = os.path.join(shader_dir, "volume.vert.glsl")
    fragment_shader = os.path.join(shader_dir, "volume.frag.glsl")

    assert os.path.exists(vertex_shader), f"Vertex shader not found at {vertex_shader}"
    assert os.path.exists(
        fragment_shader
    ), f"Fragment shader not found at {fragment_shader}"


def test_shader_path_resolution():
    """Test that the volume renderer can resolve shader paths correctly."""
    project_root = os.path.join(os.path.dirname(__file__), "..", "..")

    # Simulate the path resolution logic from volume_renderer.py
    moderngl_renderer_dir = os.path.join(project_root, "pyvr", "moderngl_renderer")
    pyvr_dir = os.path.dirname(moderngl_renderer_dir)
    shader_dir = os.path.join(pyvr_dir, "shaders")

    vertex_shader_path = os.path.join(shader_dir, "volume.vert.glsl")
    fragment_shader_path = os.path.join(shader_dir, "volume.frag.glsl")

    assert os.path.exists(
        vertex_shader_path
    ), "Volume renderer cannot resolve vertex shader path"
    assert os.path.exists(
        fragment_shader_path
    ), "Volume renderer cannot resolve fragment shader path"


def test_shader_content_not_empty():
    """Test that shader files have content."""
    project_root = os.path.join(os.path.dirname(__file__), "..", "..")
    shader_dir = os.path.join(project_root, "pyvr", "shaders")

    vertex_shader = os.path.join(shader_dir, "volume.vert.glsl")
    fragment_shader = os.path.join(shader_dir, "volume.frag.glsl")

    with open(vertex_shader, "r") as f:
        vertex_content = f.read().strip()
    assert len(vertex_content) > 0, "Vertex shader is empty"
    assert "#version" in vertex_content, "Vertex shader missing version directive"

    with open(fragment_shader, "r") as f:
        fragment_content = f.read().strip()
    assert len(fragment_content) > 0, "Fragment shader is empty"
    assert "#version" in fragment_content, "Fragment shader missing version directive"
