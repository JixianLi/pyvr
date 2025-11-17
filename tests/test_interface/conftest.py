"""Shared fixtures for interface tests."""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock

from pyvr.volume import Volume
from pyvr.datasets import create_sample_volume


@pytest.fixture
def sample_volume():
    """Create a small sample volume for testing."""
    data = create_sample_volume(32, "sphere")
    return Volume(data=data)


@pytest.fixture
def mock_renderer():
    """Create a mock renderer for testing without OpenGL."""
    renderer = Mock()
    renderer.render_to_pil.return_value = Mock()
    renderer.render_to_pil.return_value.__array__ = lambda: np.zeros(
        (512, 512, 3), dtype=np.uint8
    )
    return renderer


@pytest.fixture
def mock_axes():
    """Create a mock matplotlib axes."""
    ax = MagicMock()
    ax.figure = MagicMock()
    ax.figure.canvas = MagicMock()
    return ax
