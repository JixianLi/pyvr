"""Tests for interface state management."""

import pytest
from pyvr.interface.state import InterfaceState


def test_interface_state_initialization():
    """Test default initialization."""
    state = InterfaceState()
    assert len(state.control_points) == 2
    assert state.control_points[0] == (0.0, 0.0)
    assert state.control_points[1] == (1.0, 1.0)
    assert state.selected_control_point is None
    assert state.current_colormap == "viridis"
    assert not state.is_dragging_camera
    assert not state.is_dragging_control_point


def test_add_control_point():
    """Test adding control points."""
    state = InterfaceState()
    state.add_control_point(0.5, 0.5)
    assert len(state.control_points) == 3
    assert (0.5, 0.5) in state.control_points
    assert state.needs_tf_update


def test_add_control_point_maintains_order():
    """Test control points are kept sorted."""
    state = InterfaceState()
    state.add_control_point(0.7, 0.3)
    state.add_control_point(0.3, 0.7)
    scalars = [cp[0] for cp in state.control_points]
    assert scalars == sorted(scalars)


def test_add_control_point_validation():
    """Test validation of control point values."""
    state = InterfaceState()
    with pytest.raises(ValueError):
        state.add_control_point(-0.1, 0.5)  # Scalar out of range
    with pytest.raises(ValueError):
        state.add_control_point(0.5, 1.5)  # Opacity out of range


def test_remove_control_point():
    """Test removing middle control points."""
    state = InterfaceState(control_points=[(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)])
    state.remove_control_point(1)
    assert len(state.control_points) == 2
    assert (0.5, 0.5) not in state.control_points


def test_cannot_remove_first_last_control_point():
    """Test first and last control points cannot be removed."""
    state = InterfaceState()
    with pytest.raises(ValueError):
        state.remove_control_point(0)
    with pytest.raises(ValueError):
        state.remove_control_point(1)


def test_update_control_point():
    """Test updating control point position."""
    state = InterfaceState(control_points=[(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)])
    state.update_control_point(1, 0.6, 0.8)
    assert state.control_points[1] == (0.6, 0.8)


def test_update_first_last_locks_x():
    """Test first and last control points have locked x positions."""
    state = InterfaceState()
    state.update_control_point(0, 0.5, 0.3)  # Try to change first x
    assert state.control_points[0][0] == 0.0  # X locked to 0.0
    assert state.control_points[0][1] == 0.3  # Opacity changed

    state.update_control_point(1, 0.5, 0.7)  # Try to change last x
    assert state.control_points[1][0] == 1.0  # X locked to 1.0
    assert state.control_points[1][1] == 0.7  # Opacity changed


def test_select_control_point():
    """Test control point selection."""
    state = InterfaceState()
    state.select_control_point(0)
    assert state.selected_control_point == 0
    state.select_control_point(None)
    assert state.selected_control_point is None


def test_set_colormap():
    """Test colormap changes."""
    state = InterfaceState()
    state.set_colormap("plasma")
    assert state.current_colormap == "plasma"
    assert state.needs_tf_update
    assert state.needs_render


def test_initial_state_validation():
    """Test validation during initialization."""
    with pytest.raises(ValueError):
        InterfaceState(control_points=[(0.0, 0.0)])  # Too few points

    with pytest.raises(ValueError):
        InterfaceState(control_points=[(0.0, 0.0), (1.5, 0.5)])  # Scalar out of range

    with pytest.raises(ValueError):
        InterfaceState(control_points=[(0.0, 0.0), (0.5, 1.5)])  # Opacity out of range


def test_remove_control_point_updates_selection():
    """Test that removing a control point updates selected index."""
    state = InterfaceState(control_points=[(0.0, 0.0), (0.3, 0.3), (0.5, 0.5), (1.0, 1.0)])
    state.select_control_point(1)

    # Remove the selected control point
    state.remove_control_point(1)
    assert state.selected_control_point is None

    # Test removing point before selected
    state = InterfaceState(control_points=[(0.0, 0.0), (0.3, 0.3), (0.5, 0.5), (1.0, 1.0)])
    state.select_control_point(2)
    state.remove_control_point(1)
    assert state.selected_control_point == 1  # Index shifted down


def test_update_control_point_maintains_order():
    """Test updating control point re-sorts when needed."""
    state = InterfaceState(control_points=[(0.0, 0.0), (0.3, 0.3), (0.5, 0.5), (1.0, 1.0)])

    # Move middle point past another
    state.update_control_point(1, 0.7, 0.3)  # Move 0.3 to 0.7

    scalars = [cp[0] for cp in state.control_points]
    assert scalars == sorted(scalars)


def test_control_points_sorted_on_init():
    """Test that control points are sorted during initialization."""
    state = InterfaceState(control_points=[(1.0, 1.0), (0.0, 0.0), (0.5, 0.5)])
    scalars = [cp[0] for cp in state.control_points]
    assert scalars == [0.0, 0.5, 1.0]


def test_select_invalid_index():
    """Test selecting invalid control point index."""
    state = InterfaceState()
    with pytest.raises(IndexError):
        state.select_control_point(5)
    with pytest.raises(IndexError):
        state.select_control_point(-1)


def test_needs_flags():
    """Test needs_render and needs_tf_update flags."""
    state = InterfaceState()

    # Initial state
    assert state.needs_render
    assert not state.needs_tf_update

    # Reset flags
    state.needs_render = False
    state.needs_tf_update = False

    # Add control point sets needs_tf_update
    state.add_control_point(0.5, 0.5)
    assert state.needs_tf_update

    # Update control point sets needs_tf_update
    state.needs_tf_update = False
    state.update_control_point(1, 0.6, 0.6)
    assert state.needs_tf_update

    # Set colormap sets both flags
    state.needs_render = False
    state.needs_tf_update = False
    state.set_colormap("hot")
    assert state.needs_render
    assert state.needs_tf_update
