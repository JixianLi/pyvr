"""State management for interactive interface."""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class InterfaceState:
    """
    Manages state for the interactive volume renderer interface.

    This class centralizes all mutable state to make the interface
    easier to reason about and test.

    Attributes:
        control_points: List of (scalar, opacity) tuples for opacity transfer function
        selected_control_point: Index of currently selected control point (None if none selected)
        current_colormap: Name of currently selected matplotlib colormap
        is_dragging_camera: Whether user is currently dragging to orbit camera
        is_dragging_control_point: Whether user is currently dragging a control point
        drag_start_pos: (x, y) position where drag started (in axes coordinates)
        camera_control_mode: Current camera control mode ('trackball' or 'orbit')
        needs_render: Flag indicating volume needs to be re-rendered
        needs_tf_update: Flag indicating transfer function needs update
        show_fps: Flag indicating whether FPS counter should be displayed
        current_preset_name: Name of currently selected RenderConfig preset
        light_linked_to_camera: Flag indicating whether light is linked to camera movement
    """

    # Transfer function state
    control_points: List[Tuple[float, float]] = field(
        default_factory=lambda: [(0.0, 0.0), (1.0, 1.0)]
    )
    selected_control_point: Optional[int] = None
    current_colormap: str = "viridis"

    # Interaction state
    is_dragging_camera: bool = False
    is_dragging_control_point: bool = False
    drag_start_pos: Optional[Tuple[float, float]] = None
    camera_control_mode: str = "trackball"  # or 'orbit'

    # Update flags
    needs_render: bool = True
    needs_tf_update: bool = False

    # Display flags
    show_fps: bool = True
    show_histogram: bool = True

    # Rendering configuration
    current_preset_name: str = "fast"  # Default to fast for interactivity

    # Light linking
    light_linked_to_camera: bool = False

    # Automatic quality adjustment
    auto_quality_enabled: bool = True
    saved_preset_name: Optional[str] = None  # For restoring after interaction

    def __post_init__(self):
        """Validate initial state."""
        if len(self.control_points) < 2:
            raise ValueError("Must have at least 2 control points")

        # Ensure control points are sorted by scalar value
        self.control_points = sorted(self.control_points, key=lambda cp: cp[0])

        # Validate control points are in [0, 1] range
        for scalar, opacity in self.control_points:
            if not (0.0 <= scalar <= 1.0):
                raise ValueError(f"Control point scalar {scalar} out of range [0, 1]")
            if not (0.0 <= opacity <= 1.0):
                raise ValueError(f"Control point opacity {opacity} out of range [0, 1]")

    def add_control_point(self, scalar: float, opacity: float) -> None:
        """
        Add a control point and maintain sorted order.

        Args:
            scalar: Scalar value in [0, 1]
            opacity: Opacity value in [0, 1]

        Raises:
            ValueError: If scalar or opacity out of range
        """
        if not (0.0 <= scalar <= 1.0):
            raise ValueError(f"Scalar {scalar} out of range [0, 1]")
        if not (0.0 <= opacity <= 1.0):
            raise ValueError(f"Opacity {opacity} out of range [0, 1]")

        self.control_points.append((scalar, opacity))
        self.control_points = sorted(self.control_points, key=lambda cp: cp[0])
        self.needs_tf_update = True

    def remove_control_point(self, index: int) -> None:
        """
        Remove a control point by index.

        Args:
            index: Index of control point to remove

        Raises:
            ValueError: If trying to remove first or last control point
            IndexError: If index out of range
        """
        if index == 0 or index == len(self.control_points) - 1:
            raise ValueError("Cannot remove first or last control point")
        if index < 0 or index >= len(self.control_points):
            raise IndexError(f"Control point index {index} out of range")

        del self.control_points[index]
        if self.selected_control_point == index:
            self.selected_control_point = None
        elif (
            self.selected_control_point is not None
            and self.selected_control_point > index
        ):
            self.selected_control_point -= 1

        self.needs_tf_update = True

    def update_control_point(self, index: int, scalar: float, opacity: float) -> None:
        """
        Update a control point's position and opacity.

        For first and last control points, only opacity can be changed.

        Args:
            index: Index of control point to update
            scalar: New scalar value in [0, 1]
            opacity: New opacity value in [0, 1]

        Raises:
            ValueError: If scalar or opacity out of range
            IndexError: If index out of range
        """
        if index < 0 or index >= len(self.control_points):
            raise IndexError(f"Control point index {index} out of range")
        if not (0.0 <= opacity <= 1.0):
            raise ValueError(f"Opacity {opacity} out of range [0, 1]")

        # Lock first and last control points to x=0.0 and x=1.0
        if index == 0:
            scalar = 0.0
        elif index == len(self.control_points) - 1:
            scalar = 1.0
        else:
            if not (0.0 <= scalar <= 1.0):
                raise ValueError(f"Scalar {scalar} out of range [0, 1]")

        self.control_points[index] = (scalar, opacity)

        # Re-sort if middle control points changed
        if index != 0 and index != len(self.control_points) - 1:
            self.control_points = sorted(self.control_points, key=lambda cp: cp[0])
            # Update selected index if it changed due to re-sorting
            if self.selected_control_point == index:
                self.selected_control_point = self.control_points.index(
                    (scalar, opacity)
                )

        self.needs_tf_update = True

    def select_control_point(self, index: Optional[int]) -> None:
        """
        Select a control point by index.

        Args:
            index: Index to select, or None to deselect
        """
        if index is not None and (index < 0 or index >= len(self.control_points)):
            raise IndexError(f"Control point index {index} out of range")
        self.selected_control_point = index

    def set_colormap(self, colormap_name: str) -> None:
        """
        Change the current colormap.

        Args:
            colormap_name: Name of matplotlib colormap
        """
        self.current_colormap = colormap_name
        self.needs_tf_update = True
        self.needs_render = True

    def set_preset(self, preset_name: str) -> None:
        """
        Change the current rendering preset.

        Args:
            preset_name: Name of RenderConfig preset

        Raises:
            ValueError: If preset_name is not valid
        """
        valid_presets = ["preview", "fast", "balanced", "high_quality", "ultra_quality"]
        if preset_name not in valid_presets:
            raise ValueError(
                f"Invalid preset '{preset_name}'. Choose from: {valid_presets}"
            )

        self.current_preset_name = preset_name
        self.needs_render = True
