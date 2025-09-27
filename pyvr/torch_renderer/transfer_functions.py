import numpy as np
import torch
import math
import matplotlib
import matplotlib.pyplot as plt

class OpacityTransferFunction:
    def __init__(self, control_points=None, resolution=256, device='cpu'):
        """
        Initialize opacity transfer function from control points.

        Args:
            control_points: List of (scalar_value, opacity) tuples. If None, defaults to linear [0,1]
            resolution: Number of points in the opacity function
            device: Device to run computations on
        """
        self.resolution = resolution
        self.device = device
        self.scalar_range = torch.linspace(0, 1, resolution, device=device)

        # Set default control points if none provided (linear 0 to 1)
        if control_points is None:
            control_points = [(0.0, 0.0), (1.0, 1.0)]

        self.set_control_points(control_points)

    def set_control_points(self, control_points):
        """
        Set opacity transfer function from control points.

        Args:
            control_points: List of (scalar_value, opacity) tuples
        """
        # Ensure we have at least 2 control points
        if len(control_points) < 2:
            raise ValueError("Need at least 2 control points")

        # Sort control points by scalar value
        control_points = sorted(control_points)

        # Validate control points
        for scalar_val, opacity in control_points:
            if not (0.0 <= scalar_val <= 1.0):
                raise ValueError(
                    f"Scalar value {scalar_val} must be in range [0, 1]")
            if not (0.0 <= opacity <= 1.0):
                raise ValueError(
                    f"Opacity value {opacity} must be in range [0, 1]")

        self.control_points = control_points
        self._interpolate_values()
        return self

    def _interpolate_values(self):
        """Interpolate opacity values from control points."""
        scalars = torch.tensor(
            [cp[0] for cp in self.control_points], device=self.device)
        opacities = torch.tensor(
            [cp[1] for cp in self.control_points], device=self.device)

        # Use PyTorch's interpolation for efficiency
        self.values = torch.zeros(self.resolution, device=self.device)

        for i in range(self.resolution):
            scalar_val = i / (self.resolution - 1)

            # Find surrounding control points
            if scalar_val <= scalars[0]:
                self.values[i] = opacities[0]
            elif scalar_val >= scalars[-1]:
                self.values[i] = opacities[-1]
            else:
                # Linear interpolation between control points
                for j in range(len(scalars) - 1):
                    if scalars[j] <= scalar_val <= scalars[j + 1]:
                        t = (scalar_val - scalars[j]) / \
                            (scalars[j + 1] - scalars[j])
                        self.values[i] = opacities[j] * \
                            (1 - t) + opacities[j + 1] * t
                        break

    def add_control_point(self, scalar_value, opacity):
        """Add a new control point and re-interpolate."""
        if not (0.0 <= scalar_value <= 1.0):
            raise ValueError(
                f"Scalar value {scalar_value} must be in range [0, 1]")
        if not (0.0 <= opacity <= 1.0):
            raise ValueError(
                f"Opacity value {opacity} must be in range [0, 1]")

        self.control_points.append((scalar_value, opacity))
        self.control_points = sorted(self.control_points)
        self._interpolate_values()
        return self

    def remove_control_point(self, index):
        """Remove a control point by index."""
        if len(self.control_points) <= 2:
            raise ValueError(
                "Cannot remove control point: need at least 2 control points")
        if not (0 <= index < len(self.control_points)):
            raise ValueError(f"Index {index} out of range")

        self.control_points.pop(index)
        self._interpolate_values()
        return self

    def get_control_points(self):
        """Get current control points."""
        return self.control_points.copy()

    @classmethod
    def linear(cls, min_opacity=0.0, max_opacity=1.0, resolution=256, device='cpu'):
        """Create a linear opacity transfer function."""
        control_points = [(0.0, min_opacity), (1.0, max_opacity)]
        return cls(control_points, resolution, device)

    @classmethod
    def step(cls, threshold=0.5, low_opacity=0.0, high_opacity=1.0, resolution=256, device='cpu'):
        """Create a step function opacity transfer function."""
        # Create a sharp step using two very close control points
        epsilon = 1e-6
        control_points = [
            (0.0, low_opacity),
            (threshold - epsilon, low_opacity),
            (threshold + epsilon, high_opacity),
            (1.0, high_opacity)
        ]
        return cls(control_points, resolution, device)

    @classmethod
    def ramp(cls, start=0.2, end=0.8, max_opacity=1.0, resolution=256, device='cpu'):
        """Create a ramp function opacity transfer function."""
        control_points = [
            (0.0, 0.0),
            (start, 0.0),
            (end, max_opacity),
            (1.0, max_opacity)
        ]
        return cls(control_points, resolution, device)

    @classmethod
    def bell_curve(cls, center=0.5, width=0.3, max_opacity=1.0, resolution=256, device='cpu'):
        """Create a bell curve (Gaussian-like) opacity transfer function."""

        # Create multiple points to approximate a bell curve
        n_points = 9
        control_points = []

        for i in range(n_points):
            scalar_val = i / (n_points - 1)
            # Gaussian-like function
            distance = abs(scalar_val - center) / width
            opacity = max_opacity * math.exp(-distance**2)
            control_points.append((scalar_val, opacity))

        return cls(control_points, resolution, device)

    def apply(self, scalar_values, value_range=None):
        """
        Apply opacity transfer function to scalar values.

        Args:
            scalar_values: Tensor of scalar values
            value_range: Optional (min, max) range for normalization

        Returns:
            opacity_values: Tensor of opacity values [0, 1]
        """
        if value_range is None:
            value_range = (scalar_values.min(), scalar_values.max())

        # Avoid division by zero
        if value_range[1] == value_range[0]:
            return torch.zeros_like(scalar_values)

        # Normalize scalar values to [0, 1]
        normalized = (scalar_values -
                      value_range[0]) / (value_range[1] - value_range[0])
        normalized = torch.clamp(normalized, 0, 1)

        # Map to opacity transfer function indices
        indices = (normalized * (self.resolution - 1)).long()
        indices = torch.clamp(indices, 0, self.resolution - 1)

        return self.values[indices]

    def plot(self, ax=None, title="Opacity Transfer Function", show_control_points=True, color_transfer_function=None):
        """
        Plot the current opacity transfer function as a line plot.

        Args:
            ax: Matplotlib axis to plot on. If None, creates new figure
            title: Title for the plot
            show_control_points: Whether to show control points as markers
            color_transfer_function: Optional ColorTransferFunction to show colorbar
        """
        # Convert to numpy for plotting
        scalar_vals = self.scalar_range.cpu().numpy()
        opacity_vals = self.values.cpu().numpy()

        ax.plot(scalar_vals, opacity_vals, 'b-', linewidth=2, label='Opacity Transfer Function')

        if show_control_points:
            cp_scalars = [cp[0] for cp in self.control_points]
            cp_opacities = [cp[1] for cp in self.control_points]
            ax.scatter(cp_scalars, cp_opacities, color='red', s=50, zorder=5, label='Control Points')

        ax.set_xlabel('Scalar Value')
        ax.set_ylabel('Opacity')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend()
        
        return ax

    def plot_opacity_bar(self, ax, title="Opacity Transfer Function"):
        """
        Plot opacity transfer function as a horizontal bar.
        
        Args:
            ax: Matplotlib axis to plot on
            title: Title for the opacity bar
        """
        opacity_vals = self.values.cpu().numpy()
        
        # Create grayscale visualization based on opacity values
        opacity_colors = np.zeros((1, len(opacity_vals), 3))
        for i, opacity in enumerate(opacity_vals):
            opacity_colors[0, i] = [opacity, opacity, opacity]
        
        ax.imshow(opacity_colors, aspect='auto', extent=[0, 1, 0, 1])
        ax.set_xlim(0, 1)
        ax.set_title(title)
        ax.set_ylabel('Opacity')
        ax.set_yticks([])
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        
        # Optionally show control points as markers
        if hasattr(self, 'control_points'):
            cp_scalars = [cp[0] for cp in self.control_points]
            cp_opacities = [cp[1] for cp in self.control_points]
            # Map control points to the colorbar
            for scalar, opacity in zip(cp_scalars, cp_opacities):
                ax.axvline(x=scalar, color='red', linestyle='--', alpha=0.7)
        
        return ax

class ColorTransferFunction:
    def __init__(self, colormap='viridis', resolution=256, device='cpu'):
        """
        Initialize color transfer function.
        
        Args:
            colormap: Name of colormap ('viridis', 'hot', 'cool', 'plasma', etc.)
            resolution: Number of points in the color function
            device: Device to run computations on
        """
        self.resolution = resolution
        self.device = device
        self.colormap_name = colormap
        self._build_colormap()
    
    def _build_colormap(self):
        """Build RGB color lookup table from matplotlib colormap."""
        try:
            # Get matplotlib colormap
            cmap = matplotlib.colormaps.get_cmap(self.colormap_name)
            
            # Sample colormap at regular intervals
            x = np.linspace(0, 1, self.resolution)
            colors = cmap(x)[:, :3]  # RGB only (no alpha)
            
            # Convert to torch tensor
            self.colors = torch.tensor(colors, dtype=torch.float32, device=self.device)
            
        except:
            # Fallback to simple grayscale if matplotlib colormap fails
            print(f"Warning: Could not load colormap '{self.colormap_name}', using grayscale")
            gray_values = torch.linspace(0, 1, self.resolution, device=self.device)
            self.colors = gray_values.unsqueeze(1).repeat(1, 3)
    
    def set_custom_colors(self, color_points):
        """
        Set custom color mapping from control points.
        
        Args:
            color_points: List of (scalar_value, (r, g, b)) tuples
        """
        color_points = sorted(color_points)
        self.colors = torch.zeros(self.resolution, 3, device=self.device)
        
        for i in range(self.resolution):
            scalar_val = i / (self.resolution - 1)
            
            # Find surrounding color points
            if scalar_val <= color_points[0][0]:
                self.colors[i] = torch.tensor(color_points[0][1], device=self.device)
            elif scalar_val >= color_points[-1][0]:
                self.colors[i] = torch.tensor(color_points[-1][1], device=self.device)
            else:
                # Linear interpolation between color points
                for j in range(len(color_points) - 1):
                    if color_points[j][0] <= scalar_val <= color_points[j + 1][0]:
                        t = (scalar_val - color_points[j][0]) / (color_points[j + 1][0] - color_points[j][0])
                        color1 = torch.tensor(color_points[j][1], device=self.device)
                        color2 = torch.tensor(color_points[j + 1][1], device=self.device)
                        self.colors[i] = color1 * (1 - t) + color2 * t
                        break
        return self
    
    def apply(self, scalar_values, value_range=None):
        """
        Apply color mapping to scalar values.
        
        Args:
            scalar_values: Tensor of scalar values
            value_range: Optional (min, max) range for normalization
            
        Returns:
            color_values: Tensor of RGB color values [..., 3]
        """
        if value_range is None:
            value_range = (scalar_values.min(), scalar_values.max())
        
        # Avoid division by zero
        if value_range[1] == value_range[0]:
            return torch.zeros(*scalar_values.shape, 3, device=self.device)
        
        # Normalize scalar values to [0, 1]
        normalized = (scalar_values - value_range[0]) / (value_range[1] - value_range[0])
        normalized = torch.clamp(normalized, 0, 1)
        
        # Map to color lookup table indices
        indices = (normalized * (self.resolution - 1)).long()
        indices = torch.clamp(indices, 0, self.resolution - 1)
        
        return self.colors[indices]

    def plot_colorbar(self, ax, title="Color Transfer Function"):
        """
        Plot color transfer function as a horizontal colorbar.
        
        Args:
            ax: Matplotlib axis to plot on
            title: Title for the colorbar
        """
        colors = self.colors.cpu().numpy()
        colors_reshaped = colors.reshape(1, -1, 3)
        
        ax.imshow(colors_reshaped, aspect='auto', extent=[0, 1, 0, 1])
        ax.set_xlim(0, 1)
        ax.set_title(title)
        ax.set_ylabel('Color')
        ax.set_yticks([])
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        return ax