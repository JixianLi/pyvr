import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from camera import Camera
from transfer_function import OpacityTransferFunction, ColorTransferFunction

class VolumeRenderer:
    def __init__(self, volume_shape, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the volume renderer.

        Args:
            volume_shape: Tuple (z, y, x) representing the 3D volume dimensions
            device: Device to run computations on
        """
        self.volume_shape = volume_shape
        self.device = device

        # Default transfer functions
        self.opacity_transfer_function = OpacityTransferFunction(device=device)
        self.color_transfer_function = ColorTransferFunction(device=device)
        
        # Default lighting parameters
        self.light_direction = torch.tensor([1.0, 1.0, 1.0], device=device)
        self.light_direction = self.light_direction / torch.norm(self.light_direction)
        self.ambient_intensity = 0.3
        self.diffuse_intensity = 0.7

    def set_opacity_transfer_function(self, opacity_transfer_function):
        """Set custom opacity transfer function."""
        self.opacity_transfer_function = opacity_transfer_function
        return self
    
    def set_color_transfer_function(self, color_transfer_function):
        """Set custom color transfer function."""
        self.color_transfer_function = color_transfer_function
        return self
    
    def set_lighting(self, light_direction=(1.0, 1.0, 1.0), ambient=0.3, diffuse=0.7):
        """
        Set lighting parameters.
        
        Args:
            light_direction: (x, y, z) direction of light source
            ambient: Ambient lighting intensity [0, 1]
            diffuse: Diffuse lighting intensity [0, 1]
        """
        self.light_direction = torch.tensor(light_direction, device=self.device, dtype=torch.float32)
        self.light_direction = self.light_direction / torch.norm(self.light_direction)
        self.ambient_intensity = ambient
        self.diffuse_intensity = diffuse
        return self

    def compute_gradient(self, volume, points):
        """
        Compute volume gradient at given 3D points using finite differences.
        
        Args:
            volume: 3D tensor (Z, Y, X)
            points: Points to sample (N, 3) in world coordinates [0, volume_shape]
            
        Returns:
            gradients: (N, 3) gradient vectors
        """
        Z, Y, X = volume.shape
        N = points.shape[0]
        
        # Small offset for finite differences
        h = 1.0
        
        # Create offset points for gradient computation
        offsets = torch.tensor([
            [h, 0, 0], [-h, 0, 0],  # x direction
            [0, h, 0], [0, -h, 0],  # y direction  
            [0, 0, h], [0, 0, -h]   # z direction
        ], device=self.device, dtype=torch.float32)
        
        # Expand points for all offset directions
        points_expanded = points.unsqueeze(1).expand(-1, 6, -1)  # (N, 6, 3)
        offset_points = points_expanded + offsets.unsqueeze(0)   # (N, 6, 3)
        
        # Clamp to volume bounds
        offset_points[:, :, 0] = torch.clamp(offset_points[:, :, 0], 0, Z-1)
        offset_points[:, :, 1] = torch.clamp(offset_points[:, :, 1], 0, Y-1)
        offset_points[:, :, 2] = torch.clamp(offset_points[:, :, 2], 0, X-1)
        
        # Sample volume at offset points
        offset_points_flat = offset_points.reshape(-1, 3)
        sampled_values = self.sample_volume(volume, offset_points_flat)
        sampled_values = sampled_values.reshape(N, 6)
        
        # Compute gradients using central differences
        grad_x = (sampled_values[:, 0] - sampled_values[:, 1]) / (2 * h)
        grad_y = (sampled_values[:, 2] - sampled_values[:, 3]) / (2 * h)
        grad_z = (sampled_values[:, 4] - sampled_values[:, 5]) / (2 * h)
        
        gradients = torch.stack([grad_x, grad_y, grad_z], dim=1)  # (N, 3)
        
        return gradients

    def compute_normals(self, gradients):
        """
        Compute normalized surface normals from gradients.
        
        Args:
            gradients: (N, 3) gradient vectors
            
        Returns:
            normals: (N, 3) normalized normal vectors
        """
        # Compute gradient magnitude
        grad_magnitude = torch.norm(gradients, dim=1, keepdim=True)
        
        # Avoid division by zero
        grad_magnitude = torch.clamp(grad_magnitude, min=1e-8)
        
        # Normalize gradients to get normals (negative because gradients point toward increasing values)
        normals = -gradients / grad_magnitude
        
        return normals

    def compute_diffuse_lighting(self, normals):
        """
        Compute diffuse lighting intensity based on surface normals.
        
        Args:
            normals: (N, 3) normalized normal vectors
            
        Returns:
            lighting: (N,) diffuse lighting intensities [0, 1]
        """
        # Compute dot product between normals and light direction
        dot_product = torch.sum(normals * self.light_direction.unsqueeze(0), dim=1)
        
        # Clamp to positive values (no negative lighting)
        diffuse = torch.clamp(dot_product, min=0.0)
        
        # Combine ambient and diffuse lighting
        lighting = self.ambient_intensity + self.diffuse_intensity * diffuse
        
        # Clamp final lighting to [0, 1]
        lighting = torch.clamp(lighting, 0.0, 1.0)
        
        return lighting

    def sample_volume(self, volume, points):
        """
        Sample volume at given 3D points using trilinear interpolation.

        Args:
            volume: 3D tensor (Z, Y, X)
            points: Points to sample (N, 3) in world coordinates [0, volume_shape]

        Returns:
            sampled_values: (N,) sampled values
        """
        Z, Y, X = volume.shape

        # Normalize coordinates to [-1, 1] for grid_sample
        norm_points = points.clone()
        norm_points[:, 0] = 2 * (points[:, 0] / (Z - 1)) - 1  # z
        norm_points[:, 1] = 2 * (points[:, 1] / (Y - 1)) - 1  # y
        norm_points[:, 2] = 2 * (points[:, 2] / (X - 1)) - 1  # x

        # Reshape for grid_sample: (1, 1, 1, N, 3)
        grid = norm_points.unsqueeze(0).unsqueeze(0).unsqueeze(0)

        # Add batch and channel dimensions to volume: (1, 1, Z, Y, X)
        volume_batch = volume.unsqueeze(0).unsqueeze(0)

        # Sample using trilinear interpolation
        sampled = F.grid_sample(
            volume_batch, grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )

        return sampled.squeeze()

    def ray_march(self, volume, ray_origins, ray_directions, near=0.1, far=None, n_samples=128,
                  value_range=None, use_opacity=True, use_color=True, use_lighting=True):
        """
        Perform ray marching through the volume with vectorized alpha compositing.

        Args:
            volume: 3D tensor (Z, Y, X)
            ray_origins: (H, W, 3) ray origins
            ray_directions: (H, W, 3) ray directions
            near: Near clipping distance
            far: Far clipping distance (auto if None)
            n_samples: Number of samples per ray
            value_range: Optional tuple (min, max) for normalization
            use_opacity: Whether to use opacity transfer function
            use_color: Whether to use color transfer function
            use_lighting: Whether to use gradient-based lighting

        Returns:
            rendered_image: (H, W) grayscale or (H, W, 3) RGB rendered image
        """
        H, W = ray_origins.shape[:2]

        # Auto-calculate far distance if not provided
        if far is None:
            max_dim = max(self.volume_shape)
            far = max_dim * 1.5

        # Generate sample distances along rays
        t_vals = torch.linspace(near, far, n_samples, device=self.device)
        t_vals = t_vals.expand(H, W, n_samples)

        # Calculate step size for proper alpha compositing
        step_size = (far - near) / n_samples

        # Calculate 3D sample points
        # (H, W, n_samples, 3)
        sample_points = (ray_origins.unsqueeze(2) +
                         ray_directions.unsqueeze(2) * t_vals.unsqueeze(-1))

        # Flatten for volume sampling
        points_flat = sample_points.reshape(-1, 3)

        # Sample volume at all points
        sampled_values = self.sample_volume(volume, points_flat)

        # Compute lighting if requested
        if use_lighting:
            # Compute gradients and normals
            gradients = self.compute_gradient(volume, points_flat)
            normals = self.compute_normals(gradients)
            lighting_intensity = self.compute_diffuse_lighting(normals)
            
            # Reshape lighting back to (H, W, n_samples)
            lighting_intensity = lighting_intensity.reshape(H, W, n_samples)
        else:
            lighting_intensity = torch.ones(H, W, n_samples, device=self.device)

        # Reshape back to (H, W, n_samples)
        sampled_values = sampled_values.reshape(H, W, n_samples)

        if use_opacity:
            # Alpha compositing with opacity transfer function
            
            # Get opacity values
            opacity_values = self.opacity_transfer_function.apply(sampled_values, value_range)
            
            # Convert opacity to alpha (accounting for step size)
            alpha = 1 - torch.exp(-opacity_values * step_size)
            
            if use_color:
                # Get color values (H, W, n_samples, 3)
                color_values = self.color_transfer_function.apply(sampled_values, value_range)
                
                # Apply lighting to colors
                if use_lighting:
                    color_values = color_values * lighting_intensity.unsqueeze(-1)
                
                # Vectorized alpha compositing with color
                rendered_image = self._vectorized_alpha_composite_color(color_values, alpha)
            else:
                # Grayscale values with lighting
                if use_lighting:
                    lit_values = sampled_values * lighting_intensity
                else:
                    lit_values = sampled_values
                
                # Vectorized alpha compositing for grayscale
                rendered_image = self._vectorized_alpha_composite_grayscale(lit_values, alpha)
        else:
            # Simple average intensity (fallback)
            rendered_image = torch.mean(sampled_values, dim=-1)
            
            if use_lighting:
                avg_lighting = torch.mean(lighting_intensity, dim=-1)
                rendered_image = rendered_image * avg_lighting
            
            if use_color:
                # Apply color mapping to average
                rendered_image = self.color_transfer_function.apply(rendered_image, value_range)

        return rendered_image

    def render(self, volume, camera, image_size=(256, 256), near=0.1, far=None, n_samples=128,
               value_range=None, use_opacity=True, use_color=True, use_lighting=True):
        """
        Complete volume rendering pipeline.

        Args:
            volume: 3D tensor (Z, Y, X) to render
            camera: Camera object
            image_size: Output image size (height, width)
            near: Near clipping distance
            far: Far clipping distance
            n_samples: Number of samples per ray
            value_range: Optional tuple (min, max) for normalization
            use_opacity: Whether to use opacity transfer function
            use_color: Whether to use color transfer function
            use_lighting: Whether to use gradient-based lighting

        Returns:
            rendered_image: (H, W) grayscale or (H, W, 3) RGB rendered image
        """
        volume = volume.to(self.device)

        # Generate rays from camera
        ray_origins, ray_directions = camera.generate_rays(image_size)

        # Perform ray marching
        rendered_image = self.ray_march(
            volume, ray_origins, ray_directions, near, far, n_samples,
            value_range=value_range, use_opacity=use_opacity, use_color=use_color, use_lighting=use_lighting
        )

        return rendered_image

    def _vectorized_alpha_composite_color(self, colors, alpha):
        """
        Vectorized front-to-back alpha compositing for color values.
        
        Args:
            colors: (H, W, n_samples, 3) color values
            alpha: (H, W, n_samples) alpha values
            
        Returns:
            rendered_image: (H, W, 3) composited color image
        """
        H, W, n_samples, _ = colors.shape
        
        # Compute transmittance: T(i) = prod(1 - alpha[0:i])
        # Use log-space for numerical stability
        log_transmittance = torch.cumsum(torch.log(1 - alpha + 1e-8), dim=2)
        transmittance = torch.exp(log_transmittance)
        
        # Shift transmittance: T(i-1) for front-to-back compositing
        transmittance_shifted = torch.cat([
            torch.ones(H, W, 1, device=self.device),
            transmittance[:, :, :-1]
        ], dim=2)
        
        # Compute weights: w(i) = alpha(i) * T(i-1)
        weights = alpha * transmittance_shifted  # (H, W, n_samples)
        
        # Expand weights for color channels
        weights_expanded = weights.unsqueeze(-1)  # (H, W, n_samples, 1)
        
        # Weighted sum of colors
        rendered_image = torch.sum(weights_expanded * colors, dim=2)  # (H, W, 3)
        
        return rendered_image

    def _vectorized_alpha_composite_grayscale(self, values, alpha):
        """
        Vectorized front-to-back alpha compositing for grayscale values.
        
        Args:
            values: (H, W, n_samples) grayscale values
            alpha: (H, W, n_samples) alpha values
            
        Returns:
            rendered_image: (H, W) composited grayscale image
        """
        H, W, n_samples = values.shape
        
        # Compute transmittance: T(i) = prod(1 - alpha[0:i])
        # Use log-space for numerical stability
        log_transmittance = torch.cumsum(torch.log(1 - alpha + 1e-8), dim=2)
        transmittance = torch.exp(log_transmittance)
        
        # Shift transmittance: T(i-1) for front-to-back compositing
        transmittance_shifted = torch.cat([
            torch.ones(H, W, 1, device=self.device),
            transmittance[:, :, :-1]
        ], dim=2)
        
        # Compute weights: w(i) = alpha(i) * T(i-1)
        weights = alpha * transmittance_shifted  # (H, W, n_samples)
        
        # Weighted sum of values
        rendered_image = torch.sum(weights * values, dim=2)  # (H, W)
        
        return rendered_image

    def _vectorized_alpha_composite_with_early_termination(self, colors, alpha, termination_threshold=0.99):
        """
        Vectorized alpha compositing with early ray termination for better performance.
        
        Args:
            colors: (H, W, n_samples, 3) color values  
            alpha: (H, W, n_samples) alpha values
            termination_threshold: Stop when accumulated alpha exceeds this value
            
        Returns:
            rendered_image: (H, W, 3) composited color image
        """
        H, W, n_samples = alpha.shape
        
        # Find early termination points for each ray
        accumulated_alpha = torch.zeros(H, W, device=self.device)
        final_colors = torch.zeros(H, W, 3, device=self.device)
        active_rays = torch.ones(H, W, dtype=torch.bool, device=self.device)
        
        # Process samples in chunks for better memory efficiency
        chunk_size = min(32, n_samples)
        
        for start_idx in range(0, n_samples, chunk_size):
            end_idx = min(start_idx + chunk_size, n_samples)
            chunk_size_actual = end_idx - start_idx
            
            if not active_rays.any():
                break
                
            # Get current chunk
            alpha_chunk = alpha[active_rays, start_idx:end_idx]  # (active_rays, chunk_size)
            colors_chunk = colors[active_rays, start_idx:end_idx]  # (active_rays, chunk_size, 3)
            
            # Vectorized compositing for this chunk
            chunk_colors = self._vectorized_alpha_composite_color(
                colors_chunk.unsqueeze(0), 
                alpha_chunk.unsqueeze(0)
            ).squeeze(0)
            
            # Update accumulated values
            weight = 1 - accumulated_alpha[active_rays]
            final_colors[active_rays] += chunk_colors * weight.unsqueeze(-1)
            
            # Update accumulated alpha
            chunk_alpha_sum = 1 - torch.prod(1 - alpha_chunk + 1e-8, dim=1)
            accumulated_alpha[active_rays] += chunk_alpha_sum * weight
            
            # Update active rays
            active_rays = active_rays & (accumulated_alpha < termination_threshold)
        
        return final_colors

    # Add a method to choose compositing strategy
    def _alpha_composite(self, colors, alpha, method='vectorized'):
        """
        Choose alpha compositing method.
        
        Args:
            colors: Color values (H, W, n_samples, 3) or (H, W, n_samples)
            alpha: (H, W, n_samples) alpha values
            method: 'vectorized', 'early_termination', or 'sequential'
            
        Returns:
            rendered_image: Composited image
        """
        if method == 'vectorized':
            if colors.dim() == 4:  # Color
                return self._vectorized_alpha_composite_color(colors, alpha)
            else:  # Grayscale
                return self._vectorized_alpha_composite_grayscale(colors, alpha)
        elif method == 'early_termination':
            return self._vectorized_alpha_composite_with_early_termination(colors, alpha)
        elif method == 'sequential':
            # Fallback to original sequential method
            return self._sequential_alpha_composite(colors, alpha)
        else:
            raise ValueError(f"Unknown compositing method: {method}")

    def _sequential_alpha_composite(self, colors, alpha):
        """Original sequential alpha compositing for comparison."""
        H, W, n_samples = alpha.shape
        
        if colors.dim() == 4:  # Color
            accumulated_color = torch.zeros(H, W, 3, device=self.device)
        else:  # Grayscale
            accumulated_color = torch.zeros(H, W, device=self.device)
            
        accumulated_alpha = torch.zeros(H, W, device=self.device)
        
        for i in range(n_samples):
            if colors.dim() == 4:
                current_color = colors[:, :, i]  # (H, W, 3)
            else:
                current_color = colors[:, :, i]  # (H, W)
                
            current_alpha = alpha[:, :, i]  # (H, W)
            
            # Composite with accumulated values
            weight = current_alpha * (1 - accumulated_alpha)
            
            if colors.dim() == 4:
                weight = weight.unsqueeze(-1)  # (H, W, 1) for broadcasting
                
            accumulated_color += weight * current_color
            accumulated_alpha += current_alpha * (1 - accumulated_alpha)
        
        return accumulated_color


# Simplified performance demo
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time
    from volume_data import create_test_volume

    # Configuration variables
    DEVICE = 'cpu'
    VOLUME_SIZE = 256
    CAMERA_DISTANCE_FACTOR = 1.5  # Multiplier for camera distance from volume center
    IMAGE_WIDTH = 256
    IMAGE_HEIGHT = 256
    N_SAMPLES = 1000
    CAMERA_FOV = 45

    # VOLUME_TYPE
    VOLUME_TYPE = 'torus'  # Options: 'simple', 'complex', 'medical_phantom'
    
    # Lighting parameters
    LIGHT_DIRECTION = (1.0, 1.0, 1.0)
    AMBIENT_INTENSITY = 0.2
    DIFFUSE_INTENSITY = 0.8
    
    # Transfer function parameters
    COLORMAP = 'hot' 
    OPACITY_MIN = 0.0
    OPACITY_MAX = 0.1
    
    # Rendering options
    USE_OPACITY = True
    USE_COLOR = True
    USE_LIGHTING = True
    VALUE_RANGE = (0, 1)
    
    # Display options
    FIGURE_SIZE = (8, 8)

    print("Volume Renderer Performance Test")
    print("=" * 40)

    # Create renderer with configurable volume size
    volume_shape = (VOLUME_SIZE, VOLUME_SIZE, VOLUME_SIZE)
    renderer = VolumeRenderer(volume_shape=volume_shape, device=DEVICE)
    print(f"Device: {renderer.device}")
    print(f"Volume size: {VOLUME_SIZE}³ = {VOLUME_SIZE**3:,} voxels")

    # Create camera with configurable parameters
    volume_center = VOLUME_SIZE // 2
    camera_distance = VOLUME_SIZE * CAMERA_DISTANCE_FACTOR
    camera = Camera(
        position=(camera_distance, camera_distance//2, camera_distance),
        target=(volume_center, volume_center, volume_center),
        fov=CAMERA_FOV,
        device=renderer.device
    )

    # Create test volume
    print("Creating test volume...")
    volume = create_test_volume(volume_type=VOLUME_TYPE, shape=volume_shape)

    # Set up rendering parameters
    renderer.set_lighting(
        light_direction=LIGHT_DIRECTION, 
        ambient=AMBIENT_INTENSITY, 
        diffuse=DIFFUSE_INTENSITY
    )
    renderer.set_color_transfer_function(
        ColorTransferFunction(COLORMAP, device=renderer.device)
    )
    renderer.set_opacity_transfer_function(
        OpacityTransferFunction.linear(OPACITY_MIN, OPACITY_MAX, device=renderer.device)
    )

    # Render image with configurable parameters
    print(f"\nRendering {IMAGE_WIDTH}×{IMAGE_HEIGHT} image...")
    start_time = time.time()
    
    rendered = renderer.render(
        volume=volume,
        camera=camera,
        image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        n_samples=N_SAMPLES,
        use_opacity=USE_OPACITY,
        use_color=USE_COLOR,
        use_lighting=USE_LIGHTING,
        value_range=VALUE_RANGE
    )
    
    end_time = time.time()
    render_time_ms = (end_time - start_time) * 1000
    
    # Calculate performance metrics
    total_samples = IMAGE_WIDTH * IMAGE_HEIGHT * N_SAMPLES
    samples_per_ms = total_samples / render_time_ms
    
    print(f"\nPerformance Results:")
    print(f"Render time: {render_time_ms:.1f} ms")
    print(f"Total samples: {total_samples:,}")
    print(f"Samples/ms: {samples_per_ms:,.0f}")
    print(f"Resolution: {IMAGE_WIDTH}×{IMAGE_HEIGHT} pixels")
    print(f"Ray samples: {N_SAMPLES} per ray")
    print(f"Volume size: {VOLUME_SIZE}³ = {VOLUME_SIZE**3:,} voxels")

    # Display the result with transfer function maps
    fig = plt.figure(figsize=(FIGURE_SIZE[0], FIGURE_SIZE[1] * 1.5))
    
    # Create a grid layout: main image on top, transfer functions below
    gs = fig.add_gridspec(3, 1, height_ratios=[4, 0.5, 0.5], hspace=0.3)
    
    # Main rendered image
    ax_main = fig.add_subplot(gs[0])
    ax_main.imshow(rendered.cpu().numpy())
    ax_main.set_title(f'Volume Rendering ({VOLUME_SIZE}³ volume)\n{IMAGE_WIDTH}×{IMAGE_HEIGHT}, {N_SAMPLES} samples, {render_time_ms:.1f}ms')
    ax_main.axis('off')
    
    # Color transfer function map
    ax_color = fig.add_subplot(gs[1])
    color_tf = renderer.color_transfer_function
    colors = color_tf.colors.cpu().numpy()
    colors_reshaped = colors.reshape(1, -1, 3)
    ax_color.imshow(colors_reshaped, aspect='auto', extent=[0, 1, 0, 1])
    ax_color.set_xlim(0, 1)
    ax_color.set_title(f'Color Transfer Function ({COLORMAP})')
    ax_color.set_ylabel('Color')
    ax_color.set_yticks([])
    ax_color.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_color.set_xticklabels([])  # Remove x-labels from color map
    
    # Opacity transfer function map
    ax_opacity = fig.add_subplot(gs[2])
    opacity_tf = renderer.opacity_transfer_function
    opacity_tf.plot(ax=ax_opacity)
    
    plt.show()