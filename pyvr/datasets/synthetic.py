"""Synthetic Volume Dataset Generation

This module combines volume generation functionality from both the PyTorch 
and ModernGL renderers, providing a unified interface for creating synthetic 
3D volume datasets.
"""

import numpy as np
import torch


def create_test_volume(shape=(64, 64, 64), volume_type='complex', backend='torch'):
    """Create test volumes with various patterns.
    
    Args:
        shape: Tuple of (depth, height, width) for volume dimensions
        volume_type: Type of volume to create ('sphere', 'complex', 'turbulence', 
                    'torus', 'helix')
        backend: Backend to use ('torch' or 'numpy')
        
    Returns:
        Volume data as torch.Tensor or np.ndarray depending on backend
    """
    z, y, x = shape
    
    if backend == 'torch':
        zz, yy, xx = torch.meshgrid(
            torch.arange(z, dtype=torch.float32),
            torch.arange(y, dtype=torch.float32), 
            torch.arange(x, dtype=torch.float32),
            indexing='ij'
        )
        volume = torch.zeros(shape, dtype=torch.float32)
    else:
        zz, yy, xx = np.meshgrid(
            np.arange(z, dtype=np.float32),
            np.arange(y, dtype=np.float32), 
            np.arange(x, dtype=np.float32),
            indexing='ij'
        )
        volume = np.zeros(shape, dtype=np.float32)
    
    if volume_type == 'sphere':
        # Simple sphere
        if backend == 'torch':
            center = torch.tensor([z//2, y//2, x//2], dtype=torch.float32)
            radius = min(shape) // 4
            distance = torch.sqrt((zz - center[0])**2 + 
                                 (yy - center[1])**2 + 
                                 (xx - center[2])**2)
            volume = torch.exp(-(distance - radius)**2 / (2 * (radius/3)**2))
        else:
            center = np.array([z//2, y//2, x//2], dtype=np.float32)
            radius = min(shape) // 4
            distance = np.sqrt((zz - center[0])**2 + 
                              (yy - center[1])**2 + 
                              (xx - center[2])**2)
            volume = np.exp(-(distance - radius)**2 / (2 * (radius/3)**2))
        
    elif volume_type == 'complex':
        # Complex volume with multiple features
        if backend == 'torch':
            volume = torch.zeros(shape, dtype=torch.float32)
            
            # Central bright sphere
            center1 = torch.tensor([z//2, y//2, x//2], dtype=torch.float32)
            radius1 = min(shape) // 6
            dist1 = torch.sqrt((zz - center1[0])**2 + 
                              (yy - center1[1])**2 + 
                              (xx - center1[2])**2)
            sphere1 = torch.exp(-(dist1 - radius1)**2 / (2 * (radius1/2)**2))
            volume += sphere1 * 1.0
            
            # Secondary smaller spheres
            centers = [
                [z//4, y//4, x//4],
                [3*z//4, 3*y//4, 3*x//4],
                [z//4, 3*y//4, x//2],
                [3*z//4, y//4, x//2]
            ]
            
            for center in centers:
                center_t = torch.tensor(center, dtype=torch.float32)
                radius = min(shape) // 8
                dist = torch.sqrt((zz - center_t[0])**2 + 
                                 (yy - center_t[1])**2 + 
                                 (xx - center_t[2])**2)
                sphere = torch.exp(-(dist - radius)**2 / (2 * (radius/2)**2))
                volume += sphere * 0.6
            
            # Add some noise/texture
            noise = torch.randn(shape) * 0.1
            volume += torch.abs(noise)
            
            # Add spiral pattern
            theta = torch.atan2(yy - y//2, xx - x//2)
            r = torch.sqrt((yy - y//2)**2 + (xx - x//2)**2)
            spiral = torch.sin(3 * theta + 0.2 * r - 0.1 * zz)
            spiral_mask = (r < min(shape)//3) & (r > min(shape)//6)
            volume += spiral * spiral_mask.float() * 0.3
        else:
            volume = np.zeros(shape, dtype=np.float32)
            
            # Central bright sphere
            center1 = np.array([z//2, y//2, x//2], dtype=np.float32)
            radius1 = min(shape) // 6
            dist1 = np.sqrt((zz - center1[0])**2 + 
                           (yy - center1[1])**2 + 
                           (xx - center1[2])**2)
            sphere1 = np.exp(-(dist1 - radius1)**2 / (2 * (radius1/2)**2))
            volume += sphere1 * 1.0
            
            # Secondary smaller spheres
            centers = [
                [z//4, y//4, x//4],
                [3*z//4, 3*y//4, 3*x//4],
                [z//4, 3*y//4, x//2],
                [3*z//4, y//4, x//2]
            ]
            
            for center in centers:
                center_t = np.array(center, dtype=np.float32)
                radius = min(shape) // 8
                dist = np.sqrt((zz - center_t[0])**2 + 
                              (yy - center_t[1])**2 + 
                              (xx - center_t[2])**2)
                sphere = np.exp(-(dist - radius)**2 / (2 * (radius/2)**2))
                volume += sphere * 0.6
            
            # Add some noise/texture
            noise = np.random.randn(*shape) * 0.1
            volume += np.abs(noise)
            
            # Add spiral pattern
            theta = np.arctan2(yy - y//2, xx - x//2)
            r = np.sqrt((yy - y//2)**2 + (xx - x//2)**2)
            spiral = np.sin(3 * theta + 0.2 * r - 0.1 * zz)
            spiral_mask = (r < min(shape)//3) & (r > min(shape)//6)
            volume += spiral * spiral_mask.astype(np.float32) * 0.3
        
    elif volume_type == 'turbulence':
        # Turbulent/cloud-like volume
        if backend == 'torch':
            volume = torch.zeros(shape, dtype=torch.float32)
            
            for octave in range(4):
                scale = 2 ** octave
                amplitude = 1.0 / scale
                
                # Create noise at different frequencies
                noise_z = (zz / (z / scale)).long() % z
                noise_y = (yy / (y / scale)).long() % y
                noise_x = (xx / (x / scale)).long() % x
                
                # Simple procedural noise
                noise = torch.sin(noise_z * 0.5) * torch.cos(noise_y * 0.7) * torch.sin(noise_x * 0.3)
                noise += torch.cos(noise_z * 0.3) * torch.sin(noise_y * 0.5) * torch.cos(noise_x * 0.7)
                
                volume += noise * amplitude
            
            # Apply smoothing and make positive
            volume = torch.abs(volume)
            
            # Add density falloff from center
            center = torch.tensor([z//2, y//2, x//2], dtype=torch.float32)
            distance = torch.sqrt((zz - center[0])**2 + 
                                 (yy - center[1])**2 + 
                                 (xx - center[2])**2)
            falloff = torch.exp(-distance**2 / (2 * (min(shape)//2)**2))
            volume = volume * falloff
        else:
            volume = np.zeros(shape, dtype=np.float32)
            
            for octave in range(4):
                scale = 2 ** octave
                amplitude = 1.0 / scale
                
                # Create noise at different frequencies
                noise_z = (zz / (z / scale)).astype(int) % z
                noise_y = (yy / (y / scale)).astype(int) % y
                noise_x = (xx / (x / scale)).astype(int) % x
                
                # Simple procedural noise
                noise = np.sin(noise_z * 0.5) * np.cos(noise_y * 0.7) * np.sin(noise_x * 0.3)
                noise += np.cos(noise_z * 0.3) * np.sin(noise_y * 0.5) * np.cos(noise_x * 0.7)
                
                volume += noise * amplitude
            
            # Apply smoothing and make positive
            volume = np.abs(volume)
            
            # Add density falloff from center
            center = np.array([z//2, y//2, x//2], dtype=np.float32)
            distance = np.sqrt((zz - center[0])**2 + 
                              (yy - center[1])**2 + 
                              (xx - center[2])**2)
            falloff = np.exp(-distance**2 / (2 * (min(shape)//2)**2))
            volume = volume * falloff
        
    elif volume_type == 'torus':
        # Torus shape
        if backend == 'torch':
            center = torch.tensor([z//2, y//2, x//2], dtype=torch.float32)
            major_radius = min(shape) // 4
            minor_radius = min(shape) // 8
            
            # Distance from z-axis
            r_xy = torch.sqrt((yy - center[1])**2 + (xx - center[2])**2)
            
            # Distance from torus ring
            torus_dist = torch.sqrt((r_xy - major_radius)**2 + (zz - center[0])**2)
            
            volume = torch.exp(-(torus_dist - minor_radius)**2 / (2 * (minor_radius/2)**2))
        else:
            center = np.array([z//2, y//2, x//2], dtype=np.float32)
            major_radius = min(shape) // 4
            minor_radius = min(shape) // 8
            
            # Distance from z-axis
            r_xy = np.sqrt((yy - center[1])**2 + (xx - center[2])**2)
            
            # Distance from torus ring
            torus_dist = np.sqrt((r_xy - major_radius)**2 + (zz - center[0])**2)
            
            volume = np.exp(-(torus_dist - minor_radius)**2 / (2 * (minor_radius/2)**2))
        
    elif volume_type == 'helix':
        # Helical structure
        if backend == 'torch':
            center = torch.tensor([z//2, y//2, x//2], dtype=torch.float32)
            
            # Parametric helix
            t = (zz - center[0]) / (z / 4)  # Parameter along helix
            helix_x = center[2] + (min(shape)//4) * torch.cos(t)
            helix_y = center[1] + (min(shape)//4) * torch.sin(t)
            
            # Distance from helix curve
            helix_dist = torch.sqrt((xx - helix_x)**2 + (yy - helix_y)**2)
            
            volume = torch.exp(-helix_dist**2 / (2 * (min(shape)//16)**2))
            
            # Modulate intensity along helix
            intensity = 0.5 + 0.5 * torch.sin(t * 2)
            volume = volume * intensity
        else:
            center = np.array([z//2, y//2, x//2], dtype=np.float32)
            
            # Parametric helix
            t = (zz - center[0]) / (z / 4)  # Parameter along helix
            helix_x = center[2] + (min(shape)//4) * np.cos(t)
            helix_y = center[1] + (min(shape)//4) * np.sin(t)
            
            # Distance from helix curve
            helix_dist = np.sqrt((xx - helix_x)**2 + (yy - helix_y)**2)
            
            volume = np.exp(-helix_dist**2 / (2 * (min(shape)//16)**2))
            
            # Modulate intensity along helix
            intensity = 0.5 + 0.5 * np.sin(t * 2)
            volume = volume * intensity
        
    else:
        raise ValueError(f"Unknown volume type: {volume_type}")
    
    # Normalize to [0, 1]
    if backend == 'torch':
        volume = torch.clamp(volume, 0, None)
        if volume.max() > 0:
            volume = volume / volume.max()
    else:
        volume = np.clip(volume, 0, None)
        if volume.max() > 0:
            volume = volume / volume.max()
    
    return volume


def create_medical_phantom(shape=(64, 64, 64), backend='torch'):
    """Create a medical imaging phantom with various tissue types.
    
    Args:
        shape: Tuple of (depth, height, width) for volume dimensions
        backend: Backend to use ('torch' or 'numpy')
        
    Returns:
        Volume data as torch.Tensor or np.ndarray depending on backend
    """
    z, y, x = shape
    
    if backend == 'torch':
        zz, yy, xx = torch.meshgrid(
            torch.arange(z, dtype=torch.float32),
            torch.arange(y, dtype=torch.float32), 
            torch.arange(x, dtype=torch.float32),
            indexing='ij'
        )
        volume = torch.zeros(shape, dtype=torch.float32)
    else:
        zz, yy, xx = np.meshgrid(
            np.arange(z, dtype=np.float32),
            np.arange(y, dtype=np.float32), 
            np.arange(x, dtype=np.float32),
            indexing='ij'
        )
        volume = np.zeros(shape, dtype=np.float32)
    
    # Background tissue (soft tissue)
    if backend == 'torch':
        center = torch.tensor([z//2, y//2, x//2], dtype=torch.float32)
        outer_radius = min(shape) // 2.2
        distance = torch.sqrt((zz - center[0])**2 + 
                             (yy - center[1])**2 + 
                             (xx - center[2])**2)
        
        # Soft tissue background
        soft_tissue = (distance < outer_radius).float() * 0.3
        volume += soft_tissue
        
        # Bone structures (higher density)
        bone_centers = [
            [z//2, y//2 - y//4, x//2],  # Upper bone
            [z//2, y//2 + y//4, x//2],  # Lower bone
        ]
        
        for bone_center in bone_centers:
            bone_center_t = torch.tensor(bone_center, dtype=torch.float32)
            bone_radius = min(shape) // 8
            bone_dist = torch.sqrt((zz - bone_center_t[0])**2 + 
                                  (yy - bone_center_t[1])**2 + 
                                  (xx - bone_center_t[2])**2)
            bone = (bone_dist < bone_radius).float() * 0.9
            volume = torch.max(volume, bone)
        
        # Air cavities (lower density)
        air_centers = [
            [z//2, y//2, x//2 - x//4],
            [z//2, y//2, x//2 + x//4],
        ]
        
        for air_center in air_centers:
            air_center_t = torch.tensor(air_center, dtype=torch.float32)
            air_radius = min(shape) // 10
            air_dist = torch.sqrt((zz - air_center_t[0])**2 + 
                                 (yy - air_center_t[1])**2 + 
                                 (xx - air_center_t[2])**2)
            air_mask = (air_dist < air_radius)
            volume[air_mask] = 0.05
        
        # Small high-density spots (contrast agent or calcifications)
        spot_centers = [
            [z//2 + z//8, y//2 + y//8, x//2],
            [z//2 - z//8, y//2 - y//8, x//2],
        ]
        
        for spot_center in spot_centers:
            spot_center_t = torch.tensor(spot_center, dtype=torch.float32)
            spot_radius = min(shape) // 20
            spot_dist = torch.sqrt((zz - spot_center_t[0])**2 + 
                                  (yy - spot_center_t[1])**2 + 
                                  (xx - spot_center_t[2])**2)
            spot = (spot_dist < spot_radius).float() * 1.0
            volume = torch.max(volume, spot)
    else:
        center = np.array([z//2, y//2, x//2], dtype=np.float32)
        outer_radius = min(shape) // 2.2
        distance = np.sqrt((zz - center[0])**2 + 
                          (yy - center[1])**2 + 
                          (xx - center[2])**2)
        
        # Soft tissue background
        soft_tissue = (distance < outer_radius).astype(np.float32) * 0.3
        volume += soft_tissue
        
        # Bone structures (higher density)
        bone_centers = [
            [z//2, y//2 - y//4, x//2],  # Upper bone
            [z//2, y//2 + y//4, x//2],  # Lower bone
        ]
        
        for bone_center in bone_centers:
            bone_center_t = np.array(bone_center, dtype=np.float32)
            bone_radius = min(shape) // 8
            bone_dist = np.sqrt((zz - bone_center_t[0])**2 + 
                               (yy - bone_center_t[1])**2 + 
                               (xx - bone_center_t[2])**2)
            bone = (bone_dist < bone_radius).astype(np.float32) * 0.9
            volume = np.maximum(volume, bone)
        
        # Air cavities (lower density)
        air_centers = [
            [z//2, y//2, x//2 - x//4],
            [z//2, y//2, x//2 + x//4],
        ]
        
        for air_center in air_centers:
            air_center_t = np.array(air_center, dtype=np.float32)
            air_radius = min(shape) // 10
            air_dist = np.sqrt((zz - air_center_t[0])**2 + 
                              (yy - air_center_t[1])**2 + 
                              (xx - air_center_t[2])**2)
            air_mask = (air_dist < air_radius)
            volume[air_mask] = 0.05
        
        # Small high-density spots (contrast agent or calcifications)
        spot_centers = [
            [z//2 + z//8, y//2 + y//8, x//2],
            [z//2 - z//8, y//2 - y//8, x//2],
        ]
        
        for spot_center in spot_centers:
            spot_center_t = np.array(spot_center, dtype=np.float32)
            spot_radius = min(shape) // 20
            spot_dist = np.sqrt((zz - spot_center_t[0])**2 + 
                               (yy - spot_center_t[1])**2 + 
                               (xx - spot_center_t[2])**2)
            spot = (spot_dist < spot_radius).astype(np.float32) * 1.0
            volume = np.maximum(volume, spot)
    
    return volume


def create_sample_volume(size=64, shape='sphere'):
    """Create sample 3D volume data with various shapes.
    
    This function provides compatibility with the ModernGL renderer interface.
    
    Args:
        size: Volume size (creates size x size x size volume)
        shape: Shape type ('sphere', 'torus', 'double_sphere', 'cube', 'helix', 'random_blob')
        
    Returns:
        Volume data as np.ndarray
    """
    x, y, z = np.meshgrid(
        np.linspace(-1, 1, size),
        np.linspace(-1, 1, size),
        np.linspace(-1, 1, size)
    )

    if shape == 'sphere':
        # Simple sphere
        distance = np.sqrt(x*x + y*y + z*z)
        volume = np.exp(-(distance * 3)**2).astype(np.float32)

    elif shape == 'torus':
        # Torus shape
        R = 0.6  # Major radius
        r = 0.3  # Minor radius
        distance_to_center = np.sqrt(x*x + y*y)
        torus_distance = np.sqrt((distance_to_center - R)**2 + z*z)
        volume = np.exp(-(torus_distance / r * 4)**2).astype(np.float32)

    elif shape == 'double_sphere':
        # Two overlapping spheres
        distance1 = np.sqrt((x - 0.3)**2 + y*y + z*z)
        distance2 = np.sqrt((x + 0.3)**2 + y*y + z*z)
        sphere1 = np.exp(-(distance1 * 4)**2)
        sphere2 = np.exp(-(distance2 * 4)**2)
        volume = np.maximum(sphere1, sphere2).astype(np.float32)

    elif shape == 'cube':
        # Rounded cube
        cube_dist = np.maximum(np.maximum(np.abs(x), np.abs(y)), np.abs(z))
        volume = np.exp(-((cube_dist - 0.4) * 8)**2).astype(np.float32)
        volume[cube_dist > 0.6] = 0

    elif shape == 'helix':
        # Helical structure
        theta = np.arctan2(y, x)
        height = z
        radius = np.sqrt(x*x + y*y)

        # Parametric helix
        helix_radius = 0.5
        helix_thickness = 0.15
        turns = 3

        # Distance to helix centerline
        helix_x = helix_radius * np.cos(height * turns * 2 * np.pi)
        helix_y = helix_radius * np.sin(height * turns * 2 * np.pi)

        distance_to_helix = np.sqrt((x - helix_x)**2 + (y - helix_y)**2)
        volume = np.exp(-(distance_to_helix / helix_thickness * 3)
                        ** 2).astype(np.float32)

    elif shape == 'random_blob':
        # Non-symmetric random blob using noise and spatial gradient
        from scipy.ndimage import gaussian_filter

        # Random offset for non-symmetry
        offset = np.random.uniform(-1, 1, size=3)
        x_off = x + offset[0] * 0.5
        y_off = y + offset[1] * 0.5
        z_off = z + offset[2] * 0.5

        # Generate random noise
        noise = np.random.random((size, size, size)).astype(np.float32)
        noise = gaussian_filter(noise, sigma=size/18)

        # Apply a spatial gradient for non-symmetry
        gradient = (x_off + 1.5) * (y_off + 1.2) * (z_off + 0.8)
        gradient = gradient / np.max(np.abs(gradient))

        # Combine noise and gradient, threshold for structure
        volume = noise * (0.7 + 0.3 * gradient)
        volume = np.maximum(0, volume - 0.25) * 2.5

        # Optional: add a few random "hotspots"
        for _ in range(3):
            cx, cy, cz = np.random.uniform(-0.7, 0.7, 3)
            dist = np.sqrt((x - cx)**2 + (y - cy)**2 + (z - cz)**2)
            volume += np.exp(-(dist * 6)**2) * np.random.uniform(0.5, 1.2)

        volume = np.clip(volume, 0, 1).astype(np.float32)

    else:
        raise ValueError(
            f"Unknown shape: {shape}. Available shapes: sphere, torus, double_sphere, cube, helix, random_blob")
    return volume


def compute_normal_volume(volume):
    """Compute normalized gradient (normal) for a 3D volume.
    
    Args:
        volume: 3D volume array with shape (D, H, W)
        
    Returns:
        Normal vectors array with shape (D, H, W, 3)
    """
    gx, gy, gz = np.gradient(volume)
    normals = np.stack((gx, gy, gz), axis=-1)
    norm = np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-8
    normals = normals / norm
    return normals.astype(np.float32)