import numpy as np
import torch


def create_test_volume(shape=(64, 64, 64), volume_type='complex'):
    """Create test volumes with various patterns."""
    z, y, x = shape
    zz, yy, xx = torch.meshgrid(
        torch.arange(z, dtype=torch.float32),
        torch.arange(y, dtype=torch.float32), 
        torch.arange(x, dtype=torch.float32),
        indexing='ij'
    )
    
    if volume_type == 'sphere':
        # Simple sphere
        center = torch.tensor([z//2, y//2, x//2], dtype=torch.float32)
        radius = min(shape) // 4
        distance = torch.sqrt((zz - center[0])**2 + 
                             (yy - center[1])**2 + 
                             (xx - center[2])**2)
        volume = torch.exp(-(distance - radius)**2 / (2 * (radius/3)**2))
        
    elif volume_type == 'complex':
        # Complex volume with multiple features
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
        
    elif volume_type == 'turbulence':
        # Turbulent/cloud-like volume
        # Create multiple octaves of noise
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
        
    elif volume_type == 'torus':
        # Torus shape
        center = torch.tensor([z//2, y//2, x//2], dtype=torch.float32)
        major_radius = min(shape) // 4
        minor_radius = min(shape) // 8
        
        # Distance from z-axis
        r_xy = torch.sqrt((yy - center[1])**2 + (xx - center[2])**2)
        
        # Distance from torus ring
        torus_dist = torch.sqrt((r_xy - major_radius)**2 + (zz - center[0])**2)
        
        volume = torch.exp(-(torus_dist - minor_radius)**2 / (2 * (minor_radius/2)**2))
        
    elif volume_type == 'helix':
        # Helical structure
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
        raise ValueError(f"Unknown volume type: {volume_type}")
    
    # Normalize to [0, 1]
    volume = torch.clamp(volume, 0, None)
    if volume.max() > 0:
        volume = volume / volume.max()
    
    return volume


def create_medical_phantom(shape=(64, 64, 64)):
    """Create a medical imaging phantom with various tissue types."""
    z, y, x = shape
    zz, yy, xx = torch.meshgrid(
        torch.arange(z, dtype=torch.float32),
        torch.arange(y, dtype=torch.float32), 
        torch.arange(x, dtype=torch.float32),
        indexing='ij'
    )
    
    volume = torch.zeros(shape, dtype=torch.float32)
    
    # Background tissue (soft tissue)
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
    
    return volume