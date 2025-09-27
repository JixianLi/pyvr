import numpy as np
import torch
import torch.nn.functional as F


class Camera:
    def __init__(self, position, target, up=(0, 1, 0), fov=45, device='cpu'):
        """
        Initialize camera with position, target, and up vector.
        
        Args:
            position: Camera position (3,)
            target: Camera target/look-at point (3,)
            up: Camera up vector (3,)
            fov: Field of view in degrees
            device: Device to run computations on
        """
        self.device = device
        self.position = torch.tensor(position, dtype=torch.float32, device=device)
        self.target = torch.tensor(target, dtype=torch.float32, device=device)
        self.up = torch.tensor(up, dtype=torch.float32, device=device)
        self.fov = fov
        
        # Build camera coordinate system
        self._build_coordinate_system()
    
    def _build_coordinate_system(self):
        """Build the camera's coordinate system vectors."""
        self.forward = F.normalize(self.target - self.position, dim=0)
        self.right = F.normalize(torch.linalg.cross(self.forward, self.up), dim=0)
        self.up_corrected = torch.linalg.cross(self.right, self.forward)
    
    def update_position(self, position):
        """Update camera position and rebuild coordinate system."""
        self.position = torch.tensor(position, dtype=torch.float32, device=self.device)
        self._build_coordinate_system()
    
    def update_target(self, target):
        """Update camera target and rebuild coordinate system."""
        self.target = torch.tensor(target, dtype=torch.float32, device=self.device)
        self._build_coordinate_system()
    
    def update_fov(self, fov):
        """Update field of view."""
        self.fov = fov
    
    def look_at(self, position, target, up=(0, 1, 0)):
        """Set camera to look at target from position."""
        self.position = torch.tensor(position, dtype=torch.float32, device=self.device)
        self.target = torch.tensor(target, dtype=torch.float32, device=self.device)
        self.up = torch.tensor(up, dtype=torch.float32, device=self.device)
        self._build_coordinate_system()
    
    def generate_rays(self, image_size):
        """
        Generate rays from camera through each pixel.
        
        Args:
            image_size: Tuple (height, width)
            
        Returns:
            ray_origins: (H, W, 3) ray origins
            ray_directions: (H, W, 3) ray directions
        """
        H, W = image_size
        
        # Calculate pixel coordinates
        fov_rad = torch.deg2rad(torch.tensor(self.fov, device=self.device))
        aspect_ratio = W / H
        
        # Generate pixel grid
        i, j = torch.meshgrid(
            torch.linspace(-1, 1, H, device=self.device),
            torch.linspace(-1, 1, W, device=self.device),
            indexing='ij'
        )
        
        # Convert to camera space
        x = j * aspect_ratio * torch.tan(fov_rad / 2)
        y = -i * torch.tan(fov_rad / 2)  # Negative for correct orientation
        z = torch.ones_like(x)
        
        # Ray directions in world space
        ray_dirs = (x.unsqueeze(-1) * self.right + 
                   y.unsqueeze(-1) * self.up_corrected + 
                   z.unsqueeze(-1) * self.forward)
        ray_dirs = F.normalize(ray_dirs, dim=-1)
        
        # Ray origins (all from camera position)
        ray_origins = self.position.expand(H, W, 3)
        
        return ray_origins, ray_dirs

