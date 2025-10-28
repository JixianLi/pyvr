# Camera Class Refactoring Plan

## Overview
This plan outlines the modifications needed to rename `CameraParameters` to `Camera`, add a `Camera` class variable to `VolumeRenderer`, and move camera-related logic from `VolumeRenderer.set_camera` to the `Camera` class.

## 📋 Modification Plan

### 1. Rename `CameraParameters` to `Camera`

**Files to modify:**
- `/Users/jixianli/projects/pyvr/pyvr/camera/parameters.py`
- `/Users/jixianli/projects/pyvr/pyvr/camera/control.py`
- `/Users/jixianli/projects/pyvr/pyvr/camera/__init__.py`
- `/Users/jixianli/projects/pyvr/pyvr/moderngl_renderer/__init__.py`

**Changes:**
- Rename the `CameraParameters` class to `Camera` in `parameters.py`
- Update all references to `CameraParameters` throughout the codebase
- Update import statements and exports
- Maintain backward compatibility by creating an alias: `CameraParameters = Camera`

### 2. Add Camera Class Variable to VolumeRenderer

**Files to modify:**
- `/Users/jixianli/projects/pyvr/pyvr/moderngl_renderer/renderer.py`

**Changes:**
- Add a `camera: Camera` class variable to `VolumeRenderer.__init__()`
- Initialize with a default `Camera` instance
- Provide methods to get/set the camera instance

### 3. Move Camera Logic from VolumeRenderer to Camera Class

**Files to modify:**
- `/Users/jixianli/projects/pyvr/pyvr/camera/parameters.py` (now containing `Camera` class)
- `/Users/jixianli/projects/pyvr/pyvr/moderngl_renderer/renderer.py`

**New Camera methods to add:**
- `Camera.create_view_matrix()`
- `Camera.create_projection_matrix(width, height)`
- `Camera.get_camera_position_and_up()` (enhanced version of existing method)
- `Camera.apply_to_renderer(renderer)` - method that configures renderer matrices

**VolumeRenderer changes:**
- Modify `set_camera()` to accept a `Camera` instance or position/target/up parameters
- Delegate matrix creation to the `Camera` class
- Simplify the `set_camera` implementation

## 🔧 Detailed Implementation Steps

### Step 1: Rename CameraParameters to Camera
1. **In `parameters.py`:**
   - Rename `class CameraParameters` → `class Camera`
   - Add backward compatibility alias at end of file: `CameraParameters = Camera`
   - Update all docstrings to reference "Camera" instead of "CameraParameters"

2. **In `control.py`:**
   - Update type hints: `CameraParameters` → `Camera`
   - Update function parameters and return types
   - Update docstrings and comments

3. **In `__init__.py`:**
   - Update imports to include both `Camera` and `CameraParameters` (alias)
   - Update `__all__` list to include both names

### Step 2: Add Camera to VolumeRenderer
1. **Modify `VolumeRenderer.__init__()`:**
   ```python
   def __init__(self, ..., camera=None):
       # existing initialization...
       
       # Initialize camera
       if camera is None:
           self.camera = Camera.front_view(distance=3.0)
       else:
           self.camera = camera
   ```

2. **Add camera property methods:**
   ```python
   def get_camera(self) -> Camera:
       """Get current camera instance."""
       return self.camera
       
   def set_camera_instance(self, camera: Camera):
       """Set camera instance and update renderer matrices."""
       self.camera = camera
       self.camera.apply_to_renderer(self)
   ```

### Step 3: Move Camera Logic to Camera Class
1. **Add matrix creation methods to Camera:**
   ```python
   def create_view_matrix(self) -> np.ndarray:
       """Create view matrix from camera parameters."""
       # Move existing view matrix logic from VolumeRenderer.set_camera
       
   def create_projection_matrix(self, width: int, height: int) -> np.ndarray:
       """Create projection matrix using camera's FOV and aspect ratio."""
       # Move existing projection matrix logic from VolumeRenderer.set_camera
       
   def apply_to_renderer(self, renderer):
       """Apply camera matrices and settings to a VolumeRenderer."""
       # Calculate position and up vectors
       position, up = self.get_camera_vectors()
       
       # Create matrices
       view_matrix = self.create_view_matrix()
       projection_matrix = self.create_projection_matrix(renderer.width, renderer.height)
       
       # Apply to renderer
       renderer.gl_manager.set_uniform_matrix("view_matrix", view_matrix)
       renderer.gl_manager.set_uniform_matrix("projection_matrix", projection_matrix)
       renderer.gl_manager.set_uniform_vector("camera_pos", tuple(position))
   ```

2. **Simplify VolumeRenderer.set_camera():**
   ```python
   def set_camera(self, camera_or_position, target=None, up=None):
       """Set camera using Camera instance or position/target/up vectors."""
       if isinstance(camera_or_position, Camera):
           # Use Camera instance
           self.set_camera_instance(camera_or_position)
       else:
           # Legacy interface: convert position/target/up to Camera
           position = camera_or_position
           if target is None:
               target = (0, 0, 0)
           if up is None:
               up = (0, 1, 0)
           
           # Create temporary Camera instance and apply
           # (This maintains backward compatibility)
           temp_camera = Camera()  # Would need logic to convert pos/target/up
           temp_camera.apply_to_renderer(self)
   ```

## 📁 Files to Update for Tests and Examples

### Test Files:
- `/Users/jixianli/projects/pyvr/tests/test_camera/test_parameters.py`
- `/Users/jixianli/projects/pyvr/tests/test_moderngl_renderer/test_volume_renderer.py`
- All test files that import or use `CameraParameters`

### Example Files:
- `/Users/jixianli/projects/pyvr/example/ModernglRender/enhanced_camera_demo.py`
- `/Users/jixianli/projects/pyvr/example/ModernglRender/multiview_example.py`
- `/Users/jixianli/projects/pyvr/example/ModernglRender/rgba_demo.py`
- `/Users/jixianli/projects/pyvr/example/benchmark.py`

### Documentation:
- `/Users/jixianli/projects/pyvr/README.md` - Update API examples and documentation

## 🔄 Backward Compatibility Strategy

1. **Alias Approach:** Keep `CameraParameters = Camera` alias to maintain existing code compatibility
2. **Dual Interface:** `VolumeRenderer.set_camera()` should accept both:
   - New interface: `Camera` instance
   - Legacy interface: `position, target, up` parameters
3. **Deprecation Warnings:** Add deprecation warnings for old parameter names in future versions
4. **Import Compatibility:** Ensure both `Camera` and `CameraParameters` can be imported from existing modules

## ✅ Validation Steps

1. **Unit Tests:** All existing tests should pass without modification
2. **Examples:** All example scripts should work with minimal changes
3. **API Consistency:** New `Camera` class should provide all functionality of old `CameraParameters`
4. **Performance:** No performance regression in camera operations
5. **Documentation:** All docstrings and examples updated to reflect new naming

## 🎯 Benefits of This Refactoring

1. **Cleaner Architecture:** Camera logic is centralized in the `Camera` class
2. **Better Separation of Concerns:** `VolumeRenderer` focuses on rendering, `Camera` handles camera logic
3. **Enhanced Functionality:** `Camera` class can be extended with more sophisticated camera operations
4. **Improved Testability:** Camera logic can be tested independently
5. **Future Extensibility:** Easier to add features like camera animations, different projection types, etc.

This plan ensures a smooth transition while maintaining backward compatibility and improving the overall architecture of the camera system.