# PyVR: Python Volume Rendering Toolkit

PyVR is a Python-based toolkit for interactive and high-quality 3D volume rendering. It provides both a ModernGL (OpenGL) renderer for real-time visualization and a PyTorch-based renderer for flexible, differentiable, and research-oriented workflows.

## Features

- **Multiple Rendering Backends**:  
  - `moderngl_renderer`: GPU-accelerated, real-time OpenGL rendering using ModernGL.
  - `torch_renderer`: Fully vectorized, differentiable volume rendering in PyTorch.

- **Flexible Transfer Functions**:  
  - Piecewise linear opacity and color transfer functions.
  - Matplotlib colormap support.

- **Camera Controls**:  
  - Quaternion-based camera orbit and roll (ModernGL).
  - Fully programmable camera in PyTorch.

- **Lighting and Shading**:  
  - Ambient and diffuse lighting.
  - Gradient-based normal computation.

- **Sample Volumes**:  
  - Built-in synthetic datasets (sphere, torus, helix, medical phantom, etc.).

- **Visualization Utilities**:  
  - Matplotlib integration for rendered images and transfer function plots.

## Installation

### Requirements

- Python 3.8+
- See [torch_renderer/requirements.txt](torch_renderer/requirements.txt) and [moderngl_renderer/requirements.txt](moderngl_renderer/requirements.txt) for dependencies.

### Install dependencies

```sh
pip install -r torch_renderer/requirements.txt
pip install -r moderngl_renderer/requirements.txt
```

## Usage

### PyTorch Renderer

Run the performance demo:

```sh
python torch_renderer/volume_renderer.py
```

This will render a configurable 3D volume using the PyTorch backend and display the result with transfer function visualizations.

### ModernGL Renderer

Run the ModernGL multi-view demo:

```sh
python moderngl_renderer/volume_renderer.py
```

This will render multiple views of a sample volume using the OpenGL backend and display the results with transfer function plots.

> **Note:**  
> All volume data is assumed to be in **(D, H, W)** order, corresponding to **(z, y, x)** axes.

## Project Structure

```
pyvr/
├── torch_renderer/
│   ├── camera.py
│   ├── transfer_function.py
│   ├── volume_data.py
│   ├── volume_renderer.py
│   └── requirements.txt
├── moderngl_renderer/
│   ├── camera_control.py
│   ├── datasets.py
│   ├── transfer_functions.py
│   ├── volume_renderer.py
│   ├── requirements.txt
│   └── shaders/
├── README.md
```

## Customization

- **Transfer Functions**:  
  Edit or extend [`torch_renderer/transfer_function.py`](torch_renderer/transfer_function.py) or [`moderngl_renderer/transfer_functions.py`](moderngl_renderer/transfer_functions.py) to create custom opacity/color mappings.

- **Volume Data**:  
  Add new synthetic or real datasets in [`torch_renderer/volume_data.py`](torch_renderer/volume_data.py) or [`moderngl_renderer/datasets.py`](moderngl_renderer/datasets.py).

- **Camera Controls**:  
  Modify camera logic in [`torch_renderer/camera.py`](torch_renderer/camera.py) or [`moderngl_renderer/camera_control.py`](moderngl_renderer/camera_control.py).

## License

No license; do whatever you want. 

---

**Author:**  
[GitHub Copilot - Claude Sonnet 4 (preview) and GPT-4.1] 

I made minimal changes to fix some API signatures.

**Acknowledgments:**  
- [GitHub Copilot](https://github.com/features/copilot)
- [ModernGL](https://github.com/moderngl/moderngl)
- [PyTorch](https://pytorch.org/)  
- [Matplotlib](https://matplotlib.org/)
- [NumPy](https://numpy.org/)
- [SciPy](https://scipy.org/)