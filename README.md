# PyVR: Python Volume Rendering Toolkit

PyVR is a Python-based toolkit for interactive and high-quality 3D volume rendering. It provides both a ModernGL (OpenGL) renderer for real-time visualization and a PyTorch-based renderer for flexible, differentiable, and research-oriented workflows.

## Features

- **🔥 Multiple Rendering Backends**:
  - `moderngl_renderer`: GPU-accelerated, real-time OpenGL rendering using ModernGL
  - `torch_renderer`: Fully vectorized, differentiable volume rendering in PyTorch

- **🎨 Flexible Transfer Functions**:
  - Piecewise linear opacity and color transfer functions
  - Matplotlib colormap support

- **📷 Camera Controls**:
  - Quaternion-based camera orbit and roll (ModernGL)
  - Fully programmable camera in PyTorch

- **💡 Lighting and Shading**:
  - Ambient and diffuse lighting
  - Gradient-based normal computation

- **🧪 Sample Volumes**:
  - Built-in synthetic datasets (sphere, torus, helix, medical phantom, etc.)

- **📊 Visualization Utilities**:
  - Matplotlib integration for rendered images and transfer function plots

## Installation

### Requirements

- Python 3.11+
- Poetry (recommended) or pip

### Using Poetry

```sh
poetry install
```

### Using pip

Install dependencies from the respective renderer directories:

```sh
pip install -r pyvr/torch_renderer/requirements.txt
pip install -r pyvr/moderngl_renderer/requirements.txt
```

## Quick Start

### PyTorch Renderer

Run a simple demo:

```sh
python pyvr/torch_renderer/volume_renderer.py
```

This renders a 3D volume using PyTorch and displays the result with transfer function visualizations.

### ModernGL Renderer

Run the multi-view example:

```sh
python example/ModernglRender/multiview_example.py
```

This demonstrates real-time volume rendering with multiple views using OpenGL.

> **Note:**  
> All volume data is assumed to be in **(D, H, W)** order, corresponding to **(z, y, x)** axes.

## Examples

The `example/` directory contains ready-to-run demos:

- `ModernglRender/multiview_example.py`: Multi-view volume rendering with ModernGL.
- `TorchRenderer/simple_demo.py`: Basic volume rendering with PyTorch.

## Project Structure

```
pyvr/
├── __init__.py
├── datasets/                              # 🆕 Unified dataset generation
│   ├── __init__.py
│   └── synthetic.py                       # All volume generation functions
├── moderngl_renderer/                     # 🔥 Real-time OpenGL rendering
│   ├── __init__.py
│   ├── README.md                          # 🆕 ModernGL renderer documentation
│   ├── camera_control.py
│   ├── requirements.txt
│   ├── transfer_functions.py
│   ├── volume_renderer.py
│   └── shaders/
│       ├── volume.frag.glsl
│       └── volume.vert.glsl
├── torch_renderer/                        # 🧠 Differentiable PyTorch rendering
│   ├── __init__.py
│   ├── README.md                          # 🆕 PyTorch renderer documentation
│   ├── camera.py
│   ├── requirements.txt
│   ├── transfer_functions.py
│   └── volume_renderer.py
example/
├── ModernglRender/
│   └── multiview_example.py
└── TorchRenderer/
    └── simple_demo.py
tests/
pyproject.toml
poetry.lock
README.md
```

## Renderer Documentation

Each renderer has detailed documentation:

- **[📖 ModernGL Renderer](pyvr/moderngl_renderer/README.md)**: Real-time OpenGL rendering with interactive controls
- **[📖 PyTorch Renderer](pyvr/torch_renderer/README.md)**: Differentiable rendering for research and ML applications

## Customization

- **🎨 Transfer Functions**:  
  Edit [`pyvr/torch_renderer/transfer_functions.py`](pyvr/torch_renderer/transfer_functions.py) or [`pyvr/moderngl_renderer/transfer_functions.py`](pyvr/moderngl_renderer/transfer_functions.py) to create custom opacity/color mappings.

- **🧪 Volume Data**:  
  The unified [`pyvr/datasets/synthetic.py`](pyvr/datasets/synthetic.py) module provides all volume generation functions. Import with `from pyvr.datasets import create_test_volume, create_sample_volume`.

- **📷 Camera Controls**:  
  Modify camera logic in [`pyvr/torch_renderer/camera.py`](pyvr/torch_renderer/camera.py) or [`pyvr/moderngl_renderer/camera_control.py`](pyvr/moderngl_renderer/camera_control.py).

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## License

This project is licensed under the WTFPL (Do What The F*ck You Want To Public License). See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [ModernGL](https://github.com/moderngl/moderngl)
- [PyTorch](https://pytorch.org/)
- [Matplotlib](https://matplotlib.org/)
- [NumPy](https://numpy.org/)
- [SciPy](https://scipy.org/)