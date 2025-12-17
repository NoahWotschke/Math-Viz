# 2D Heat Equation Solver & Visualizer

A comprehensive visualization tool for solving and animating the 2D heat diffusion equation on rectangular domains using explicit finite differences with customizable boundary conditions.

## Features

- **Explicit Finite Difference Solver**: Efficient time-stepping with automatic CFL stability control
- **Real-time 3D Visualization**: Side-by-side numerical vs. analytical reference solutions
- **Flexible Boundary Conditions**: 35+ built-in BC functions (constant, trigonometric, exponential, Gaussian, etc.)
- **Phase Alternation**: Dynamically switch between positive and negative BC phases during simulation
- **Dual Interfaces**:
  - **CLI**: Full-featured command-line interface (`heat2d_rect_fd.py`)
  - **Web UI**: Interactive Streamlit dashboard (`app.py`)
- **MP4 Export**: Save animations directly to video files
- **Analytical Reference**: Series expansion solution for error tracking

## Installation

### Requirements
- Python 3.11+
- NumPy
- Matplotlib
- Streamlit (for web UI only)
- FFmpeg (optional, for MP4 export)

### Setup

```bash
# Clone or navigate to the project directory
cd math-viz

# Install Python dependencies
pip install numpy matplotlib streamlit

# Optional: Install FFmpeg for video export
# macOS: brew install ffmpeg
# Ubuntu: sudo apt-get install ffmpeg
# Windows: choco install ffmpeg
```

## Quick Start

### Command-Line Interface (CLI) - **Recommended for Best Quality**

For the highest quality visualization with real-time 3D rendering, run locally:

```bash
# Default simulation with 3D rotation
python heat2d_rect_fd.py --spin

# Custom domain and resolution
python heat2d_rect_fd.py --Lx 3.0 --Ly 2.0 --res 50 --spin

# Export to MP4
python heat2d_rect_fd.py --save --out simulation.mp4 --seconds 20 --fps 60

# View all options
python heat2d_rect_fd.py --help
```

**Why run locally:**
- Real-time GPU-accelerated 3D rendering
- Interactive rotation, zoom, and pan
- Crystal clear visualization (no image compression)
- Full performance on your machine

### Web Interface (Streamlit)

For quick testing in a browser (lower quality):

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

## CLI Usage

```bash
python heat2d_rect_fd.py [options]
```

### Available Arguments

| Argument             | Type  | Default            | Description                                        |
| -------------------- | ----- | ------------------ | -------------------------------------------------- |
| `--save`             | flag  | -                  | Export animation to MP4 instead of displaying live |
| `--out`              | str   | `heat_3d_rect.mp4` | Output MP4 filename                                |
| `--fps`              | int   | 60                 | Frames per second for animation/export             |
| `--seconds`          | int   | 15                 | Duration in seconds                                |
| `--dpi`              | int   | 200                | DPI for export                                     |
| `--bitrate`          | int   | 6000               | Bitrate for MP4 export (kbps)                      |
| `--spin`             | flag  | -                  | Slowly rotate both 3D views                        |
| `--deg_per_sec`      | float | 10.0               | Rotation speed (degrees/second)                    |
| `--steps_per_frame`  | int   | 25                 | Time steps to advance per frame                    |
| `--enforce_BC`       | flag  | -                  | Enforce BCs on analytic solution (validation)      |
| `--alternate`        | flag  | -                  | Alternate BC sign (positive/negative phases)       |
| `--steps_per_cycles` | int   | 1000               | Steps per alternation cycle                        |
| `--alpha`            | float | 1.0                | Diffusion coefficient (thermal diffusivity)        |
| `--height_scale`     | float | 1.0                | Vertical scaling factor for 3D visualization       |
| `--hide_colorbar`    | flag  | -                  | Hide colorbar on plots                             |
| `--hide_function`    | flag  | -                  | Hide analytic solution formula title               |
| `--n_terms`          | int   | 200                | Number of terms in series expansion                |
| `--res`              | int   | 21                 | Grid resolution (points per unit length)           |
| `--Lx`               | float | 2.0                | Domain length in x-direction                       |
| `--Ly`               | float | 1.0                | Domain length in y-direction                       |
| `--skip_error`       | flag  | -                  | Skip L∞ error calculation (faster)                 |

## Web UI Features

- **Domain Controls**: Interactive sliders for Lx and Ly
- **BC Selection**: Dropdown menus for all 4 boundaries with parameter input
- **Physics Parameters**: Adjustable diffusion coefficient, series terms, resolution
- **Visualization Options**: Colormap selection, colorbar/formula toggle
- **Live Animation**: Run and stop solver in real-time browser window

## Boundary Condition Functions

35 usable BC functions organized by category:

### Basic
`const_0`, `const_1`, `linear_up`, `linear_down`, `quadratic_bowl`, `quadratic_cap`, `parabola_peak_mid`

### Trigonometric
`sin_k`, `cos_k`, `sin_2pi_k`, `cos_2pi_k`

### Exponential/Hyperbolic
`exp_growth`, `exp_decay`, `sinh_shape`, `cosh_shape`

### Root/Log/Rational
`sqrt_shape`, `log_shape`, `rational`, `x_pow_x`

### Shapes
`abs_centered`, `triangle`, `heaviside_step`, `piecewise_step`

### Gaussian/Bump
`gaussian`, `bump_cos`, `smooth_bump`

### Waveforms
`sawtooth`, `square_wave`, `pulse`, `hermite_smoothstep`

### Noise
`noise_bc`

### Parametrized Example
```bash
# Sine wave on right boundary with k=2
python heat2d_rect_fd.py --right_bc sin_k --right_params '{"k": 2}'
```

In the web UI, select `sin_k` from the BC dropdown and enter `{"k": 2}` in the parameters field.

## Grid Resolution System

The grid uses a base resolution of **21 points per unit length**:
- Calculation: `Nx = 21 × int(Lx)`, `Ny = 21 × int(Ly)`
- Example: Lx=2.0, Ly=1.0 → Nx=42, Ny=21 points
- Override with `--res` parameter to adjust density (e.g., `--res 50` for finer grid)
- Wireframe spacing is automatically adjusted: `ccount = 0.2 × Nx`

## Phase Alternation

When `--alternate` is enabled:
- **Positive Phase (0)**: Uses positive boundary conditions
- **Negative Phase (1)**: Automatically negates BC values
- Switches occur periodically based on `--steps_per_cycles`
- Visualization updates: wireframe and boundary lines toggle between phases

## Architecture

```
heat2d/
├── __init__.py
├── grid.py              # Grid management and resolution control
├── bc_funcs.py          # 35+ boundary condition functions
├── solver.py            # FD solver and analytic reference
├── vis_settings.py      # Matplotlib styling templates
└── math_settings.py     # Physical constants

heat2d_rect_fd.py        # CLI solver with matplotlib animation
app.py                   # Streamlit web interface
```

## Example Workflows

### Explore Gaussian Boundary Condition
```bash
python heat2d_rect_fd.py \
  --right_bc gaussian \
  --right_params '{"mu": 0.5, "sigma": 0.1, "amp": 1.0}' \
  --Lx 2.0 --Ly 1.0 \
  --spin
```

### Create High-Resolution Video
```bash
python heat2d_rect_fd.py \
  --save --out hires_simulation.mp4 \
  --res 100 \
  --fps 60 --seconds 30 \
  --dpi 300 --bitrate 10000
```

### Validate Solver Against Analytic Solution
```bash
python heat2d_rect_fd.py \
  --enforce_BC \
  --n_terms 500
```

### Phase Alternation with Custom Domain
```bash
python heat2d_rect_fd.py \
  --Lx 3.0 --Ly 2.0 \
  --alternate \
  --steps_per_cycles 500 \
  --spin
```

## Stability & Performance

- **CFL Stability**: dt = 0.45 × dt_max (automatically computed from α and grid spacing)
- **Performance Tips**:
  - Use `--skip_error` for faster animation without error calculation
  - Reduce `--n_terms` for faster analytic solution computation
  - Lower `--res` for coarser grids (faster, less detail)
  - Use `--steps_per_frame` to control frame rate vs. simulation accuracy

## Troubleshooting

**Animation is too slow**
- Increase `--steps_per_frame`, reduce `--res`, use `--skip_error`

**Visual artifacts or instability**
- Reduce `--alpha` (diffusion coefficient), check printed CFL output
- Use higher `--res` for better stability

**BC parameters not working in CLI**
- Ensure JSON is properly formatted: `'{"param": value}'` (note single quotes!)

**Streamlit won't start**
- Reinstall: `pip install --upgrade streamlit`

**FFmpeg not found for video export**
- Install FFmpeg: `brew install ffmpeg` (macOS) or `apt-get install ffmpeg` (Ubuntu)

## Mathematical Background

### Heat Equation
$$\frac{\partial u}{\partial t} = \alpha \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right)$$

### Finite Difference Discretization
- **Space**: Central differences (2nd order)
- **Time**: Forward Euler (1st order)
- **Stability**: CFL criterion with safety factor 0.45

### Analytical Solution
Series expansion via separation of variables with Legendre quadrature for computing Fourier coefficients.

### Error Metric
L∞ norm (max absolute error) between numerical and analytical solutions.

## Files Guide

| File                      | Purpose                                        |
| ------------------------- | ---------------------------------------------- |
| `heat2d_rect_fd.py`       | CLI solver with matplotlib 3D animation        |
| `app.py`                  | Interactive Streamlit web dashboard            |
| `heat2d/grid.py`          | Grid initialization and resolution management  |
| `heat2d/bc_funcs.py`      | Library of 35+ boundary condition functions    |
| `heat2d/solver.py`        | PDE solver, time-stepping, analytic solution   |
| `heat2d/vis_settings.py`  | Matplotlib styling (colors, line widths, etc.) |
| `heat2d/math_settings.py` | Physical constants (α, CFL factor)             |
| `README.md`               | This file                                      |

## License

Open source for educational and research use.

## Author

Noah Wotschke  
MathVis Project - 2D Heat Equation Visualization Suite
