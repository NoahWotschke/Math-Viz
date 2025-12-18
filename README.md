# Math-Viz: PDE Visualization Framework

Interactive visualization tool for solving and animating partial differential equations (PDEs) using finite differences and numerical methods.

**Current:** 2D Heat Equation on rectangular domains  
**Coming Soon:** Heat & Wave equations on disc/polar and 1D bar domains

## Current Features

- **Universal CLI Solver** (`solve.py`) - Single entry point for all PDEs and domains
- **Modular Architecture** - Domain, Solver, Visualization, and BC abstractions
- **Explicit Finite Difference Solver** with stability control
- **Real-time 3D Visualization** with analytical reference comparison
- **40+ Boundary Condition Functions** (composable and extensible)
- **MP4 Export** for animations with FFmpeg
- **Web Interface** (Streamlit)
- **Phase Toggling** - Alternate between positive/negative BC phases during animation
- **Rotation & Spin** - Visualize 3D solutions from multiple angles

## Setup

### Prerequisites
- Python 3.11 or higher
- Git

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/NoahWotschke/Math-Viz.git
   cd Math-Viz
   ```

2. **Create virtual environment and install dependencies:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install numpy matplotlib streamlit tqdm
   ```

3. **Optional: Install FFmpeg for video export**
   ```bash
   # macOS
   brew install ffmpeg
   
   # Ubuntu/Debian
   sudo apt-get install ffmpeg
   
   # Windows (with Chocolatey)
   choco install ffmpeg
   ```

## Quick Start

### Interactive 3D Visualization (Recommended)
```bash
python solve.py --pde heat --domain rect --spin
```

### Generate MP4 Animation
```bash
python solve.py --pde heat --domain rect --save --out animation.mp4 --seconds 20 --fps 60
```

### Alternate BC Phases
```bash
python solve.py --pde heat --domain rect --spin --alternate --steps_per_cycles 1000
```

### Web Interface
```bash
streamlit run app.py
```
Then open `http://localhost:8501` in your browser

## CLI Reference

```bash
python solve.py --help
```

**Key Options:**
- `--pde {heat, wave}` - PDE type (default: heat)
- `--domain {rect, disc, bar}` - Domain type (default: rect)
- `--res RES` - Grid resolution in points/unit (default: 21)
- `--Lx LX --Ly LY` - Domain dimensions (default: 1.0 × 1.0)
- `--fps FPS` - Animation frame rate (default: 60)
- `--seconds SECONDS` - Duration in seconds (default: 15)
- `--spin` - Rotate 3D view during animation
- `--deg_per_sec DEG` - Rotation speed (default: 10)
- `--alternate` - Toggle BC phase (positive/negative)
- `--steps_per_cycles STEPS` - Steps per phase cycle (default: 1000)
- `--skip_error` - Skip error calculation (faster)
- `--save` - Export to MP4 instead of live display
- `--out FILE` - Output MP4 filename (default: pde_solution.mp4)

## Examples

**Rectangle with high resolution:**
```bash
python solve.py --pde heat --domain rect --res 51 --Lx 2.0 --Ly 1.0 --spin
```

**Quick test animation:**
```bash
python solve.py --pde heat --domain rect --fps 30 --seconds 5 --save --out test.mp4
```

**Phase toggling:**
```bash
python solve.py --pde heat --domain rect --alternate --steps_per_cycles 500 --spin
```

## Project Structure

```
math-viz/
├── solve.py                    # Universal CLI dispatcher
├── app.py                      # Streamlit web interface
├── heat2d/
│   ├── math_settings.py       # Configuration
│   ├── vis_settings.py        # Visualization defaults
│   ├── bc/                    # 40+ boundary condition functions
│   ├── domains/               # Domain abstractions (rectangle, disc, bar)
│   ├── solvers/               # PDE solver implementations
│   ├── analytic/              # Analytical reference solutions
│   └── visualization/         # Generic 3D visualizer
└── Project/                   # Documentation and plans
```

## Mathematical Background

For more detailed mathematical derivations and numerical methods, see the companion page on my personal website:

**[Math-Viz: Mathematical Derivations & Methods](https://noahwotschke.github.io/math-viz.html)**

## License

Open source for educational and research use.