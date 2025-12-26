# MathVis: PDE Solver & Visualization

A modular system for solving PDEs with both CLI and web interfaces. Split into three independent projects sharing a common solver library.

## Quick Start

### Web UI (Recommended)
```bash
cd mathvis-web
pip install -e ../mathvis-core -q
pip install -e . -q
streamlit run app.py
```

### Command Line
```bash
cd mathvis-cli
pip install -e ../mathvis-core -q
python3 solve.py --pde heat --domain rect --save
```

## Project Structure

This is a monorepo with three independent subprojects:

### **mathvis-core** - Shared Solver Library
- Core PDE solvers (Heat, Wave equations)
- Domain abstractions (Rectangle, Disc, Bar)
- 40+ customizable boundary condition functions
- Analytical reference solutions
- 3D visualization system
- Base package that both CLI and web depend on
- Dependencies: numpy, matplotlib, tqdm

### **mathvis-cli** - Command-Line Interface  
- Universal solver entry point (`solve.py`)
- Full argument parsing for all parameters
- MP4 video export with FFmpeg
- Batch processing and automation
- Can run without GUI or browser
- Interactive 3D visualization via matplotlib

### **mathvis-web** - Streamlit Web UI
- Browser-based interactive interface (http://localhost:8501)
- Real-time parameter adjustment with live updates
- 3D visualization in web browser
- Performance presets (Fast, Balanced, High Quality)
- Animation playback and frame export

## Features

- **Multiple PDEs:** Heat, Wave equations (extensible)
- **Flexible Domains:** Rectangle (disc coming soon)
- **Advanced BCs:** Customizable boundary conditions
- **3D Visualization:** Surface plots with animations
- **Dual Interfaces:** CLI for automation, web for exploration  
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
- Optional: FFmpeg for MP4 video export

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/NoahWotschke/Math-Viz.git
   cd math-viz
   ```

2. **Create and activate virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install mathvis-core (shared library):**
   ```bash
   cd mathvis-core
   pip install -e .
   cd ..
   ```

4. **Choose your interface:**

   **Option A: Web UI (Recommended for exploration)**
   ```bash
   cd mathvis-web
   pip install -e .
   streamlit run app.py
   # Opens at http://localhost:8501
   ```

   **Option B: Command-Line (for automation)**
   ```bash
   cd mathvis-cli
   pip install -e .
   python3 solve.py --pde heat --domain rect --spin
   ```

5. **Optional: Install FFmpeg for video export**
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

## Directory Structure

```
math-viz/
├── mathvis-core/               # Shared solver library
│   ├── PDEs/
│   │   ├── solvers/           # PDE solver implementations (Heat, Wave)
│   │   ├── domains/           # Domain abstractions (Rectangle, Disc, Bar)
│   │   ├── bc/                # 40+ boundary condition functions
│   │   ├── analytic/          # Analytical reference solutions
│   │   ├── visualization/     # 3D visualization system
│   │   ├── math_settings.py   # Mathematical configuration
│   │   └── vis_settings.py    # Visualization defaults
│   └── setup.py
│
├── mathvis-cli/                # Command-line interface
│   ├── solve.py               # Universal CLI dispatcher
│   └── setup.py
│
├── mathvis-web/                # Streamlit web interface
│   ├── app.py                 # Streamlit application
│   └── setup.py
│
├── Project/                    # Documentation and architecture plans
├── Backups/                    # Legacy backup files
└── README.md                   # This file
```

## Mathematical Background

For more detailed mathematical derivations and numerical methods, see the companion page on my personal website:

**[Math-Viz: Mathematical Derivations & Methods](https://noahwotschke.github.io/math-viz.html)**

## License

Open source for educational and research use.