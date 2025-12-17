# Math-Viz: PDE Visualization Framework

Interactive visualization tool for solving and animating partial differential equations (PDEs) using finite differences and numerical methods.

**Current:** 2D Heat Equation on rectangular domains  
**Future:** Heat & Wave equations on various domains

## Current Features

- **Explicit Finite Difference Solver** with stability control
- **Real-time 3D Visualization** & analytical reference comparison
- **35+ Boundary Condition Functions**
- **MP4 Export** for animations
- **CLI + Web Interface**

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

2. **Install dependencies:**
   ```bash
   pip install numpy matplotlib streamlit
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

### Running Locally

**CLI with 3D visualization (recommended):**
```bash
python heat2d_rect_fd.py --spin
```

**Web interface:**
```bash
streamlit run app.py
```
Then open `http://localhost:8501` in your browser

**Generate MP4 animation:**
```bash
python heat2d_rect_fd.py --save --out simulation.mp4 --seconds 20 --fps 60
```

View all options with:
```bash
python heat2d_rect_fd.py --help
```

## License

Open source for educational and research use.