## Streamlit UI Guide

### Installation

If you haven't installed Streamlit yet:
```bash
pip install streamlit
```

### Running the UI

From the project directory:
```bash
streamlit run app.py
```

This will open a web browser at `http://localhost:8501` with the interactive interface.

### Using the Interface

**Left sidebar - Solver Parameters:**
- **FPS**: Animation frame rate (10-120)
- **Duration**: How long the animation runs (5-60 seconds)
- **Steps per frame**: How many time steps between each frame (higher = more accurate but slower)
- **Diffusion coefficient (α)**: Controls how fast heat diffuses (lower = slower diffusion)
- **Series terms**: Number of terms in the analytic solution expansion (higher = more accurate reference)

**Right sidebar - Display Options:**
- **Show colorbar**: Display temperature color scale
- **Show analytic formula**: Display the mathematical formula at top
- **Rotate view**: Slowly rotate the 3D camera around the solution
- **Height scale**: Vertical magnification of the 3D plot
- **Skip error calculation**: Disable L∞ error display (faster animation)
- **Rotation speed**: How fast the camera rotates (if enabled)

**Main button:**
Click **🚀 Run Animation** to generate and display the animation.

### What You're Looking At

- **Left plot**: Numerical solution (finite difference method)
- **Right plot**: Analytic reference solution (series expansion)
- **Colors**: Temperature (blue = cold, red = hot)
- **Wireframes**: Help visualize the 3D surface
- **Edge lines**: Boundary conditions

The bottom left corner shows:
- `t=...`: Current simulation time
- `||u-u*||∞`: Maximum error between numerical and analytic solution

### Tips

- Start with default settings to see the basic behavior
- Increase **Steps per frame** for higher accuracy
- Increase **Series terms** for a more accurate reference solution
- Use **Rotate view** to better understand the 3D structure
- Disable **Skip error calculation** to see convergence to the analytic solution

### Troubleshooting

**Safari says "Safari can't open the page"**

Safari has HTTPS-Only security enabled by default. You have two options:

1. **Use Chrome, Firefox, or Edge instead** (easiest)
   - Any of these browsers will work without issues

2. **Disable HTTPS-Only in Safari**
   - Safari menu → Settings (or Preferences)
   - Click **Privacy** tab
   - Uncheck **"Require HTTPS Only"**
   - Reload `http://localhost:8501`

**Animation runs very slowly**

- Reduce **FPS** (try 30 or 15)
- Reduce **Steps per frame** (try 10-15)
- Enable **Skip error calculation** to disable error computation
- Close other applications to free up CPU

**Import errors when running**

Make sure you have all dependencies installed:
```bash
pip install streamlit numpy matplotlib scipy
```
