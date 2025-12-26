#!/usr/bin/env python3
"""Test visualization wireframe rendering."""

import sys
sys.path.insert(0, 'mathvis-core')

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from PDEs.visualization.visualizer import Visualizer3D, VisualizerConfig
import PDEs.vis_settings as vis

# Setup basic domain
Lx_grid = 1.0
Ly_grid = 1.0
res = 21
x = np.linspace(0, Lx_grid, res)
y = np.linspace(0, Ly_grid, res)
X, Y = np.meshgrid(x, y)
Nx, Ny = X.shape

print(f"Grid: Nx={Nx}, Ny={Ny}")

# Create config and visualizer
vis_config = VisualizerConfig(
    fps=10,
    seconds=1,
    skip_error=False,
    save=False,
    show_function=True,
    show_colorbar=True,
)

visualizer = Visualizer3D(vis_config, domain_name="Rectangle", pde_name="Heat")

# Create dummy data
u_init = np.sin(np.pi * X) * np.sin(np.pi * Y)
u_star_pos = np.sin(np.pi * X) * np.sin(np.pi * Y)
u_star_neg = -u_star_pos

# Update wireframe kwargs
wire_kwargs = vis.wire_kwargs.copy()
surf_kwargs = vis.surf_kwargs.copy()
wire_kwargs["ccount"] = 0.2 * Nx
wire_kwargs["rcount"] = 0.2 * Ny
surf_kwargs["ccount"] = 0.2 * Nx
surf_kwargs["rcount"] = 0.2 * Ny

print(f"Setting up visualization...")

# Dummy boundary lines
boundary_lines_data = {
    "num": [],
    "ref": [],
}

analytic_title = r"$u(x,y) = \sin(\pi x)\sin(\pi y)$"

visualizer.setup(
    X,
    Y,
    u_init,
    u_star_pos=u_star_pos,
    u_star_neg=u_star_neg,
    analytic_title=analytic_title,
    surf_kwargs=surf_kwargs,
    wire_kwargs=wire_kwargs,
    line_kwargs=vis.line_kwargs,
    boundary_lines_data=boundary_lines_data,
)

print(f"wf_pos visible: {visualizer.wf_pos is not None}")
print(f"wf_neg visible: {visualizer.wf_neg is not None}")
print(f"surf visible: {visualizer.surf is not None}")
print(f"ax_ref title: {visualizer.ax_ref.get_title() if visualizer.ax_ref else 'No ax_ref'}")

# Save to image
if visualizer.fig:
    visualizer.fig.savefig('test_wireframe.png', dpi=100, bbox_inches='tight')
    print("Saved to test_wireframe.png")
    plt.close(visualizer.fig)
