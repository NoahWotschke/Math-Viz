#!/usr/bin/env python3
"""Quick test to see if visualization is working."""

import sys
sys.path.insert(0, 'mathvis-core')

from PDEs.solvers.heat2d_rect import Heat2DRectSolver, Heat2DRectConfig
from PDEs.domains.rectangle import RectangleDomain
from PDEs.domains.base import Grid as DomainGrid
from PDEs.visualization.visualizer import Visualizer3D, VisualizerConfig
import PDEs.vis_settings as vis
import PDEs.bc.funcs as bc
import numpy as np

# Setup basic domain
Lx_grid = 1.0
Ly_grid = 1.0
res = 21
x = np.linspace(0, Lx_grid, res)
y = np.linspace(0, Ly_grid, res)
X, Y = np.meshgrid(x, y)
Nx, Ny = X.shape

print(f"Grid: Nx={Nx}, Ny={Ny}")
print(f"Creating visualizer...")

# Create config and visualizer
vis_config = VisualizerConfig(
    fps=10,
    seconds=1,
    skip_error=False,
    save=False,
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

print(f"wire_kwargs: {wire_kwargs}")
print(f"surf_kwargs: {surf_kwargs}")
print(f"vis.line_kwargs: {vis.line_kwargs}")

# Dummy boundary lines
boundary_lines_data = {
    "num": [],
    "ref": [],
}

analytic_title = r"$u(x,y) = \sin(\pi x)\sin(\pi y)$"

print("Calling visualizer.setup()...")
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

print("Setup complete!")
print(f"wf_pos: {visualizer.wf_pos}")
print(f"wf_neg: {visualizer.wf_neg}")
print("Done!")
