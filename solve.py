#!/usr/bin/env python3
"""Universal PDE Solver CLI Dispatcher.

Unified command-line interface for solving various PDEs on different domains.
Replaces separate heat2d_rect_fd.py, heat2d_disc_fd.py, etc.

Usage:
    python solve.py --pde heat --domain rect [options]
    python solve.py --pde heat --domain disc [options]
    python solve.py --pde wave --domain rect [options]

For help:
    python solve.py --help
"""

import argparse
import sys
from typing import Optional

# Import solver modules
from heat2d.solvers.heat2d_rect import Heat2DRectSolver, Heat2DRectConfig
from heat2d.domains.rectangle import RectangleDomain
from heat2d.domains.base import Grid as DomainGrid
from heat2d.visualization.visualizer import Visualizer3D, VisualizerConfig
import heat2d.vis_settings as vis
import heat2d.bc.funcs as bc
import heat2d.math_settings as math_settings
from heat2d.bc.builder import build_bc_from_spec

import numpy as np


def solve_heat_rect(args) -> None:
    """Solve 2D heat equation on rectangle."""

    print(f"Solving Heat Equation on Rectangle Domain")
    print(f"Lx={args.Lx}, Ly={args.Ly}, resolution={args.res} pts/unit")

    # =========== DOMAIN AND GRID ===========
    Lx_grid = args.Lx
    Ly_grid = args.Ly
    res = args.res

    Nx = res * int(Lx_grid)
    Ny = res * int(Ly_grid)
    x = np.linspace(0, Lx_grid, Nx)
    y = np.linspace(0, Ly_grid, Ny)
    dx = x[1] - x[0] if Nx > 1 else 1.0
    dy = y[1] - y[0] if Ny > 1 else 1.0
    X, Y = np.meshgrid(x, y, indexing="xy")

    print(f"Grid: Nx={Nx}, Ny={Ny}, dx={dx:.4g}, dy={dy:.4g}")

    # =========== BOUNDARY CONDITIONS ===========
    f_left = build_bc_from_spec(math_settings.bc_left_spec, Ly_grid, bc)
    f_right = build_bc_from_spec(math_settings.bc_right_spec, Ly_grid, bc)
    f_bottom = build_bc_from_spec(math_settings.bc_bottom_spec, Lx_grid, bc)
    f_top = build_bc_from_spec(math_settings.bc_top_spec, Lx_grid, bc)

    f_left_n = bc.neg_bc(f_left)
    f_right_n = bc.neg_bc(f_right)
    f_bottom_n = bc.neg_bc(f_bottom)
    f_top_n = bc.neg_bc(f_top)

    bc_functions = {
        "pos": [f_left, f_right, f_bottom, f_top],
        "neg": [f_left_n, f_right_n, f_bottom_n, f_top_n],
    }

    # =========== SOLVER SETUP ===========
    alpha = args.alpha
    dt_max = 1.0 / (2.0 * alpha * (1.0 / dx**2 + 1.0 / dy**2))
    dt = 0.45 * dt_max

    print(f"Thermal diffusivity alpha={alpha}, dt_max={dt_max:.4g}, dt={dt:.4g}")

    domain = RectangleDomain(Lx_grid, Ly_grid)
    rect_grid = DomainGrid.cartesian_2d(Lx_grid, Ly_grid, res_per_unit=len(x) / Lx_grid)

    config = Heat2DRectConfig(
        dt=dt,
        alpha=alpha,
        cycles_per_frame=args.steps_per_frame,
        alternate=args.alternate,
        steps_per_cycles=args.steps_per_cycles,
    )

    heat_solver = Heat2DRectSolver(domain, rect_grid, config, bc_functions)
    heat_solver.apply_bc()

    # =========== ANALYTIC SOLUTION ===========
    if not args.skip_error:
        u_star_pos = heat_solver.get_analytic_solution(0.0)
        u_star_neg = -u_star_pos
    else:
        u_star_pos = None
        u_star_neg = None

    if args.enforce_BC and u_star_pos is not None:
        phase_key = "pos"
        f_left, f_right, f_bottom, f_top = bc_functions[phase_key]
        u_star_pos[:, 0] = f_left(heat_solver.y)
        u_star_pos[:, -1] = f_right(heat_solver.y)
        u_star_pos[0, :] = f_bottom(heat_solver.x)
        u_star_pos[-1, :] = f_top(heat_solver.x)
        u_star_neg = -u_star_pos

    if heat_solver.u_curr is None:
        raise RuntimeError("Solver u_curr not initialized")
    u_init = heat_solver.u_curr.copy()

    # =========== VISUALIZATION SETUP ===========
    vis_config = VisualizerConfig(
        fps=args.fps,
        seconds=args.seconds,
        dpi=args.dpi,
        bitrate=args.bitrate,
        spin=args.spin,
        deg_per_sec=args.deg_per_sec,
        height_scale=args.height_scale,
        show_colorbar=args.show_colorbar,
        show_function=args.show_function,
        skip_error=args.skip_error,
        save=args.save,
        out=args.out,
    )

    visualizer = Visualizer3D(vis_config, domain_name="Rectangle", pde_name="Heat")

    # Update wireframe kwargs based on grid resolution
    wire_kwargs = vis.wire_kwargs.copy()
    surf_kwargs = vis.surf_kwargs.copy()
    wire_kwargs["ccount"] = 0.2 * Nx
    wire_kwargs["rcount"] = 0.2 * Ny
    surf_kwargs["ccount"] = 0.2 * Nx
    surf_kwargs["rcount"] = 0.2 * Ny

    # Boundary lines data
    y_samp = y
    x_samp = x
    f_left_vals = f_left(y_samp)
    f_right_vals = f_right(y_samp)
    f_bottom_vals = f_bottom(x_samp)
    f_top_vals = f_top(x_samp)

    boundary_lines_data = {
        "num": [
            (0 * y_samp, y_samp, f_left_vals, "left"),
            (Lx_grid + 0 * y_samp, y_samp, f_right_vals, "right"),
            (x_samp, 0 * x_samp, f_bottom_vals, "bottom"),
            (x_samp, Ly_grid + 0 * x_samp, f_top_vals, "top"),
        ],
        "ref": [
            (0 * y_samp, y_samp, f_left_vals, "left"),
            (Lx_grid + 0 * y_samp, y_samp, f_right_vals, "right"),
            (x_samp, 0 * x_samp, f_bottom_vals, "bottom"),
            (x_samp, Ly_grid + 0 * x_samp, f_top_vals, "top"),
        ],
    }

    analytic_title = (
        r"$u(x,y) = u_n + u_m + v \quad\text{where}$"
        "\n"
        r"$u_n=\sum_{n=1}^{N_x}\left[B_n\sinh\left(\frac{n\pi y}{L_x}\right)+B_n^*\sinh\left(\frac{n\pi(L_y-y)}{L_x}\right)\right]\sin\left(\frac{n\pi x}{L_x}\right)$"
        "\n"
        r"$u_m=\sum_{m=1}^{N_y}\left[B_m\sinh\left(\frac{m\pi x}{L_y}\right)+B_m^*\sinh\left(\frac{m\pi(L_x-x)}{L_y}\right)\right]\sin\left(\frac{m\pi y}{L_y}\right)$"
        "\n"
        r"$v = C_{00}\frac{(a-x)(b-y)}{ab} + C_{a0}\frac{x(b-y)}{ab} + C_{0b}\frac{(a-x)y}{ab} + C_{ab}\frac{xy}{ab}$"
    )

    visualizer.setup(
        X,
        Y,
        u_init,
        u_star_pos=u_star_pos,
        u_star_neg=u_star_neg,
        analytic_title=analytic_title if args.show_function else None,
        surf_kwargs=surf_kwargs,
        wire_kwargs=wire_kwargs,
        line_kwargs=vis.line_kwargs,
        boundary_lines_data=boundary_lines_data,
    )

    # Create dict with actual line objects for update_phase_and_lines
    boundary_lines = {
        "num": visualizer.boundary_lines_num,
        "ref": visualizer.boundary_lines_ref,
    }

    # =========== ANIMATION ===========
    t = [0.0]  # Use list for nonlocal in nested function
    steps_per_frame = args.steps_per_frame

    def update_func(frame: int) -> dict:
        """Update function called once per animation frame."""
        for _ in range(steps_per_frame):
            heat_solver.apply_bc()
            heat_solver.step_once()
            t[0] += dt

        # Update phase and boundary lines if needed
        assert heat_solver.x is not None and heat_solver.y is not None
        heat_solver.update_phase_and_lines(
            boundary_lines,
            heat_solver.y,
            heat_solver.x,
            args.height_scale,
        )

        # Prepare state update
        state = {}

        # Phase toggle
        state["phase"] = heat_solver.phase

        # Time display
        if not args.skip_error and u_star_pos is not None and u_star_neg is not None:
            err_inf = heat_solver.compute_error(u_star_pos, u_star_neg)
            state["time_text"] = (
                rf"$t={t[0]:.2f}\qquad \|u-u^*\|_\infty={err_inf:.2e}\qquad$"
            )
        else:
            state["time_text"] = rf"$t={t[0]:.2f}$"

        return state

    def u_curr_getter():
        return heat_solver.u_curr

    visualizer.create_animation(
        update_func,
        u_curr_getter,
        X,
        Y,
    )

    # =========== OUTPUT ===========
    if args.save:
        visualizer.save()
    else:
        visualizer.show()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Universal PDE Solver CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python solve.py --pde heat --domain rect --save --spin
  python solve.py --pde heat --domain rect --fps 30 --seconds 10
  python solve.py --pde heat --domain rect --res 30 --Lx 2.0 --Ly 1.0
        """,
    )

    # PDE and domain selection
    parser.add_argument(
        "--pde",
        type=str,
        default="heat",
        choices=["heat", "wave"],
        help="PDE type (default: heat)",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="rect",
        choices=["rect", "disc", "bar"],
        help="Domain type (default: rect)",
    )

    # Domain parameters
    parser.add_argument(
        "--res",
        type=int,
        default=21,
        help="Grid resolution in points per unit length (default: 21)",
    )
    parser.add_argument(
        "--Lx",
        type=float,
        default=1.0,
        help="Domain length in x-direction (default: 1.0)",
    )
    parser.add_argument(
        "--Ly",
        type=float,
        default=1.0,
        help="Domain length in y-direction (default: 1.0)",
    )

    # Solver parameters
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Diffusion/wave coefficient (default: 1.0)",
    )
    parser.add_argument(
        "--steps_per_frame",
        type=int,
        default=25,
        help="Time steps per animation frame (default: 25)",
    )
    parser.add_argument(
        "--enforce_BC",
        action="store_true",
        help="Enforce BC on analytic solution (for validation)",
    )
    parser.add_argument(
        "--alternate",
        action="store_true",
        help="Alternate boundary condition sign during animation",
    )
    parser.add_argument(
        "--steps_per_cycles",
        type=int,
        default=1000,
        help="Steps per alternation cycle (default: 1000)",
    )

    # Visualization parameters
    parser.add_argument(
        "--fps", type=int, default=60, help="Animation frames per second (default: 60)"
    )
    parser.add_argument(
        "--seconds",
        type=int,
        default=15,
        help="Animation duration in seconds (default: 15)",
    )
    parser.add_argument(
        "--spin", action="store_true", help="Rotate 3D views during animation"
    )
    parser.add_argument(
        "--deg_per_sec",
        type=float,
        default=10.0,
        help="Rotation speed in degrees/second (default: 10.0)",
    )
    parser.add_argument(
        "--height_scale",
        type=float,
        default=1.0,
        help="Vertical scaling for visualization (default: 1.0)",
    )
    parser.add_argument(
        "--hide_colorbar",
        action="store_false",
        dest="show_colorbar",
        help="Hide colorbar",
    )
    parser.add_argument(
        "--hide_function",
        action="store_false",
        dest="show_function",
        help="Hide analytic solution formula",
    )
    parser.add_argument(
        "--skip_error", action="store_true", help="Skip error calculation (faster)"
    )
    parser.add_argument(
        "--dpi", type=int, default=200, help="DPI for export (default: 200)"
    )
    parser.add_argument(
        "--bitrate",
        type=int,
        default=6000,
        help="Bitrate for MP4 export (default: 6000)",
    )

    # Output options
    parser.add_argument(
        "--save",
        action="store_true",
        help="Export MP4 instead of displaying live animation",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="pde_solution.mp4",
        help="Output MP4 filename (default: pde_solution.mp4)",
    )

    args = parser.parse_args()

    # Route to appropriate solver
    if args.pde == "heat" and args.domain == "rect":
        solve_heat_rect(args)
    elif args.pde == "heat" and args.domain == "disc":
        print("ERROR: Disc domain not yet implemented in solve.py")
        print("      Use heat2d_disc_fd.py for now")
        sys.exit(1)
    else:
        print(
            f"ERROR: Unsupported combination: --pde {args.pde} --domain {args.domain}"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
