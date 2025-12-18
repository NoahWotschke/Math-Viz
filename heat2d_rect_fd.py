"""2D Heat Equation Solver and Visualizer (Rectangular Domain)

Solves the 2D parabolic heat equation on a rectangle using finite differences:

    ∂u/∂t = α(∂²u/∂x² + ∂²u/∂y²)  on [0, Lx] × [0, Ly]

Features:
  - Explicit finite-difference time-stepping with stability control (CFL factor 0.45)
  - Customizable boundary conditions on all four edges
  - Side-by-side visualization: numerical solution vs. analytic reference
  - Analytical solution via series expansion (Legendre quadrature)
  - Real-time 3D surface rendering with rotation option
  - MP4 export capability via FFmpeg

Configuration:
  - Grid resolution: set via heat2d/math_settings.py
  - BC functions: selected via command-line arguments
  - Animation: fps, duration, dpi, bitrate via CLI flags

Usage:
    python heat2d_rect_fd.py [--save] [--spin] [--fps 60] [--seconds 15] ...

For a list of all options:
    python heat2d_rect_fd.py --help
"""

# Standard library
import argparse
import os

# Third-party
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

# Local package helpers
import heat2d.vis_settings as vis
import heat2d.bc.funcs as bc
import heat2d.math_settings as math_settings
from heat2d.bc.builder import build_bc_from_spec
from heat2d.domains.rectangle import RectangleDomain
from heat2d.domains.base import Grid as DomainGrid
from heat2d.solvers.heat2d_rect import Heat2DRectSolver, Heat2DRectConfig

""" Initial grid defaults (will be updated by args) """
Lx_initial, Ly_initial = 1.0, 1.0
res_initial = 21  # Points per unit length

# Compute initial grid
Nx = res_initial * int(Lx_initial)
Ny = res_initial * int(Ly_initial)
x = np.linspace(0, Lx_initial, Nx)
y = np.linspace(0, Ly_initial, Ny)
dx = x[1] - x[0]
dy = y[1] - y[0]
X, Y = np.meshgrid(x, y, indexing="xy")

print("x range:", x[0], "to", x[-1], "Nx=", len(x))
print("y range:", y[0], "to", y[-1], "Ny=", len(y))

""" Plotting appearance defaults (from vis_settings module) """
surf_kwargs = vis.surf_kwargs
wire_kwargs = vis.wire_kwargs
line_kwargs = vis.line_kwargs


def main():
    """Args Parser for Rectangle Heat Equation FD Solver and Visualizer"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save", action="store_true", help="Export MP4 instead of showing live."
    )
    parser.add_argument(
        "--out", type=str, default="heat_3d_rect.mp4", help="Output MP4 filename."
    )
    parser.add_argument("--fps", type=int, default=60, help="FPS for animation/export.")
    parser.add_argument("--seconds", type=int, default=15, help="Duration (seconds).")
    parser.add_argument("--dpi", type=int, default=200, help="DPI for export.")
    parser.add_argument(
        "--bitrate", type=int, default=6000, help="Bitrate for mp4 export."
    )
    parser.add_argument(
        "--spin", action="store_true", help="Slowly rotate both 3D views."
    )
    parser.add_argument(
        "--deg_per_sec", type=float, default=10.0, help="Rotation speed (deg/sec)."
    )
    parser.add_argument(
        "--steps_per_frame",
        type=int,
        default=25,
        help="Time steps to advance per frame.",
    )
    parser.add_argument(
        "--enforce_BC",
        action="store_true",
        help="Enforce boundary conditions on analytic solution (for validation).",
    )
    parser.add_argument(
        "--alternate",
        action="store_true",
        help="Alternate boundary condition sign (positive/negative phases).",
    )
    parser.add_argument(
        "--steps_per_cycles", type=int, default=1000, help="Steps per cycle."
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Diffusion coefficient (thermal diffusivity).",
    )
    parser.add_argument(
        "--height_scale",
        type=float,
        default=1.0,
        help="Vertical scaling factor for 3D visualization.",
    )
    parser.add_argument(
        "--hide_colorbar",
        action="store_false",
        dest="show_colorbar",
        help="Hide colorbar on plots.",
    )
    parser.add_argument(
        "--hide_function",
        action="store_false",
        dest="show_function",
        help="Hide analytic solution formula title.",
    )
    parser.add_argument(
        "--n_terms",
        type=int,
        default=200,
        help="Number of series expansion terms for analytic solution.",
    )
    parser.add_argument(
        "--res",
        type=int,
        default=21,
        help="Grid resolution (points per unit length; default 21).",
    )
    parser.add_argument(
        "--Lx",
        type=float,
        default=1.0,
        help="Domain length in x-direction (default 1.0).",
    )
    parser.add_argument(
        "--Ly",
        type=float,
        default=1.0,
        help="Domain length in y-direction (default 1.0).",
    )
    parser.add_argument(
        "--skip_error",
        action="store_true",
        help="Skip L∞ error calculation (faster animation).",
    )
    args = parser.parse_args()

    """ Set Domain Dimensions """
    Lx_grid = args.Lx
    Ly_grid = args.Ly
    print(f"Lx,Ly = {Lx_grid}, {Ly_grid}")

    """ Resolution Adjustment """
    # Compute grid based on resolution argument (points per unit length)
    res = args.res
    Nx = res * int(Lx_grid)
    Ny = res * int(Ly_grid)
    x = np.linspace(0, Lx_grid, Nx)
    y = np.linspace(0, Ly_grid, Ny)
    dx = x[1] - x[0] if Nx > 1 else 1.0
    dy = y[1] - y[0] if Ny > 1 else 1.0
    X, Y = np.meshgrid(x, y, indexing="xy")
    print(f"Adjusted resolution to Nx,Ny = {Nx} {Ny}")
    print(f"dx={dx:.4g}, dy={dy:.4g}")

    """ Build Boundary Conditions from specs """
    f_left = build_bc_from_spec(math_settings.bc_left_spec, Ly_grid, bc)
    f_right = build_bc_from_spec(math_settings.bc_right_spec, Ly_grid, bc)
    f_bottom = build_bc_from_spec(math_settings.bc_bottom_spec, Lx_grid, bc)
    f_top = build_bc_from_spec(math_settings.bc_top_spec, Lx_grid, bc)

    # Compute negative phase (negate all positive BCs)
    f_left_n = bc.neg_bc(f_left)
    f_right_n = bc.neg_bc(f_right)
    f_bottom_n = bc.neg_bc(f_bottom)
    f_top_n = bc.neg_bc(f_top)

    # Organize BCs into dictionary with 'pos' and 'neg' keys
    bc_functions = {
        "pos": [f_left, f_right, f_bottom, f_top],
        "neg": [f_left_n, f_right_n, f_bottom_n, f_top_n],
    }

    # Aliases for clarity
    f_left_pos, f_right_pos, f_bottom_pos, f_top_pos = f_left, f_right, f_bottom, f_top
    f_left_neg, f_right_neg, f_bottom_neg, f_top_neg = (
        f_left_n,
        f_right_n,
        f_bottom_n,
        f_top_n,
    )

    # Adjust line spacing in wireframe/surface based on grid resolution
    wire_kwargs["ccount"] = 0.2 * Nx
    wire_kwargs["rcount"] = 0.2 * Ny

    surf_kwargs["ccount"] = 0.2 * Nx
    surf_kwargs["rcount"] = 0.2 * Ny

    """ Diffusion coefficient and Stability limit """
    alpha = args.alpha
    dt_max = 1.0 / (2.0 * alpha * (1.0 / dx**2 + 1.0 / dy**2))
    dt = 0.45 * dt_max
    print(f"dx={dx:.4g}, dy={dy:.4g}, dt_max={dt_max:.4g}, using dt={dt:.4g}")

    """ Create domain, grid, and solver """
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

    # Initialize with boundary conditions (positive phase)
    heat_solver.apply_bc()

    # Get analytic solutions (computed once, reused for all frames - steady-state)
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

    # Reference to numerical solution
    u = heat_solver.u_curr
    u_new = heat_solver.u_next

    # Height scaling (so it looks nice if u is in [0,1])
    height_scale = args.height_scale

    # set surf vmin/vmax based on analytic solution (if available)
    if u_star_pos is not None:
        absmax = float(np.nanmax(np.abs(u_star_pos)))
        surf_kwargs["vmin"] = -absmax * height_scale
        surf_kwargs["vmax"] = absmax * height_scale
    else:
        # If no analytic solution, scale based on numerical solution range
        absmax = float(np.nanmax(np.abs(heat_solver.u_curr)))
        surf_kwargs["vmin"] = -absmax * height_scale
        surf_kwargs["vmax"] = absmax * height_scale

    """ Animation Setup """
    t = 0.0
    steps_per_frame = args.steps_per_frame
    step = 0

    # Figure and Axes
    aspect_xy = max(Lx_grid / Ly_grid, Ly_grid / Lx_grid)
    aspect_xy = 1.15 ** np.sqrt(aspect_xy)
    base_w, base_h = 12, 5  # Default window size
    fig = plt.figure(figsize=(base_w * aspect_xy, base_h * aspect_xy))

    fig.subplots_adjust(top=0.8)  # give more space for titles
    ax_num = fig.add_subplot(1, 2, 1, projection="3d")
    ax_ref = fig.add_subplot(1, 2, 2, projection="3d")

    # analytic wireframe
    wf_pos = ax_ref.plot_wireframe(
        X,
        Y,
        height_scale * u_star_pos,
        **wire_kwargs,
    )
    wf_neg = ax_ref.plot_wireframe(
        X,
        Y,
        height_scale * u_star_neg,
        **wire_kwargs,
    )
    wf_neg.set_visible(False)  # start in + phase

    # initial numerical surface
    surf = ax_num.plot_surface(X, Y, height_scale * u, **surf_kwargs)

    """ Set Analytic and Numeric Boundary Lines """
    y_samp = y
    x_samp = x

    # Pre-compute all boundary function values (optimization: avoid recomputation)
    f_left_vals = f_left_pos(y_samp)
    f_right_vals = f_right_pos(y_samp)
    f_bottom_vals = f_bottom_pos(x_samp)
    f_top_vals = f_top_pos(x_samp)

    # ref boundary lines
    (left_line_ref,) = ax_ref.plot(
        0 * y_samp, y_samp, height_scale * f_left_vals, **line_kwargs
    )
    (right_line_ref,) = ax_ref.plot(
        Lx_grid + 0 * y_samp, y_samp, height_scale * f_right_vals, **line_kwargs
    )
    (bottom_line_ref,) = ax_ref.plot(
        x_samp, 0 * x_samp, height_scale * f_bottom_vals, **line_kwargs
    )
    (top_line_ref,) = ax_ref.plot(
        x_samp, Ly_grid + 0 * x_samp, height_scale * f_top_vals, **line_kwargs
    )

    # numeric boundary lines
    (left_line,) = ax_num.plot(
        0 * y_samp, y_samp, height_scale * f_left_vals, **line_kwargs
    )
    (right_line,) = ax_num.plot(
        Lx_grid + 0 * y_samp, y_samp, height_scale * f_right_vals, **line_kwargs
    )
    (bottom_line,) = ax_num.plot(
        x_samp, 0 * x_samp, height_scale * f_bottom_vals, **line_kwargs
    )
    (top_line,) = ax_num.plot(
        x_samp, Ly_grid + 0 * x_samp, height_scale * f_top_vals, **line_kwargs
    )

    # Store lines in a dictionary for faster updates
    boundary_lines = {
        "num": [left_line, right_line, bottom_line, top_line],
        "ref": [left_line_ref, right_line_ref, bottom_line_ref, top_line_ref],
    }

    """ Set Axes Labels and Limits """
    ax_num.set_xlabel("x")
    ax_num.set_ylabel("y")
    ax_num.set_zlabel("u(x,y,t)")
    ax_num.set_ylim(0, Ly_grid)
    ax_num.set_xlim(0, Lx_grid)
    ax_ref.set_xlim(0, Lx_grid)
    ax_ref.set_ylim(0, Ly_grid)
    ax_num.set_zlim(-1, 1)
    ax_ref.set_zlim(-1, 1)

    elev0 = 25
    azim0 = -60
    ax_num.view_init(elev=elev0, azim=azim0)
    ax_ref.view_init(elev=elev0, azim=azim0)

    # x/y box matches Lx:Ly, z box fixed (does NOT change your zlim values)
    ax_num.set_box_aspect((Lx_grid / Ly_grid, 1.0, 1.0))
    ax_ref.set_box_aspect((Lx_grid / Ly_grid, 1.0, 1.0))

    """ Set Colorbar and Time Text """
    # show colorbar if requested
    if args.show_colorbar:
        fig.colorbar(surf, ax=[ax_num, ax_ref], shrink=0.7, pad=0.08, label="u (temp)")

    # show analytic solution function if requested
    if args.show_function:
        ax_ref.set_title(
            r"$u(x,y) = u_n + u_m + v \quad\text{where}$"
            "\n"
            r"$u_n=\sum_{n=1}^{N_x}\left[B_n\sinh\left(\frac{n\pi y}{L_x}\right)+B_n^*\sinh\left(\frac{n\pi(L_y-y)}{L_x}\right)\right]\sin\left(\frac{n\pi x}{L_x}\right)$"
            "\n"
            r"$u_m=\sum_{m=1}^{N_y}\left[B_m\sinh\left(\frac{m\pi x}{L_y}\right)+B_m^*\sinh\left(\frac{m\pi(L_x-x)}{L_y}\right)\right]\sin\left(\frac{m\pi y}{L_y}\right)$"
            "\n"
            r"$v = C_{00}\frac{(a-x)(b-y)}{ab} + C_{a0}\frac{x(b-y)}{ab} + C_{0b}\frac{(a-x)y}{ab} + C_{ab}\frac{xy}{ab}$",
            pad=12,
        )

    time_text = ax_num.text2D(0.02, 0.98, "", transform=ax_num.transAxes)

    # Animation Setup
    fps = args.fps
    seconds = args.seconds
    frames = fps * seconds

    # Prepare boundary lines dict for update_phase_and_lines
    boundary_lines = {
        "num": [left_line, right_line, bottom_line, top_line],
        "ref": [left_line_ref, right_line_ref, bottom_line_ref, top_line_ref],
    }

    def update(frame):
        nonlocal t, surf

        # Perform several time steps per frame
        for _ in range(steps_per_frame):
            heat_solver.apply_bc()
            heat_solver.step_once()
            t += dt

        # Get current solution (after array swap in step_once)
        u_current = heat_solver.u_curr

        # Update phase and boundary lines if needed
        # Type assertion: we verified in __init__ that x and y are not None
        assert heat_solver.x is not None and heat_solver.y is not None
        heat_solver.update_phase_and_lines(
            boundary_lines,
            heat_solver.y,
            heat_solver.x,
            height_scale,
        )

        # Switch between wireframes if phase changed (simple toggle)
        if heat_solver.phase == 1:
            wf_pos.set_visible(False)
            wf_neg.set_visible(True)
        else:
            wf_pos.set_visible(True)
            wf_neg.set_visible(False)

        # Update surface with current solution
        assert heat_solver.X is not None and heat_solver.Y is not None
        surf.remove()
        surf = ax_num.plot_surface(
            heat_solver.X, heat_solver.Y, height_scale * u_current, **surf_kwargs
        )

        # Compute and display error
        if not args.skip_error:
            err_inf = heat_solver.compute_error(u_star_pos, u_star_neg)
            time_text.set_text(
                rf"$t={t:.2f}\qquad \|u-u^*\|_\infty={err_inf:.2e}\qquad$"
            )
        else:
            time_text.set_text(rf"$t={t:.2f}$")

        # Rotate views if requested
        if args.spin:
            az = azim0 + args.deg_per_sec * (frame / fps)
            ax_num.view_init(elev=elev0, azim=az)
            ax_ref.view_init(elev=elev0, azim=az)

        return (surf, time_text)

    """
        python heat2d_rect_fd.py \
        --save \
        --out rectangle_solution.mp4 \
        --fps 60 \
        --seconds 15 \
        --dpi 300 \
        --bitrate 8000 \
        --spin \
        --deg_per_sec 5 \
        --enforce_BC False \
        --steps_per_frame 100 \
        --alternate \
        --steps_per_cycles 1000
    """
    anim = FuncAnimation(fig, update, interval=1000 / fps, frames=frames, blit=False)

    if args.save:
        from matplotlib.animation import FFMpegWriter

        plt.ioff()
        outpath = args.out
        # Use simple manual prints instead.
        writer = FFMpegWriter(
            fps=fps,
            bitrate=args.bitrate,
            codec="libx264",
        )
        # Manual robust saving with tqdm
        with writer.saving(fig, outpath, args.dpi):
            for _, frame in enumerate(
                tqdm(anim.frame_seq, total=frames, desc="Exporting MP4")
            ):
                update(frame)
                fig.canvas.draw()
                writer.grab_frame()
        print(f"Saved animation to: {os.path.abspath(outpath)}")
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
