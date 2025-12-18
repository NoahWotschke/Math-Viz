"""Streamlit UI for 2D Heat Equation Solver.

Simple, expandable web interface for running the rectangular heat equation solver
without needing to use the terminal.

Run with:
    streamlit run app.py
"""

import streamlit as st
import numpy as np
import matplotlib
import json
import inspect

matplotlib.use("agg")  # Non-interactive backend (required for Streamlit)
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from io import BytesIO
import time
import heat2d.vis_settings as vis
import heat2d.bc.funcs as bc
from heat2d.domains.rectangle import RectangleDomain
from heat2d.domains.base import Grid as DomainGrid
from heat2d.solvers.heat2d_rect import Heat2DRectSolver, Heat2DRectConfig
from heat2d.bc.builder import build_bc_from_spec

# Initialize session state for stop button
if "is_animating" not in st.session_state:
    st.session_state.is_animating = False

# Get list of available BC functions (exclude combinators, utilities, and helpers)
EXCLUDE_PATTERNS = {"bc_", "make_bc", "theta_to_s", "const_c", "neg_bc"}
BC_FUNCTIONS = [
    name
    for name in dir(bc)
    if not name.startswith("_")
    and callable(getattr(bc, name))
    and not any(
        name.startswith(pattern) or name == pattern for pattern in EXCLUDE_PATTERNS
    )
]


# Create display names with parameters
def get_bc_display_name(func_name):
    """Get display name with parameters for a BC function."""
    func = getattr(bc, func_name)
    sig = inspect.signature(func)
    # Get parameter names (exclude 's' and 'L' which are always required)
    params = [p for p in sig.parameters.keys() if p not in {"s", "L"}]
    if params:
        return f"{func_name}  # params: {', '.join(params)}"
    else:
        return func_name


BC_DISPLAY_NAMES = [get_bc_display_name(name) for name in BC_FUNCTIONS]


@st.cache_data
def validate_bc_params(func_name, params_dict):
    """Validate and filter parameters for a BC function.

    Only keeps parameters that the function actually accepts.
    Returns filtered dict with valid parameters only.

    Parameters
    ----------
    func_name : str
        Name of the BC function (e.g., 'sin_k', 'const_0')
    params_dict : dict
        User-provided parameters

    Returns
    -------
    dict
        Filtered parameters dict with only valid parameters
    """
    if not hasattr(bc, func_name):
        raise ValueError(f"Unknown BC function: '{func_name}'")

    func = getattr(bc, func_name)
    sig = inspect.signature(func)

    # Get parameter names (exclude 's' and 'L' which are always required)
    valid_params = set(sig.parameters.keys()) - {"s", "L"}

    # Filter user params to only include valid ones
    filtered = {k: v for k, v in params_dict.items() if k in valid_params}

    # Warn if user provided invalid parameters
    invalid = set(params_dict.keys()) - valid_params
    if invalid:
        st.warning(
            f"Function '{func_name}' doesn't accept parameters: {', '.join(invalid)}"
        )

    return filtered


def run_solver_streamlit(
    fps,
    alpha,
    height_scale,
    show_colorbar,
    show_function,
    n_terms,
    steps_per_frame,
    skip_error,
    spin,
    deg_per_sec,
    alternate,
    cycles_per_cycle,
    left_spec,
    right_spec,
    bottom_spec,
    top_spec,
    res,
    Lx,
    Ly,
    cmap,
    t_final,
):
    """Run heat equation solver and display animation in matplotlib window.

    Precalculates all solution frames, then displays them as smooth animation
    with live progress tracking. Supports phase alternation for oscillating BCs.

    Parameters
    ----------
    fps : int
        Frames per second for animation display
    alpha : float
        Thermal diffusivity coefficient
    height_scale : float
        Scaling factor for z-axis visualization
    show_colorbar : bool
        Whether to display colorbar on surface plot
    show_function : bool
        Whether to display analytic formula on reference plot
    n_terms : int
        Number of series terms for analytic solution
    steps_per_frame : int
        Solver steps between animation frames
    skip_error : bool
        Whether to skip error computation (faster but less info)
    spin : bool
        Whether to auto-rotate 3D view
    deg_per_sec : float
        Rotation speed in degrees/second
    alternate : bool
        Whether to alternate between positive/negative BC phases
    cycles_per_cycle : int
        Solver steps per full BC cycle
    left_spec, right_spec, bottom_spec, top_spec : tuple
        BC specifications (name, parameters dict)
    res : int
        Grid resolution (points per unit length)
    Lx, Ly : float
        Domain dimensions
    cmap : str
        Colormap for numeric solution
    t_final : float
        Total simulation time in seconds
    """

    # Set domain dimensions
    Lx_grid = Lx
    Ly_grid = Ly

    # Copy kwargs to avoid modifying globals
    wire_kwargs = vis.wire_kwargs.copy()
    surf_kwargs = vis.surf_kwargs.copy()
    line_kwargs = vis.line_kwargs.copy()

    # Compute grid based on resolution argument (points per unit length)
    Nx = res * int(Lx_grid)
    Ny = res * int(Ly_grid)
    x = np.linspace(0, Lx_grid, Nx)
    y = np.linspace(0, Ly_grid, Ny)
    dx = x[1] - x[0] if Nx > 1 else 1.0
    dy = y[1] - y[0] if Ny > 1 else 1.0
    X, Y = np.meshgrid(x, y, indexing="xy")
    st.info(f"Adjusted grid resolution to {Nx} × {Ny}")

    # Adjust line spacing in wireframe/surface based on grid resolution
    wire_kwargs["ccount"] = 0.2 * Nx
    wire_kwargs["rcount"] = 0.2 * Ny
    surf_kwargs["ccount"] = 0.2 * Nx
    surf_kwargs["rcount"] = 0.2 * Ny

    # Boundary conditions - use new build_bc_from_spec approach
    f_left_pos = build_bc_from_spec(left_spec, Ly_grid, bc)
    f_right_pos = build_bc_from_spec(right_spec, Ly_grid, bc)
    f_bottom_pos = build_bc_from_spec(bottom_spec, Lx_grid, bc)
    f_top_pos = build_bc_from_spec(top_spec, Lx_grid, bc)

    # Negative phase (negated) - use default args to capture values properly
    f_left_neg = lambda s, f=f_left_pos: -f(s)
    f_right_neg = lambda s, f=f_right_pos: -f(s)
    f_bottom_neg = lambda s, f=f_bottom_pos: -f(s)
    f_top_neg = lambda s, f=f_top_pos: -f(s)

    # Solver setup
    dt_max = 1.0 / (2.0 * alpha * (1.0 / dx**2 + 1.0 / dy**2))
    dt = 0.45 * dt_max

    # Create domain, grid, and solver
    domain = RectangleDomain(Lx_grid, Ly_grid)
    domain_grid = DomainGrid.cartesian_2d(
        Lx_grid, Ly_grid, res_per_unit=len(x) / Lx_grid
    )

    # BC functions dict for solver (pos/neg phases)
    bc_functions = {
        "pos": [f_left_pos, f_right_pos, f_bottom_pos, f_top_pos],
        "neg": [f_left_neg, f_right_neg, f_bottom_neg, f_top_neg],
    }

    config = Heat2DRectConfig(
        dt=dt,
        alpha=alpha,
        cycles_per_frame=1,  # One full cycle per frame
        alternate=alternate,
        steps_per_cycles=cycles_per_cycle,  # User-controlled steps per full cycle
    )
    heat_solver = Heat2DRectSolver(domain, domain_grid, config, bc_functions)

    # Initialize with boundary conditions (positive phase)
    heat_solver.apply_bc()

    # Assert that solver arrays are initialized for type checking
    assert heat_solver.u_curr is not None, "Solver u_curr not initialized"
    assert heat_solver.x is not None, "Solver x not initialized"
    assert heat_solver.y is not None, "Solver y not initialized"
    assert heat_solver.X is not None, "Solver X not initialized"
    assert heat_solver.Y is not None, "Solver Y not initialized"

    # Get initial solution and analytic solutions
    u_star_pos = heat_solver.get_analytic_solution(0.0)
    u_star_neg = -u_star_pos

    # Set colorbar limits
    absmax = float(np.nanmax(np.abs(u_star_pos)))
    surf_kwargs["vmin"] = -absmax
    surf_kwargs["vmax"] = absmax

    # Figure setup
    aspect_xy = max(Lx_grid / Ly_grid, Ly_grid / Lx_grid)
    aspect_xy = 1.15 ** np.sqrt(aspect_xy)
    base_w, base_h = 16, 6.5  # Larger for better browser rendering
    fig = plt.figure(figsize=(base_w * aspect_xy, base_h * aspect_xy))
    fig.subplots_adjust(top=0.8)

    ax_num = fig.add_subplot(1, 2, 1, projection="3d")
    ax_ref = fig.add_subplot(1, 2, 2, projection="3d")

    # Wireframes
    wf_pos = ax_ref.plot_wireframe(X, Y, height_scale * u_star_pos, **wire_kwargs)
    wf_neg = ax_ref.plot_wireframe(X, Y, height_scale * u_star_neg, **wire_kwargs)
    wf_neg.set_visible(False)

    # Surface (with user-selected colormap for numeric solution)
    surf_kwargs["cmap"] = cmap
    surf = ax_num.plot_surface(X, Y, height_scale * heat_solver.u_curr, **surf_kwargs)

    # Boundary lines
    y_samp, x_samp = y, x
    f_left_vals = f_left_pos(y_samp)
    f_right_vals = f_right_pos(y_samp)
    f_bottom_vals = f_bottom_pos(x_samp)
    f_top_vals = f_top_pos(x_samp)

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

    boundary_lines = {
        "num": [left_line, right_line, bottom_line, top_line],
        "ref": [left_line_ref, right_line_ref, bottom_line_ref, top_line_ref],
    }

    # Axes setup - apply to both plots for consistency
    for ax in [ax_num, ax_ref]:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("u(x,y,t)")
        ax.set_xlim(0, Lx_grid)
        ax.set_ylim(0, Ly_grid)
        ax.set_zlim(-1, 1)
        ax.set_box_aspect((Lx_grid / Ly_grid, 1.0, 1.0))

    # Initial 3D view angles
    elev0, azim0 = 25, -60
    for ax in [ax_num, ax_ref]:
        ax.view_init(elev=elev0, azim=azim0)

    # Colorbar
    if show_colorbar:
        fig.colorbar(surf, ax=[ax_num, ax_ref], shrink=0.7, pad=0.08, label="u (temp)")

    # Function title
    if show_function:
        ax_ref.set_title(
            r"$u(x,y) = u_n + u_m + v$"
            "\n"
            r"$u_n=\sum_{n=1}^{N_x}\left[B_n\sinh\left(\frac{n\pi y}{L_x}\right)+B_n^*\sinh\left(\frac{n\pi(L_y-y)}{L_x}\right)\right]\sin\left(\frac{n\pi x}{L_x}\right)$"
            "\n"
            r"$u_m=\sum_{m=1}^{N_y}\left[B_m\sinh\left(\frac{m\pi x}{L_y}\right)+B_m^*\sinh\left(\frac{m\pi(L_x-x)}{L_y}\right)\right]\sin\left(\frac{m\pi y}{L_y}\right)$"
            "\n"
            r"$v = C_{00}\frac{(a-x)(b-y)}{ab} + C_{a0}\frac{x(b-y)}{ab} + C_{0b}\frac{(a-x)y}{ab} + C_{ab}\frac{xy}{ab}$",
            pad=12,
        )

    time_text = ax_num.text2D(0.02, 0.98, "", transform=ax_num.transAxes)

    # Prepare boundary lines dict for update_phase_and_lines
    boundary_lines = {
        "num": [left_line, right_line, bottom_line, top_line],
        "ref": [left_line_ref, right_line_ref, bottom_line_ref, top_line_ref],
    }

    # Precalculate all frames
    st.status("Precalculating frames...", expanded=True)
    progress_bar = st.progress(0)
    progress_text = st.empty()

    # Pre-compute analytic solutions (one per phase if alternating, otherwise once)
    u_star_solutions = {}
    if not skip_error:
        progress_bar.progress(0.05)
        progress_text.write("**5%** — Computing analytic solution...")
        # Compute for phase 0
        u_star_solutions[0] = heat_solver.get_analytic_solution(0.0)
        # Compute for phase 1 if alternating
        if alternate:
            original_phase = heat_solver.phase
            heat_solver.phase = 1
            heat_solver.apply_bc()
            u_star_solutions[1] = heat_solver.get_analytic_solution(0.0)
            heat_solver.phase = original_phase
            heat_solver.apply_bc()
    
    frames_data = []  # List of frame data tuples
    t = 0.0

    # Step through solver and collect frames until t_final
    max_steps = int(t_final / (dt * steps_per_frame)) + 1
    step_count = 0
    
    # Collect all frames
    for step in range(max_steps):
        # Run steps for this frame
        for _ in range(steps_per_frame):
            heat_solver.apply_bc()
            heat_solver.step_once()
            t += dt
            step_count += 1

            # Check if phase should toggle (this must happen after step_once)
            if alternate and step_count % cycles_per_cycle == 0:
                heat_solver.phase = 1 - heat_solver.phase
                # Apply BCs with new phase immediately to sync solution
                heat_solver.apply_bc()

        # Get the correct analytic solution for the current phase
        u_star_pos = u_star_solutions.get(heat_solver.phase) if not skip_error else None
        
        frames_data.append(
            {
                "u": heat_solver.u_curr.copy(),
                "t": t,
                "phase": heat_solver.phase,
                "x": heat_solver.x,
                "y": heat_solver.y,
                "X": heat_solver.X,
                "Y": heat_solver.Y,
                "u_star_pos": u_star_pos,
            }
        )

        progress = 0.05 + 0.9 * (step + 1) / max(1, max_steps)
        progress_bar.progress(progress)
        progress_text.write(
            f"**{progress*100:.0f}%** — Frame {step+1}/{max_steps} | Phase: {heat_solver.phase}"
        )
    
    st.success(f"✓ Precalculated {len(frames_data)} frames")

    # Animation loop - render precalculated frames
    t = 0.0
    start_time = time.time()

    def update(frame_idx):
        nonlocal surf, t

        if frame_idx >= len(frames_data):
            return (surf, time_text)

        frame_data = frames_data[frame_idx]
        u_current = frame_data["u"]
        t = frame_data["t"]
        phase = frame_data["phase"]

        # Get BC functions for current phase
        phase_key = "neg" if phase == 1 else "pos"
        bc_funcs = heat_solver.bc_functions[phase_key]
        fL, fR, fB, fT = bc_funcs

        # Update boundary lines for current phase
        y_samp = frame_data["y"]
        x_samp = frame_data["x"]

        boundary_lines["num"][0].set_data(0 * y_samp, y_samp)
        boundary_lines["num"][0].set_3d_properties(height_scale * fL(y_samp))  # type: ignore
        boundary_lines["num"][1].set_data(Lx_grid + 0 * y_samp, y_samp)
        boundary_lines["num"][1].set_3d_properties(height_scale * fR(y_samp))  # type: ignore
        boundary_lines["num"][2].set_data(x_samp, 0 * x_samp)
        boundary_lines["num"][2].set_3d_properties(height_scale * fB(x_samp))  # type: ignore
        boundary_lines["num"][3].set_data(x_samp, Ly_grid + 0 * x_samp)
        boundary_lines["num"][3].set_3d_properties(height_scale * fT(x_samp))  # type: ignore

        boundary_lines["ref"][0].set_data(0 * y_samp, y_samp)
        boundary_lines["ref"][0].set_3d_properties(height_scale * fL(y_samp))  # type: ignore
        boundary_lines["ref"][1].set_data(Lx_grid + 0 * y_samp, y_samp)
        boundary_lines["ref"][1].set_3d_properties(height_scale * fR(y_samp))  # type: ignore
        boundary_lines["ref"][2].set_data(x_samp, 0 * x_samp)
        boundary_lines["ref"][2].set_3d_properties(height_scale * fB(x_samp))  # type: ignore
        boundary_lines["ref"][3].set_data(x_samp, Ly_grid + 0 * x_samp)
        boundary_lines["ref"][3].set_3d_properties(height_scale * fT(x_samp))  # type: ignore

        # Switch between wireframes if phase changed
        if phase == 1:
            wf_pos.set_visible(False)
            wf_neg.set_visible(True)
        else:
            wf_pos.set_visible(True)
            wf_neg.set_visible(False)

        # Update surface with current solution
        surf.remove()
        surf = ax_num.plot_surface(
            frame_data["X"], frame_data["Y"], height_scale * u_current, **surf_kwargs
        )

        # Compute and display error
        if not skip_error and frame_data["u_star_pos"] is not None:
            u_star_pos = frame_data["u_star_pos"]
            # Compare the stored numerical solution against the analytic solution
            err_inf = heat_solver.compute_error(u_star_pos, u_current)
            time_text.set_text(
                rf"$t={t:.2f}\qquad \|u-u^*\|_\infty={err_inf:.2e}\qquad$"
            )
        else:
            time_text.set_text(rf"$t={t:.2f}$")

        # Rotation
        if spin:
            elapsed = time.time() - start_time
            az = azim0 + deg_per_sec * elapsed
            ax_num.view_init(elev=elev0, azim=az)
            ax_ref.view_init(elev=elev0, azim=az)

        return (surf, time_text)

    # Render animation frames from precalculated data
    placeholder = st.empty()
    progress_bar_playback = st.progress(0)
    progress_text_playback = st.empty()

    frame_num = 0
    while st.session_state.is_animating:
        update(frame_num)

        # Convert figure to image (PNG for lossless quality)
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=75, bbox_inches="tight")
        buf.seek(0)

        # Display frame
        placeholder.image(buf)

        # Update playback progress
        progress = frame_num / max(1, len(frames_data))
        progress_bar_playback.progress(progress)
        progress_text_playback.write(
            f"**{progress*100:.0f}%** — Frame {frame_num+1}/{len(frames_data)} | t={frames_data[frame_num]['t']:.3f}s"
        )

        frame_num = (frame_num + 1) % len(
            frames_data
        )  # Loop through precalculated frames

    plt.close(fig)


# ============================================================================
# Streamlit UI
# ============================================================================

st.set_page_config(page_title="PDE Solver", layout="wide")

st.title("PDE Solver")

# PDE Type Selector
pde_type = st.selectbox(
    "Select PDE Type",
    options=[
        "Heat 2D (Rectangle)",
        "Heat 1D (Bar)",
        "Heat 2D (Disc)",
        "Wave 2D (Rectangle)",
    ],
    index=0,
    help="Choose which PDE to solve. Other options coming soon.",
)

if pde_type == "Heat 2D (Rectangle)":
    st.write(
        """
    Visualize the 2D heat equation solving on a rectangle using finite differences.
    Adjust parameters below and click **Run Animation** to visualize the solution.
    """
    )
elif pde_type == "Heat 1D (Bar)":
    st.info("🚧 **Heat 1D (Bar) solver coming soon!**")
    st.stop()
elif pde_type == "Heat 2D (Disc)":
    st.info("🚧 **Heat 2D (Disc) solver coming soon!**")
    st.stop()
elif pde_type == "Wave 2D (Rectangle)":
    st.info("🚧 **Wave 2D (Rectangle) solver coming soon!**")
    st.stop()

st.info(
    "🔗 **For the best quality visualization with real-time 3D rendering, "
    "[download and run locally](https://noahwotschke.github.io/projects.html) — "
    "this web version uses compressed images.**"
)

with st.expander("📊 Cloud Performance Tips"):
    st.write(
        """
    **For faster cloud rendering:**
    - Use "Fast (Cloud)" preset (20 FPS, lower resolution)
    - Lower grid resolution manually if needed
    - Reduce animation duration for quicker computation
    - Skip error calculation if not needed
    
    **Typical cloud load times by resolution:**
    - 21 pts/unit (fast): ~5-10 sec
    - 31 pts/unit (balanced): ~10-20 sec  
    - 51+ pts/unit (high quality): ~30-60 sec
    
    **Precomputation happens once — playback is smooth!**
    """
    )

# Sidebar for parameters
st.sidebar.header("Solver Parameters")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Animation")
    fps = st.slider("FPS", 10, 60, 30, help="Frames per second")
    t_final = st.slider(
        "Animation duration (s)", 0.5, 2.0, 1.0, 0.1, help="Total simulation time"
    )
    steps_per_frame = st.slider(
        "Steps per frame", 5, 100, 30, 5, help="Time steps between frames"
    )

with col2:
    st.subheader("Physics")
    alpha = st.slider(
        "Diffusion coefficient (α)", 0.1, 5.0, 1.0, 0.1, help="Thermal diffusivity"
    )
    n_terms = st.slider(
        "Series terms", 10, 500, 200, 20, help="Terms in analytic solution expansion"
    )
    Lx = st.slider(
        "Domain length (Lx)", 0.5, 5.0, 2.0, 0.5, help="Length in x-direction"
    )
    Ly = st.slider(
        "Domain height (Ly)", 0.5, 5.0, 1.0, 0.5, help="Length in y-direction"
    )

st.sidebar.header("Boundary Conditions")

st.markdown(
    """
*Enter BC function names** 
For functions with parameters, enter as JSON in the Parameters field.  
Example: `sin_k` with parameter `k=2` → enter `{"k": 2}` in parameters
"""
)

col_bc1, col_bc2 = st.columns(2)

# Initialize negate state in session_state if not present
if "negate_left" not in st.session_state:
    st.session_state.negate_left = False
if "negate_bottom" not in st.session_state:
    st.session_state.negate_bottom = False
if "negate_right" not in st.session_state:
    st.session_state.negate_right = False
if "negate_top" not in st.session_state:
    st.session_state.negate_top = False

with col_bc1:
    st.subheader("Left & Bottom")
    left_bc_display = st.selectbox(
        "Left BC function",
        BC_DISPLAY_NAMES,
        index=(
            BC_DISPLAY_NAMES.index(get_bc_display_name("const_0"))
            if "const_0" in BC_FUNCTIONS
            else 0
        ),
        help="Select boundary condition function",
    )
    left_bc_name = (
        left_bc_display.split("  #")[0] if "  #" in left_bc_display else left_bc_display
    )
    left_col1, left_col2 = st.columns([4, 1])
    with left_col1:
        left_params = st.text_input(
            "Left BC params (JSON)", "{}", help='e.g., {"k": 2}'
        )
    with left_col2:
        st.write("")  # Spacer for alignment
        if st.button(
            "Negate L",
            key="btn_negate_left",
            use_container_width=True,
            type="primary" if st.session_state.negate_left else "secondary",
        ):
            st.session_state.negate_left = not st.session_state.negate_left
            st.rerun()

    bottom_bc_display = st.selectbox(
        "Bottom BC function",
        BC_DISPLAY_NAMES,
        index=(
            BC_DISPLAY_NAMES.index(get_bc_display_name("const_0"))
            if "const_0" in BC_FUNCTIONS
            else 0
        ),
        help="Select boundary condition function",
    )
    bottom_bc_name = (
        bottom_bc_display.split("  #")[0]
        if "  #" in bottom_bc_display
        else bottom_bc_display
    )
    bottom_col1, bottom_col2 = st.columns([4, 1])
    with bottom_col1:
        bottom_params = st.text_input(
            "Bottom BC params (JSON)", "{}", help='e.g., {"k": 1}'
        )
    with bottom_col2:
        st.write("")  # Spacer for alignment
        if st.button(
            "Negate B",
            key="btn_negate_bottom",
            use_container_width=True,
            type="primary" if st.session_state.negate_bottom else "secondary",
        ):
            st.session_state.negate_bottom = not st.session_state.negate_bottom
            st.rerun()

with col_bc2:
    st.subheader("Right & Top")
    right_bc_display = st.selectbox(
        "Right BC function",
        BC_DISPLAY_NAMES,
        index=(
            BC_DISPLAY_NAMES.index(get_bc_display_name("sin_k"))
            if "sin_k" in BC_FUNCTIONS
            else 0
        ),
        help="Select boundary condition function",
    )
    right_bc_name = (
        right_bc_display.split("  #")[0]
        if "  #" in right_bc_display
        else right_bc_display
    )
    right_col1, right_col2 = st.columns([4, 1])
    with right_col1:
        right_params = st.text_input(
            "Right BC params (JSON)", '{"k": 1}', help='e.g., {"k": 2}'
        )
    with right_col2:
        st.write("")  # Spacer for alignment
        if st.button(
            "Negate R",
            key="btn_negate_right",
            use_container_width=True,
            type="primary" if st.session_state.negate_right else "secondary",
        ):
            st.session_state.negate_right = not st.session_state.negate_right
            st.rerun()

    top_bc_display = st.selectbox(
        "Top BC function",
        BC_DISPLAY_NAMES,
        index=(
            BC_DISPLAY_NAMES.index(get_bc_display_name("const_0"))
            if "const_0" in BC_FUNCTIONS
            else 0
        ),
        help="Select boundary condition function",
    )
    top_bc_name = (
        top_bc_display.split("  #")[0] if "  #" in top_bc_display else top_bc_display
    )
    top_col1, top_col2 = st.columns([4, 1])
    with top_col1:
        top_params = st.text_input(
            "Top BC params (JSON)", "{}", help='e.g., {"beta": 3.0}'
        )
    with top_col2:
        st.write("")  # Spacer for alignment
        if st.button(
            "Negate T",
            key="btn_negate_top",
            use_container_width=True,
            type="primary" if st.session_state.negate_top else "secondary",
        ):
            st.session_state.negate_top = not st.session_state.negate_top
            st.rerun()


st.sidebar.header("Display Options")

col3, col4 = st.columns(2)

with col3:
    show_colorbar = st.checkbox("Show colorbar", True)
    show_function = st.checkbox("Show analytic formula", True)
    skip_error = st.checkbox(
        "Skip error calculation", False, help="Faster but less information"
    )
    show_boundary_lines = st.checkbox(
        "Show boundary lines", True, help="Visualize applied BC values"
    )
    cmap = st.selectbox(
        "Colormap (numeric solution)",
        ["viridis", "plasma", "inferno", "magma", "cividis"],
        index=0,
        help="Color scheme for numerical solution surface",
    )

with col4:
    height_scale = st.slider("Height scale", 0.1, 1.0, 1.0, 0.1)
    spin = st.checkbox("Rotate view", False, help="Slowly rotate 3D camera")
    alternate = st.checkbox(
        "Toggle alternate BC", False, help="Switch between two boundary conditions"
    )
    res = st.slider("Grid resolution", 21, 101, 21, 10, help="Points per unit length")
    deg_per_sec = st.slider(
        "Rotation speed (deg/sec)",
        1.0,
        30.0,
        10.0,
        1.0,
        help="Only if Rotate view is on",
    )
    cycles_per_cycle = st.slider(
        "Steps per cycle", 200, 10000, 2000, 200, help="Time steps per full cycle"
    )

# Run button
if st.button("Run Animation", use_container_width=True):
    st.session_state.is_animating = True

    try:
        # Parse BC parameters from JSON
        try:
            left_params_dict = json.loads(left_params)
            right_params_dict = json.loads(right_params)
            bottom_params_dict = json.loads(bottom_params)
            top_params_dict = json.loads(top_params)
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON in BC parameters: {e}")
            st.stop()

        # Validate and filter parameters for each function
        try:
            left_params_dict = validate_bc_params(left_bc_name, left_params_dict)
            right_params_dict = validate_bc_params(right_bc_name, right_params_dict)
            bottom_params_dict = validate_bc_params(bottom_bc_name, bottom_params_dict)
            top_params_dict = validate_bc_params(top_bc_name, top_params_dict)
        except ValueError as e:
            st.error(f"BC Function error: {e}")
            st.stop()

        # Convert BC names to specs with validated parameters
        left_spec = (left_bc_name, left_params_dict)
        right_spec = (right_bc_name, right_params_dict)
        bottom_spec = (bottom_bc_name, bottom_params_dict)
        top_spec = (top_bc_name, top_params_dict)

        # Apply negation wrapper if requested
        if st.session_state.negate_left:
            left_spec = ("neg", {"f": left_spec})
        if st.session_state.negate_right:
            right_spec = ("neg", {"f": right_spec})
        if st.session_state.negate_bottom:
            bottom_spec = ("neg", {"f": bottom_spec})
        if st.session_state.negate_top:
            top_spec = ("neg", {"f": top_spec})

        run_solver_streamlit(
            fps=fps,
            alpha=alpha,
            height_scale=height_scale,
            show_colorbar=show_colorbar,
            show_function=show_function,
            n_terms=n_terms,
            steps_per_frame=steps_per_frame,
            skip_error=skip_error,
            spin=spin,
            deg_per_sec=deg_per_sec,
            alternate=alternate,
            cycles_per_cycle=cycles_per_cycle,
            left_spec=left_spec,
            right_spec=right_spec,
            bottom_spec=bottom_spec,
            top_spec=top_spec,
            res=res,
            Lx=Lx,
            Ly=Ly,
            cmap=cmap,
            t_final=t_final,
        )
        st.success("✓ Animation stopped!")

    except Exception as e:
        st.error(f"Error running solver: {e}")
        import traceback

        st.code(traceback.format_exc())

st.sidebar.markdown("---")
st.sidebar.subheader("⚡ Performance Presets")
preset = st.sidebar.radio(
    "Quick preset",
    ["Custom", "Fast (Cloud)", "Balanced", "High Quality"],
    index=2,
    help="Presets optimize for different environments",
)

# Apply preset defaults
preset_settings = {
    "Custom": {"fps": 30, "duration": 1.0, "steps": 30, "res": 21},
    "Fast (Cloud)": {"fps": 20, "duration": 0.8, "steps": 20, "res": 21},
    "Balanced": {"fps": 30, "duration": 1.0, "steps": 30, "res": 31},
    "High Quality": {"fps": 60, "duration": 2.0, "steps": 50, "res": 51},
}

if preset != "Custom":
    settings = preset_settings[preset]
    # These will be used as defaults below
    default_fps = settings["fps"]
    default_duration = settings["duration"]
    default_steps = settings["steps"]
    default_res = settings["res"]
else:
    default_fps = 30
    default_duration = 1.0
    default_steps = 30
    default_res = 21

st.sidebar.markdown("---")

st.sidebar.write(
    """
**About:** This solver uses finite difference time-stepping to approximate 
the 2D parabolic heat equation with Dirichlet boundary conditions. 
The left plot shows the numerical solution; the right shows the analytic reference.
"""
)
