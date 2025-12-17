"""Streamlit UI for 2D Heat Equation Solver using Plotly (fast web rendering).

Clean, fast web interface using client-side Plotly rendering instead of matplotlib.

Run with:
    streamlit run app_plotly.py
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import json
import inspect
import heat2d.bc.funcs as bc
from heat2d.domains.rectangle import RectangleDomain
from heat2d.domains.base import Grid as DomainGrid
from heat2d.solvers.heat2d_rect import Heat2DRectSolver, Heat2DRectConfig
from heat2d.bc.builder import build_bc_from_spec

# Page config
st.set_page_config(page_title="Math-Viz: PDE Solver", layout="wide")
st.title("🌡️ Math-Viz: PDE Visualization")
st.write("Fast web-based visualization of the 2D heat equation using Plotly")

# Get BC functions
EXCLUDE_PATTERNS = {"bc_", "make_bc", "theta_to_s", "const_c", "neg_bc"}
BC_FUNCTIONS = [
    name for name in dir(bc)
    if not name.startswith("_") and callable(getattr(bc, name))
    and not any(name.startswith(p) or name == p for p in EXCLUDE_PATTERNS)
]

def get_bc_display_name(func_name):
    func = getattr(bc, func_name)
    sig = inspect.signature(func)
    params = [p for p in sig.parameters.keys() if p not in {"s", "L"}]
    if params:
        return f"{func_name}  # params: {', '.join(params)}"
    return func_name

BC_DISPLAY_NAMES = [get_bc_display_name(name) for name in BC_FUNCTIONS]

def validate_bc_params(func_name, params_dict):
    if not hasattr(bc, func_name):
        raise ValueError(f"Unknown BC function: '{func_name}'")
    func = getattr(bc, func_name)
    sig = inspect.signature(func)
    valid_params = set(sig.parameters.keys()) - {"s", "L"}
    filtered = {k: v for k, v in params_dict.items() if k in valid_params}
    invalid = set(params_dict.keys()) - valid_params
    if invalid:
        st.warning(f"Function '{func_name}' doesn't accept: {', '.join(invalid)}")
    return filtered

# Sidebar controls
st.sidebar.header("⚙️ Parameters")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Domain")
    Lx = st.slider("Length (Lx)", 0.5, 5.0, 2.0, 0.5)
    Ly = st.slider("Width (Ly)", 0.5, 5.0, 1.0, 0.5)
    res = st.slider("Resolution", 5, 30, 15, 1)

with col2:
    st.subheader("Physics")
    alpha = st.slider("Diffusivity (α)", 0.1, 5.0, 1.0, 0.1)
    t_final = st.slider("Simulation time", 0.5, 10.0, 3.0, 0.5)
    n_frames = st.slider("Frames to render", 10, 100, 30, 5)

# Boundary conditions
st.sidebar.subheader("Boundary Conditions")

left_bc_display = st.sidebar.selectbox("Left BC", BC_DISPLAY_NAMES, index=0)
left_bc_name = left_bc_display.split("  #")[0] if "  #" in left_bc_display else left_bc_display
left_params = st.sidebar.text_input("Left params (JSON)", "{}")

right_bc_display = st.sidebar.selectbox("Right BC", BC_DISPLAY_NAMES, index=BC_DISPLAY_NAMES.index(get_bc_display_name("sin_k")) if "sin_k" in BC_FUNCTIONS else 0)
right_bc_name = right_bc_display.split("  #")[0] if "  #" in right_bc_display else right_bc_display
right_params = st.sidebar.text_input("Right params (JSON)", '{"k": 1}')

bottom_bc_display = st.sidebar.selectbox("Bottom BC", BC_DISPLAY_NAMES, index=0)
bottom_bc_name = bottom_bc_display.split("  #")[0] if "  #" in bottom_bc_display else bottom_bc_display
bottom_params = st.sidebar.text_input("Bottom params (JSON)", "{}")

top_bc_display = st.sidebar.selectbox("Top BC", BC_DISPLAY_NAMES, index=0)
top_bc_name = top_bc_display.split("  #")[0] if "  #" in top_bc_display else top_bc_display
top_params = st.sidebar.text_input("Top params (JSON)", "{}")

if st.sidebar.button("Run Simulation", use_container_width=True, type="primary"):
    try:
        # Parse parameters
        left_params_dict = json.loads(left_params)
        right_params_dict = json.loads(right_params)
        bottom_params_dict = json.loads(bottom_params)
        top_params_dict = json.loads(top_params)
        
        # Validate
        left_params_dict = validate_bc_params(left_bc_name, left_params_dict)
        right_params_dict = validate_bc_params(right_bc_name, right_params_dict)
        bottom_params_dict = validate_bc_params(bottom_bc_name, bottom_params_dict)
        top_params_dict = validate_bc_params(top_bc_name, top_params_dict)
        
        # Setup solver
        Nx = res * int(Lx)
        Ny = res * int(Ly)
        x = np.linspace(0, Lx, Nx)
        y = np.linspace(0, Ly, Ny)
        dx = x[1] - x[0] if Nx > 1 else 1.0
        dy = y[1] - y[0] if Ny > 1 else 1.0
        X, Y = np.meshgrid(x, y, indexing="xy")
        
        # Time stepping
        dt_max = 1.0 / (2.0 * alpha * (1.0 / dx**2 + 1.0 / dy**2))
        dt = 0.45 * dt_max
        n_steps = int(t_final / dt)
        
        # Create solver
        domain = RectangleDomain(Lx, Ly)
        domain_grid = DomainGrid.cartesian_2d(Lx, Ly, res_per_unit=res)
        
        left_spec = (left_bc_name, left_params_dict)
        right_spec = (right_bc_name, right_params_dict)
        bottom_spec = (bottom_bc_name, bottom_params_dict)
        top_spec = (top_bc_name, top_params_dict)
        
        f_left = build_bc_from_spec(left_spec, Ly, bc)
        f_right = build_bc_from_spec(right_spec, Ly, bc)
        f_bottom = build_bc_from_spec(bottom_spec, Lx, bc)
        f_top = build_bc_from_spec(top_spec, Lx, bc)
        
        bc_functions = {
            "pos": [f_left, f_right, f_bottom, f_top],
            "neg": [lambda s: -f_left(s), lambda s: -f_right(s), lambda s: -f_bottom(s), lambda s: -f_top(s)],
        }
        
        config = Heat2DRectConfig(dt=dt, alpha=alpha)
        solver = Heat2DRectSolver(domain, domain_grid, config, bc_functions)
        solver.apply_bc()
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Solve and store snapshots
        snapshots = []
        times = []
        steps_per_frame = max(1, n_steps // n_frames)
        
        for step in range(n_steps):
            solver.apply_bc()
            solver.step_once()
            
            if step % steps_per_frame == 0:
                snapshots.append(solver.u_curr.copy())
                times.append(step * dt)
            
            progress_bar.progress((step + 1) / n_steps)
            status_text.text(f"Solving: {step+1}/{n_steps} steps")
        
        progress_bar.empty()
        status_text.empty()
        
        # Create Plotly animation
        st.subheader("Solution Animation")
        
        frames = []
        for i, u in enumerate(snapshots):
            frames.append(go.Frame(data=[go.Surface(z=u, x=x, y=y, colorscale="Viridis")], name=str(i)))
        
        fig = go.Figure(
            data=[go.Surface(z=snapshots[0], x=x, y=y, colorscale="Viridis", name="u(x,y,t)")],
            frames=frames
        )
        
        fig.update_layout(
            title=f"2D Heat Equation Solution",
            scene=dict(
                xaxis_title="x",
                yaxis_title="y",
                zaxis_title="u(x,y,t)",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
            ),
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(label="Play", method="animate", args=[None, {"frame": {"duration": 50, "redraw": True}, "fromcurrent": True}]),
                    dict(label="Pause", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}])
                ]
            )],
            sliders=[dict(
                active=0,
                steps=[dict(args=[[f.name], dict(frame=dict(duration=0, redraw=True), mode="immediate")], label=f"t={times[i]:.2f}") for i, f in enumerate(frames)],
                transition=dict(duration=0)
            )],
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Grid size", f"{Nx}×{Ny}")
        with col2:
            st.metric("Time steps", n_steps)
        with col3:
            st.metric("Final time", f"{times[-1]:.2f}")
        
        st.success("✓ Simulation complete!")
        
    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON in parameters: {e}")
    except Exception as e:
        st.error(f"Error: {e}")
        import traceback
        st.code(traceback.format_exc())

# Footer
st.divider()
st.markdown("""
**About:** Math-Viz uses Plotly for interactive client-side rendering (fast on web) and matplotlib for local high-quality outputs.

**For best performance:** Download and run locally with `python heat2d_rect_fd.py --spin`
""")
