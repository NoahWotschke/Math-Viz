"""Streamlit UI for 2D Heat Equation Solver - Plotly version.

Fast, cloud-optimized web interface using Plotly for interactive 3D visualization.
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
st.set_page_config(
    page_title="Math-Viz: PDE Visualizer",
    page_icon="📊",
    layout="wide",
)

# Get list of available BC functions
EXCLUDE_PATTERNS = {"bc_", "make_bc", "theta_to_s", "const_c", "neg_bc"}
BC_FUNCTIONS = [
    name
    for name in dir(bc)
    if not name.startswith("_")
    and callable(getattr(bc, name))
    and not any(name.startswith(pattern) or name == pattern for pattern in EXCLUDE_PATTERNS)
]

def get_bc_display_name(func_name):
    """Get display name with parameters for a BC function."""
    func = getattr(bc, func_name)
    sig = inspect.signature(func)
    params = [p for p in sig.parameters.keys() if p not in {"s", "L"}]
    if params:
        return f"{func_name}  # params: {', '.join(params)}"
    return func_name

BC_DISPLAY_NAMES = [get_bc_display_name(name) for name in BC_FUNCTIONS]

def validate_bc_params(func_name, params_dict):
    """Validate and filter parameters for a BC function."""
    if not hasattr(bc, func_name):
        raise ValueError(f"Unknown BC function: '{func_name}'")
    
    func = getattr(bc, func_name)
    sig = inspect.signature(func)
    valid_params = set(sig.parameters.keys()) - {"s", "L"}
    filtered = {k: v for k, v in params_dict.items() if k in valid_params}
    
    invalid = set(params_dict.keys()) - valid_params
    if invalid:
        st.warning(f"Function '{func_name}' doesn't accept parameters: {', '.join(invalid)}")
    
    return filtered

def create_plotly_figure(X, Y, u_num, u_ref, title_num="Numerical Solution", title_ref="Analytical Solution", height_scale=1.0, cmap="viridis"):
    """Create interactive Plotly 3D surface plot."""
    # Normalize for better scaling
    absmax = max(np.nanmax(np.abs(u_num)), np.nanmax(np.abs(u_ref)))
    
    fig = go.Figure()
    
    # Numerical solution
    fig.add_trace(go.Surface(
        x=X[0], y=Y[:, 0],
        z=height_scale * u_num,
        colorscale=cmap,
        showscale=True,
        name="Numerical",
        cmin=-absmax, cmax=absmax,
    ))
    
    # Analytical solution
    fig.add_trace(go.Surface(
        x=X[0], y=Y[:, 0],
        z=height_scale * u_ref,
        colorscale="Greys",
        showscale=False,
        name="Analytical",
        opacity=0.7,
        cmin=-absmax, cmax=absmax,
    ))
    
    fig.update_layout(
        title=f"{title_num} vs {title_ref}",
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="u(x,y,t)",
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            )
        ),
        width=1200,
        height=600,
        hovermode="closest",
    )
    
    return fig

# UI
st.title("🔬 PDE Visualizer")
st.markdown("**Fast cloud-native visualization of PDEs**")

st.info("""
⚡ **This version uses Plotly for fast interactive visualization.**
For real-time animated 3D surfaces, [run the CLI locally](https://github.com/NoahWotschke/Math-Viz).
""")

# Parameters
st.sidebar.header("Parameters")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Domain & Physics")
    Lx = st.slider("Domain length (Lx)", 0.5, 5.0, 2.0, 0.5)
    Ly = st.slider("Domain height (Ly)", 0.5, 5.0, 1.0, 0.5)
    alpha = st.slider("Diffusion (α)", 0.1, 5.0, 1.0, 0.1)
    res = st.slider("Resolution", 10, 40, 21, 5, help="Lower = faster")

with col2:
    st.subheader("Solver")
    n_terms = st.slider("Series terms", 10, 500, 200, 50)
    height_scale = st.slider("Height scale", 0.1, 2.0, 1.0, 0.1)
    cmap = st.selectbox("Colormap", ["Viridis", "Plasma", "Inferno", "Blues", "Greens"])
    enforce_BC = st.checkbox("Enforce BC on analytic", False)

st.sidebar.header("Boundary Conditions")

# Simple BC setup
left_bc = st.selectbox("Left BC", BC_DISPLAY_NAMES, index=0)
left_bc_name = left_bc.split("  #")[0] if "  #" in left_bc else left_bc
left_params = st.text_input("Left params", "{}", key="left_params")

right_bc = st.selectbox("Right BC", BC_DISPLAY_NAMES, index=BC_DISPLAY_NAMES.index(get_bc_display_name("sin_k")) if "sin_k" in BC_FUNCTIONS else 0)
right_bc_name = right_bc.split("  #")[0] if "  #" in right_bc else right_bc
right_params = st.text_input("Right params", '{"k": 1}', key="right_params")

bottom_bc = st.selectbox("Bottom BC", BC_DISPLAY_NAMES, index=0)
bottom_bc_name = bottom_bc.split("  #")[0] if "  #" in bottom_bc else bottom_bc
bottom_params = st.text_input("Bottom params", "{}", key="bottom_params")

top_bc = st.selectbox("Top BC", BC_DISPLAY_NAMES, index=0)
top_bc_name = top_bc.split("  #")[0] if "  #" in top_bc else top_bc
top_params = st.text_input("Top params", "{}", key="top_params")

# Run
if st.button("🚀 Visualize", use_container_width=True, type="primary"):
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
        
        left_spec = (left_bc_name, left_params_dict)
        right_spec = (right_bc_name, right_params_dict)
        bottom_spec = (bottom_bc_name, bottom_params_dict)
        top_spec = (top_bc_name, top_params_dict)
        
        # Setup solver
        with st.spinner("⏳ Setting up solver..."):
            domain = RectangleDomain(Lx, Ly)
            domain_grid = DomainGrid.cartesian_2d(Lx, Ly, res_per_unit=res)
            
            # Grid
            Nx = res * int(Lx)
            Ny = res * int(Ly)
            x = np.linspace(0, Lx, Nx)
            y = np.linspace(0, Ly, Ny)
            X, Y = np.meshgrid(x, y, indexing="xy")
            
            # BCs
            f_left = build_bc_from_spec(left_spec, Ly, bc)
            f_right = build_bc_from_spec(right_spec, Ly, bc)
            f_bottom = build_bc_from_spec(bottom_spec, Lx, bc)
            f_top = build_bc_from_spec(top_spec, Lx, bc)
            
            bc_functions = {
                "pos": [f_left, f_right, f_bottom, f_top],
                "neg": [lambda s: -f_left(s), lambda s: -f_right(s), lambda s: -f_bottom(s), lambda s: -f_top(s)],
            }
            
            # Compute dt from stability criterion
            dx = x[1] - x[0] if len(x) > 1 else 1.0
            dy = y[1] - y[0] if len(y) > 1 else 1.0
            dt_max = 1.0 / (2.0 * alpha * (1.0 / dx**2 + 1.0 / dy**2))
            dt = 0.45 * dt_max
            
            config = Heat2DRectConfig(dt=dt, alpha=alpha, cycles_per_frame=1)
            solver = Heat2DRectSolver(domain, domain_grid, config, bc_functions)
            solver.apply_bc()
            
            # Get solutions at t=0 (initial state)
            u_num = solver.u_curr
            if u_num is None:
                u_num = np.zeros_like(X)
            else:
                u_num = u_num.copy()
            u_analytical = solver.get_analytic_solution(0.0)
            
            if enforce_BC:
                u_analytical[:, 0] = f_left(y)
                u_analytical[:, -1] = f_right(y)
                u_analytical[0, :] = f_bottom(x)
                u_analytical[-1, :] = f_top(x)
        
        # Display
        st.success("✓ Solution computed!")
        
        fig = create_plotly_figure(X, Y, u_num, u_analytical, height_scale=height_scale, cmap=cmap.lower())
        st.plotly_chart(fig, use_container_width=True)
        
        # Stats
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Grid Points", f"{Nx} × {Ny}")
        with col_b:
            st.metric("Max Numerical", f"{np.max(u_num):.3f}")
        with col_c:
            st.metric("Max Analytical", f"{np.max(u_analytical):.3f}")
    
    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON: {e}")
    except Exception as e:
        st.error(f"Error: {e}")
        import traceback
        st.code(traceback.format_exc())

st.markdown("---")
st.markdown("""
**About:** Fast Plotly-based visualization of the 2D heat equation.
For animations and high-resolution 3D, use the [CLI](https://github.com/NoahWotschke/Math-Viz).
""")
