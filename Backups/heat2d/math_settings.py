"""Mathematical settings for heat equation solver.

Contains domain dimensions, grid resolution, solver parameters,
and boundary condition specifications.
"""

import numpy as np
from typing import Tuple

# ============================================================================
# DOMAIN AND GRID PARAMETERS
# ============================================================================

# Domain dimensions
Lx: float = 2.0  # Length in x-direction
Ly: float = 1.0  # Length in y-direction

# Grid resolution
Nx: int = 102  # Number of grid points in x
Ny: int = 51  # Number of grid points in y

# Derived grid parameters (read-only)
dx: float = Lx / (Nx - 1)
dy: float = Ly / (Ny - 1)
x: np.ndarray = np.linspace(0, Lx, Nx)
y: np.ndarray = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)


# ============================================================================
# PDE SOLVER PARAMETERS
# ============================================================================

# Heat equation diffusion coefficient
alpha: float = 1.0

# Time stepping
dt_safety_factor: float = 0.45  # Safety factor for CFL condition (< 0.5)
# dt is computed dynamically: dt = dt_safety_factor / (2 * alpha * (1/dx² + 1/dy²))

# Number of terms in series expansion for analytic solution
n_terms: int = 200
m_terms: int = 200


# ============================================================================
# SOLVER CONFIGURATION
# ============================================================================

# Boundary condition cycling
alternate_bc: bool = False  # Whether to alternate between positive/negative BCs
steps_per_cycle: int = 1000  # Steps before switching BC phase

# Enforce boundary conditions on analytic solution
enforce_bc_on_analytic: bool = False


# ============================================================================
# BOUNDARY CONDITIONS
# ============================================================================

# Define boundary condition specifications as nested tuples: (function_name, {kwargs})
# Available base functions: const_0, const_1, sin_k, cos_k, sin_2pi_k, cos_2pi_k,
#   linear_up, linear_down, gaussian, exp_growth, exp_decay, triangle, sawtooth, etc.
# Available operations: scale, neg, shift, mirror, normalize, derivative, mul, add
#
# Examples - simple base functions:
#   bc_left_spec = ("const_c", {"c": 1.0})
#   bc_right_spec = ("sin_k", {"k": 1})
#
# Examples - with operations:
#   bc_right_spec = ("scale", {"amp": 2.0, "f": ("sin_k", {"k": 1})})
#   bc_bottom_spec = ("neg", {"f": ("gaussian", {"mu": 0.5, "sigma": 0.1, "amp": 1.0})})
#   bc_top_spec = ("mul", {"f": ("sin_k", {"k": 1}), "g": ("cos_k", {"k": 1})})
#   bc_left_spec = ("add", {"f": ("const_1", {}), "g": ("sin_k", {"k": 2})})
#
# Chaining operations:
#   bc_right_spec = ("scale", {"amp": 0.5, "f": ("neg", {"f": ("sin_k", {"k": 1})})})

bc_left_spec: Tuple[str, dict] = (
    "add",
    {"f": ("const_c", {"c": 1.0}), "g": ("neg", {"f": ("sin_k", {"k": 1})})},
)
bc_right_spec: Tuple[str, dict] = ("neg", {"f": ("sin_k", {"k": 1})})
bc_bottom_spec: Tuple[str, dict] = ("linear_down", {})
bc_top_spec: Tuple[str, dict] = ("linear_down", {})


def get_solver_params() -> dict:
    """Get all solver parameters as a dictionary.

    Returns
    -------
    dict
        Dictionary containing all mathematical parameters
    """
    return {
        "Lx": Lx,
        "Ly": Ly,
        "Nx": Nx,
        "Ny": Ny,
        "dx": dx,
        "dy": dy,
        "alpha": alpha,
        "dt_safety_factor": dt_safety_factor,
        "n_terms": n_terms,
        "m_terms": m_terms,
        "alternate_bc": alternate_bc,
        "steps_per_cycle": steps_per_cycle,
        "enforce_bc_on_analytic": enforce_bc_on_analytic,
    }
