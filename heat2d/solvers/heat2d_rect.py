"""
Heat equation solver on rectangular domain.

Implements 2D explicit finite difference solver for the heat equation:
∂u/∂t = α(∂²u/∂x² + ∂²u/∂y²)

Uses central differences in space, forward Euler in time.
"""

from typing import Dict, List, Optional
import numpy as np

from heat2d.domains.base import Domain, Grid
from heat2d.domains.rectangle import RectangleDomain
from heat2d.solvers.base import BaseSolver, SolverConfig
from heat2d.analytic.heat_rect import analytic_dirichlet_rect_series


class Heat2DRectConfig(SolverConfig):
    """Configuration for 2D heat equation on rectangle."""

    def __init__(
        self,
        dt: float,
        alpha: float,
        cycles_per_frame: int = 1,
        alternate: bool = False,
        steps_per_cycles: int = 1000,
    ):
        """
        Args:
            dt: Time step size
            alpha: Thermal diffusivity coefficient
            cycles_per_frame: Solver cycles per visualization frame
            alternate: Whether to alternate BC phase between positive/negative
            steps_per_cycles: Number of time steps before alternating phase
        """
        super().__init__(
            dt=dt,
            cycles_per_frame=cycles_per_frame,
            alternate=alternate,
            steps_per_cycles=steps_per_cycles,
        )
        self.alpha = alpha


class Heat2DRectSolver(BaseSolver):
    """
    Heat equation solver on rectangular domain [0, Lx] × [0, Ly].

    Uses explicit finite difference scheme:
    - Space: Central differences (2nd order)
    - Time: Forward Euler (1st order)
    - CFL stability: dt < 0.5 / (2 * alpha * (1/dx² + 1/dy²))

    Provides analytic solution via Dirichlet series on rectangle.
    """

    def __init__(
        self,
        domain: RectangleDomain,
        grid: Grid,
        config: Heat2DRectConfig,
        bc_functions: Dict[str, List],
    ):
        """
        Initialize Heat 2D Rectangle solver.

        Args:
            domain: RectangleDomain instance
            grid: Cartesian grid with x, y, dx, dy, X, Y
            config: Heat2DRectConfig instance (contains alpha, dt, etc.)
            bc_functions: Dict with 'pos' and 'neg' keys, each containing
                         list [f_left, f_right, f_bottom, f_top]
        """
        super().__init__(domain, grid, config, bc_functions)

        if not isinstance(domain, RectangleDomain):
            raise TypeError("Heat2DRectSolver requires RectangleDomain")
        if not isinstance(config, Heat2DRectConfig):
            raise TypeError("Heat2DRectSolver requires Heat2DRectConfig")

        # Store typed references
        self.rect_domain = domain
        self.heat_config = config

        # Store grid details with assertions
        self.x = grid.x
        self.y = grid.y
        self.dx = grid.dx
        self.dy = grid.dy
        self.X = grid.X
        self.Y = grid.Y

        if self.x is None or self.y is None:
            raise ValueError("Grid must have Cartesian coordinates (x, y)")
        if self.dx is None or self.dy is None:
            raise ValueError("Grid must have spacing (dx, dy)")
        if self.X is None or self.Y is None:
            raise ValueError("Grid must have meshgrid arrays (X, Y)")

        # Allocate solution arrays
        self.u_curr = np.zeros(grid.shape, dtype=np.float64)
        self.u_next = np.zeros(grid.shape, dtype=np.float64)

    def step_once(self) -> None:
        """
        Advance solution by one time step using explicit FD scheme.

        Implements: u_next = u_curr + alpha*dt*(∂²u/∂x² + ∂²u/∂y²)
        """
        # Type assertions for type checkers
        assert self.heat_config.alpha is not None
        assert self.config.dt is not None
        assert self.dx is not None and self.dy is not None
        assert self.u_curr is not None and self.u_next is not None

        alpha = self.heat_config.alpha
        dt = self.config.dt
        dx2_inv = 1.0 / (self.dx**2)
        dy2_inv = 1.0 / (self.dy**2)

        # Interior points: central difference stencil
        self.u_next[1:-1, 1:-1] = self.u_curr[1:-1, 1:-1] + alpha * dt * (
            # ∂²u/∂x²
            (
                self.u_curr[1:-1, 2:]
                - 2 * self.u_curr[1:-1, 1:-1]
                + self.u_curr[1:-1, :-2]
            )
            * dx2_inv
            # ∂²u/∂y²
            + (
                self.u_curr[2:, 1:-1]
                - 2 * self.u_curr[1:-1, 1:-1]
                + self.u_curr[:-2, 1:-1]
            )
            * dy2_inv
        )

        # Swap arrays for next iteration
        self.u_curr, self.u_next = self.u_next, self.u_curr

        # Increment step counter
        self.step_count += 1

    def apply_bc(self) -> None:
        """Apply boundary conditions from current phase."""
        phase_key = "neg" if self.phase == 1 else "pos"
        f_left, f_right, f_bottom, f_top = self.bc_functions[phase_key]

        assert self.x is not None and self.y is not None
        assert self.u_curr is not None

        # Apply Dirichlet BCs on all four boundaries
        self.u_curr[:, 0] = f_left(self.y)  # Left boundary
        self.u_curr[:, -1] = f_right(self.y)  # Right boundary
        self.u_curr[0, :] = f_bottom(self.x)  # Bottom boundary
        self.u_curr[-1, :] = f_top(self.x)  # Top boundary

    def get_analytic_solution(self, t: float) -> np.ndarray:
        """
        Compute analytic solution at time t.

        Uses Laplace equation solution (equilibrium of heat equation)
        via eigenfunction series on rectangle.

        Args:
            t: Time (unused; returns steady-state Laplace solution)

        Returns:
            Array of analytic solution values on grid
        """
        phase_key = "neg" if self.phase == 1 else "pos"
        f_left, f_right, f_bottom, f_top = self.bc_functions[phase_key]

        # Type assertions
        assert self.X is not None and self.Y is not None

        # Compute Laplace solution on rectangle
        # Note: This is the steady-state solution; time-dependent heat diffusion
        # would require heat kernel integration, but for equilibrium reference:
        u_star = analytic_dirichlet_rect_series(
            self.X,
            self.Y,
            self.rect_domain.Lx,
            self.rect_domain.Ly,
            f_bottom,
            f_top,
            f_left,
            f_right,
            n_terms=200,
            m_terms=200,
            quad_pts=256,
        )

        return u_star


# Note: _analytic_dirichlet_rect_series() moved to heat2d/analytic/heat_rect.py
