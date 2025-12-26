"""
Abstract base class for PDE solvers.

Defines the interface for solving PDEs on different domains with various
boundary conditions and visualization.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from PDEs.domains.base import Domain, Grid


@dataclass
class SolverConfig:
    """Configuration container for PDE solver parameters.

    Attributes:
        dt: Time step size
        cycles_per_frame: Number of solver cycles per visualization frame
        alternate: Boolean flag for alternating phase visualization
        steps_per_cycles: Number of time steps before alternating phase

    Subclasses may extend with problem-specific parameters (alpha, c, etc.)
    """

    dt: float
    cycles_per_frame: int = 1
    alternate: bool = False
    steps_per_cycles: int = 1000


class BaseSolver(ABC):
    """
    Abstract base class for PDE solvers.

    Subclasses must implement:
    - step_once() — Advance solution by one time step
    - apply_bc() — Apply boundary conditions to solution
    - get_analytic_solution(t) — Compute exact solution for error computation

    Concrete methods provided:
    - compute_error() — L∞ error vs. analytic solution
    - update_phase_and_lines() — Handle phase switching and visualization
    - time_step() — Advance by multiple cycles with error tracking
    """

    def __init__(
        self,
        domain: Domain,
        grid: Grid,
        config: SolverConfig,
        bc_functions: Dict[str, List],
    ):
        """
        Initialize solver.

        Parameters:
            domain: Domain object (Rectangle, Disc, Bar1D, etc.)
            grid: Computational grid with coordinate arrays
            config: Solver configuration (dt, cycles_per_frame, etc.)
            bc_functions: Dict with 'pos' and 'neg' keys, each containing
                         list of boundary condition functions
        """
        self.domain = domain
        self.grid = grid
        self.config = config
        self.bc_functions = bc_functions

        # Solution arrays (allocated by subclass)
        self.u_curr: Optional[np.ndarray] = None
        self.u_next: Optional[np.ndarray] = None

        # Error tracking
        self.error_history: List[float] = []
        self.time_history: List[float] = []
        self.t_current: float = 0.0

        # Phase tracking (for oscillating BCs)
        self.phase: int = 0  # 0 or 1
        self.step_count: int = 0

    @abstractmethod
    def step_once(self) -> None:
        """
        Advance solution by one time step in-place.

        Must update self.u_next based on self.u_curr, then swap:
        self.u_curr, self.u_next = self.u_next, self.u_curr
        """
        pass

    @abstractmethod
    def apply_bc(self) -> None:
        """
        Apply boundary conditions to current solution.

        Should use self.bc_functions[phase] to apply BCs to self.u_curr.
        """
        pass

    @abstractmethod
    def get_analytic_solution(self, t: float) -> np.ndarray:
        """
        Compute exact solution at time t.

        Returns:
            Array same shape as self.u_curr with analytic solution values.
        """
        pass

    def compute_error(
        self,
        u_star_pos: np.ndarray,
        u_star_neg: np.ndarray,
        err_buf: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute L∞ error versus analytic solution.

        Parameters:
            u_star_pos: Analytic solution for positive phase
            u_star_neg: Analytic solution for negative phase
            err_buf: Optional pre-allocated buffer for error array

        Returns:
            L∞ error (max absolute difference)
        """
        u_star = u_star_neg if self.phase == 1 else u_star_pos
        error_array = np.abs(self.u_curr - u_star)

        return float(np.max(error_array))

    def update_phase_and_lines(
        self,
        boundary_lines: Dict[str, List],
        y_samp: np.ndarray,
        x_samp: np.ndarray,
        height_scale: float,
    ) -> None:
        """
        Update visualization boundary lines.

        Toggles phase between 0 and 1 periodically (every steps_per_cycles steps)
        and updates boundary line coordinates for the current phase.

        Parameters:
            boundary_lines: Dict with 'num' and 'ref' keys, each containing
                           list of Line3D objects for boundary visualization
            y_samp: y-coordinates for left/right boundary sampling
            x_samp: x-coordinates for bottom/top boundary sampling
            height_scale: Scaling factor for z-height in visualization
        """
        # Only toggle phase if alternate is enabled and enough steps have elapsed
        if (
            self.config.alternate
            and self.step_count % self.config.steps_per_cycles == 0
        ):
            self.phase = 1 - self.phase

        # Get BC functions for current phase (0='pos', 1='neg')
        phase_key = "neg" if self.phase == 1 else "pos"
        bc_funcs = self.bc_functions[phase_key]

        # Update boundary lines with new phase's BC values
        fL, fR, fB, fT = bc_funcs

        # Left boundary (x = 0, varying y)
        boundary_lines["num"][0].set_data(0 * y_samp, y_samp)
        boundary_lines["num"][0].set_3d_properties(height_scale * fL(y_samp))
        boundary_lines["ref"][0].set_data(0 * y_samp, y_samp)
        boundary_lines["ref"][0].set_3d_properties(height_scale * fL(y_samp))

        # Right boundary (x = Lx, varying y)
        # Access Lx from domain (works for RectangleDomain)
        Lx = getattr(self.domain, "Lx", None)
        if Lx is None:
            raise ValueError("Domain must have Lx attribute for rectangular boundaries")
        boundary_lines["num"][1].set_data(Lx + 0 * y_samp, y_samp)
        boundary_lines["num"][1].set_3d_properties(height_scale * fR(y_samp))
        boundary_lines["ref"][1].set_data(Lx + 0 * y_samp, y_samp)
        boundary_lines["ref"][1].set_3d_properties(height_scale * fR(y_samp))

        # Bottom boundary (y = 0, varying x)
        boundary_lines["num"][2].set_data(x_samp, 0 * x_samp)
        boundary_lines["num"][2].set_3d_properties(height_scale * fB(x_samp))
        boundary_lines["ref"][2].set_data(x_samp, 0 * x_samp)
        boundary_lines["ref"][2].set_3d_properties(height_scale * fB(x_samp))

        # Top boundary (y = Ly, varying x)
        Ly = getattr(self.domain, "Ly", None)
        if Ly is None:
            raise ValueError("Domain must have Ly attribute for rectangular boundaries")
        boundary_lines["num"][3].set_data(x_samp, Ly + 0 * x_samp)
        boundary_lines["num"][3].set_3d_properties(height_scale * fT(x_samp))
        boundary_lines["ref"][3].set_data(x_samp, Ly + 0 * x_samp)
        boundary_lines["ref"][3].set_3d_properties(height_scale * fT(x_samp))

    def time_step(
        self, num_cycles: int = 1, track_error: bool = False
    ) -> Optional[float]:
        """
        Advance solution by multiple time steps.

        Parameters:
            num_cycles: Number of step_once() calls
            track_error: If True, compute and return L∞ error

        Returns:
            L∞ error if track_error=True, else None
        """
        for _ in range(num_cycles):
            self.apply_bc()
            self.step_once()
            self.t_current += self.config.dt
            self.step_count += 1

        if track_error:
            u_star_pos = self.get_analytic_solution(self.t_current)
            u_star_neg = -u_star_pos  # For alternating BCs
            error = self.compute_error(u_star_pos, u_star_neg)
            self.error_history.append(error)
            self.time_history.append(self.t_current)
            return error

        return None
