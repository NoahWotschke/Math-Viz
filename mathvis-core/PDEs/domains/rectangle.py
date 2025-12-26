"""
Rectangular domain for 2D PDEs.

Supports rectangular domains [0, Lx] × [0, Ly] with 4 Dirichlet boundaries.
"""

from typing import List, Dict
import numpy as np

from PDEs.domains.base import Domain, Grid


class RectangleDomain(Domain):
    """
    Rectangular domain [0, Lx] × [0, Ly].

    Boundaries:
    - Index 0: Left (x=0, varying y)
    - Index 1: Right (x=Lx, varying y)
    - Index 2: Bottom (y=0, varying x)
    - Index 3: Top (y=Ly, varying x)
    """

    def __init__(self, Lx: float, Ly: float):
        """
        Initialize rectangular domain.

        Args:
            Lx: Domain length in x-direction
            Ly: Domain length in y-direction
        """
        self.Lx = Lx
        self.Ly = Ly

    @property
    def name(self) -> str:
        """Domain identifier."""
        return "rect"

    @property
    def num_boundaries(self) -> int:
        """Number of boundaries in rectangle: 4."""
        return 4

    def boundaries(self) -> List[Dict]:
        """
        Return list of boundary specifications.

        Returns:
            List of 4 dicts with keys: name, index, type, length
        """
        return [
            {
                "name": "left",
                "index": 0,
                "type": "line",
                "x": 0.0,
                "y_range": (0.0, self.Ly),
                "length": self.Ly,
            },
            {
                "name": "right",
                "index": 1,
                "type": "line",
                "x": self.Lx,
                "y_range": (0.0, self.Ly),
                "length": self.Ly,
            },
            {
                "name": "bottom",
                "index": 2,
                "type": "line",
                "y": 0.0,
                "x_range": (0.0, self.Lx),
                "length": self.Lx,
            },
            {
                "name": "top",
                "index": 3,
                "type": "line",
                "y": self.Ly,
                "x_range": (0.0, self.Lx),
                "length": self.Lx,
            },
        ]

    def create_grid(self, res_per_unit: float = 21.0) -> Grid:
        """
        Generate computational grid for rectangular domain.

        Args:
            res_per_unit: Points per unit length (default 21)

        Returns:
            Grid object with Cartesian coordinates
        """
        return Grid.cartesian_2d(self.Lx, self.Ly, res_per_unit)

    def apply_bc_point(
        self,
        bc_value: float,
        boundary_index: int,
        point_index: int,
        u: np.ndarray,
        grid: Grid,
    ) -> None:
        """
        Apply boundary condition at a specific point on rectangle boundary.

        Args:
            bc_value: BC function value at this point
            boundary_index: Which boundary (0=left, 1=right, 2=bottom, 3=top)
            point_index: Index along the boundary
            u: Solution array to modify in-place
            grid: Computational grid
        """
        Ny, Nx = u.shape

        if boundary_index == 0:  # Left (x=0)
            u[point_index, 0] = bc_value
        elif boundary_index == 1:  # Right (x=Lx)
            u[point_index, -1] = bc_value
        elif boundary_index == 2:  # Bottom (y=0)
            u[0, point_index] = bc_value
        elif boundary_index == 3:  # Top (y=Ly)
            u[-1, point_index] = bc_value
        else:
            raise ValueError(f"Invalid boundary index {boundary_index} for rectangle")
