"""
Abstract base class for computational domains.

Defines the interface for different domain geometries (rectangle, disc, 1D bar)
and coordinate systems (Cartesian, polar, 1D).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, List, Optional
import numpy as np


@dataclass
class Grid:
    """
    Represents a computational grid for a domain.

    Attributes:
        x, y: 2D coordinate arrays (Cartesian) or None for 1D
        r, theta: Polar coordinate arrays or None
        dx, dy: Spatial step sizes in each direction
        dr, dtheta: Polar coordinate step sizes or None
        shape: Grid shape (ny, nx) for 2D or (nx,) for 1D
        coordinate_system: 'cartesian', 'polar', or '1d'
    """

    coordinate_system: str  # 'cartesian', 'polar', '1d'
    shape: Tuple[int, ...]

    # Cartesian coordinates
    x: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None
    dx: Optional[float] = None
    dy: Optional[float] = None
    X: Optional[np.ndarray] = None
    Y: Optional[np.ndarray] = None

    # Polar coordinates
    r: Optional[np.ndarray] = None
    theta: Optional[np.ndarray] = None
    dr: Optional[float] = None
    dtheta: Optional[float] = None
    R: Optional[np.ndarray] = None
    Theta: Optional[np.ndarray] = None

    @staticmethod
    def cartesian_2d(Lx: float, Ly: float, res_per_unit: float = 21.0) -> "Grid":
        """Create a 2D Cartesian grid on [0, Lx] × [0, Ly].

        Args:
            Lx, Ly: Domain dimensions
            res_per_unit: Points per unit length (default 21)

        Returns:
            Grid object with Cartesian coordinates
        """
        Nx = max(2, int(res_per_unit * Lx))
        Ny = max(2, int(res_per_unit * Ly))

        x = np.linspace(0, Lx, Nx)
        y = np.linspace(0, Ly, Ny)
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        X, Y = np.meshgrid(x, y, indexing="xy")

        return Grid(
            coordinate_system="cartesian",
            shape=(Ny, Nx),
            x=x,
            y=y,
            dx=dx,
            dy=dy,
            X=X,
            Y=Y,
        )

    @staticmethod
    def cartesian_1d(L: float, res_per_unit: float = 21.0) -> "Grid":
        """Create a 1D Cartesian grid on [0, L].

        Args:
            L: Domain length
            res_per_unit: Points per unit length (default 21)

        Returns:
            Grid object with 1D Cartesian coordinates
        """
        Nx = max(2, int(res_per_unit * L))
        x = np.linspace(0, L, Nx)
        dx = x[1] - x[0]

        return Grid(coordinate_system="1d", shape=(Nx,), x=x, dx=dx)

    @staticmethod
    def polar(R: float, res_r: int = 21, res_theta: int = 64) -> "Grid":
        """Create a 2D polar grid on [0, R] × [0, 2π].

        Args:
            R: Domain radius
            res_r: Number of radial points
            res_theta: Number of angular points

        Returns:
            Grid object with polar coordinates
        """
        r = np.linspace(0, R, res_r)
        theta = np.linspace(0, 2 * np.pi, res_theta, endpoint=False)
        dr = r[1] - r[0] if len(r) > 1 else 1.0
        dtheta = theta[1] - theta[0] if len(theta) > 1 else 1.0

        Rad, Theta_grid = np.meshgrid(r, theta, indexing="ij")

        return Grid(
            coordinate_system="polar",
            shape=(res_r, res_theta),
            r=r,
            theta=theta,
            dr=dr,
            dtheta=dtheta,
            R=Rad,
            Theta=Theta_grid,
        )


class Domain(ABC):
    """
    Abstract base class for computational domains.

    Subclasses must implement:
    - boundaries() — Return list of boundary specifications
    - create_grid(res_per_unit) — Generate Grid with coordinate arrays
    - apply_bc_point(bc_value, boundary_index, point_index) — Apply BC at specific point

    Properties:
    - name — Domain type identifier (e.g., 'rect', 'disc', 'bar1d')
    - num_boundaries — Number of distinct boundaries
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Domain type identifier (e.g., 'rect', 'disc', 'bar1d')."""
        pass

    @property
    @abstractmethod
    def num_boundaries(self) -> int:
        """Number of distinct boundaries in this domain."""
        pass

    @abstractmethod
    def boundaries(self) -> List[dict]:
        """
        Return list of boundary specifications.

        Returns:
            List of dicts with keys:
            - 'name': boundary identifier (e.g., 'left', 'right', 'bottom', 'top')
            - 'index': integer index for reference in BC lists
            - 'type': boundary type (e.g., 'line', 'curve', 'arc')
            - 'points': coordinate arrays defining the boundary
        """
        pass

    @abstractmethod
    def create_grid(self, res_per_unit: float = 21.0) -> Grid:
        """
        Generate computational grid for this domain.

        Args:
            res_per_unit: Points per unit length (determines dx, dy, dr, etc.)

        Returns:
            Grid object with coordinate arrays and spacing info.
        """
        pass

    @abstractmethod
    def apply_bc_point(
        self,
        bc_value: float,
        boundary_index: int,
        point_index: int,
        u: np.ndarray,
        grid: Grid,
    ) -> None:
        """
        Apply boundary condition at a specific point.

        Args:
            bc_value: BC function value at this point
            boundary_index: Which boundary (0, 1, 2, 3 for rect)
            point_index: Index along the boundary
            u: Solution array to modify in-place
            grid: Computational grid
        """
        pass
