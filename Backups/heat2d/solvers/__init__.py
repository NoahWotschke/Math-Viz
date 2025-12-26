"""
Solver classes for different PDEs and domains.
"""

from .base import BaseSolver, SolverConfig
from .heat2d_rect import Heat2DRectSolver, Heat2DRectConfig

__all__ = ["BaseSolver", "SolverConfig", "Heat2DRectSolver", "Heat2DRectConfig"]
