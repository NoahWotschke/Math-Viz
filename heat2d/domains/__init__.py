"""
Domain classes for different computational geometries.
"""

from .base import Domain, Grid
from .rectangle import RectangleDomain

__all__ = ["Domain", "Grid", "RectangleDomain"]
