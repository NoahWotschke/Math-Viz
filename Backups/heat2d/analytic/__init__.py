"""
Analytic solutions for PDEs on various domains.

Submodules:
    heat_rect: Laplace equation solution on rectangular domain
"""

from .heat_rect import analytic_dirichlet_rect_series

__all__ = ["analytic_dirichlet_rect_series"]
