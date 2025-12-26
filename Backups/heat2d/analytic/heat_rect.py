"""
Analytic solution for Laplace equation on rectangular domain [0,a] × [0,b].

Solves: Δu = 0 with Dirichlet BCs
  u(x, 0) = f₁(x)     (bottom)
  u(x, b) = f₂(x)     (top)
  u(0, y) = f₃(y)     (left)
  u(a, y) = f₄(y)     (right)

Uses eigenfunction series expansion with bilinear corner correction v(x,y).

Mathematical Decomposition:
  u(x,y) = w(x,y) + v(x,y)
  where w = Σₙ [B_n sinh(nπy/a) + B_n* sinh(nπ(b-y)/a)] sin(nπx/a)
          + Σₘ [B_m sinh(mπx/b) + B_m* sinh(mπ(a-x)/b)] sin(mπy/b)
        v = bilinear interpolation through corner values C₀₀, Cₐ₀, C₀ᵦ, Cₐᵦ

The coefficients B_n, B_n*, B_m, B_m* are computed from corrected boundary
functions that subtract the bilinear edge contributions.
"""

from typing import Callable
import numpy as np


def analytic_dirichlet_rect_series(
    X: np.ndarray,
    Y: np.ndarray,
    a: float,
    b: float,
    f1: Callable,
    f2: Callable,
    f3: Callable,
    f4: Callable,
    n_terms: int = 200,
    m_terms: int = 200,
    quad_pts: int = 256,
) -> np.ndarray:
    """
    Compute analytic solution to Laplace equation on rectangle with corner correction.

    Parameters
    ----------
    X, Y : ndarray
        2D mesh grids from np.meshgrid()
    a, b : float
        Rectangle dimensions: [0, a] × [0, b]
    f1, f2 : callable
        Bottom and top boundary functions f(x), defined on [0, a]
    f3, f4 : callable
        Left and right boundary functions f(y), defined on [0, b]
    n_terms : int
        Number of terms in x-series
    m_terms : int
        Number of terms in y-series
    quad_pts : int
        Number of Legendre-Gauss quadrature points

    Returns
    -------
    ndarray
        Analytic solution u(x,y) on the mesh
    """
    x = X[0, :]
    y = Y[:, 0]

    # Legendre-Gauss quadrature nodes and weights
    xi, wi = np.polynomial.legendre.leggauss(int(quad_pts))

    # ========== Corner values for bilinear correction ==========
    C_00 = f1(0.0)  # u(0, 0)
    C_a0 = f1(a)  # u(a, 0)
    C_0b = f3(b)  # u(0, b)
    C_ab = f4(b)  # u(a, b)

    def integ_sin_on_0L(f: Callable, L: float, k_vec: np.ndarray) -> np.ndarray:
        """
        Compute integral of f * sin(k*pi*s/L) over [0, L] via quadrature.
        """
        s = 0.5 * (xi + 1.0) * L
        w = 0.5 * L * wi
        fs = np.asarray(f(s), dtype=float)
        if fs.ndim == 0:
            fs = np.full_like(s, float(fs))
        k_vec = np.asarray(k_vec, dtype=float)
        S = np.sin((k_vec[:, None] * np.pi * s[None, :]) / L)
        return S @ (fs * w)

    def sinh_stable(x: np.ndarray) -> np.ndarray:
        """
        Compute sinh(x) with numerical stability for large |x|.
        For |x| > 100, returns exp(|x|)/2 capped at exp(100)/2 to prevent overflow.
        """
        x = np.asarray(x, dtype=float)
        out = np.empty_like(x)

        # For small arguments, use exact computation
        small = np.abs(x) < 50.0
        if np.any(small):
            out[small] = np.sinh(x[small])

        # For large arguments, use asymptotic approximation but cap at large value
        if np.any(~small):
            xs = x[~small]
            # sinh(x) ≈ sign(x) * exp(|x|) / 2, but cap at exp(100)/2
            # to avoid overflow while maintaining correct behavior
            abs_xs = np.abs(xs)
            abs_xs = np.minimum(abs_xs, 100.0)  # Cap at 100 to prevent overflow
            out[~small] = np.sign(xs) * np.exp(abs_xs) / 2.0

        return out

    # ========== Bilinear corner correction v(x,y) ==========
    # v(x,y) = C₀₀·(a-x)(b-y)/(ab) + Cₐ₀·x(b-y)/(ab)
    #        + C₀ᵦ·(a-x)y/(ab) + Cₐᵦ·xy/(ab)
    v = (
        C_00 * (a - X) * (b - Y) / (a * b)
        + C_a0 * X * (b - Y) / (a * b)
        + C_0b * (a - X) * Y / (a * b)
        + C_ab * X * Y / (a * b)
    )

    # ========== Corrected boundary functions for coefficients ==========
    # Subtract linear interpolations of corner values along each edge

    def f1_corrected(x_eval):
        """f₁(x) - [C₀₀·(a-x)/a + Cₐ₀·x/a]"""
        return (
            np.asarray(f1(x_eval), dtype=float)
            - (C_00 * (a - x_eval) + C_a0 * x_eval) / a
        )

    def f2_corrected(x_eval):
        """f₂(x) - [C₀ᵦ·(a-x)/a + Cₐᵦ·x/a]"""
        return (
            np.asarray(f2(x_eval), dtype=float)
            - (C_0b * (a - x_eval) + C_ab * x_eval) / a
        )

    def f3_corrected(y_eval):
        """f₃(y) - [C₀₀·(b-y)/b + C₀ᵦ·y/b]"""
        return (
            np.asarray(f3(y_eval), dtype=float)
            - (C_00 * (b - y_eval) + C_0b * y_eval) / b
        )

    def f4_corrected(y_eval):
        """f₄(y) - [Cₐ₀·(b-y)/b + Cₐᵦ·y/b]"""
        return (
            np.asarray(f4(y_eval), dtype=float)
            - (C_a0 * (b - y_eval) + C_ab * y_eval) / b
        )

    # ========== Part 1: X-series with sinh in y-direction ==========
    # u_n(x,y) = Σₙ [B_n sinh(nπy/a) + B_n* sinh(nπ(b-y)/a)] sin(nπx/a)
    #
    # B_n = (2 / [a·sinh(nπb/a)]) ∫₀ᵃ f₂_corrected(x) sin(nπx/a) dx
    # B_n* = (2 / [a·sinh(nπb/a)]) ∫₀ᵃ f₁_corrected(x) sin(nπx/a) dx

    n = np.arange(1, int(n_terms) + 1, dtype=float)
    lam_n = (n * np.pi) / a

    # Skip terms where sinh argument gets too large (they contribute negligibly)
    max_sinh_arg_n = np.max(lam_n * b)
    if max_sinh_arg_n > 100:
        n_max = np.searchsorted(lam_n * b, 100.0)
        n = n[:n_max]
        lam_n = lam_n[:n_max]

    # Compute integrals
    I_f2 = integ_sin_on_0L(f2_corrected, a, n)
    I_f1 = integ_sin_on_0L(f1_corrected, a, n)

    # Compute normalization: 1 / sinh(nπb/a)
    sinh_nb_a = sinh_stable(lam_n * b)
    norm_n = 2.0 / (a * sinh_nb_a)

    B_n = norm_n * I_f2
    B_n_star = norm_n * I_f1

    # Evaluate series
    sin_nx = np.sin(lam_n[:, None] * x[None, :])
    sinh_ny = sinh_stable(lam_n[:, None] * y[None, :])
    sinh_n_b_minus_y = sinh_stable(lam_n[:, None] * (b - y)[None, :])

    u_n = np.einsum("n,ny,nx->yx", B_n, sinh_ny, sin_nx)
    u_n += np.einsum("n,ny,nx->yx", B_n_star, sinh_n_b_minus_y, sin_nx)

    # ========== Part 2: Y-series with sinh in x-direction ==========
    # u_m(x,y) = Σₘ [B_m sinh(mπx/b) + B_m* sinh(mπ(a-x)/b)] sin(mπy/b)
    #
    # B_m = (2 / [b·sinh(mπa/b)]) ∫₀ᵇ f₄_corrected(y) sin(mπy/b) dy
    # B_m* = (2 / [b·sinh(mπa/b)]) ∫₀ᵇ f₃_corrected(y) sin(mπy/b) dy

    m = np.arange(1, int(m_terms) + 1, dtype=float)
    lam_m = (m * np.pi) / b

    # Skip terms where sinh argument gets too large (they contribute negligibly)
    max_sinh_arg_m = np.max(lam_m * a)
    if max_sinh_arg_m > 100:
        m_max = np.searchsorted(lam_m * a, 100.0)
        m = m[:m_max]
        lam_m = lam_m[:m_max]

    # Compute integrals
    I_f4 = integ_sin_on_0L(f4_corrected, b, m)
    I_f3 = integ_sin_on_0L(f3_corrected, b, m)

    # Compute normalization: 1 / sinh(mπa/b)
    sinh_ma_b = sinh_stable(lam_m * a)
    norm_m = 2.0 / (b * sinh_ma_b)

    B_m = norm_m * I_f4
    B_m_star = norm_m * I_f3

    # Evaluate series
    sin_my = np.sin(lam_m[:, None] * y[None, :])
    sinh_mx = sinh_stable(lam_m[:, None] * x[None, :])
    sinh_m_a_minus_x = sinh_stable(lam_m[:, None] * (a - x)[None, :])

    u_m = np.einsum("m,mx,my->yx", B_m, sinh_mx, sin_my)
    u_m += np.einsum("m,mx,my->yx", B_m_star, sinh_m_a_minus_x, sin_my)

    # ========== Total solution ==========
    # u(x,y) = u_n(x,y) + u_m(x,y) + v(x,y)
    return u_n + u_m + v
