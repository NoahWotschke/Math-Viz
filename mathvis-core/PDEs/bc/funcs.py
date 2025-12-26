"""Boundary condition functions for heat equation.

Provides a comprehensive library of parameterizable boundary condition functions
for use with rectangular and disk geometries. Each function accepts:
  - s: scalar or array of positions along the boundary [0, L]
  - L: domain length (optional for some functions)
  - Additional parameters: k (frequency), beta (growth rate), etc.

Functions are organized by category: polynomial, trigonometric, exponential,
logarithmic, and composite shapes. Use make_bc() to bind parameters and
combinator functions to create custom boundary conditions.
"""

import numpy as np


# ---------- CONSTANT / LINEAR / POLYNOMIAL ----------
def const_0(s, L=None):
    """Zero boundary condition: u = 0 everywhere.

    Args:
        s: Position along boundary (scalar or array).
        L: Domain length (unused).

    Returns:
        Array of zeros with same shape as input.
    """
    return np.zeros_like(s, dtype=float)


def const_1(s, L=None):
    """Constant boundary condition: u = 1 everywhere.

    Args:
        s: Position along boundary (scalar or array).
        L: Domain length (unused).

    Returns:
        Array of ones with same shape as input.
    """
    return np.ones_like(s, dtype=float)


def const_c(s, L, c=0.5):
    """Constant boundary condition with arbitrary value.

    Args:
        s: Position along boundary [0, L].
        L: Domain extent (ignored).
        c: Constant value.

    Returns:
        Array of constant values matching shape of s.
    """
    return c * np.ones_like(s, dtype=float)


def linear_up(s, L):
    """Linear increasing boundary condition: u = s/L (0 → 1).

    Args:
        s: Position along boundary [0, L].
        L: Domain length.

    Returns:
        Linearly increasing values from 0 to 1.
    """
    return s / L


def linear_down(s, L):
    """Linear decreasing boundary condition: u = 1 - s/L (1 → 0).

    Args:
        s: Position along boundary [0, L].
        L: Domain length.

    Returns:
        Linearly decreasing values from 1 to 0.
    """
    return 1.0 - s / L


def quadratic_bowl(s, L):
    """Convex quadratic (bowl shape): u = (s/L)² (0 → 1).

    Args:
        s: Position along boundary [0, L].
        L: Domain length.

    Returns:
        Quadratic values, increasing from 0 to 1.
    """
    return (s / L) ** 2


def quadratic_cap(s, L):
    """Concave quadratic (cap shape): u = 1 - (s/L)² (1 → 0).

    Args:
        s: Position along boundary [0, L].
        L: Domain length.

    Returns:
        Quadratic values, decreasing from 1 to 0.
    """
    return 1.0 - (s / L) ** 2


def parabola_peak_mid(s, L):
    """Parabolic peak at midpoint: u = 4z(1-z) where z = s/L (0 at ends, 1 at mid).

    Args:
        s: Position along boundary [0, L].
        L: Domain length.

    Returns:
        Parabolic bell curve peaking at s = L/2.
    """
    z = s / L
    return 4.0 * z * (1.0 - z)


# ---------- TRIGONOMETRIC ----------
def sin_k(s, L, k=1):
    """Sine wave boundary condition: u = sin(kπs/L).

    Args:
        s: Position along boundary [0, L].
        L: Domain length.
        k: Number of half-wavelengths (default 1 = half-wave from 0 to 0).

    Returns:
        Sinusoidal values. For k=1, half-wave returning to 0 at boundaries.
    """
    return np.sin(k * np.pi * s / L)


def cos_k(s, L, k=1):
    """Cosine wave boundary condition: u = cos(kπs/L).

    Args:
        s: Position along boundary [0, L].
        L: Domain length.
        k: Number of half-wavelengths (default 1).

    Returns:
        Cosine values. For k=1, peaks at both ends.
    """
    return np.cos(k * np.pi * s / L)


def sin_2pi_k(s, L, k=1):
    """Full-wavelength sine: u = sin(2πks/L).

    Args:
        s: Position along boundary [0, L].
        L: Domain length.
        k: Number of full wavelengths (default 1).

    Returns:
        Sine values with k complete oscillations over [0, L].
    """
    return np.sin(2.0 * np.pi * k * s / L)


def cos_2pi_k(s, L, k=1):
    """Full-wavelength cosine: u = cos(2πks/L).

    Args:
        s: Position along boundary [0, L].
        L: Domain length.
        k: Number of full wavelengths (default 1).

    Returns:
        Cosine values with k complete oscillations over [0, L].
    """
    return np.cos(2.0 * np.pi * k * s / L)


# ---------- EXPONENTIAL / HYPERBOLIC ----------
def exp_growth(s, L, beta=3.0):
    """Exponential growth: u = exp(βs/L) (1 → e^β).

    Args:
        s: Position along boundary [0, L].
        L: Domain length.
        beta: Growth rate parameter (default 3.0).

    Returns:
        Exponentially increasing values.
    """
    return np.exp(beta * (s / L))


def exp_decay(s, L, beta=3.0):
    """Exponential decay: u = exp(-βs/L) (1 → e^-β).

    Args:
        s: Position along boundary [0, L].
        L: Domain length.
        beta: Decay rate parameter (default 3.0).

    Returns:
        Exponentially decreasing values.
    """
    return np.exp(-beta * (s / L))


def sinh_shape(s, L, beta=3.0):
    """Hyperbolic sine: u = sinh(βs/L).

    Args:
        s: Position along boundary [0, L].
        L: Domain length.
        beta: Growth rate parameter (default 3.0).

    Returns:
        Hyperbolic sine values (odd-symmetric growth).
    """
    return np.sinh(beta * (s / L))


def cosh_shape(s, L, beta=3.0):
    """Hyperbolic cosine: u = cosh(βs/L) (always ≥ 1).

    Args:
        s: Position along boundary [0, L].
        L: Domain length.
        beta: Growth rate parameter (default 3.0).

    Returns:
        Hyperbolic cosine values (even-symmetric, minimum 1).
    """
    return np.cosh(beta * (s / L))


# ---------- LOG / ROOT / RATIONAL ----------
def sqrt_shape(s, L):
    """Square root boundary condition: u = √(s/L) (0 → 1).

    Args:
        s: Position along boundary [0, L].
        L: Domain length.

    Returns:
        Square root values, concave increasing from 0 to 1.
    """
    return np.sqrt(np.maximum(s, 0.0) / L)


def log_shape(s, L, eps=1e-9):
    """Logarithmic shape: u = log(1 + s/L + ε).

    Args:
        s: Position along boundary [0, L].
        L: Domain length.
        eps: Small regularization constant to avoid log(0).

    Returns:
        Logarithmically increasing values.
    """
    return np.log1p((s / L) + eps)


def rational(s, L):
    """Rational function: u = z/(1+z) where z = s/L (0 → 1, sub-exponential).

    Args:
        s: Position along boundary [0, L].
        L: Domain length.

    Returns:
        Rational saturation curve from 0 to 1.
    """
    z = s / L
    return z / (1.0 + z)


def x_pow_x(s, L, amp=1.0, shift=0.0, eps=1e-12):
    """
    Returns amp * (z+shift)^(z+shift) with z = s/L in [0,1].
    Safe at 0 via eps; supports arrays.
    """
    z = np.asarray(s, dtype=float) / L
    z = np.clip(z + shift, 0.0, 1.0)  # keep it in [0,1] unless you want to allow >1
    # compute z**z safely (0**0 -> 1 by convention here)
    zz = np.where(z > eps, np.exp(z * np.log(z)), 1.0)
    return amp * zz


# ---------- ABS / TRIANGLE / STEP ----------
def abs_centered(s, L):
    """V-shaped absolute value: u = |z - 0.5| where z = s/L (minimum at center).

    Args:
        s: Position along boundary [0, L].
        L: Domain length.

    Returns:
        V-shaped values peaking at both ends, minimum 0 at s = L/2.
    """
    z = s / L
    return np.abs(z - 0.5)


def triangle(s, L, k=1.0):
    """Triangular peak: u = k(1 - 2|z - 0.5|) where z = s/L (0 at ends, k at mid).

    Args:
        s: Position along boundary [0, L].
        L: Domain length.
        k: Peak height (default 1.0).

    Returns:
        Triangular wave peaking at s = L/2.
    """
    z = s / L
    return k - 2.0 * k * np.abs(z - 0.5)


def heaviside_step(s, L):
    """Heaviside step function: u = 0 for s < L/2, u = 1 for s ≥ L/2.

    Args:
        s: Position along boundary [0, L].
        L: Domain length.

    Returns:
        Binary step function (discontinuous at midpoint).
    """
    z = s / L
    return (z >= 0.5).astype(float)


def piecewise_step(s, L, c=0.3):
    """Step function at arbitrary position: u = 0 for s < cL, u = 1 for s ≥ cL.

    Args:
        s: Position along boundary [0, L].
        L: Domain length.
        c: Relative step position in [0, 1] (default 0.3).

    Returns:
        Binary step function at position cL.
    """
    z = s / L
    return (z >= c).astype(float)


# ---------- GAUSSIAN / BUMP ----------
def gaussian(s, L, mu=0.5, sigma=0.1, amp=1.0):
    """Gaussian bump: u = amp * exp(-0.5((z-μ)/σ)²) where z = s/L.

    Args:
        s: Position along boundary [0, L].
        L: Domain length.
        mu: Center position in [0, 1] (default 0.5).
        sigma: Standard deviation in [0, 1] (default 0.1).
        amp: Peak amplitude (default 1.0).

    Returns:
        Gaussian bell curve centered at μL.
    """
    z = s / L
    return amp * np.exp(-0.5 * ((z - mu) / sigma) ** 2)


def bump_cos(s, L):
    """Cosine bump: u = 0.5(1 - cos(2πz)) where z = s/L (smooth, 0 at ends).

    Args:
        s: Position along boundary [0, L].
        L: Domain length.

    Returns:
        Smooth bump from 0 at ends to 1 at center, using cosine.
    """
    z = s / L
    return 0.5 * (1.0 - np.cos(2.0 * np.pi * z))


# ---------- NEW WAVEFORMS ----------
def sawtooth(s, L, k=1):
    """Sawtooth wave: repeated linear ramps (-1 → 1) with k periods.

    Args:
        s: Position along boundary [0, L].
        L: Domain length.
        k: Number of periods (default 1).

    Returns:
        Sawtooth wave with k complete ramps.
    """
    z = (k * s / L) % 1.0
    return 2.0 * (z - 0.5)


def square_wave(s, L, k=1):
    """Square wave: ±1 alternating with k periods.

    Args:
        s: Position along boundary [0, L].
        L: Domain length.
        k: Number of periods (default 1).

    Returns:
        Square wave oscillating between -1 and 1.
    """
    return np.sign(np.sin(2 * np.pi * k * s / L))


def pulse(s, L, start=0.2, end=0.8, amp=1.0):
    """Rectangular pulse: u = amp for start·L ≤ s ≤ end·L, else 0.

    Args:
        s: Position along boundary [0, L].
        L: Domain length.
        start: Relative pulse start in [0, 1] (default 0.2).
        end: Relative pulse end in [0, 1] (default 0.8).
        amp: Pulse amplitude (default 1.0).

    Returns:
        Rectangular pulse between start·L and end·L.
    """
    z = s / L
    return amp * ((z >= start) & (z <= end)).astype(float)


def smooth_bump(s, L, center=0.5, width=0.2):
    """Smooth exponential bump using analytic function: exp(-1/(1-z²)) inside, 0 outside.

    Args:
        s: Position along boundary [0, L].
        L: Domain length.
        center: Center position in [0, 1] (default 0.5).
        width: Relative width in [0, 1] (default 0.2).

    Returns:
        Smooth C^∞ bump centered at center·L with given width.
    """
    z = (s / L - center) / width
    mask = np.abs(z) < 1
    out = np.zeros_like(z)
    out[mask] = np.exp(-1 / (1 - z[mask] ** 2))
    return out


def hermite_smoothstep(s, L):
    """Hermite smoothstep: u = z²(3 - 2z) where z = s/L (smooth 0 → 1).

    Args:
        s: Position along boundary [0, L].
        L: Domain length.

    Returns:
        Smooth step with zero derivatives at endpoints (C¹).
    """
    z = s / L
    return z * z * (3 - 2 * z)


def noise_bc(s, L, seed=0):
    """Random Gaussian noise boundary condition.

    Args:
        s: Position along boundary (scalar or array).
        L: Domain length (unused).
        seed: Random seed for reproducibility (default 0).

    Returns:
        Array of normally distributed random values N(0, 1).
    """
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, size=len(np.atleast_1d(s)))


# ---------- COMBINATORS ----------
def bc_add(f, g):
    """Compose two boundary conditions by addition: (f + g)(s) = f(s) + g(s).

    Args:
        f, g: Callable boundary condition functions.

    Returns:
        New boundary condition computing f + g.
    """
    return lambda s: f(s) + g(s)


def bc_mul(f, g):
    """Compose two boundary conditions by multiplication: (f·g)(s) = f(s)·g(s).

    Args:
        f, g: Callable boundary condition functions.

    Returns:
        New boundary condition computing f × g.
    """
    return lambda s: f(s) * g(s)


def bc_scale(f, amp):
    """Scale boundary condition by constant: (c·f)(s) = c·f(s).

    Args:
        f: Callable boundary condition function.
        amp: Scaling amplitude.

    Returns:
        Scaled boundary condition.
    """
    return lambda s: amp * f(s)


def bc_shift(f, offset, L):
    """Shift boundary condition: f_shifted(s) = f((s - offset) mod L).

    Args:
        f: Callable boundary condition function.
        offset: Shift distance.
        L: Domain length (for periodic boundary).

    Returns:
        Shifted boundary condition.
    """
    return lambda s: f((s - offset) % L)


def bc_mirror(f, L):
    """Mirror boundary condition: f_mirror(s) = f(L - s).

    Args:
        f: Callable boundary condition function.
        L: Domain length.

    Returns:
        Reflected boundary condition.
    """
    return lambda s: f(L - s)


def bc_normalize(f):
    """Normalize boundary condition to [0, 1]: u_norm = (u - min) / (max - min).

    Args:
        f: Callable boundary condition function.

    Returns:
        Normalized boundary condition mapped to [0, 1].
    """
    return lambda s: (f(s) - np.min(f(s))) / (np.max(f(s)) - np.min(f(s)) + 1e-12)


# ---------- ANALYSIS UTILITIES ----------
def bc_derivative(f, h=1e-6):
    """Compute numerical derivative of boundary condition using finite differences.

    Args:
        f: Callable boundary condition function.
        h: Finite difference step size (default 1e-6).

    Returns:
        Boundary condition representing df/ds via central difference.
    """
    return lambda s: (f(s + h) - f(s - h)) / (2 * h)


def neg_bc(f):
    """Negate boundary condition: (-f)(s) = -f(s).

    Args:
        f: Callable boundary condition function.

    Returns:
        Negated boundary condition.
    """
    return lambda s, f=f: -np.asarray(f(s))


# --------- Bind L/k/etc into a callable boundary function ---------
def make_bc(func, *, L, **params):
    """Factory function: bind domain parameters into a shape function.

    Creates a callable boundary condition by fixing domain length L and any
    additional parameters (k, beta, etc.) for a base function. The returned
    function can be called with only position argument(s).

    Args:
        func: Base boundary shape function (e.g., sin_k, gaussian).
        L: Domain length (keyword-only argument).
        **params: Additional parameters (k, beta, mu, sigma, etc.).

    Returns:
        Callable g(s) that applies func(s, L, **params) with type safety.

    Example:
        >>> f = make_bc(sin_k, L=1.0, k=2)
        >>> f(0.5)  # calls sin_k(0.5, L=1.0, k=2)
    """

    def g(s):
        s = np.asarray(s, dtype=float)
        return func(s, L, **params)

    g.__doc__ = func.__doc__
    return g


# --- Coordinate transform: rectangle BC -> disk BC ---
def theta_to_s(theta, L):
    """Map angular coordinate θ ∈ [0, 2π] to arc length s ∈ [0, L].

    Converts polar angle to arc length on unit disk boundary, allowing
    reuse of rectangular boundary condition functions for circular domains.

    Args:
        theta: Angle in radians [0, 2π].
        L: Corresponding arc length at radius r=1 (typically L = 2π).

    Returns:
        Arc length s = (L/2π)·θ in [0, L].
    """
    return (L / (2 * np.pi)) * theta


def make_bc_for_disk(rect_bc_func, L):
    """Adapt rectangular boundary condition to disk geometry.

    Wraps a rectangular boundary condition function to accept angular
    coordinates on a disk boundary, internally converting θ → s arc length.

    Args:
        rect_bc_func: Rectangular boundary condition expecting s ∈ [0, L].
        L: Arc length parameter (default 2π for unit disk).

    Returns:
        Callable f(θ) for disk boundary, θ ∈ [0, 2π].

    Example:
        >>> rect_bc = make_bc(sin_k, L=2*π, k=3)
        >>> disk_bc = make_bc_for_disk(rect_bc, L=2*π)
        >>> disk_bc(π/2)  # evaluate at θ = π/2
    """
    return lambda theta: rect_bc_func(theta_to_s(theta, L))
