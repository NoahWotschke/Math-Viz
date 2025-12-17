# How to Implement a New PDE in the Framework

A step-by-step guide to extend the modular solver framework with a new partial differential equation.

## Overview

The framework uses an **extensible template pattern** where:
- **Domains** handle geometry (grid, boundaries)
- **Solvers** implement the PDE (time-stepping, boundary application)
- **Analytic solutions** provide reference values
- **BC library** is independent of both

To add a new PDE, you'll create **one new solver class** that plugs into the existing domain infrastructure.

---

## Step 1: Understand the Template

### BaseSolver ABC (the blueprint)
```python
# heat2d/solvers/base.py

class BaseSolver(ABC):
    def __init__(self, domain: Domain, grid: Grid, config: SolverConfig, 
                 bc_functions: Dict[str, List]):
        """All solvers take these 4 arguments"""
        self.domain = domain
        self.grid = grid
        self.config = config
        self.bc_functions = bc_functions
    
    # YOUR IMPLEMENTATION (3 required methods)
    @abstractmethod
    def step_once(self) -> None:
        """Advance solution by one time step"""
        
    @abstractmethod
    def apply_bc(self) -> None:
        """Apply boundary conditions from current phase"""
        
    @abstractmethod
    def get_analytic_solution(self, t: float) -> np.ndarray:
        """Compute reference solution at time t"""
    
    # SHARED BY ALL SOLVERS (don't override these)
    def compute_error(self, u_ref_pos: np.ndarray, u_ref_neg: np.ndarray) -> float:
        """Returns L∞ norm ||u_numerical - u_analytic||"""
    
    def update_phase_and_lines(self, boundary_lines: Dict, y: np.ndarray, 
                               x: np.ndarray, scale: float) -> None:
        """Toggles visualization between positive/negative phases"""
```

### Key Insight
You only need to implement **3 required methods**. The rest is handled by the base class.

### The 3 Required Methods

1. **`step_once() -> None`**
   - Advance the solution by one time step
   - Implement your PDE's numerical scheme here (finite differences, RK methods, etc.)
   - Update `self.u_curr` with new values from `self.u_next`
   - Increment `self.step_count += 1`
   - Called repeatedly by the framework

2. **`apply_bc() -> None`**
   - Apply boundary conditions to `self.u_curr`
   - Called after each `step_once()` to enforce constraints
   - Choose type: Dirichlet (values), Neumann (gradients), Periodic (wraparound), or Robin (mixed)
   - Read from `self.bc_functions[phase_key]` where phase_key is "pos" or "neg"

3. **`get_analytic_solution(t: float) -> np.ndarray`**
   - Compute the reference solution at time `t`
   - Used for error computation: L∞ norm ||u_numerical - u_analytic||
   - Return shape must match `self.X.shape` (the grid)
   - Can return zeros if no analytic solution is available

### What BaseSolver Provides (You Don't Implement)
- **Phase tracking** (`self.phase = 0` for 'pos', `1` for 'neg')
- **Step counting** (`self.step_count` incremented automatically)
- **Error computation** (L∞ norm between numerical and analytical)
- **Visualization updates** (boundary line toggling, phase switching)
- **Configuration access** (all parameters via `self.config`)

### What You Must Provide
- **Domain knowledge** (know your geometry constraints)
- **Numerical scheme** (FD stencil, stability conditions)
- **Boundary application** (how to enforce constraints)
- **Reference solution** (for validation)

---

## Step 2: Choose a Domain

You can use any existing domain:
- **RectangleDomain** - `[0, Lx] × [0, Ly]` with 4 boundaries
- **Bar1DDomain** - `[0, L]` with 2 boundaries (when created)
- **DiscDomain** - Polar `[0, R] × [0, 2π]` (when created)

For this guide, we'll use **RectangleDomain** (most common).

The domain handles grid generation, so you get:
```python
self.grid.x        # 1D array of x coordinates: shape (Nx,)
self.grid.y        # 1D array of y coordinates: shape (Ny,)
self.grid.X, Y     # 2D meshgrid arrays: shape (Ny, Nx)
self.grid.dx, dy   # Grid spacing (float, float)
self.grid.shape    # (Ny, Nx) - indexing is [y_index, x_index]
```

**Critical:** Array indexing is `array[row_index, col_index]` = `array[y_index, x_index]`
- Interior points: `u[1:-1, 1:-1]` (all except boundaries)
- Left edge: `u[:, 0]` (all y, x=0)
- Right edge: `u[:, -1]` (all y, x=Lx)
- Bottom edge: `u[0, :]` (y=0, all x)
- Top edge: `u[-1, :]` (y=Ly, all x)

---

## Step 3: Create Your Solver File

### File: `heat2d/solvers/your_pde_domain.py`

Start with this template:

```python
"""
Your PDE description.

Mathematical formulation:
∂u/∂t = f(u, ∇u, ∇²u, ...)  on domain with boundary conditions

Features:
  - Time-stepping scheme (e.g., explicit, implicit, semi-implicit)
  - Stability constraints (CFL condition)
  - Boundary condition type (Dirichlet, Neumann, periodic)
"""

from typing import Dict, List, Optional
import numpy as np

from heat2d.domains.base import Domain, Grid
from heat2d.domains.rectangle import RectangleDomain  # or your domain
from heat2d.solvers.base import BaseSolver, SolverConfig
# from heat2d.analytic.your_pde_domain import analytic_function  # when ready


class YourPDEConfig(SolverConfig):
    """Configuration specific to your PDE."""
    
    def __init__(
        self,
        dt: float,
        param1: float,           # e.g., diffusivity, wave speed
        param2: float = 1.0,     # optional parameter
        cycles_per_frame: int = 1,
        alternate: bool = False,
        steps_per_cycles: int = 1000,
    ):
        """Initialize configuration."""
        super().__init__(
            dt=dt,
            cycles_per_frame=cycles_per_frame,
            alternate=alternate,
            steps_per_cycles=steps_per_cycles,
        )
        self.param1 = param1
        self.param2 = param2


class YourPDESolver(BaseSolver):
    """Solver for: ∂u/∂t = ...
    
    Uses: [describe numerical scheme]
    Stability: [describe CFL or other constraints]
    """
    
    def __init__(
        self,
        domain: RectangleDomain,
        grid: Grid,
        config: YourPDEConfig,
        bc_functions: Dict[str, List],
    ):
        """Initialize solver with domain, grid, config, and BC functions.
        
        Args:
            domain: RectangleDomain (or your domain type)
            grid: Cartesian grid with x, y, dx, dy, X, Y
            config: YourPDEConfig instance
            bc_functions: Dict with 'pos' and 'neg' keys, each containing
                         list [f_left, f_right, f_bottom, f_top]
        """
        super().__init__(domain, grid, config, bc_functions)
        
        # Type checking
        if not isinstance(domain, RectangleDomain):
            raise TypeError("YourPDESolver requires RectangleDomain")
        if not isinstance(config, YourPDEConfig):
            raise TypeError("YourPDESolver requires YourPDEConfig")
        
        # Store typed references
        self.rect_domain = domain
        self.pde_config = config
        
        # Extract grid details with assertions
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
        """Advance solution by one time step using [scheme name].
        
        Implements: [write out the numerical update formula]
        
        Example for heat equation:
        u_next = u_curr + α*dt*(∂²u/∂x² + ∂²u/∂y²)
        """
        # Type assertions for type checkers
        assert self.pde_config.param1 is not None
        assert self.config.dt is not None
        assert self.dx is not None and self.dy is not None
        assert self.u_curr is not None and self.u_next is not None
        
        # Extract parameters
        param1 = self.pde_config.param1
        dt = self.config.dt
        
        # YOUR NUMERICAL SCHEME HERE
        # This is where the actual PDE solving happens
        
        # Example (heat equation):
        # self.u_next[1:-1, 1:-1] = self.u_curr[1:-1, 1:-1] + param1 * dt * (
        #     (self.u_curr[1:-1, 2:] - 2*self.u_curr[1:-1, 1:-1] + self.u_curr[1:-1, :-2]) / self.dx**2 +
        #     (self.u_curr[2:, 1:-1] - 2*self.u_curr[1:-1, 1:-1] + self.u_curr[:-2, 1:-1]) / self.dy**2
        # )
        
        # Swap arrays for next iteration
        self.u_curr, self.u_next = self.u_next, self.u_curr
        
        # Increment step counter (required for phase toggling)
        self.step_count += 1
    
    
    def apply_bc(self) -> None:
        """Apply boundary conditions from current phase.
        
        Reads from self.phase (0 = 'pos', 1 = 'neg') and applies
        Dirichlet conditions on all four edges.
        """
        phase_key = "neg" if self.phase == 1 else "pos"
        f_left, f_right, f_bottom, f_top = self.bc_functions[phase_key]
        
        assert self.x is not None and self.y is not None
        assert self.u_curr is not None
        
        # Apply Dirichlet BCs on all four boundaries
        self.u_curr[:, 0] = f_left(self.y)      # Left boundary (x=0)
        self.u_curr[:, -1] = f_right(self.y)    # Right boundary (x=Lx)
        self.u_curr[0, :] = f_bottom(self.x)    # Bottom boundary (y=0)
        self.u_curr[-1, :] = f_top(self.x)      # Top boundary (y=Ly)
    
    
    def get_analytic_solution(self, t: float) -> np.ndarray:
        """Compute analytic reference solution at time t.
        
        Args:
            t: Time value
        
        Returns:
            2D array of solution values on grid
        """
        phase_key = "neg" if self.phase == 1 else "pos"
        f_left, f_right, f_bottom, f_top = self.bc_functions[phase_key]
        
        assert self.X is not None and self.Y is not None
        
        # COMPUTE YOUR ANALYTIC SOLUTION HERE
        # For heat equation: Laplace series
        # For wave equation: d'Alembert or product series
        # For Burgers: Cole-Hopf transformation
        
        # Example (return zeros for now):
        u_star = np.zeros_like(self.X)
        
        return u_star
```

---

## Step 4: Implement `step_once()`

This is the core of your PDE solver. Here's how to approach it:

### 4a. Understand Your PDE

Write out the mathematical formulation:
```
∂u/∂t = f(u, ∇u, ∇²u, ...)
```

**Classification by type:**
1. **Parabolic (Diffusive)** - Smoothing, equilibrium-seeking:
   - **Heat:** ∂u/∂t = α(∂²u/∂x² + ∂²u/∂y²)
   - **Reaction-Diffusion:** ∂u/∂t = D∇²u + f(u)
   - CFL: dt < dx²/(4αβ) where β depends on dimension
   - 1-layer time stepping (u^n → u^{n+1})
   
2. **Hyperbolic (Propagating)** - Wave-like transport:
   - **Wave:** ∂²u/∂t² = c²(∂²u/∂x² + ∂²u/∂y²)
   - **Advection:** ∂u/∂t + c·∂u/∂x = 0
   - CFL: dt < dx/c (or dx/(2c) for 2D)
   - 2-layer time stepping (u^{n-1}, u^n, u^{n+1})
   
3. **Mixed (Advection-Diffusion)** - Transport with dissipation:
   - **Burgers:** ∂u/∂t + u·∂u/∂x = ν·∂²u/∂x²
   - **Navier-Stokes:** momentum + mass conservation
   - CFL: dt < min(dx²/(4ν), dx/(2|u|_max))
   - Often semi-implicit (explicit advection, implicit diffusion)

**Determine:** Is your PDE parabolic, hyperbolic, or mixed? This dictates time-stepping.

### 4b. Choose a Discretization Scheme

**Five main approaches:**

1. **Explicit Forward Euler (simplest, restrictive CFL):**
   ```python
   u_next = u_curr + dt * f(u_curr)  # Single derivative evaluation
   ```
   - **Accuracy:** O(dt) time, O(dx²) space (2nd order FD)
   - **Stability:** Conditional (CFL constraint applies)
   - **Pros:** Simple, easy to debug, fast per step
   - **Cons:** Tiny dt required for stability
   - **Use for:** Learning, parabolic PDEs on coarse grids

2. **Explicit Runge-Kutta (RK2, RK3, RK4 - better accuracy):**
   ```python
   # RK2: two derivative evaluations
   k1 = f(u_curr)
   k2 = f(u_curr + 0.5*dt*k1)
   u_next = u_curr + dt * k2
   ```
   - **Accuracy:** O(dt²) time for RK2, O(dt³) for RK3, O(dt⁴) for RK4
   - **Stability:** Still conditional on CFL
   - **Pros:** Better accuracy, not much more complex
   - **Cons:** Multiple function evaluations per step
   - **Use for:** Parabolic PDEs where accuracy matters

3. **Implicit Backward Euler (unconditionally stable):**
   ```python
   # (I - dt*L)u_next = u_curr  where L is differential operator
   # Requires solving linear system Ax = b
   ```
   - **Accuracy:** O(dt) time, O(dx²) space
   - **Stability:** Unconditional (all dt allowed)
   - **Pros:** Large dt possible, always stable
   - **Cons:** Requires linear solver (sparse matrix), ~100x slower
   - **Use for:** Stiff problems, need large dt, parabolic only

4. **Semi-Implicit (Additive Splitting):**
   ```python
   # Split PDE: explicit_part + implicit_part
   # e.g., Burgers: advection (explicit) + diffusion (implicit)
   ```
   - **Accuracy:** O(dt) or O(dt²) depending on approach
   - **Stability:** Better than explicit, not as good as implicit
   - **Pros:** Balanced stability/complexity
   - **Cons:** More code, requires tuning
   - **Use for:** Advection-diffusion, reaction-diffusion

5. **Predictor-Corrector (multiple passes, better accuracy):**
   ```python
   # Estimate u_next, then refine using corrector step
   u_pred = u_curr + dt * f(u_curr)
   u_next = u_curr + 0.5*dt*(f(u_curr) + f(u_pred))
   ```
   - **Accuracy:** O(dt³) with right scheme
   - **Stability:** Can improve stability slightly
   - **Pros:** Higher accuracy than forward Euler
   - **Cons:** Two evaluations per step
   - **Use for:** When accuracy critical, want explicit simplicity

**Recommendation for this framework:**
Use **explicit RK2 or RK4** for best balance of simplicity and accuracy. Start with explicit; only go implicit if explicit is too slow.

### 4c. Compute Derivatives Using Finite Differences

**First derivatives - Three main options:**

1. **Central difference (2nd order, symmetric, most accurate):**
   ```python
   du_dx = (u[i, j+1] - u[i, j-1]) / (2*dx)  # Interior points
   du_dy = (u[i+1, j] - u[i-1, j]) / (2*dy)
   ```
   - **Truncation error:** O(dx²) - good accuracy
   - **Use for:** Most smooth problems
   - **Issue:** Can oscillate for steep gradients
   - **Boundary points:** Cannot use (needs u[j±1]). Use forward/backward instead.

2. **Forward difference (1st order, biased right):**
   ```python
   du_dx = (u[i, j+1] - u[i, j]) / dx
   ```
   - **Truncation error:** O(dx) - lower accuracy
   - **Use for:** Upwind schemes (advection with flow direction right)
   - **Stability:** Helps hyperbolic equations

3. **Backward difference (1st order, biased left):**
   ```python
   du_dx = (u[i, j] - u[i, j-1]) / dx
   ```
   - **Truncation error:** O(dx) - lower accuracy
   - **Use for:** Upwind schemes (advection with flow direction left)
   - **Stability:** Opposite of forward difference

**Second derivatives - Central difference always:**

```python
# 5-point stencil (O(dx²) accuracy)
d2u_dx2 = (u[i, j+1] - 2*u[i, j] + u[i, j-1]) / dx**2
d2u_dy2 = (u[i+1, j] - 2*u[i, j] + u[i-1, j]) / dy**2

# Laplacian: ∇²u = ∂²u/∂x² + ∂²u/∂y²
laplacian = d2u_dx2 + d2u_dy2

# 9-point stencil (O(dx⁴) accuracy, rarely needed)
d2u_dx2 = (-u[i, j+2] + 16*u[i, j+1] - 30*u[i, j] + 16*u[i, j-1] - u[i, j-2]) / (12*dx**2)
```

**Mixed derivatives (less common):**
```python
# ∂²u/∂x∂y using central difference in both directions
d2u_dxdy = (u[i+1, j+1] - u[i+1, j-1] - u[i-1, j+1] + u[i-1, j-1]) / (4*dx*dy)
```

**Accuracy comparison:**
| Scheme               | Truncation Error | When to Use                  |
| -------------------- | ---------------- | ---------------------------- |
| Forward/Backward 1st | O(dx)            | Upwind advection             |
| Central 2nd          | O(dx²)           | Most PDEs (heat, wave)       |
| Compact 4th          | O(dx⁴)           | High accuracy needed         |
| 9-point 4th          | O(dx⁴)           | 2D problems, smooth solution |

### 4d. Implement Vectorized (NumPy - ~100x faster than loops)

**CRITICAL:** Always use NumPy vectorization instead of Python loops.

```python
# NEVER DO THIS (Python loops are 100x slower):
for i in range(1, Ny-1):
    for j in range(1, Nx-1):
        self.u_next[i, j] = self.u_curr[i, j] + dt * (...)

# ALWAYS DO THIS (NumPy vectorized):
self.u_next[1:-1, 1:-1] = (
    self.u_curr[1:-1, 1:-1] + dt * (...)
)
```

**Why vectorization matters:**
- For 100×100 grid (10,000 interior points):
  - Python loops: ~1 second/step → 1000 seconds for 1000 steps
  - NumPy: ~10 ms/step → 10 seconds for 1000 steps
- NumPy uses compiled C/Fortran → 100x speed improvement

**Array slicing reference:**
```python
# Interior only (what you compute)
u[1:-1, 1:-1]    # All except boundaries

# Shifted versions (for stencils)
u[1:-1, 2:]      # Interior, shifted right in x (for u[i, j+1])
u[1:-1, :-2]     # Interior, shifted left in x (for u[i, j-1])
u[2:, 1:-1]      # Interior, shifted down in y (for u[i+1, j])
u[:-2, 1:-1]     # Interior, shifted up in y (for u[i-1, j])

# Diagonal neighbors (for mixed derivatives)
u[2:, 2:]        # Interior, shifted down-right
u[2:, :-2]       # Interior, shifted down-left
u[:-2, 2:]       # Interior, shifted up-right
u[:-2, :-2]      # Interior, shifted up-left
```

**Shape tracking (critical for debugging):**
```python
# If u.shape = (Ny, Nx) = (101, 101), then interior is (99, 99)
assert u[1:-1, 1:-1].shape == (Ny-2, Nx-2)

# All stencil slices must produce same interior shape!
d2u_dx2 = (u[1:-1, 2:] - 2*u[1:-1, 1:-1] + u[1:-1, :-2]) / dx**2  # (99, 99)
d2u_dy2 = (u[2:, 1:-1] - 2*u[1:-1, 1:-1] + u[:-2, 1:-1]) / dy**2  # (99, 99)
assert d2u_dx2.shape == d2u_dy2.shape  # Must match!
```

**Vectorization composition (real example):**
```python
# Heat equation: u^{n+1} = u^n + α·dt·∇²u
laplacian = (
    (self.u_curr[1:-1, 2:] - 2*self.u_curr[1:-1, 1:-1] + self.u_curr[1:-1, :-2]) / (self.dx**2) +
    (self.u_curr[2:, 1:-1] - 2*self.u_curr[1:-1, 1:-1] + self.u_curr[:-2, 1:-1]) / (self.dy**2)
)  # Shape: (Ny-2, Nx-2)

self.u_next[1:-1, 1:-1] = self.u_curr[1:-1, 1:-1] + self.alpha * self.dt * laplacian
```

### Example Implementation (Heat Equation)

**Option 1: Explicit Forward Euler (simplest)**
```python
def step_once(self) -> None:
    """Explicit forward Euler for heat equation: ∂u/∂t = α∇²u
    
    Accuracy: O(dt) + O(dx²)
    Stability: dt < dx_min²/(4α) where dx_min = min(dx, dy)
    """
    assert self.pde_config.param1 is not None, "param1 (diffusivity) not set"
    assert self.config.dt is not None, "dt not set"
    
    alpha = self.pde_config.param1  # Diffusivity coefficient
    dt = self.config.dt              # Time step
    
    # Compute Laplacian at interior points ONLY
    # Note: Boundaries handled by apply_bc()
    d2u_dx2 = (
        (self.u_curr[1:-1, 2:] - 2*self.u_curr[1:-1, 1:-1] + self.u_curr[1:-1, :-2]) 
        / (self.dx**2)
    )
    d2u_dy2 = (
        (self.u_curr[2:, 1:-1] - 2*self.u_curr[1:-1, 1:-1] + self.u_curr[:-2, 1:-1]) 
        / (self.dy**2)
    )
    laplacian = d2u_dx2 + d2u_dy2  # Shape: (Ny-2, Nx-2)
    
    # Forward Euler: u^{n+1} = u^n + dt·α·∇²u^n
    self.u_next[1:-1, 1:-1] = (
        self.u_curr[1:-1, 1:-1] + alpha * dt * laplacian
    )
    
    # Boundaries will be overwritten by apply_bc() before next step
    # No need to update them here
    
    # Array swap (efficient: avoids copying)
    self.u_curr, self.u_next = self.u_next, self.u_curr
    
    # CRITICAL: Increment step counter for phase tracking
    self.step_count += 1
```

**Stability check (add to `__init__`):**
```python
def __init__(self, ...):
    # ... existing init code ...
    
    # CFL stability warning
    dx_min = min(self.dx, self.dy)
    dt_max = dx_min**2 / (4 * alpha)
    
    if self.config.dt >= dt_max:
        print(f"WARNING: dt={self.config.dt} >= dt_max={dt_max}")
        print(f"Solution will likely be unstable!")
        print(f"Reduce dt to < {dt_max}")
```

**Option 2: Explicit RK2 (better accuracy, O(dt²) + O(dx²))**
```python
def step_once(self) -> None:
    """Runge-Kutta 2nd order for heat equation.
    
    Accuracy: O(dt²) + O(dx²) (twice as accurate as forward Euler)
    Stability: Same CFL constraint as forward Euler
    Cost: ~2x more function evaluations
    """
    alpha = self.pde_config.param1
    dt = self.config.dt
    
    # Stage 1: Compute Laplacian at current state
    d2u_dx2_curr = (
        (self.u_curr[1:-1, 2:] - 2*self.u_curr[1:-1, 1:-1] + self.u_curr[1:-1, :-2]) 
        / (self.dx**2)
    )
    d2u_dy2_curr = (
        (self.u_curr[2:, 1:-1] - 2*self.u_curr[1:-1, 1:-1] + self.u_curr[:-2, 1:-1]) 
        / (self.dy**2)
    )
    laplacian_curr = d2u_dx2_curr + d2u_dy2_curr
    
    # Intermediate solution (half step)
    self.u_next[1:-1, 1:-1] = (
        self.u_curr[1:-1, 1:-1] + 0.5*alpha*dt*laplacian_curr
    )
    
    # Copy boundaries temporarily for next evaluation
    self.u_next[1:-1, 0] = self.u_curr[1:-1, 0]
    self.u_next[1:-1, -1] = self.u_curr[1:-1, -1]
    self.u_next[0, 1:-1] = self.u_curr[0, 1:-1]
    self.u_next[-1, 1:-1] = self.u_curr[-1, 1:-1]
    
    # Stage 2: Compute Laplacian at intermediate state
    d2u_dx2_next = (
        (self.u_next[1:-1, 2:] - 2*self.u_next[1:-1, 1:-1] + self.u_next[1:-1, :-2]) 
        / (self.dx**2)
    )
    d2u_dy2_next = (
        (self.u_next[2:, 1:-1] - 2*self.u_next[1:-1, 1:-1] + self.u_next[:-2, 1:-1]) 
        / (self.dy**2)
    )
    laplacian_next = d2u_dx2_next + d2u_dy2_next
    
    # Full step using intermediate Laplacian
    self.u_next[1:-1, 1:-1] = (
        self.u_curr[1:-1, 1:-1] + alpha*dt*laplacian_next
    )
    
    self.u_curr, self.u_next = self.u_next, self.u_curr
    self.step_count += 1
```

---

## Step 5: Implement `apply_bc()`

This enforces boundary conditions. Four main types:

### Type 1: Dirichlet (most common - specify value on boundary)

```python
def apply_bc(self) -> None:
    """Apply Dirichlet boundary conditions: u = f(position)
    
    Use when: You know the solution values on boundaries
    Example: Heat equation with fixed temperature on edges
    """
    phase_key = "neg" if self.phase == 1 else "pos"
    f_left, f_right, f_bottom, f_top = self.bc_functions[phase_key]
    
    # Set values at boundaries
    self.u_curr[:, 0] = f_left(self.y)      # Left edge (x=0)
    self.u_curr[:, -1] = f_right(self.y)    # Right edge (x=Lx)
    self.u_curr[0, :] = f_bottom(self.x)    # Bottom edge (y=0)
    self.u_curr[-1, :] = f_top(self.x)      # Top edge (y=Ly)
```

**Example boundary functions:**
```python
# Constant
f_left = lambda y: np.full_like(y, 1.0)

# Linear
f_right = lambda y: 0.5 + 0.5*y  # Varies from 0.5 to 1.0

# Trigonometric
f_top = lambda x: np.sin(np.pi * x)

# From heat2d.bc library
from heat2d.bc import const_bc, linear_bc, sin_bc

f_left = const_bc(1.0)  # Temperature = 1.0
f_bottom = sin_bc(1.0, 1)  # sin(πx) amplitude 1.0
```

### Type 2: Neumann (specify derivative on boundary)

Used when you know the gradient/flux at boundaries.

```python
def apply_bc(self) -> None:
    """Apply Neumann boundary conditions: ∂u/∂n = g(position)
    
    Use when: You know the gradient on boundaries
    Example: Insulated boundary (∂u/∂n = 0) in heat equation
    """
    # LEFT BOUNDARY: ∂u/∂x = g_left
    # Backward difference: (u[i,1] - u[i,0])/dx = g_left
    # Solve: u[i,0] = u[i,1] - dx*g_left(y[i])
    g_left_vals = g_left(self.y)
    self.u_curr[:, 0] = self.u_curr[:, 1] - self.dx * g_left_vals
    
    # RIGHT BOUNDARY: ∂u/∂x = g_right
    # Forward difference: (u[i,-1] - u[i,-2])/dx = g_right
    # Solve: u[i,-1] = u[i,-2] + dx*g_right(y[i])
    g_right_vals = g_right(self.y)
    self.u_curr[:, -1] = self.u_curr[:, -2] + self.dx * g_right_vals
    
    # BOTTOM BOUNDARY: ∂u/∂y = g_bottom
    g_bottom_vals = g_bottom(self.x)
    self.u_curr[0, :] = self.u_curr[1, :] - self.dy * g_bottom_vals
    
    # TOP BOUNDARY: ∂u/∂y = g_top
    g_top_vals = g_top(self.x)
    self.u_curr[-1, :] = self.u_curr[-2, :] + self.dy * g_top_vals
```

**Common Neumann conditions:**
- Insulated boundary (heat): ∂u/∂n = 0 → `self.u_curr[:, 0] = self.u_curr[:, 1]`
- Zero flux everywhere: Just use 0 for all g_* functions

### Type 3: Periodic (wraparound at boundaries)

Used for circular/periodic domains or when solution wraps around.

```python
def apply_bc(self) -> None:
    """Apply periodic boundary conditions: u[edge] = u[opposite_edge]
    
    Use when: Solution is periodic in a direction
    Example: 2π periodic in angle for polar coordinates
    """
    # X-periodic: left = right
    self.u_curr[:, 0] = self.u_curr[:, -2]   # Copy from second-to-right
    self.u_curr[:, -1] = self.u_curr[:, 1]   # Copy from second-to-left
    
    # Y-periodic: bottom = top
    self.u_curr[0, :] = self.u_curr[-2, :]   # Copy from second-to-top
    self.u_curr[-1, :] = self.u_curr[1, :]   # Copy from second-to-bottom
```

**Note:** Use `-2` and `1` (not `-1` and `0`) to avoid self-reference in periodic wraparound.

### Type 4: Robin (mixed Dirichlet-Neumann)

Combines Dirichlet and Neumann: α·u + β·∂u/∂n = γ

```python
def apply_bc(self) -> None:
    """Apply Robin boundary conditions: α·u + β·∂u/∂n = γ
    
    Use when: Boundary involves both value and flux
    Example: Heat loss proportional to temperature: -k·∂u/∂n = h·(u - u_ambient)
    """
    # LEFT BOUNDARY: u + h*∂u/∂x = f_left
    # Rearrange: u[i,0] = (f_left - h*(u[i,1] - u[i,0])/dx) / (1 + h/dx)
    h = 0.5  # Robin coefficient
    f_left_vals = f_left(self.y)
    
    self.u_curr[:, 0] = (f_left_vals - h*(self.u_curr[:, 1] - self.u_curr[:, 0])/self.dx) / (1 + h/self.dx)
```

**Choosing BC type:**
| Type      | Know                  | Use              | Example             |
| --------- | --------------------- | ---------------- | ------------------- |
| Dirichlet | Values at boundary    | Most common      | Fixed temperature   |
| Neumann   | Gradients at boundary | Insulated/flux   | ∂u/∂n=0 (insulated) |
| Periodic  | Solution wraps        | Circular domains | Polar coordinates   |
| Robin     | Mixed condition       | Heat exchange    | Newton's cooling    |

---

## Step 6: Implement `get_analytic_solution()`

This provides a reference solution for error computation:

```python
def get_analytic_solution(self, t: float) -> np.ndarray:
    """Return analytical solution at time t."""
    phase_key = "neg" if self.phase == 1 else "pos"
    f_left, f_right, f_bottom, f_top = self.bc_functions[phase_key]
    
    # For Laplace equation (steady state):
    from heat2d.analytic.your_pde import analytic_solution
    return analytic_solution(self.X, self.Y, self.rect_domain.Lx, 
                             self.rect_domain.Ly, f_bottom, f_top, 
                             f_left, f_right)
    
    # OR if no analytic solution available:
    return np.zeros_like(self.X)
```

For simple cases, you might have a closed-form solution:
```python
def get_analytic_solution(self, t: float) -> np.ndarray:
    """Gaussian diffusion: u(x,y,t) = exp(-(x²+y²)/(4αt))"""
    alpha = self.pde_config.param1
    u = np.exp(-(self.X**2 + self.Y**2) / (4*alpha*t + 1e-8))
    return u
```

---

## Step 7: Understanding Stability (Critical!)

**Most numerical schemes fail due to instability, not inaccuracy.** This section explains why.

### Von Neumann Stability Analysis

For linear PDEs, we analyze how small perturbations grow/decay using Fourier modes:

```
u(x,t) = e^{ikx} * e^{λ(k)t}
```

where:
- k = wavenumber (0 to π/dx)
- λ(k) = growth factor (real) or amplification factor (complex)

**Stability condition:** |λ(k)| ≤ 1 for all k

If |λ(k)| > 1 for any k, those modes grow exponentially → **instability**.

### CFL (Courant-Friedrichs-Lewy) Constraints

**For explicit schemes, stability requires:**

**Heat equation** (Parabolic):
```
α·dt/dx² ≤ 1/4  (2D)
α·dt/dx² ≤ 1/6  (3D)

Example: α=0.1, dx=0.01
dt_max = 0.01² / (4*0.1) = 0.0025

Typical choice: dt = 0.8 * dt_max = 0.002
```

**Wave equation** (Hyperbolic):
```
c·dt/dx ≤ 1/√2  (2D)
c·dt/dx ≤ 1/2   (1D)

Example: c=1.0, dx=0.01
dt_max = 0.01 / 1.0 = 0.01
dt_max_2d = 0.01 / sqrt(2) ≈ 0.007

Typical choice: dt = 0.8 * dt_max
```

**Advection equation**:
```
|u|·dt/dx ≤ 1  (upwind scheme)

Example: |u|=1.0, dx=0.01
dt_max = 0.01

Typical choice: dt = 0.8 * dt_max = 0.008
```

**Burgers' equation** (mixed):
```
dt < min(α·dx²/(4), dx/(2|u|_max))

Take the smaller of the two constraints!
```

### Practical Stability Checks

```python
def check_stability(self) -> None:
    """Verify CFL stability before running simulation."""
    alpha = self.pde_config.param1  # Diffusivity
    c = self.pde_config.param2      # Wave speed
    
    dx_min = min(self.dx, self.dy)
    dt = self.config.dt
    
    # Heat equation: α·dt/dx² ≤ 1/4
    if c is None:  # Pure diffusion
        dt_max = dx_min**2 / (4 * alpha)
        fourier_num = alpha * dt / dx_min**2
        print(f"Heat equation: Fourier number = {fourier_num:.4f} (max 0.25)")
        if fourier_num > 0.25:
            print("WARNING: Unstable! Reduce dt or increase resolution.")
    
    # Wave equation: c·dt/dx ≤ 1/√2
    else:  # Hyperbolic
        dt_max = dx_min / (np.sqrt(2) * c)
        courant_num = c * dt / dx_min
        print(f"Wave equation: Courant number = {courant_num:.4f} (max 0.707)")
        if courant_num > 1/np.sqrt(2):
            print("WARNING: Unstable! Reduce dt or increase resolution.")
```

### Debugging Instability

If your solution "blows up" or oscillates wildly:

1. **Check CFL constraint:** Print Courant/Fourier number
2. **Reduce dt:** Use dt = 0.5 * dt_theoretical (very safe)
3. **Increase resolution:** More grid points can help stability
4. **Switch scheme:** RK2/RK4 can be more stable than forward Euler
5. **Check boundary conditions:** Wrong BC application causes instability
6. **Verify derivatives:** Arithmetic errors in FD stencils cause instability

---

## Step 8: (Optional) Create Analytic Solution Module

If you have a complex analytic solution, create a separate module:

### File: `heat2d/analytic/your_pde_domain.py`

```python
"""Analytic solution for your PDE on rectangle."""

from typing import Callable
import numpy as np


def analytic_your_pde(X: np.ndarray, Y: np.ndarray, 
                     Lx: float, Ly: float,
                     f_bottom: Callable, f_top: Callable,
                     f_left: Callable, f_right: Callable,
                     **kwargs) -> np.ndarray:
    """
    Compute analytical solution.
    
    For Laplace equation: eigenfunction series expansion
    For heat equation: heat kernel convolution
    For wave equation: d'Alembert formula
    """
    # YOUR IMPLEMENTATION HERE
    u = np.zeros_like(X)
    return u
```

Then import it in your solver:
```python
from heat2d.analytic.your_pde_domain import analytic_your_pde
```

---

## Step 9: Register Your Solver

Update `heat2d/solvers/__init__.py`:

```python
from .base import BaseSolver, SolverConfig
from .heat2d_rect import Heat2DRectSolver, Heat2DRectConfig
from .your_pde_domain import YourPDESolver, YourPDEConfig  # ADD THIS

__all__ = [
    "BaseSolver", "SolverConfig",
    "Heat2DRectSolver", "Heat2DRectConfig",
    "YourPDESolver", "YourPDEConfig",  # ADD THIS
]
```

---

## Step 10: Test Your Solver

Create a simple test script:

```python
# test_your_pde.py

import numpy as np
from heat2d.domains.rectangle import RectangleDomain
from heat2d.domains.base import Grid
from heat2d.solvers.your_pde_domain import YourPDESolver, YourPDEConfig
from heat2d.bc.builder import build_bc_from_spec
import heat2d.bc.funcs as bc

# Setup domain and grid
Lx, Ly = 1.0, 1.0
res_per_unit = 21
domain = RectangleDomain(Lx, Ly)
grid = Grid.cartesian_2d(Lx, Ly, res_per_unit)

# Create BC functions (use any BC from heat2d.bc.funcs)
f_left = bc.const_0
f_right = bc.const_0
f_bottom = bc.const_1
f_top = bc.const_0

bc_functions = {
    "pos": [f_left, f_right, f_bottom, f_top],
    "neg": [bc.neg_bc(f) for f in [f_left, f_right, f_bottom, f_top]],
}

# Create solver
config = YourPDEConfig(dt=0.001, param1=1.0)
solver = YourPDESolver(domain, grid, config, bc_functions)

# Initial conditions
solver.u_curr[:, :] = 0.0
solver.apply_bc()

# Run a few steps
for step in range(10):
    solver.apply_bc()
    solver.step_once()
    
    # Check solution is finite
    assert np.all(np.isfinite(solver.u_curr)), f"NaN at step {step}"
    print(f"Step {step}: min={solver.u_curr.min():.4f}, max={solver.u_curr.max():.4f}")

print("✓ Solver runs without crashing!")
```

Run it:
```bash
cd heat2d_rect_fd  # or wherever your project is
python test_your_pde.py
```

---

## Step 10: Integrate with CLI

Update `heat2d_rect_fd.py` (or create a new CLI) to use your solver:

```python
from heat2d.solvers.your_pde_domain import YourPDESolver, YourPDEConfig

# In main():
config = YourPDEConfig(
    dt=dt,
    param1=args.param1,  # add to argparse
    cycles_per_frame=args.steps_per_frame,
)
solver = YourPDESolver(domain, rect_grid, config, bc_functions)

# Rest of animation loop stays the same!
```

---

## Checklist

- [ ] Created `heat2d/solvers/your_pde_domain.py` with `YourPDESolver` class
- [ ] Implemented `step_once()` with correct numerical scheme
- [ ] Implemented `apply_bc()` with boundary conditions
- [ ] Implemented `get_analytic_solution()` (can return zeros for now)
- [ ] Updated `heat2d/solvers/__init__.py` to export new solver
- [ ] (Optional) Created `heat2d/analytic/your_pde_domain.py` if analytic solution is complex
- [ ] (Optional) Updated `heat2d/analytic/__init__.py` to export analytic function
- [ ] Tested solver doesn't crash with small test script
- [ ] Integrated with CLI tool

---

## Common Pitfalls

### 1. Forgetting `self.step_count += 1`
The base class tracks this for phase toggling. Don't forget it!

### 2. Array swap issues
Always swap after computation:
```python
self.u_curr, self.u_next = self.u_next, self.u_curr
```

### 3. CFL violation (instability)
Too large time step causes oscillations. Check:
- Heat: `dt < dx² / (4*α)`
- Wave: `dt < dx / (2*c)`

### 4. Incorrect indexing
- `u[i, j]` where i = y-index, j = x-index
- Interior points: `u[1:-1, 1:-1]` (skip boundaries)
- Boundaries: `u[:, 0]`, `u[:, -1]`, `u[0, :]`, `u[-1, :]`

### 5. Type errors
Use assertions:
```python
assert self.pde_config.param1 is not None
assert self.X is not None and self.Y is not None
```

---

## Troubleshooting Guide: Common Failure Modes

### Problem 1: Solution "Blows Up" (Values → ∞)

**Symptoms:** After a few steps, u values become NaN or ±∞

**Causes & Solutions:**

1. **Instability (most common):**
   - Check CFL constraint: print `dt / (dx**2)` (should be << 1 for heat)
   - Solution: Reduce dt by factor of 10
   - ```python
     dx_min = min(self.dx, self.dy)
     dt_safe = 0.1 * dx_min**2 / (4 * alpha)  # Very conservative
     if self.config.dt > dt_safe:
         print(f"ERROR: dt too large. Use dt <= {dt_safe}")
     ```

2. **Derivative computation error:**
   - Check stencil indexing: u[1:-1, 2:] should be one cell right
   - Check spacing: Using dx=1 instead of actual grid spacing
   - Verify interior vs boundary: step_once() modifies only [1:-1, 1:-1]

3. **Boundary condition error:**
   - apply_bc() not being called before step_once()
   - BC function returning huge values
   - Wrong BC type for the problem

**Debug steps:**
```python
# In step_once(), add checks:
laplacian = ...
assert not np.any(np.isnan(laplacian)), "NaN in Laplacian!"
assert np.max(np.abs(laplacian)) < 1e6, f"Huge Laplacian: {np.max(np.abs(laplacian))}"

# After update:
assert not np.any(np.isnan(self.u_next)), "NaN in u_next!"
assert np.max(np.abs(self.u_next)) < 1e10, "Solution too large!"
```

### Problem 2: Solution Doesn't Change (Freezes)

**Symptoms:** u remains constant after each step, error → 0 but solution wrong

**Causes & Solutions:**

1. **dt = 0 or empty update:**
   - Check: `if dt == 0: print("ERROR: dt is zero!")`
   - Verify config passed correctly to solver

2. **step_once() doesn't update u_curr:**
   - Forgetting array swap: `self.u_curr, self.u_next = self.u_next, self.u_curr`
   - Or not incrementing: `self.step_count += 1`
   - All interior values getting zeroed

3. **Initial condition issue:**
   - Is u_curr initialized with correct initial condition?
   - Is IC satisfying boundary conditions?

**Debug:**
```python
def step_once(self) -> None:
    u_before = self.u_curr[1:-1, 1:-1].copy()
    # ... your code ...
    u_after = self.u_curr[1:-1, 1:-1].copy()
    
    diff = np.max(np.abs(u_after - u_before))
    if diff < 1e-15:
        print(f"WARNING: Interior not changing! Max change: {diff}")
```

### Problem 3: Solution Oscillates Wildly (Not Smooth)

**Symptoms:** Checkerboard or high-frequency oscillations

**Causes & Solutions:**

1. **Central difference for advection:**
   - Central differences can oscillate on advection-dominated problems
   - Solution: Use upwind scheme instead
   ```python
   # DON'T: du_dx = (u[1:-1, 2:] - u[1:-1, :-2]) / (2*dx)  # Central
   
   # DO:
   u_vel = ...  # Advection velocity
   du_dx = np.where(
       u_vel >= 0,
       (u[1:-1, 1:-1] - u[1:-1, :-2]) / dx,     # Backward (velocity >= 0)
       (u[1:-1, 2:] - u[1:-1, 1:-1]) / dx       # Forward (velocity < 0)
   )
   ```

2. **Overly large dt:**
   - Even if CFL satisfied, dt near limit causes oscillations
   - Solution: Use dt = 0.5 * dt_max instead of 0.99 * dt_max

3. **High-frequency boundary noise:**
   - BC functions introducing sharp features
   - Solution: Smooth BC functions with low-pass filter
   ```python
   from scipy.ndimage import gaussian_filter
   bc_vals = gaussian_filter(bc_vals, sigma=1)
   ```

### Problem 4: Boundary Artifacts (Values Wrong at Edges)

**Symptoms:** Good solution in interior, but wrong at/near boundaries

**Causes & Solutions:**

1. **Boundary points not being updated:**
   - Your step_once() only updates interior [1:-1, 1:-1]
   - Verify apply_bc() is called every step before visualization

2. **Wrong boundary function:**
   - BC function doesn't match problem specification
   - Check: `print(self.bc_functions[phase_key])` to verify it's correct

3. **Neumann BC implementation wrong:**
   - Using forward instead of backward difference at left boundary
   - Remember: left boundary needs backward difference (uses interior)
   - Right boundary needs forward difference

**Debug:**
```python
print(f"Left BC values: {self.u_curr[5, 0]}")  # Should match BC function
print(f"BC function value: {f_left(self.grid.y[5])}")
assert np.allclose(self.u_curr[:, 0], f_left(self.grid.y)), "BC not applied!"
```

### Problem 5: Array Shape Mismatch (Shape Error)

**Symptoms:** ValueError about incompatible shapes when adding arrays

**Causes & Solutions:**

1. **Stencil slices produce different shapes:**
   ```python
   # WRONG: These aren't the same shape!
   d2u_dx2 = (u[1:-1, 2:] - 2*u[1:-1, 1:-1] + u[1:-1, :-2]) / dx**2  # Shape (Ny-2, Nx-3)
   d2u_dy2 = (u[2:, 1:-1] - 2*u[1:-1, 1:-1] + u[:-2, 1:-1]) / dy**2  # Shape (Ny-3, Nx-2)
   # Can't add these!
   
   # RIGHT: All should be (Ny-2, Nx-2)
   d2u_dx2 = (u[1:-1, 2:] - 2*u[1:-1, 1:-1] + u[1:-1, :-2]) / dx**2
   d2u_dy2 = (u[2:, 1:-1] - 2*u[1:-1, 1:-1] + u[:-2, 1:-1]) / dy**2
   ```

2. **Update assignment shape mismatch:**
   ```python
   # WRONG: Interior is (Ny-2, Nx-2), but laplacian might be different
   self.u_next[1:-1, 1:-1] = laplacian  # Only works if shapes match
   
   # RIGHT: Always verify shapes
   assert laplacian.shape == (self.grid.shape[0]-2, self.grid.shape[1]-2)
   assert self.u_next[1:-1, 1:-1].shape == laplacian.shape
   ```

**Debug:**
```python
print(f"u_curr shape: {self.u_curr.shape}")
print(f"Interior shape: {self.u_curr[1:-1, 1:-1].shape}")
print(f"d2u_dx2 shape: {d2u_dx2.shape}")
print(f"Laplacian shape: {laplacian.shape}")
assert laplacian.shape == self.u_curr[1:-1, 1:-1].shape
```

### Problem 6: Error Stays at 1.0 or Doesn't Decrease

**Symptoms:** Error computation returns 1.0 every step, no improvement

**Causes & Solutions:**

1. **get_analytic_solution() returns zeros:**
   - If you return np.zeros_like(X), error = ||numerical||, not decreasing
   - Implement actual analytic solution

2. **Reference solution wrong:**
   - Verify analytic solution satisfies same BCs and PDE
   - Check: Does it match initial condition?

3. **Time dependence missing:**
   - get_analytic_solution(t) must depend on actual time t
   - Check: Does u_star change as solver advances?

**Debug:**
```python
u_analytic = self.get_analytic_solution(self.step_count * self.config.dt)
print(f"Analytic solution max: {np.max(u_analytic)}")
print(f"Numerical solution max: {np.max(self.u_curr)}")
print(f"Difference: {np.max(np.abs(self.u_curr - u_analytic))}")
```

---

## Example: Implementing Burgers' Equation

For reference, here's a more complex example:

```python
class BurgersConfig(SolverConfig):
    def __init__(self, dt: float, nu: float, **kwargs):
        super().__init__(dt=dt, **kwargs)
        self.nu = nu  # Viscosity


class BurgersSolver(BaseSolver):
    def __init__(self, domain, grid, config, bc_functions):
        super().__init__(domain, grid, config, bc_functions)
        self.u_curr = np.zeros(grid.shape)
        self.u_next = np.zeros(grid.shape)
    
    def step_once(self) -> None:
        """Burgers' equation: ∂u/∂t + u·∂u/∂x = ν·∂²u/∂x²"""
        nu = self.pde_config.nu
        dt = self.config.dt
        
        # u·∂u/∂x (central difference for ∂u/∂x)
        u_mid = self.u_curr[1:-1, 1:-1]
        du_dx = (self.u_curr[1:-1, 2:] - self.u_curr[1:-1, :-2]) / (2*self.dx)
        advection = u_mid * du_dx
        
        # ν·∂²u/∂x²
        d2u_dx2 = (self.u_curr[1:-1, 2:] - 2*self.u_curr[1:-1, 1:-1] + 
                   self.u_curr[1:-1, :-2]) / (self.dx**2)
        diffusion = nu * d2u_dx2
        
        # Update
        self.u_next[1:-1, 1:-1] = u_mid + dt * (-advection + diffusion)
        
        self.u_curr, self.u_next = self.u_next, self.u_curr
        self.step_count += 1
    
    def apply_bc(self) -> None:
        """Periodic or zero BCs"""
        phase_key = "neg" if self.phase == 1 else "pos"
        f_left, f_right, f_bottom, f_top = self.bc_functions[phase_key]
        self.u_curr[:, 0] = f_left(self.y)
        self.u_curr[:, -1] = f_right(self.y)
    
    def get_analytic_solution(self, t: float) -> np.ndarray:
        # For now, return zeros (Burgers' has complex solutions)
        return np.zeros_like(self.X)
```

---

## Next Steps

1. **Pick a PDE** (Wave, Advection, Burgers, etc.)
2. **Create your solver class** following this template
3. **Implement the three required methods**
4. **Test with small examples**
5. **Integrate with visualization/CLI**
6. **Create analytic solution module** (if needed)

The framework does all the heavy lifting—you just need to fill in the numerical scheme!

---

*For questions, see ARCHITECTURE.md for class diagrams and API reference.*
