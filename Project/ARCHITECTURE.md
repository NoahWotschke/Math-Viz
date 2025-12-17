# Multi-PDE Modular Solver Architecture

## File Structure After Completion

```
math-viz/
├── solve.py                          # TODO: Universal CLI dispatcher
├── heat2d_rect_fd.py                 # ✅ COMPLETE: Rect heat solver (refactored to use OOP)
├── heat2d/
│   ├── __init__.py
│   ├── math_settings.py              # Configuration (will extend for wave/disc)
│   ├── vis_settings.py               # Visualization settings
│   │
│   ├── bc/                           # ✅ COMPLETE: Boundary conditions
│   │   ├── __init__.py
│   │   ├── funcs.py                 # 40+ BC shape functions + combinators
│   │   └── builder.py               # BC spec constructor (recursive)
│   │
│   ├── domains/                      # In Progress
│   │   ├── __init__.py
│   │   ├── base.py                  # ✅ Grid dataclass, Domain ABC
│   │   ├── rectangle.py             # ✅ RectangleDomain (4 boundaries)
│   │   ├── bar1d.py                 # TODO: Bar1DDomain (2 boundaries, 1D)
│   │   └── disc.py                  # TODO: DiscDomain (radial + angular)
│   │
│   ├── solvers/                      # In Progress
│   │   ├── __init__.py
│   │   ├── base.py                  # ✅ SolverConfig, BaseSolver ABC
│   │   ├── heat2d_rect.py           # ✅ Heat2DRectSolver (explicit FD)
│   │   ├── heat1d_bar.py            # TODO: Heat1DBarSolver (1D stencil)
│   │   ├── heat2d_disc.py           # TODO: Heat2DDiscSolver (polar Laplacian)
│   │   ├── wave1d_bar.py            # TODO: Wave1DBarSolver (2-layer)
│   │   ├── wave2d_rect.py           # TODO: Wave2DRectSolver (2-layer rect)
│   │   └── wave2d_disc.py           # TODO: Wave2DDiscSolver (2-layer polar)
│   │
│   ├── analytic/                     # In Progress
│   │   ├── __init__.py
│   │   ├── heat_rect.py             # ✅ analytic_dirichlet_rect_series()
│   │   ├── heat_bar.py              # TODO: 1D sine series
│   │   ├── heat_disc.py             # TODO: 2D Bessel series
│   │   ├── wave_bar.py              # TODO: 1D sine product
│   │   ├── wave_rect.py             # TODO: 2D sine product
│   │   └── wave_disc.py             # TODO: 2D Bessel product
│   │
│   └── visualization/               # TODO: Generalized rendering
│       └── visualizer.py            # Domain-agnostic Visualizer class
```

---

## Current File Structure (Status - Updated)

```
heat2d/
├── __init__.py
├── math_settings.py            # Configuration: domain dims, BC specs, solver params
├── vis_settings.py             # Visualization: plot/animation settings
│
├── bc/                         # ✅ REFACTORED: Boundary conditions module
│   ├── __init__.py
│   ├── funcs.py               # 40+ BC shape functions + combinators (domain-independent)
│   └── builder.py             # BC specification-based constructor (recursive)
│
├── domains/                    # ✅ CREATED: Domain abstraction layer
│   ├── __init__.py
│   ├── base.py                # Grid dataclass, Domain ABC with boundaries()
│   └── rectangle.py           # RectangleDomain with 4 boundaries (left/right/bottom/top)
│
├── solvers/                    # ✅ CREATED: Solver abstraction layer
│   ├── __init__.py
│   ├── base.py                # SolverConfig dataclass, BaseSolver ABC
│   └── heat2d_rect.py         # Heat2DRectConfig, Heat2DRectSolver (2D FD stencil)
│
└── analytic/                   # ✅ CREATED: Analytic solutions module
    ├── __init__.py
    └── heat_rect.py           # analytic_dirichlet_rect_series() for Laplace on rectangle
```


## Core Design

### Domain Abstraction (`heat2d/domains/`)
Encapsulates geometry and grid generation for different coordinate systems:
- **Domain ABC** - Abstract base class with boundaries() and create_grid() interface
- **Grid dataclass** - Stores coordinates (x, y), spacing (dx, dy), and mesh (X, Y)

**Current Implementations:**
- **RectangleDomain** - [0, Lx] × [0, Ly] with 4 boundaries (left, right, bottom, top)

**Planned:**
- **Bar1DDomain** - [0, L] with 2 boundaries (left, right)
- **DiscDomain** - Polar [0, R] × [0, 2π] with radial boundary + periodic angular

### Solver Abstraction (`heat2d/solvers/`)
Implements time-stepping PDEs with configurable boundary conditions:
- **SolverConfig dataclass** - Parameters (dt, alpha, cycles_per_frame, alternate, steps_per_cycles)
- **BaseSolver ABC** - Abstract methods (step_once, apply_bc, get_analytic_solution) + concrete utilities (compute_error, update_phase_and_lines)

**Current Implementations:**
- **Heat2DRectSolver** - Explicit finite difference on rectangles with 2D Laplacian

**Planned:**
- **Heat1DBarSolver** - 1D heat equation on bar
- **Heat2DDiscSolver** - Heat equation in polar coordinates
- **Wave1DBarSolver**, **Wave2DRectSolver**, **Wave2DDiscSolver** - Wave variants

### Boundary Condition Library (`heat2d/bc/`)
Domain-independent collection of parameterizable BC functions:
- **40+ functions** - const, linear, polynomial, trig, exponential, Gaussian, noise, etc.
- **Combinators** - scale, shift, mirror, normalize, multiply, add, negate
- **Builder** - Construct complex BCs from recursive specifications

### Analytic Solutions (`heat2d/analytic/`)
Reference solutions for error computation and validation:
- **heat_rect.py** - Laplace equation on rectangle via eigenfunction series (Legendre quadrature)

**Planned:**
- **heat_bar.py** - 1D heat series expansion
- **heat_disc.py** - Bessel function series for polar disc
- **wave_rect.py**, **wave_bar.py**, **wave_disc.py** - Wave equation solutions

## File Dependency Graph

```
heat2d_rect_fd.py (CLI)
    ↓
heat2d/solvers/heat2d_rect.py (Heat2DRectSolver)
    ↓
heat2d/solvers/base.py (BaseSolver ABC)
    ├─→ heat2d/domains/base.py (Domain ABC, Grid)
    │       └─→ heat2d/domains/rectangle.py (RectangleDomain)
    ├─→ heat2d/analytic/heat_rect.py (analytic_dirichlet_rect_series)
    └─→ heat2d/bc/builder.py (build_bc_from_spec)
            └─→ heat2d/bc/funcs.py (40+ BC functions)
```

## Key Classes

### Domain (Base Class)
```python
class Domain(ABC):
    def boundaries(self) -> List[str]: ...
    def create_grid(self, res_per_unit: float) -> Grid: ...
```

### BaseSolver (Base Class)
```python
class BaseSolver(ABC):
    def __init__(self, domain: Domain, grid: Grid, config: SolverConfig, bc_functions: Dict): ...
    
    # Abstract methods (implement in subclass)
    def step_once(self) -> None: ...
    def apply_bc(self) -> None: ...
    def get_analytic_solution(self, t: float) -> np.ndarray: ...
    
    # Concrete methods (shared by all solvers)
    def compute_error(self, u_ref_pos: np.ndarray, u_ref_neg: np.ndarray) -> float: ...
    def update_phase_and_lines(self, boundary_lines: Dict, y: np.ndarray, x: np.ndarray, scale: float) -> None: ...
```

### Heat2DRectSolver (Concrete Implementation)
```python
class Heat2DRectSolver(BaseSolver):
    def step_once(self) -> None:  # 2D FD stencil
    def apply_bc(self) -> None:   # Dirichlet boundaries
    def get_analytic_solution(self, t: float) -> np.ndarray:  # Laplace series
```

## Extension Pattern

### Adding a New Domain
1. Create `heat2d/domains/newdomain.py` with class extending `Domain`
2. Implement `boundaries()` → list of boundary names
3. Implement `create_grid()` → return Grid with coordinates and spacing

### Adding a New Solver
1. Create `heat2d/solvers/pde_domain.py` with class extending `BaseSolver`
2. Implement `step_once()` → numerical time-stepping logic
3. Implement `apply_bc()` → enforce boundary conditions
4. Implement `get_analytic_solution()` → return reference solution

### Adding a New Analytic Solution
1. Create `heat2d/analytic/pde_domain.py` with function computing reference solution
2. Return numpy array on same grid as numerical solution
3. Export from `heat2d/analytic/__init__.py`

## Current CLI Usage
```bash
python heat2d_rect_fd.py --spin --res 51 --seconds 10 --Lx 2.0 --Ly 1.0 --alterate --steps_per_cycles 500
```

## Future Unified CLI
```bash
python solve.py --pde heat --domain rect --Lx 2.0 --Ly 1.0 --spin
python solve.py --pde heat --domain bar --L 1.0
python solve.py --pde heat --domain disc --R 1.0
python solve.py --pde wave --domain rect --Lx 1.0 --Ly 1.0 --c 1.0
```

## Implementation Statistics

| Component  | Files | Lines      | Status                |
| ---------- | ----- | ---------- | --------------------- |
| Domains    | 2     | ~250       | ✅ 1/3 planned         |
| Solvers    | 2     | ~600       | ✅ 1/7 planned         |
| Analytic   | 1     | ~150       | ✅ 1/6 planned         |
| BC Library | 2     | ~750       | ✅ Complete            |
| CLI        | 1     | ~490       | ✅ Working (rect only) |
| **TOTAL**  | **8** | **~2,240** | **Phase 2/8**         |

## Next Priorities

1. **Bar1D (Step 4)** - Simplest extension; introduces 1D stencil and line plotting
2. **Disc (Step 5)** - Introduces polar coordinates and Bessel functions
3. **Wave (Step 6)** - Introduces 2-layer time stepping and CFL constraints
4. **Visualization (Step 7)** - Generalize 3D and 1D rendering
5. **Unified CLI (Step 8)** - Dispatcher supporting all 6 (PDE, domain) combinations

---
*Last updated: December 16, 2025*
