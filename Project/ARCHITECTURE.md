# Multi-PDE Modular Solver Architecture

## File Structure After Completion

```
math-viz/
├── solve.py                          # ✅ COMPLETE: Universal CLI dispatcher (routes all PDEs + domains)
├── heat2d_rect_fd.py                 # ⚠️ DEPRECATED: Use solve.py instead (solve.py --pde heat --domain rect)
├── heat2d_disc_fd.py                 # ⚠️ DEPRECATED: Use solve.py instead (solve.py --pde heat --domain disc)
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
│   └── visualization/               # ✅ COMPLETE: Generalized rendering
│       ├── __init__.py
│       └── visualizer.py            # ✅ Visualizer3D + VisualizerConfig (domain-agnostic)
```

---

## Current File Structure (Status - Updated Dec 19, 2025)

```
solve.py                           # ✅ NEW: Universal dispatcher for all PDE/domain combinations

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
├── analytic/                   # ✅ CREATED: Analytic solutions module
│   ├── __init__.py
│   └── heat_rect.py           # analytic_dirichlet_rect_series() for Laplace on rectangle
│
└── visualization/             # ✅ NEW: Generic 3D visualization module
    ├── __init__.py
    └── visualizer.py          # Visualizer3D class (reusable for all PDEs/domains)
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
solve.py (Universal CLI Dispatcher)
    ↓
heat2d/solvers/heat2d_rect.py (Heat2DRectSolver for --pde heat --domain rect)
    ↓
heat2d/solvers/base.py (BaseSolver ABC)
    ├─→ heat2d/domains/base.py (Domain ABC, Grid)
    │       └─→ heat2d/domains/rectangle.py (RectangleDomain)
    ├─→ heat2d/analytic/heat_rect.py (analytic_dirichlet_rect_series)
    ├─→ heat2d/bc/builder.py (build_bc_from_spec)
    │       └─→ heat2d/bc/funcs.py (40+ BC functions)
    └─→ heat2d/visualization/visualizer.py (Visualizer3D - REUSABLE FOR ALL SOLVERS)
```

**Key advantage:** All future solvers (Heat1D, Heat2DDisc, Wave*) will reuse `Visualizer3D` via `solve.py`.
Only need to implement domain-specific logic in solver classes.


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

## Current Unified CLI (✅ Now Available)
```bash
python solve.py --pde heat --domain rect --spin --res 21 --seconds 10
python solve.py --pde heat --domain rect --save --out animation.mp4

# Future (after implementing new domains/PDEs):
python solve.py --pde heat --domain bar --L 1.0
python solve.py --pde heat --domain disc --R 1.0
python solve.py --pde wave --domain rect --Lx 1.0 --Ly 1.0 --c 1.0
```

## Legacy Scripts (⚠️ Deprecated)
```bash
# Old way (still works but not recommended):
python heat2d_rect_fd.py --spin --res 51 --seconds 10
python heat2d_disc_fd.py --spin --seconds 10

# New way (unified):
python solve.py --pde heat --domain rect --spin --res 51 --seconds 10
python solve.py --pde heat --domain disc --spin --seconds 10
```

## Implementation Statistics

| Component  | Files  | Lines      | Status                     |
| ---------- | ------ | ---------- | -------------------------- |
| Domains    | 2      | ~250       | ✅ 1/3 planned              |
| Solvers    | 2      | ~600       | ✅ 1/7 planned              |
| Analytic   | 1      | ~150       | ✅ 1/6 planned              |
| BC Library | 2      | ~750       | ✅ Complete                 |
| Vis Layer  | 2      | ~350       | ✅ NEW: Generic visualizer  |
| CLI        | 1      | ~390       | ✅ Complete: solve.py       |
| **TOTAL**  | **10** | **~2,490** | **Phase 3/8 (↑ from 2/8)** |

**What changed:** Added generic `Visualizer3D` class (+350 LOC) and refactored CLI into `solve.py` (+390 LOC).
Replaces scattered visualization code in heat2d_rect_fd.py, heat2d_disc_fd.py, etc.
Net result: Reduced duplication, easier to extend with new domains/PDEs.

## Next Priorities

1. **Disc Domain (Step 5)** - Implement heat2d/domains/disc.py + heat2d/solvers/heat2d_disc.py
   - Add route in solve.py: `--pde heat --domain disc`
   - Reuses Visualizer3D automatically
2. **Bar1D (Step 4)** - 1D solver with line plot visualization extension
3. **Wave PDEs (Step 6)** - 2-layer time stepping solvers (wave1d_bar, wave2d_rect, wave2d_disc)
4. **1D Visualization** - Extend Visualizer3D or create Visualizer1D for bar/wave1d
5. **Full solver matrix** - All 6 (3 PDEs × 2 domains) = heat rect/disc, wave rect/disc/bar, advection, etc.

---
*Last updated: December 19, 2025 - Refactored CLI and visualization layer*

