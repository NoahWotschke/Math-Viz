# Multi-PDE Modular Solver Framework

## Next Steps & Implementations
- [ ] Update Website UI
  - [ ] Add performance presets (low/medium/high)
  - [ ] remove Emojies
  - [ ] Restructure layout for clarity
- [ ] Start Dokumentation on Personal Website
  - [ ] Derivation of Heat on Rectangle
  - [ ] Explanation of Numerical Methods
  - [ ] Explanation of Analytic Solution
- [ ] Put Wesbite Link on LinkedIn Profile


## To-DO
- Step 1: Create Abstract Base Classes (Phase 1.1-1.2)
    - [x] **1.1a** Create `heat2d/domains/base.py` with `Domain` abstract base class
      - [x] Methods: `boundaries()`, `create_grid()`, `apply_bc_point()`
      - [x] Properties: `name`, `num_boundaries`
    
    - [x] **1.1b** Create `heat2d/solvers/base.py` with `BaseSolver` abstract base class
      - [x]  Abstract methods: `step_once()`, `apply_bc()`, `get_analytic_solution()`
      - [x] Concrete: `compute_error()`, `update_phase_and_lines()`, error tracking
      - [x] Attributes: `config`, `domain`, `grid`, `bc_functions`

    - [x] **1.2** Update `heat2d/grid.py` to add `Grid` class
      - [x] Constructor accepts coordinate system (cartesian/polar/1d)
      - [x] Attributes: `x`, `y` (or `r`, `theta`), `dx`, `dy`, shape info

- Step 2: Implement Rectangle Domain (Phase 1.3)
  - [x] **2.1** Create `heat2d/domains/rectangle.py` with `RectangleDomain`
      - [x] Constructor: `Lx`, `Ly`, `res_per_unit`
      - [x] Methods: `create_grid()`, `boundaries()` → [left, right, bottom, top]
      - [x] BC sides: 4 boundaries indexed [0, 1, 2, 3]

  - [x] **2.2** Update `heat2d/math_settings.py`
      - [x] Add domain specification: `domain_type = "rect"`, `Lx`, `Ly`
      - [x] Existing BC specs already in correct format

- Step 3: Refactor Heat 2D Rectangle Solver (Phase 2.1)
  - [x] **3.1** Create `heat2d/solvers/heat2d_rect.py` with `Heat2DRectSolver(BaseSolver)`
      - [x] Constructor takes `domain: RectangleDomain`, `config: SolverConfig`, `bc_functions`
      - [x] `step_once()` — Copy 2D FD stencil logic from existing solver
      - [x] `apply_bc()` — Copy Dirichlet application logic
      - [x] Store BC in 4-element lists (index 0=left, 1=right, 2=bottom, 3=top)

  - [x] **3.2** Extract analytic solution to `heat2d/analytic/heat_rect.py`
    - [x] Move `analytic_dirichlet_rect_series()` function
    - [x] Keep all eigenvalue/integration logic intact

  - [x] **3.3** Update `heat2d_rect_fd.py` to use new solver
    - [x] Import `Heat2DRectSolver`, `RectangleDomain` from new modules
    - [x] Instantiate: `domain = RectangleDomain(Lx, Ly, res_per_unit)`
    - [x] Instantiate: `solver = Heat2DRectSolver(domain, config, bc_functions)`
    - [x] Replace main loop to call `solver.step_once()`, `solver.compute_error()`
    - [x] **Test:** Run `python heat2d_rect_fd.py --spin` — output must match original exactly

- Step 4: Create 1D Bar Domain & Solver (Phase 2.2 - Simplest New Solver)
  - [ ] **4.1** Create `heat2d/domains/bar1d.py` with `Bar1DDomain`
    - [ ] Constructor: `L`, `res_per_unit`
    - [ ] Boundaries: [left, right] (2 boundaries)
    - [ ] Grid: 1D array (just `x`, no `y`)

  - [ ] **4.2** Create `heat2d/solvers/heat1d_bar.py` with `Heat1DBarSolver(BaseSolver)`
    - [ ] `step_once()` — 1D Laplacian stencil: `d²u/dx²`
    - [ ] `apply_bc()` — 2 Dirichlet BCs at x=0, x=L
    - [ ]Analytic solution: Sine series (from eigenfunction expansion)

  - [ ] **4.3** Create `heat2d/analytic/heat_bar.py`
    - [ ] `analytic_heat_1d_bar(x, t, L, alpha, fourier_coeffs)` function
    - [ ] Use sine series: $u(x,t) = \sum b_n \sin(n\pi x/L) e^{-\alpha (n\pi/L)^2 t}$

  - [ ] **4.4** Create minimal CLI entry in `solve.py` (temporary)
    - [ ] Test: `python solve.py --pde heat --domain bar --L 1.0`

  - [ ] **4.5** Create simple 1D visualization
    - [ ] Line plot (x-axis = space, y-axis = temperature)
    - [ ] Animate over time steps with Matplotlib

- Step 5: Create Disc Domain (Phase 1.3, 2.1)
  - [ ] **5.1** Create `heat2d/domains/disc.py` with `DiscDomain`
    - [ ] Constructor: `R`, `res_r`, `res_theta`
    - [ ] Boundary: radial edge (1 boundary, θ ∈ [0, 2π])
    - [ ] Grid: polar coordinates `r`, `theta` using `numpy.meshgrid()`

  - [ ] **5.2** Create `heat2d/solvers/heat2d_disc.py` with `Heat2DDiscSolver(BaseSolver)`
    - [ ] `step_once()` — 2D Laplacian in polar: $\frac{1}{r}\frac{\partial}{\partial r}(r\frac{\partial u}{\partial r}) + \frac{1}{r^2}\frac{\partial^2 u}{\partial\theta^2}$
    - [ ] `apply_bc()` — Dirichlet at r=R, periodic in θ
    - [ ] Use finite differences (account for singular point at r=0)

  - [ ] **5.3** Create `heat2d/analytic/heat_disc.py`
    - [ ] Bessel function series: $u(r,t,\theta) = \sum b_{n,m} J_n(j_{n,m} r/R) \cos(n\theta) e^{-\alpha (j_{n,m}/R)^2 t}$
    - [ ] Use `scipy.special.jn()` for Bessel functions

- Step 6: Implement Wave Solvers (Phase 2.3)
  - [ ] **6.1** Create base `wave2d/` package (or extend `heat2d/`)
    - [ ] Copy domain classes (reuse from heat2d)
    - [ ] Create new solver directory: `heat2d/solvers/wave_*.py`

  - [ ] **6.2** Create `heat2d/solvers/wave1d_bar.py` with `Wave1DBarSolver(BaseSolver)`
    - [ ] 2-layer time stepping: $u^{n+1} = 2u^n - u^{n-1} + c^2(dt/dx)^2(u^{n}_{i+1} - 2u^{n}_i + u^{n}_{i-1})$
    - [ ] CFL: $c \cdot dt / dx \leq 1$
    - [ ] `apply_bc()` — Dirichlet at x=0, x=L

  - [ ] **6.3** Create `heat2d/solvers/wave2d_rect.py` with `Wave2DRectSolver(BaseSolver)`
    - [ ] 2-layer time stepping on rectangle
    - [ ] CFL: $c \cdot dt / \min(dx, dy) \leq 1/\sqrt{2}$

  - [ ] **6.4** Create `heat2d/solvers/wave2d_disc.py` with `Wave2DDiscSolver(BaseSolver)`
    - [ ] 2-layer time stepping in polar coordinates

  - [ ] **6.5** Create analytic solution modules
    - [ ]`heat2d/analytic/wave_bar.py`, `wave_rect.py`, `wave_disc.py`

- Step 7: Generalize Visualization (Phase 3.1)
  - [ ] **7.1** Create `heat2d/visualization/visualizer.py` with `Visualizer` class
    - Methods:
      - [ ] `plot_3d_surface(domain, u)` — For 2D domains (rect/disc)
      - [ ] `plot_1d_line(domain, u)` — For 1D bar
      - [ ] `update_wireframe(ax, domain, u)` — Update 3D surface
      - [ ] `update_boundary_lines(ax, domain, bc_values)` — Draw BC on edges
      - [ ] Support all domain types via type checking or virtual methods

  - [ ] **7.2** Update `heat2d_rect_fd.py` to use `Visualizer`
    - [ ] Remove embedded `recreate_surface()`, use `visualizer.update_wireframe()`

- Step 8: Build Unified CLI (Phase 4.1)
  - [ ] **8.1** Create `solve.py` with argparse
    - [ ] Arguments: `--pde {heat,wave}`, `--domain {rect,disc,bar}`, domain params, solver params
    - [ ] Example: `python solve.py --pde heat --domain rect --Lx 2.0 --Ly 1.0 --alpha 0.1 --spin`
    - [ ] Dispatcher: Load domain, create solver, run simulation

  - [ ] **8.2** Support all 6 combinations
    - [ ] Heat: rect, disc, bar (3)
    - [ ] Wave: rect, disc, bar (3)
    - [ ] Total: 6 solvers

- Step 9: Testing & Validation
  - [ ] **9.1** Regression test: Heat 2D Rect
    - [ ] Run refactored solver, compare output to original `heat2d_rect_fd.py`
    - [ ] Check: grid values, analytic error, animation frames

  - [ ] **9.2** Unit tests for new domain classes
    - [ ] Test: `grid creation, boundary indexing, BC point lookup`

  - [ ] **9.3** Integration test for each solver
    - [ ] Run short simulation (5 steps), check stability & error computation

  - [ ] **9.4** CLI test all 6 combinations
    - [ ] `python solve.py` for each (pde, domain) pair

### Target PDEs & Domains
- **Heat**
    - Rectangle (✅ exists → Step 3)
    - Disc (→ Step 5)
    - 1D Bar (→ Step 4)
- **Wave** 
    - Rectangle (→ Step 6.3)
    - Disc (→ Step 6.4)
    - 1D Bar (→ Step 6.2)

---

## Phase 1: Abstract Solver Architecture

### 1.1 Create BaseSolver Class
aa**File:** `heat2d/solvers/base.py`

Abstract base class for all PDE solvers with core time-stepping logic, BC caching, phase switching, and error computation.

### 1.2 Generalize Domain Representation
**Files:** `heat2d/domains/base.py`, `rectangle.py`, `disc.py`, `bar1d.py`

Create `Domain` class hierarchy supporting arbitrary geometries:
- **RectangleDomain** — 4 boundaries (left/right/bottom/top)
- **DiscDomain** — Radial boundary + periodic angular
- **Bar1DDomain** — 2 boundaries (left/right)

### 1.3 Refactor Grid Management
**File:** `heat2d/grid.py` (update existing)

Replace global rectangular grid with `Grid` class supporting multiple coordinate systems (Cartesian, polar, 1D).

---

## Phase 2: PDE-Specific Solver Subclasses

### 2.1 Heat 2D Solvers
- `heat2d/solvers/heat2d_rect.py` — Heat on rectangle (refactored existing)
- `heat2d/solvers/heat2d_disc.py` — Heat on disc (Bessel series)

### 2.2 Heat 1D Solver
- `heat2d/solvers/heat1d_bar.py` — Heat on bar (sine series)

### 2.3 Wave Solvers
- `heat2d/solvers/wave2d_rect.py` — Wave on rectangle
- `heat2d/solvers/wave2d_disc.py` — Wave on disc
- `heat2d/solvers/wave1d_bar.py` — Wave on bar

---

## Phase 3: Generalized Visualization

### 3.1 Create Visualizer Class
**File:** `heat2d/visualization/visualizer.py`

Domain-agnostic visualizer supporting:
- **Rectangle:** 3D surface + wireframe + boundary lines (existing)
- **Disc:** Polar 3D surface or contour plot
- **1D Bar:** Time-series line plot or heatmap

---

## Phase 4: Unified CLI

### 4.1 Universal Entry Point
**File:** `solve.py` (new)

Single command handles all (PDE, domain) combinations:
```bash
python solve.py --pde heat --domain rect --Lx 2.0 --Ly 1.0 --spin
python solve.py --pde heat --domain disc --R 1.0
python solve.py --pde wave --domain rect --Lx 1.0 --Ly 1.0 --seconds 10
python solve.py --pde heat --domain bar --L 1.0
python solve.py --pde wave --domain bar --L 1.0 --c 1.0
```

---

## Phase 5: Analytic Solution Organization

### 5.1 Separate Modules by (PDE, Domain)
**Directory:** `heat2d/analytic/`

- `heat_rect.py` — Heat on rectangle (existing series)
- `heat_disc.py` — Heat on disc (Bessel series)
- `heat_bar.py` — Heat on bar (sine series)
- `wave_rect.py` — Wave on rectangle (sine product)
- `wave_disc.py` — Wave on disc (Bessel/sine product)
- `wave_bar.py` — Wave on bar (sine product)

---

## Phase 6: Implementation Order

1. **Refactor Heat2DRect** — Extract to `BaseSolver` + `RectangleDomain` + `Heat2DRectSolver`
2. **Implement domains** — `DiscDomain`, `Bar1DDomain` with grid generation
3. **Add Heat1D solver** — `Heat1DBarSolver` + sine series analytic solution
4. **Add Heat2D Disc** — `Heat2DDiscSolver` + Bessel series + polar visualization
5. **Add Wave solvers** — All three variants with 2-layer time stepping
6. **Build unified CLI** — `solve.py` dispatcher with all combinations

---

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

## File Structure After Completion

```

## Current File Structure (Status - Updated)

### Completed Implementation ✅
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

### Key Components by File (Current State)

| File                     | Purpose                             | Status     | Reusable?    |
| ------------------------ | ----------------------------------- | ---------- | ------------ |
| `bc/funcs.py`            | BC shape functions + composition    | ✅ Complete | **100%**     |
| `bc/builder.py`          | BC spec constructor                 | ✅ Complete | **100%**     |
| `domains/base.py`        | Domain ABC + Grid dataclass         | ✅ Complete | **100%**     |
| `domains/rectangle.py`   | Rectangle domain with 4 boundaries  | ✅ Complete | **100%**     |
| `solvers/base.py`        | BaseSolver ABC + SolverConfig       | ✅ Complete | **100%**     |
| `solvers/heat2d_rect.py` | Heat2D rectangle FD solver          | ✅ Complete | ✅ Base model |
| `analytic/heat_rect.py`  | Laplace equation series solution    | ✅ Complete | ✅ Reference  |
| `heat2d_rect_fd.py`      | Rectangle solver CLI (uses new OOP) | ✅ Working  | ✅ Reference  |

### Architecture Summary

**Phase 1 (Abstract Architecture):** ✅ COMPLETE
- `Domain` ABC with grid creation and boundary specification
- `BaseSolver` ABC with time-stepping, BC application, error computation
- `Grid` dataclass supporting Cartesian (extensible to polar/1D)

**Phase 2 (Rectangle Heat Solver):** ✅ COMPLETE
- `RectangleDomain` — 4-sided rectangular domain with indexed boundaries
- `Heat2DRectConfig` + `Heat2DRectSolver` — 2D explicit FD heat equation on rectangles
- `analytic_dirichlet_rect_series()` — Eigenfunction series solution (Laplace)
- Fully integrated into `heat2d_rect_fd.py` CLI

**Phase 3+ (Remaining Work):** See "To-DO" section above

---

## Design Decisions

**Q1: BC Parameterization for Disc?**
- Use angle-parameterized functions: single `bc_radial_spec` applies to all θ

**Q2: Grid Resolution?**
- "points per unit length" for all (uniform across geometries)
- Disc: separate `res_r` (radial) and `res_theta` (angular)

**Q3: Analytic Solutions?**
- Separate modules (clean separation, not coupled to solvers)

**Q4: 1D Visualization?**
- Time-series line plot (x-axis = space, animate over time)
- Optional heatmap export (space × time 2D)

**Q5: Backward Compatibility?**
- Keep `heat2d_rect_fd.py` as reference; gradually migrate to `solve.py`

---

## Implementation Status Summary (December 16, 2025)

### ✅ COMPLETED (Steps 1-3)
1. ✅ **Step 1:** Abstract architecture (Domain ABC, BaseSolver ABC, Grid dataclass)
2. ✅ **Step 2:** Rectangle domain (RectangleDomain with 4 boundaries)
3. ✅ **Step 3:** Heat 2D rectangle solver refactoring
   - `Heat2DRectSolver` fully implemented
   - `heat2d_rect_fd.py` refactored to use new classes
   - All functionality working correctly

### 📦 Current Package Structure
```
heat2d/
├── bc/                    # ✅ Boundary conditions (40+ functions)
├── domains/               # ✅ Domain abstractions
│   ├── base.py
│   └── rectangle.py      # ✅ RectangleDomain
├── solvers/              # ✅ Solver abstractions
│   ├── base.py
│   └── heat2d_rect.py    # ✅ Heat2DRectSolver
└── analytic/             # ✅ Analytic solutions
    └── heat_rect.py      # ✅ Laplace series
```

---

## 🌐 Deployment & Portfolio (Proposed)

### Step 9: Create Personal Website (NEW)
- [ ] **9.1** Set up static site generator or hosting
  - [ ] Options: GitHub Pages, Vercel, Netlify (free tier)
  - [ ] Or custom domain with simple HTML/CSS
  
- [ ] **9.2** Portfolio page structure
  - [ ] Project overview (problem, solution, results)
  - [ ] Download link to GitHub repository
  - [ ] Installation & quick-start instructions
  - [ ] Screenshots/GIFs of visualizations
  - [ ] Technical documentation links

- [ ] **9.3** Interactive demo (optional)
  - [ ] Embed Streamlit app via iframe
  - [ ] Or link to hosted Streamlit instance
  - [ ] Alternative: WebGL 3D viewer (Babylon.js, Three.js)

- [ ] **9.4** GitHub repository setup
  - [ ] Clean folder structure for distribution
  - [ ] Comprehensive README (already done)
  - [ ] requirements.txt for easy pip install
  - [ ] License file (MIT recommended)

### Step 10: Deploy to Streamlit Cloud (NEW)
- [ ] **10.1** Prepare for Streamlit Cloud
  - [ ] Create `requirements.txt` with all dependencies
  - [ ] Ensure GitHub repo is public (or private with access granted)
  - [ ] Test locally with `streamlit run app.py`

- [ ] **10.2** Set up Streamlit Cloud
  - [ ] Sign up at [streamlit.io/cloud](https://streamlit.io/cloud)
  - [ ] Connect GitHub account
  - [ ] Select repo and app.py file
  - [ ] Deploy automatically on push

- [ ] **10.3** Domain configuration (optional)
  - [ ] Map custom domain to Streamlit Cloud app
  - [ ] SSL certificate (automatic with Streamlit)

- [ ] **10.4** Update README
  - [ ] Add link to live Streamlit app
  - [ ] Note about image compression vs local version
  - [ ] Clear instructions: "Best quality: download locally"

**Benefits:** 
- Free hosting with auto-scaling
- 24/7 uptime
- Auto-deploys on GitHub push
- Users can run `app.py` without setup

**Rationale:** Portfolio website is essential for showcasing work to employers/collaborators. Users can download locally for best-quality visualization.