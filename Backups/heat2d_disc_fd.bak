"""2D Heat Equation Solver and Visualizer (Circular Disk Domain)

Solves the 2D heat equation on a unit disk using polar coordinates and
finite differences:

    ∂u/∂t = α(∂²u/∂r² + (1/r)∂u/∂r + (1/r²)∂²u/∂θ²)  on [0, 1] × [0, 2π]

Features:
  - Radial and angular finite difference discretization
  - Neumann boundary condition at center (∂u/∂r = 0 for symmetry)
  - Dirichlet boundary condition on disk edge
  - Analytical reference solution via series expansion
  - 3D polar coordinate visualization
  - MP4 export capability via FFmpeg

Configuration:
  - Grid resolution: Nr (radial), Nθ (angular) points
  - Boundary condition: customizable function f(θ)
  - Animation: fps, duration, dpi, bitrate via CLI flags

Usage:
    python heat2d_disc_fd.py [--save] [--spin] [--fps 60] [--seconds 15] ...

For a list of all options:
    python heat2d_disc_fd.py --help
"""

# ---------------------------
# Visualization parameters
# ---------------------------
surf_kwargs: dict[str, Any] = dict(
    cmap="coolwarm",
    vmin=-1.0,
    vmax=1.0,
    edgecolor="black",
    linewidth=0.2,
    antialiased=True,
    alpha=1.0,
)

# ---------------------------
# Boundary condition f(θ)
# ---------------------------
f_s = bc_scale(make_bc(sin_k, L=1.0, k=4), 2)
f_theta = make_bc_for_disk(f_s, L=1.0)


# ================================================================
# Analytic Dirichlet solution on disk
# ================================================================
def analytic_dirichlet_disk(X, Y, R, f_theta, n_terms=200, quad_pts=512):
    """
    Solve Laplace's equation on the disk {x^2 + y^2 <= R^2}
    with Dirichlet boundary u(R,θ) = f_theta(θ).

    u(r,θ) = a0 + Σ r^n ( a_n cos(nθ) + b_n sin(nθ) ).
    """

    r = np.sqrt(X**2 + Y**2)
    theta = np.mod(np.arctan2(Y, X), 2 * np.pi)

    # Gauss–Legendre quadrature on [0,2π]
    xi, wi = np.polynomial.legendre.leggauss(int(quad_pts))
    th_nodes = 0.5 * (xi + 1.0) * 2 * np.pi
    w = 0.5 * (2 * np.pi) * wi

    f_vals = f_theta(th_nodes)

    # Fourier coefficients
    a0 = (1.0 / (2.0 * np.pi)) * np.sum(f_vals * w)

    n = np.arange(1, n_terms + 1, dtype=float)
    cos_n = np.cos(n[:, None] * th_nodes[None, :])
    sin_n = np.sin(n[:, None] * th_nodes[None, :])

    a_n = (1.0 / np.pi) * np.sum(f_vals[None, :] * cos_n * w[None, :], axis=1)
    b_n = (1.0 / np.pi) * np.sum(f_vals[None, :] * sin_n * w[None, :], axis=1)

    # Evaluate series
    u = np.full_like(X, np.nan, dtype=float)
    inside = r <= R

    rr = r[inside] / R
    tt = theta[inside]

    u_val = np.full(rr.shape, a0)
    for k in range(1, n_terms + 1):
        u_val += rr**k * (a_n[k - 1] * np.cos(k * tt) + b_n[k - 1] * np.sin(k * tt))

    u[inside] = u_val
    return u


# ================================================================
# Dirichlet boundary enforcement on disk
# ================================================================
def apply_disk_bc(u, r, R, theta, f_theta):
    boundary = np.abs(r - R) < 1e-2 * R
    u[boundary] = f_theta(theta[boundary])


# ================================================================
# Main program (fully equivalent structure to rectangle version)
# ================================================================
def main():

    # ------------------ Arguments ------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--out", type=str, default="heat_3d_disk.mp4")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--seconds", type=int, default=15)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--bitrate", type=int, default=6000)
    parser.add_argument("--spin", action="store_true")
    parser.add_argument("--deg_per_sec", type=float, default=25.0)
    parser.add_argument(
        "--steps_per_frame", type=int, default=20, help="Steps Per Frame"
    )
    args = parser.parse_args()

    # ------------------ Domain ------------------
    R = 1.0
    Nx = Ny = 201
    x = np.linspace(-R, R, Nx)
    y = np.linspace(-R, R, Ny)
    X, Y = np.meshgrid(x, y, indexing="xy")
    r = np.sqrt(X**2 + Y**2)
    theta = np.mod(np.arctan2(Y, X), 2 * np.pi)
    inside = r <= R

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # ------------------ Stability ------------------
    alpha = 1.0
    dt_max = 1.0 / (2 * alpha * (1 / dx**2 + 1 / dy**2))
    dt = 0.45 * dt_max
    print(f"dx={dx:.4g}, dy={dy:.4g}, dt_max={dt_max:.4g}, dt={dt:.4g}")

    # ------------------ Initial condition ------------------
    u = np.zeros((Ny, Nx), dtype=float)
    u_new = np.zeros_like(u)

    apply_disk_bc(u, r, R, theta, f_theta)

    # ------------------ Analytic reference ------------------
    u_star = analytic_dirichlet_disk(X, Y, R, f_theta, n_terms=150)

    umin = np.nanmin(u_star)
    umax = np.nanmax(u_star)
    surf_kwargs["vmin"] = umin
    surf_kwargs["vmax"] = umax

    # ------------------ Heat step ------------------
    def step_once(u_curr, u_next):
        u_next[:] = u_curr

        lap = (
            u_curr[1:-1, 2:] - 2 * u_curr[1:-1, 1:-1] + u_curr[1:-1, :-2]
        ) / dx**2 + (
            u_curr[2:, 1:-1] - 2 * u_curr[1:-1, 1:-1] + u_curr[:-2, 1:-1]
        ) / dy**2

        u_next[1:-1, 1:-1] = u_curr[1:-1, 1:-1] + dt * alpha * lap
        apply_disk_bc(u_next, r, R, theta, f_theta)

    # =========================================================
    # Visualization (structurally identical to rectangle script)
    # =========================================================
    fig = plt.figure(figsize=(12, 5))
    fig.subplots_adjust(top=0.85)  # give more space for titles
    ax_num = fig.add_subplot(121, projection="3d")
    ax_ref = fig.add_subplot(122, projection="3d")

    height_scale = 1.0

    # analytic wireframe
    ax_ref.plot_wireframe(
        X,
        Y,
        height_scale * u_star,
        rstride=6,
        cstride=6,
        linewidth=0.8,
        color="black",
        alpha=0.9,
    )

    # boundary curve on rim
    th_samp = np.linspace(0, 2 * np.pi, 400)
    bx = R * np.cos(th_samp)
    by = R * np.sin(th_samp)
    bz = height_scale * f_theta(th_samp)
    ax_ref.plot(bx, by, bz, color="red", linewidth=2.0)

    # numeric initial surface
    u_plot0 = u.copy()
    u_plot0[~inside] = np.nan
    surf = ax_num.plot_surface(X, Y, height_scale * u_plot0, **surf_kwargs)

    # ----- Draw boundary curve on numerical plot (Dirichlet rim) -----
    theta_samples = np.linspace(0, 2 * np.pi, 400)
    bx = R * np.cos(theta_samples)
    by = R * np.sin(theta_samples)
    bz = height_scale * f_theta(theta_samples)

    # Plot boundary on numerical solution
    boundary_line_num = ax_num.plot(
        bx, by, bz, color="red", linewidth=2.0, label="Boundary f(θ)"
    )

    ax_num.set_zlim(-1, 1)
    ax_ref.set_zlim(-1, 1)

    elev0, azim0 = 25, -60
    ax_num.view_init(elev=elev0, azim=azim0)
    ax_ref.view_init(elev=elev0, azim=azim0)

    # Colorbar
    fig.colorbar(surf, ax=[ax_num, ax_ref], shrink=0.7, pad=0.08, label="u (temp)")

    time_text = ax_num.text2D(0.02, 0.98, "", transform=ax_num.transAxes)

    # analytic title
    ax_ref.set_title(
        r"$u^*(r,\theta)\approx a_0"
        r"+\sum_{n=1}^{200}\left(\frac{r}{R}\right)^{n}"
        r"\!\left(a_n\cos(n\theta)+b_n\sin(n\theta)\right)$"
        "\n"
        r"$a_0=\frac{1}{2\pi}\int_0^{2\pi}f(\phi)d\phi,\quad"
        r"a_n=\frac{1}{\pi}\int_0^{2\pi}f(\phi)\cos(n\phi)d\phi,\quad"
        r"b_n=\frac{1}{\pi}\int_0^{2\pi}f(\phi)\sin(n\phi)d\phi$",
        pad=12,
    )

    fps = args.fps
    frames = fps * args.seconds
    steps_per_frame = args.steps_per_frame

    t = 0.0

    # ------------------ Animation update ------------------
    def update(frame):
        nonlocal u, u_new, t, surf

        for _ in range(steps_per_frame):
            step_once(u, u_new)
            u, u_new = u_new, u
            t += dt

        u_plot = u.copy()
        u_plot[~inside] = np.nan

        surf.remove()
        surf = ax_num.plot_surface(X, Y, height_scale * u_plot, **surf_kwargs)

        err_inf = np.nanmax(np.abs(u_star[inside] - u[inside]))
        time_text.set_text(rf"$t={t:.2f}\quad\|u-u^*\|_\infty={err_inf:.2e}$")

        if args.spin:
            az = azim0 + args.deg_per_sec * (frame / fps)
            ax_num.view_init(elev=elev0, azim=az)
            ax_ref.view_init(elev=elev0, azim=az)

        return surf, time_text

    ani = FuncAnimation(fig, update, frames=frames, interval=1000 / fps)

    """
        python heat2d_disc_fd.py \
        --save \
        --out disk_solution.mp4 \
        --fps 60 \
        --seconds 15 \
        --dpi 300 \
        --bitrate 8000 \
        --spin \
        --deg_per_sec 5 \
        --steps_per_frame 20 
    """
    if args.save:
        writer = animation.FFMpegWriter(fps=fps, bitrate=args.bitrate)
        ani.save(args.out, writer=writer, dpi=args.dpi)
        print("Saved to:", os.path.abspath(args.out))
    else:
        plt.show()


if __name__ == "__main__":
    main()
