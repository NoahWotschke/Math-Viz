"""Domain-agnostic 3D Visualization for PDE Solvers.

Provides a generic Visualizer class that handles animation, surface/wireframe rendering,
boundary condition visualization, and MP4 export for different domain types
(rectangle, disc, bar, etc.).
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation, FFMpegWriter
from tqdm import tqdm
import os


@dataclass
class VisualizerConfig:
    """Configuration for 3D PDE visualization."""

    fps: int = 60
    seconds: int = 15
    dpi: int = 200
    bitrate: int = 6000
    spin: bool = False
    deg_per_sec: float = 10.0
    height_scale: float = 1.0
    show_colorbar: bool = True
    show_function: bool = True
    elev: float = 25
    azim: float = -60
    skip_error: bool = False
    save: bool = False
    out: str = "pde_solution.mp4"


class Visualizer3D:
    """
    Generic 3D visualizer for PDE solvers on different domains.

    Handles:
    - Surface and wireframe rendering
    - Boundary condition visualization as 3D curves
    - Phase toggling (for alternating BC problems)
    - Rotation, colorbar, titles
    - MP4 export via FFmpeg
    """

    def __init__(
        self,
        config: VisualizerConfig,
        domain_name: str = "Unknown Domain",
        pde_name: str = "PDE",
    ):
        """
        Initialize visualizer.

        Args:
            config: VisualizerConfig with visualization parameters
            domain_name: Display name for the domain (e.g., "Rectangle", "Disk")
            pde_name: Display name for the PDE (e.g., "Heat", "Wave")
        """
        self.config = config
        self.domain_name = domain_name
        self.pde_name = pde_name

        # Will be initialized in setup()
        self.fig: Optional[Figure] = None
        self.ax_num: Optional[Any] = None  # 3D Axes
        self.ax_ref: Optional[Any] = None  # 3D Axes

        # Drawable objects
        self.surf: Optional[Any] = None
        self.wf_pos: Optional[Any] = None
        self.wf_neg: Optional[Any] = None
        self.boundary_lines_num: list = []
        self.boundary_lines_ref: list = []
        self.time_text: Optional[Any] = None

        # Animation state
        self.frame_count = 0
        self.animation: Optional[FuncAnimation] = None
        self.animate_func: Optional[Callable] = None

    def setup(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        u_init: np.ndarray,
        u_star_pos: Optional[np.ndarray] = None,
        u_star_neg: Optional[np.ndarray] = None,
        analytic_title: Optional[str] = None,
        surf_kwargs: Optional[dict] = None,
        wire_kwargs: Optional[dict] = None,
        line_kwargs: Optional[dict] = None,
        boundary_lines_data: Optional[dict] = None,
    ):
        """
        Set up the figure, axes, and initial plots.

        Args:
            X, Y: 2D coordinate grids
            u_init: Initial solution
            u_star_pos: Analytical reference solution (positive phase)
            u_star_neg: Analytical reference solution (negative phase)
            analytic_title: LaTeX title for analytic solution panel
            surf_kwargs: Keyword args for plot_surface
            wire_kwargs: Keyword args for plot_wireframe
            line_kwargs: Keyword args for boundary curves
            boundary_lines_data: Dict with 'num' and 'ref' lists of boundary line data
                Each entry: (x_coords, y_coords, z_coords, name)
        """
        if surf_kwargs is None:
            surf_kwargs = {}
        if wire_kwargs is None:
            wire_kwargs = {}
        if line_kwargs is None:
            line_kwargs = {}

        self.surf_kwargs = surf_kwargs
        self.wire_kwargs = wire_kwargs
        self.line_kwargs = line_kwargs

        # Create figure and axes
        aspect_xy = 1.0
        if X.shape != Y.shape:
            ratio = X.shape[1] / X.shape[0]
        else:
            ratio = (X.max() - X.min()) / (Y.max() - Y.min())
        aspect_xy = 1.15 ** np.sqrt(max(ratio, 1.0 / ratio))

        base_w, base_h = 12, 5
        self.fig = plt.figure(figsize=(base_w * aspect_xy, base_h * aspect_xy))
        self.fig.subplots_adjust(top=0.75)

        self.ax_num = self.fig.add_subplot(1, 2, 1, projection="3d")
        self.ax_ref = self.fig.add_subplot(1, 2, 2, projection="3d")

        # Set vmin/vmax based on analytic solution if available
        if u_star_pos is not None:
            absmax = float(np.nanmax(np.abs(u_star_pos)))
            self.surf_kwargs["vmin"] = -absmax * self.config.height_scale
            self.surf_kwargs["vmax"] = absmax * self.config.height_scale
        else:
            absmax = float(np.nanmax(np.abs(u_init)))
            self.surf_kwargs["vmin"] = -absmax * self.config.height_scale
            self.surf_kwargs["vmax"] = absmax * self.config.height_scale

        # Wireframes for analytical solution (positive and negative phases)
        if u_star_pos is not None:
            self.wf_pos = self.ax_ref.plot_wireframe(
                X, Y, self.config.height_scale * u_star_pos, **self.wire_kwargs
            )
            if u_star_neg is not None:
                self.wf_neg = self.ax_ref.plot_wireframe(
                    X, Y, self.config.height_scale * u_star_neg, **self.wire_kwargs
                )
                self.wf_neg.set_visible(False)  # type: ignore  # Start with positive phase

        # Initial numerical surface
        self.surf = self.ax_num.plot_surface(
            X, Y, self.config.height_scale * u_init, **self.surf_kwargs
        )

        # Boundary lines
        self._setup_boundary_lines(boundary_lines_data)

        # Axes labels and limits
        self._setup_axes(X, Y)

        # Colorbar
        if self.config.show_colorbar and self.surf is not None and self.fig is not None:
            self.fig.colorbar(
                self.surf,
                ax=[self.ax_num, self.ax_ref],
                shrink=0.7,
                pad=0.08,
                label="u (temp)",
            )

        # Analytic title
        if self.config.show_function and analytic_title and self.ax_ref is not None:
            self.ax_ref.set_title(analytic_title, pad=12)

        # Time text
        self.time_text = self.ax_num.text2D(
            0.02, 0.98, "", transform=self.ax_num.transAxes
        )

    def _setup_boundary_lines(self, boundary_lines_data: Optional[dict]) -> None:
        """Set up 3D curves representing boundary conditions."""
        if boundary_lines_data is None:
            return

        self.boundary_lines_num = []
        self.boundary_lines_ref = []

        for label, lines_list in boundary_lines_data.items():
            if label == "num":
                target_list = self.boundary_lines_num
                ax = self.ax_num
            elif label == "ref":
                target_list = self.boundary_lines_ref
                ax = self.ax_ref
            else:
                continue

            if ax is None:
                continue

            for x_coords, y_coords, z_coords, name in lines_list:
                (line,) = ax.plot(
                    x_coords,
                    y_coords,
                    self.config.height_scale * z_coords,
                    **self.line_kwargs,
                )
                target_list.append(line)

    def _setup_axes(self, X: np.ndarray, Y: np.ndarray) -> None:
        """Configure axis labels, limits, view angle."""
        Lx = X.max() - X.min()
        Ly = Y.max() - Y.min()

        for ax in [self.ax_num, self.ax_ref]:
            if ax is not None:
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_zlabel("u")  # type: ignore
                ax.set_xlim(X.min(), X.max())
                ax.set_ylim(Y.min(), Y.max())
                ax.set_zlim(-1, 1)  # type: ignore
                ax.view_init(elev=self.config.elev, azim=self.config.azim)  # type: ignore
                ax.set_box_aspect((Lx / Ly, 1.0, 1.0))  # type: ignore

    def create_animation(
        self,
        update_func: Callable,
        u_curr_getter: Callable,
        X: np.ndarray,
        Y: np.ndarray,
    ) -> FuncAnimation:
        """
        Create animation with the given update function.

        Args:
            update_func: Callable(frame) that performs time stepping and returns state dict
            u_curr_getter: Callable() that returns current solution array
            X, Y: Coordinate grids for surface updates

        Returns:
            FuncAnimation object
        """
        fps = self.config.fps
        frames = fps * self.config.seconds

        def animate(frame: int) -> tuple[Any, Any]:
            # Call user's update (handles time stepping)
            state = update_func(frame)

            # Get current solution
            u_current = u_curr_getter()

            # Update surface
            if self.surf is not None:
                self.surf.remove()
            if self.ax_num is not None:
                self.surf = self.ax_num.plot_surface(  # type: ignore
                    X, Y, self.config.height_scale * u_current, **self.surf_kwargs
                )

            # Update phase toggling if in state
            if "phase" in state:
                if state["phase"] == 1:
                    if self.wf_pos is not None:
                        self.wf_pos.set_visible(False)  # type: ignore
                    if self.wf_neg is not None:
                        self.wf_neg.set_visible(True)  # type: ignore
                else:
                    if self.wf_pos is not None:
                        self.wf_pos.set_visible(True)  # type: ignore
                    if self.wf_neg is not None:
                        self.wf_neg.set_visible(False)  # type: ignore

            # Update boundary lines if provided
            if "boundary_lines_num_z" in state and self.boundary_lines_num:
                for i, z_vals in enumerate(state["boundary_lines_num_z"]):
                    self.boundary_lines_num[i].set_data(
                        self.boundary_lines_num[i].get_xdata(),
                        self.boundary_lines_num[i].get_ydata(),
                    )
                    self.boundary_lines_num[i].set_3d_properties(
                        self.config.height_scale * z_vals
                    )

            # Update time display
            if "time_text" in state and self.time_text is not None:
                self.time_text.set_text(state["time_text"])

            # Update view angle if spinning
            if self.config.spin and self.ax_num is not None and self.ax_ref is not None:
                azim = self.config.azim + self.config.deg_per_sec * (frame / fps)
                self.ax_num.view_init(elev=self.config.elev, azim=azim)  # type: ignore
                self.ax_ref.view_init(elev=self.config.elev, azim=azim)  # type: ignore

            return (self.surf, self.time_text)

        if self.fig is not None:
            self.animation = FuncAnimation(
                self.fig, animate, interval=1000 / fps, frames=frames, blit=False
            )
            self.animate_func = animate
            self.frame_count = frames
            return self.animation
        else:
            raise RuntimeError("Figure not initialized. Call setup() first.")

    def show(self):
        """Display the animation interactively."""
        if self.fig:
            plt.show()

    def save(self):
        """Save animation to MP4 file."""
        if not self.animation or not self.fig or not self.animate_func:
            raise ValueError("Animation not created or figure not set up")

        outpath = self.config.out
        fps = self.config.fps

        plt.ioff()
        writer = FFMpegWriter(
            fps=fps,
            bitrate=self.config.bitrate,
            codec="libx264",
        )

        with writer.saving(self.fig, outpath, self.config.dpi):
            for frame in tqdm(
                range(self.frame_count),
                total=self.frame_count,
                desc=f"Exporting {outpath}",
            ):
                self.animate_func(frame)
                self.fig.canvas.draw()
                writer.grab_frame()

        print(f"Saved animation to: {os.path.abspath(outpath)}")
        plt.close(self.fig)
