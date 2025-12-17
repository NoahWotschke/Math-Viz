"""Visualization settings for heat equation solver.

Contains all plotting styles, animation parameters, display options,
and configuration dataclasses.
"""

from dataclasses import dataclass
from typing import Tuple, Any

from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ============================================================================
# FIGURE AND WINDOW SETTINGS
# ============================================================================

window_size: Tuple[float, float] = (12, 5)  # Base window size (width, height)


# ============================================================================
# SURFACE PLOT STYLING
# ============================================================================

surf_kwargs: dict[str, Any] = dict(
    cmap="magma",  # Colormap: 'magma', 'coolwarm', 'viridis', etc.
    vmin=-1.0,  # Min value for color scaling (set dynamically if None)
    vmax=1.0,  # Max value for color scaling (set dynamically if None)
    edgecolor="black",  # Edge color for surface facets
    linewidth=0.2,  # Edge line width
    antialiased=True,  # Enable antialiasing
    alpha=1.0,  # Transparency (0-1)
    ccount=10,  # Column count (updated based on resolution)
    rcount=10,  # Row count (updated based on resolution)
)


# ============================================================================
# WIREFRAME PLOT STYLING
# ============================================================================

wire_kwargs: dict[str, Any] = dict(
    linewidth=0.8,  # Wireframe line width
    color="blue",  # Wireframe color
    alpha=0.9,  # Transparency
    ccount=10,  # Column count (updated based on resolution)
    rcount=10,  # Row count (updated based on resolution)
)


# ============================================================================
# BOUNDARY LINE STYLING
# ============================================================================

line_kwargs: dict[str, Any] = dict(
    color="red",  # Boundary line color
    linewidth=2.0,  # Line width
)


# ============================================================================
# 3D VIEW SETTINGS
# ============================================================================

# Default viewing angles
view_elev: int = 25  # Elevation angle (degrees)
view_azim: int = -60  # Azimuth angle (degrees)

# Height scaling for u values
height_scale: float = 1.0  # Multiplies solution values for better visualization


# ============================================================================
# ANIMATION SETTINGS
# ============================================================================

fps: int = 60  # Frames per second
seconds: int = 15  # Animation duration in seconds
steps_per_frame: int = 25  # Solver steps executed per animation frame

# Rotation animation
spin_enabled: bool = False  # Slowly rotate the 3D view
deg_per_sec: float = 10.0  # Rotation speed in degrees per second


# ============================================================================
# VIDEO EXPORT SETTINGS
# ============================================================================

save_video: bool = False  # Save animation to MP4
video_filename: str = "heat_3d_rect.mp4"  # Output filename
video_dpi: int = 200  # DPI for export
video_bitrate: int = 6000  # Bitrate in kbps


# ============================================================================
# DISPLAY OPTIONS
# ============================================================================

show_colorbar: bool = True  # Display colorbar
show_analytic_formula: bool = True  # Show analytic solution formula as title
skip_error_display: bool = False  # Skip error computation (faster)


def get_visualization_params() -> dict:
    """Get all visualization parameters as a dictionary.

    Returns
    -------
    dict
        Dictionary containing all visualization settings
    """
    return {
        "window_size": window_size,
        "view_elev": view_elev,
        "view_azim": view_azim,
        "height_scale": height_scale,
        "fps": fps,
        "seconds": seconds,
        "steps_per_frame": steps_per_frame,
        "spin_enabled": spin_enabled,
        "deg_per_sec": deg_per_sec,
        "save_video": save_video,
        "video_filename": video_filename,
        "video_dpi": video_dpi,
        "video_bitrate": video_bitrate,
        "show_colorbar": show_colorbar,
        "show_analytic_formula": show_analytic_formula,
        "skip_error_display": skip_error_display,
    }


# ============================================================================
# CONFIGURATION DATACLASSES
# ============================================================================


@dataclass
class AnimationConfig:
    """Bundle animation parameters for cleaner function signatures."""

    fps: int
    seconds: int
    dpi: int
    bitrate: int
    steps_per_frame: int
    skip_error: bool
    spin: bool
    deg_per_sec: float
    height_scale: float
    save: bool
    out: str


@dataclass
class ViewConfig:
    """3D view parameters for both axes."""

    elev: int = 25
    azim: int = -60


@dataclass
class VisualizationConfig:
    """Contains figure, axes, and plot objects."""

    fig: Figure
    ax_num: Any  # Axes3DSubplot
    ax_ref: Any  # Axes3DSubplot
    surf: Any  # Poly3DCollection
    wf_pos: Any  # Poly3DCollection
    wf_neg: Any  # Poly3DCollection
    left_line: Any  # Line3D
    right_line: Any  # Line3D
    bottom_line: Any  # Line3D
    top_line: Any  # Line3D
    left_line_ref: Any  # Line3D
    right_line_ref: Any  # Line3D
    bottom_line_ref: Any  # Line3D
    top_line_ref: Any  # Line3D
    time_text: Any  # Text
    boundary_lines: dict
