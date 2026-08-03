"""Figures drawn from a trace.

Only the Agg canvas is used, never ``pyplot``, so nothing here depends on an
interactive backend or on global figure state. Every function returns a
:class:`matplotlib.figure.Figure` and leaves saving to the caller through
:func:`save_figure`.

The categorical colours are the first four slots of the Okabe-Ito palette
(M. Okabe and K. Ito, 'Color universal design', 2008,
https://jfly.uni-koeln.de/color/ ). They are assigned to series in a fixed order
so a series keeps its colour across figures, and every chart with more than one
series carries a legend, so identity is never signalled by colour alone.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Final

import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from manipulator_kinematics.analysis.metrics import SolverSummary
from manipulator_kinematics.pipeline.trace import SingularityScan, Trace

__all__ = [
    "PALETTE",
    "convergence_figure",
    "residual_tail_figure",
    "save_figure",
    "singularity_figure",
    "success_figure",
]

PALETTE: Final[tuple[str, ...]] = ("#0072b2", "#d55e00", "#009e73", "#7b4fa8")

_GRID: Final[str] = "#c9ccd1"
_INK: Final[str] = "#22262b"
_MUTED: Final[str] = "#5b6069"
_LINE_WIDTH: Final[float] = 1.8


def _colour(index: int) -> str:
    if index >= len(PALETTE):
        raise ValueError(
            f"only {len(PALETTE)} categorical colours are defined; "
            "group the extra series rather than generating a new hue"
        )
    return PALETTE[index]


def _style(axes: Axes, *, title: str, xlabel: str, ylabel: str) -> None:
    axes.set_title(title, color=_INK, fontsize=10, loc="left")
    axes.set_xlabel(xlabel, color=_MUTED, fontsize=9)
    axes.set_ylabel(ylabel, color=_MUTED, fontsize=9)
    axes.grid(True, color=_GRID, linewidth=0.6, alpha=0.7)
    axes.set_axisbelow(True)
    axes.tick_params(colors=_MUTED, labelsize=8)
    for edge in ("top", "right"):
        axes.spines[edge].set_visible(False)
    for edge in ("left", "bottom"):
        axes.spines[edge].set_color(_GRID)


def save_figure(figure: Figure, path: Path, *, dpi: int = 150) -> Path:
    """Write ``figure`` to ``path`` as a PNG, creating parent directories.

    Args:
        figure: The figure to write.
        path: Destination file.
        dpi: Output resolution.

    Returns:
        The path that was written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    FigureCanvasAgg(figure)
    figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    return path


def convergence_figure(trace: Trace) -> Figure:
    """Plot the pose error against iteration for every solver in ``trace``.

    Individual trials are drawn faintly, each ending where that trial stopped,
    and the per-iteration median across trials is drawn on top. A trial that has
    stopped is held at its final residual when the median is formed, so the
    median curve does not jump when fast trials drop out of the pool.

    The vertical axis is logarithmic because the residual spans many decades. The
    horizontal axis is symmetric-logarithmic, linear below one iteration and
    logarithmic above, because the methods that converge do so within about ten
    iterations while the Jacobian transpose runs for hundreds.

    Args:
        trace: The campaign to plot.

    Returns:
        A single-axes figure.
    """
    figure = Figure(figsize=(7.2, 4.4))
    axes = figure.add_subplot(111)

    for index, name in enumerate(trace.solvers):
        trials = trace.for_solver(name)
        if not trials:
            continue
        colour = _colour(index)
        longest = max(len(trial.result.residuals) for trial in trials)
        padded = np.full((len(trials), longest), np.nan, dtype=np.float64)
        for row, trial in enumerate(trials):
            series = np.asarray(trial.result.residuals, dtype=np.float64)
            padded[row, : series.size] = series
            padded[row, series.size :] = series[-1]
            axes.plot(
                np.arange(series.size),
                np.maximum(series, 1e-16),
                color=colour,
                alpha=0.12,
                linewidth=0.8,
            )
        median = np.nanmedian(padded, axis=0)
        axes.plot(
            np.arange(longest),
            np.maximum(median, 1e-16),
            color=colour,
            linewidth=_LINE_WIDTH,
            label=name,
        )

    axes.axhline(
        trace.tolerance.position,
        color=_MUTED,
        linewidth=1.0,
        linestyle="--",
    )
    axes.annotate(
        "position tolerance",
        xy=(0.99, trace.tolerance.position),
        xycoords=("axes fraction", "data"),
        ha="right",
        va="bottom",
        color=_MUTED,
        fontsize=8,
    )
    axes.set_yscale("log")
    axes.set_xscale("symlog", linthresh=1.0)
    axes.set_xlim(left=0.0, right=float(trace.max_iterations))
    _style(
        axes,
        title=f"Pose error against iteration, {trace.robot}, {len(trace.targets)} targets",
        xlabel="iteration (linear below one, logarithmic above)",
        ylabel="pose error norm (m and rad combined)",
    )
    axes.legend(frameon=False, fontsize=9, labelcolor=_INK, loc="lower left")
    figure.tight_layout()
    return figure


def success_figure(summaries: tuple[SolverSummary, ...], *, robot: str) -> Figure:
    """Plot the fraction of targets each solver reached, as horizontal bars.

    Args:
        summaries: One row per solver.
        robot: Name of the chain, used in the title.

    Returns:
        A single-axes figure.
    """
    figure = Figure(figsize=(7.2, 0.7 * max(len(summaries), 1) + 1.8))
    axes = figure.add_subplot(111)

    positions = np.arange(len(summaries))
    for index, item in enumerate(summaries):
        axes.barh(
            positions[index],
            100.0 * item.success_rate,
            height=0.55,
            color=_colour(index),
            edgecolor="white",
            linewidth=1.0,
        )
        axes.annotate(
            f"{item.solved}/{item.trials}",
            xy=(100.0 * item.success_rate, positions[index]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            color=_INK,
            fontsize=9,
        )

    axes.set_yticks(positions)
    axes.set_yticklabels([item.solver for item in summaries])
    axes.set_xlim(0.0, 112.0)
    axes.invert_yaxis()
    _style(
        axes,
        title=f"Targets reached within tolerance, {robot}",
        xlabel="percent of targets solved",
        ylabel="",
    )
    axes.grid(axis="y", visible=False)
    figure.tight_layout()
    return figure


def residual_tail_figure(labelled: Sequence[tuple[str, Trace]], *, solver: str) -> Figure:
    """Plot the sorted final residual of one solver across several campaigns.

    A success count says how many targets a solver reached. It cannot say how
    badly it missed the rest, and a median hides the tail by construction. Sorting
    the final residual of every target and drawing it on a logarithmic axis shows
    the whole distribution: where it crosses the tolerance, how many targets lie
    beyond it, and how far beyond they lie.

    Args:
        labelled: One ``(label, trace)`` pair per campaign being compared. Every
            trace must contain ``solver`` and cover the same targets.
        solver: Which solver of each trace to draw.

    Returns:
        A single-axes figure.

    Raises:
        ValueError: If ``labelled`` is empty or a trace has no such solver.
    """
    if not labelled:
        raise ValueError("at least one labelled trace is needed")

    figure = Figure(figsize=(7.2, 4.0))
    axes = figure.add_subplot(111)

    tolerance = labelled[0][1].tolerance.position
    for index, (label, trace) in enumerate(labelled):
        trials = trace.for_solver(solver)
        if not trials:
            raise ValueError(f"trace {trace.robot!r} holds no trials for solver {solver!r}")
        residuals = np.sort(
            np.maximum([trial.result.final_residual for trial in trials], 1e-16)
        )
        missed = sum(not trial.result.converged for trial in trials)
        axes.plot(
            np.arange(1, residuals.size + 1),
            residuals,
            color=_colour(index),
            linewidth=_LINE_WIDTH,
            label=f"{label}, {missed} of {residuals.size} unsolved",
        )

    axes.axhline(tolerance, color=_MUTED, linewidth=1.0, linestyle="--")
    axes.annotate(
        "position tolerance",
        xy=(0.01, tolerance),
        xycoords=("axes fraction", "data"),
        ha="left",
        va="bottom",
        color=_MUTED,
        fontsize=8,
    )
    axes.set_yscale("log")
    _style(
        axes,
        title=f"Residual of the returned configuration, {solver} on {labelled[0][1].robot}",
        xlabel="target, ordered by residual",
        ylabel="pose error norm at the answer (m and rad combined)",
    )
    axes.legend(frameon=False, fontsize=9, labelcolor=_INK, loc="upper left")
    figure.tight_layout()
    return figure


def singularity_figure(scan: SingularityScan) -> Figure:
    """Plot conditioning and requested step size across a joint sweep.

    Three stacked panels share the horizontal axis. Measures with different
    scales are given their own panel rather than a second vertical axis, so no
    panel carries two scales.

    Args:
        scan: The sweep to plot.

    Returns:
        A three-panel figure.
    """
    figure = Figure(figsize=(7.2, 8.0))
    top, middle, bottom = figure.subplots(3, 1, sharex=True)

    values = scan.column("value")
    floor = 1e-18

    manipulabilities = np.maximum(scan.column("manipulability"), floor)
    smallest = np.maximum(scan.column("smallest_singular_value"), floor)
    top.plot(
        values,
        manipulabilities,
        color=_colour(0),
        linewidth=_LINE_WIDTH,
        label="manipulability",
    )
    top.plot(
        values,
        smallest,
        color=_colour(1),
        linewidth=_LINE_WIDTH,
        label="smallest singular value",
    )
    top.set_yscale("log")
    # At an exact singularity both indices fall to the floating point floor, which
    # would stretch the axis over sixteen decades and hide the approach. The axis
    # is clipped and the clipping is stated, rather than the data being altered.
    ceiling = float(max(manipulabilities.max(), smallest.max()))
    top.set_ylim(bottom=1e-5 * ceiling, top=3.0 * ceiling)
    top.annotate(
        "both indices reach the floating point floor at the singularity",
        xy=(0.5, 0.06),
        xycoords="axes fraction",
        ha="center",
        color=_MUTED,
        fontsize=8,
    )
    _style(
        top,
        title=f"Conditioning of {scan.robot} sweeping joint {scan.joint_index + 1}",
        xlabel="",
        ylabel="index value",
    )
    top.legend(frameon=False, fontsize=9, labelcolor=_INK, loc="upper right")

    middle.plot(
        values,
        scan.column("condition_number"),
        color=_colour(2),
        linewidth=_LINE_WIDTH,
    )
    middle.set_yscale("log")
    _style(middle, title="Condition number", xlabel="", ylabel="s_max / s_min")

    for index, (field, label) in enumerate(
        (
            ("pseudoinverse_step_norm", "pseudoinverse"),
            ("damped_step_norm", "damped least squares"),
            ("transpose_step_norm", "Jacobian transpose"),
        )
    ):
        bottom.plot(
            values,
            np.maximum(scan.column(field), floor),
            color=_colour(index),
            linewidth=_LINE_WIDTH,
            label=label,
        )
    bottom.set_yscale("log")
    _style(
        bottom,
        title="Joint step requested for a unit task velocity",
        xlabel=f"joint {scan.joint_index + 1} value (rad)",
        ylabel="step norm (rad)",
    )
    bottom.legend(frameon=False, fontsize=9, labelcolor=_INK)

    figure.tight_layout()
    return figure
