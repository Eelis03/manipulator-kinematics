"""Turns a trace into per-solver summary statistics and a printable table."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from manipulator_kinematics.model.joints import JointLimit, limit_span
from manipulator_kinematics.pipeline.trace import SingularityScan, Trace

__all__ = [
    "SolverSummary",
    "format_failure_table",
    "format_scan_table",
    "format_summary_table",
    "summarise",
]

_COLUMNS: tuple[tuple[str, int], ...] = (
    ("solver", 22),
    ("solved", 8),
    ("rate", 7),
    ("med iter", 9),
    ("med pos err", 12),
    ("med rot err", 12),
    ("worst resid", 12),
)


@dataclass(frozen=True, slots=True)
class SolverSummary:
    """Aggregate performance of one solver over one campaign.

    Iteration counts are taken over converged trials only, because the count of a
    failed trial is the iteration budget and says nothing about speed. Errors are
    taken over every trial, so a solver cannot look accurate by failing often.

    Attributes:
        solver: Name of the solver.
        trials: How many targets it was given.
        solved: How many of them met both tolerances.
        median_iterations: Median iteration count among converged trials, or
            ``nan`` when none converged.
        median_position_error: Median position error over every trial, in metres.
        median_orientation_error: Median orientation error over every trial, in
            radians.
        worst_residual: Largest final pose error norm over every trial.
        median_manipulability: Median Yoshikawa index at the returned
            configurations.
    """

    solver: str
    trials: int
    solved: int
    median_iterations: float
    median_position_error: float
    median_orientation_error: float
    worst_residual: float
    median_manipulability: float

    @property
    def success_rate(self) -> float:
        """Fraction of trials that met both tolerances, in ``[0, 1]``."""
        return self.solved / self.trials if self.trials else 0.0


def summarise(trace: Trace) -> tuple[SolverSummary, ...]:
    """Return one summary per solver, in the order the solvers were run.

    Args:
        trace: The campaign to summarise.

    Returns:
        One :class:`SolverSummary` per solver named in ``trace.solvers``.
    """
    summaries: list[SolverSummary] = []
    for name in trace.solvers:
        trials = trace.for_solver(name)
        if not trials:
            continue
        converged = [trial for trial in trials if trial.result.converged]
        iterations = [float(trial.result.iterations) for trial in converged]
        summaries.append(
            SolverSummary(
                solver=name,
                trials=len(trials),
                solved=len(converged),
                median_iterations=float(np.median(iterations)) if iterations else float("nan"),
                median_position_error=float(
                    np.median([trial.result.position_error for trial in trials])
                ),
                median_orientation_error=float(
                    np.median([trial.result.orientation_error for trial in trials])
                ),
                worst_residual=float(max(trial.result.final_residual for trial in trials)),
                median_manipulability=float(
                    np.median([trial.conditioning.manipulability for trial in trials])
                ),
            )
        )
    return tuple(summaries)


def _row(values: tuple[str, ...]) -> str:
    return "  ".join(value.ljust(width) for value, (_, width) in zip(values, _COLUMNS, strict=True))


def format_summary_table(summaries: tuple[SolverSummary, ...]) -> str:
    """Render summaries as a fixed-width text table.

    Args:
        summaries: The rows to render.

    Returns:
        A string with a header, a rule, and one line per solver.
    """
    header = _row(tuple(name for name, _ in _COLUMNS))
    rule = "-" * len(header)
    lines = [header, rule]
    for item in summaries:
        iterations = (
            "n/a" if np.isnan(item.median_iterations) else f"{item.median_iterations:.0f}"
        )
        lines.append(
            _row(
                (
                    item.solver,
                    f"{item.solved}/{item.trials}",
                    f"{100.0 * item.success_rate:.0f}%",
                    iterations,
                    f"{item.median_position_error:.3e}",
                    f"{item.median_orientation_error:.3e}",
                    f"{item.worst_residual:.3e}",
                )
            )
        )
    return "\n".join(lines)


def format_failure_table(
    trace: Trace,
    limits: tuple[JointLimit, ...],
    *,
    tolerance: float = 1e-9,
) -> str:
    """Render every trial that did not converge, with the evidence for why.

    A success count says how many targets were missed. This says what the misses
    have in common: whether the answer is pressed against a joint limit, which
    points at a constrained stationary point, or whether the Jacobian has lost
    rank there, which points at a singularity instead. The two call for different
    remedies and only one of them is a step rule.

    Args:
        trace: The campaign to inspect.
        limits: Travel range of each joint, used to count active bounds.
        tolerance: How close to a bound counts as sitting on it.

    Returns:
        A string with a header, a rule, and one line per failed trial, or a
        single line saying that every trial converged.
    """
    lower, upper = limit_span(limits)
    header = (
        f"{'solver':>22}  {'target':>7}  {'residual':>11}  {'on bound':>9}  "
        f"{'manip':>11}  {'cond':>11}  reason"
    )
    lines = [header, "-" * len(header)]
    for trial in trace.trials:
        if trial.result.converged:
            continue
        on_bound = int(
            np.count_nonzero(
                (trial.result.q <= lower + tolerance) | (trial.result.q >= upper - tolerance)
            )
        )
        lines.append(
            f"{trial.solver:>22}  {trial.target_index:7d}  "
            f"{trial.result.final_residual:11.4e}  {on_bound:9d}  "
            f"{trial.conditioning.manipulability:11.4e}  "
            f"{trial.conditioning.condition_number:11.4e}  {trial.result.message}"
        )
    if len(lines) == 2:
        return "every trial converged"
    return "\n".join(lines)


def format_scan_table(scan: SingularityScan, *, stride: int = 1) -> str:
    """Render a singularity scan as a fixed-width text table.

    Args:
        scan: The sweep to render.
        stride: Print every ``stride``-th point.

    Returns:
        A string with a header, a rule, and one line per printed point.
    """
    if stride < 1:
        raise ValueError("stride must be at least one")
    header = (
        f"{'q value':>10}  {'manipulability':>15}  {'cond number':>13}  "
        f"{'sigma_min':>11}  {'lambda':>9}  {'|dq| pinv':>11}  {'|dq| dls':>11}"
    )
    lines = [header, "-" * len(header)]
    for point in scan.points[::stride]:
        metrics = point.conditioning
        lines.append(
            f"{point.value:10.4f}  {metrics.manipulability:15.6e}  "
            f"{metrics.condition_number:13.4e}  {metrics.smallest_singular_value:11.4e}  "
            f"{point.damping:9.4f}  {point.pseudoinverse_step_norm:11.4e}  "
            f"{point.damped_step_norm:11.4e}"
        )
    return "\n".join(lines)
