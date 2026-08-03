"""The analysis layer, exercised in process rather than through a script.

Summaries and figures are built from traces assembled by hand, so every expected
value is a hand computation rather than whatever the solvers happened to produce.
The figure tests read the objects matplotlib built, never the pixels, because the
pixels are not reproducible across platforms while the artist tree is.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from manipulator_kinematics.algorithm import IKResult, Tolerance, conditioning, geometric_jacobian
from manipulator_kinematics.analysis import (
    PALETTE,
    convergence_figure,
    format_failure_table,
    format_scan_table,
    format_summary_table,
    residual_tail_figure,
    save_figure,
    singularity_figure,
    success_figure,
    summarise,
)
from manipulator_kinematics.model import JointLimit, chain_reach, puma560
from manipulator_kinematics.pipeline import ScanPoint, SingularityScan, Target, Trace, Trial

TOLERANCE = Tolerance(position=1e-6, orientation=1e-6)


def _result(
    solver: str, *, converged: bool, iterations: int, position: float, orientation: float
) -> IKResult:
    return IKResult(
        solver=solver,
        q=np.zeros(6, dtype=np.float64),
        converged=converged,
        iterations=iterations,
        position_error=position,
        orientation_error=orientation,
        residuals=tuple(float(10.0**-k) for k in range(iterations + 1)),
        message="converged" if converged else "iteration limit reached",
    )


def _trial(result: IKResult, index: int, *, manipulability: float = 0.5) -> Trial:
    return Trial(
        target_index=index,
        seed=np.zeros(6, dtype=np.float64),
        result=result,
        conditioning=conditioning(
            np.diag([manipulability, 1.0, 1.0, 1.0, 1.0, 1.0]),
            characteristic_length=1.0,
        ),
    )


def _trace(solvers: tuple[str, ...], results: list[IKResult], *, robot: str = "toy") -> Trace:
    """Assemble a trace in which every solver saw the same targets, in order."""
    count = max(len(results) // max(len(solvers), 1), 1)
    return Trace(
        robot=robot,
        solvers=solvers,
        tolerance=TOLERANCE,
        max_iterations=50,
        characteristic_length=1.0,
        targets=tuple(
            Target(index=index, pose=np.eye(4, dtype=np.float64)) for index in range(count)
        ),
        trials=tuple(
            _trial(result, index % count) for index, result in enumerate(results)
        ),
    )


def _campaign_trace() -> Trace:
    """A four-target trace with one converging and one failing solver."""
    fast = [
        _result("fast", converged=True, iterations=2 + index, position=1e-9, orientation=1e-10)
        for index in range(4)
    ]
    slow = [
        _result("slow", converged=False, iterations=50, position=0.1 * (index + 1),
                orientation=0.01)
        for index in range(4)
    ]
    return _trace(("fast", "slow"), [*fast, *slow])


# --------------------------------------------------------------------------
# Summary statistics
# --------------------------------------------------------------------------


def test_summarise_returns_one_row_per_solver_in_order() -> None:
    """Rows follow ``trace.solvers``, not the order the trials happen to be in."""
    summaries = summarise(_campaign_trace())
    assert [item.solver for item in summaries] == ["fast", "slow"]
    assert [item.trials for item in summaries] == [4, 4]
    assert [item.solved for item in summaries] == [4, 0]


def test_summarise_takes_iterations_over_converged_trials_only() -> None:
    """A failed trial contributes its error but not its iteration count."""
    mixed = [
        _result("mixed", converged=True, iterations=3, position=1e-9, orientation=1e-9),
        _result("mixed", converged=True, iterations=7, position=1e-9, orientation=1e-9),
        _result("mixed", converged=False, iterations=50, position=1.0, orientation=1.0),
    ]
    (summary,) = summarise(_trace(("mixed",), mixed))
    assert summary.median_iterations == pytest.approx(5.0)
    assert summary.median_position_error == pytest.approx(1e-9)
    assert summary.success_rate == pytest.approx(2.0 / 3.0)


def test_summarise_reports_not_a_number_when_nothing_converged() -> None:
    """A solver that never converged has no iteration statistic to report."""
    (summary,) = summarise(_trace(("slow",), [
        _result("slow", converged=False, iterations=50, position=0.2, orientation=0.02)
    ]))
    assert np.isnan(summary.median_iterations)
    assert "n/a" in format_summary_table((summary,))


def test_summarise_skips_a_solver_with_no_trials() -> None:
    """A name in ``solvers`` with nothing recorded produces no row."""
    trace = _trace(("present", "absent"), [
        _result("present", converged=True, iterations=1, position=1e-9, orientation=1e-9)
    ])
    assert [item.solver for item in summarise(trace)] == ["present"]


def test_worst_residual_is_the_error_of_the_returned_configuration() -> None:
    """The worst residual is measured at the answer, not at the last iterate."""
    result = _result("slow", converged=False, iterations=4, position=0.3, orientation=0.4)
    assert result.residuals[-1] == pytest.approx(1e-4)
    assert result.final_residual == pytest.approx(0.5)
    (summary,) = summarise(_trace(("slow",), [result]))
    assert summary.worst_residual == pytest.approx(0.5)


def test_success_rate_of_an_empty_summary_is_zero() -> None:
    """A summary over no trials reports zero rather than dividing by zero."""
    (summary,) = summarise(_trace(("solo",), [
        _result("solo", converged=True, iterations=1, position=0.0, orientation=0.0)
    ]))
    empty = type(summary)(
        solver="none",
        trials=0,
        solved=0,
        median_iterations=float("nan"),
        median_position_error=0.0,
        median_orientation_error=0.0,
        worst_residual=0.0,
        median_manipulability=0.0,
    )
    assert empty.success_rate == 0.0


# --------------------------------------------------------------------------
# Text tables
# --------------------------------------------------------------------------


def test_summary_table_has_a_header_a_rule_and_one_row_per_solver() -> None:
    """The table is a header, an underline of the same width, and the rows."""
    table = format_summary_table(summarise(_campaign_trace())).splitlines()
    assert len(table) == 4
    assert table[0].split()[:3] == ["solver", "solved", "rate"]
    assert set(table[1]) == {"-"}
    assert len(table[1]) == len(table[0])
    assert table[2].startswith("fast")
    assert "4/4" in table[2]
    assert "0/4" in table[3]


def test_failure_table_lists_only_the_unsolved_trials() -> None:
    """One line per failure, none for the successes, and a rule under the header."""
    limits = tuple(JointLimit(-1.0, 1.0) for _ in range(6))
    lines = format_failure_table(_campaign_trace(), limits).splitlines()
    assert len(lines) == 6
    assert lines[0].split()[:3] == ["solver", "target", "residual"]
    assert set(lines[1]) == {"-"}
    assert all("slow" in line for line in lines[2:])
    assert [line.split()[1] for line in lines[2:]] == ["0", "1", "2", "3"]


def test_failure_table_counts_the_joints_pressed_against_a_bound() -> None:
    """The bound count is what separates a constrained stop from a singular one."""
    result = _result("stuck", converged=False, iterations=9, position=0.2, orientation=0.1)
    trial = Trial(
        target_index=7,
        seed=np.zeros(6, dtype=np.float64),
        result=IKResult(
            solver="stuck",
            q=np.array([-1.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            converged=False,
            iterations=result.iterations,
            position_error=result.position_error,
            orientation_error=result.orientation_error,
            residuals=result.residuals,
            message="step size collapsed",
        ),
        conditioning=conditioning(np.eye(6, dtype=np.float64), characteristic_length=1.0),
    )
    trace = Trace(
        robot="toy",
        solvers=("stuck",),
        tolerance=TOLERANCE,
        max_iterations=50,
        characteristic_length=1.0,
        targets=(Target(index=7, pose=np.eye(4, dtype=np.float64)),),
        trials=(trial,),
    )
    limits = tuple(JointLimit(-1.0, 1.0) for _ in range(6))
    row = format_failure_table(trace, limits).splitlines()[2]
    assert row.split()[3] == "2"
    assert row.endswith("step size collapsed")


def test_failure_table_says_so_when_nothing_failed() -> None:
    """A clean campaign gets a sentence rather than an empty table."""
    limits = tuple(JointLimit(-1.0, 1.0) for _ in range(6))
    clean = _trace(("fast",), [
        _result("fast", converged=True, iterations=2, position=1e-9, orientation=1e-9)
    ])
    assert format_failure_table(clean, limits) == "every trial converged"


def _scan(points: int) -> SingularityScan:
    robot = puma560()
    base = np.array([0.3, -0.9, 0.6, 0.4, 0.0, 0.2], dtype=np.float64)
    length = chain_reach(robot)
    entries = []
    for value in np.linspace(-0.2, 0.2, points):
        q = base.copy()
        q[4] = float(value)
        entries.append(
            ScanPoint(
                value=float(value),
                conditioning=conditioning(
                    geometric_jacobian(robot, q), characteristic_length=length
                ),
                transpose_step_norm=0.5,
                pseudoinverse_step_norm=1.0 / max(abs(float(value)), 1e-6),
                damped_step_norm=2.0,
                damping=0.01,
            )
        )
    return SingularityScan(
        robot=robot.name,
        joint_index=4,
        base_q=base,
        twist=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        characteristic_length=length,
        points=tuple(entries),
    )


def test_scan_table_prints_every_stride_th_point() -> None:
    """A stride of three over nine points prints three rows under the header."""
    lines = format_scan_table(_scan(9), stride=3).splitlines()
    assert len(lines) == 5
    assert "manipulability" in lines[0]
    assert set(lines[1]) == {"-"}


def test_scan_table_rejects_a_non_positive_stride() -> None:
    """A stride below one would print nothing or loop, so it is refused."""
    with pytest.raises(ValueError, match="stride"):
        format_scan_table(_scan(3), stride=0)


def test_scan_column_reaches_into_the_conditioning_record() -> None:
    """``column`` reads both scan point fields and conditioning fields."""
    scan = _scan(5)
    assert scan.column("value").shape == (5,)
    assert np.allclose(scan.column("damped_step_norm"), 2.0)
    manipulabilities = scan.column("manipulability")
    assert manipulabilities.shape == (5,)
    assert manipulabilities.min() < manipulabilities.max()
    assert np.all(scan.column("condition_number") > 1.0)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def test_convergence_figure_labels_one_median_curve_per_solver() -> None:
    """Every solver contributes one labelled median line over its faint trials."""
    trace = _campaign_trace()
    figure = convergence_figure(trace)
    (axes,) = figure.axes
    labelled = [line for line in axes.lines if line.get_label() in trace.solvers]
    assert [line.get_label() for line in labelled] == ["fast", "slow"]
    assert [line.get_color() for line in labelled] == list(PALETTE[:2])
    assert axes.get_yscale() == "log"
    assert axes.get_xlim()[1] == pytest.approx(float(trace.max_iterations))


def test_convergence_figure_skips_a_solver_with_no_trials() -> None:
    """A declared solver that recorded nothing contributes no curve."""
    trace = _trace(("present", "absent"), [
        _result("present", converged=True, iterations=3, position=1e-9, orientation=1e-9)
    ])
    (axes,) = convergence_figure(trace).axes
    assert [line.get_label() for line in axes.lines if line.get_label() == "absent"] == []


def test_figures_refuse_more_series_than_the_palette_defines() -> None:
    """A fifth series would need a colour the accessible palette does not define."""
    names = ("a", "b", "c", "d", "e")
    results = [
        _result(name, converged=True, iterations=2, position=1e-9, orientation=1e-9)
        for name in names
    ]
    with pytest.raises(ValueError, match="categorical colours"):
        convergence_figure(_trace(names, results))


def test_success_figure_draws_one_bar_per_solver_at_its_rate() -> None:
    """Bar widths are the success percentages and the ticks name the solvers."""
    summaries = summarise(_campaign_trace())
    figure = success_figure(summaries, robot="toy")
    (axes,) = figure.axes
    bars = [patch for patch in axes.patches if isinstance(patch, Rectangle)]
    assert sorted(bar.get_width() for bar in bars) == pytest.approx([0.0, 100.0])
    assert [label.get_text() for label in axes.get_yticklabels()] == ["fast", "slow"]


def test_singularity_figure_stacks_three_panels_on_one_axis() -> None:
    """Three panels share the horizontal axis and only the last one labels it."""
    figure = singularity_figure(_scan(41))
    top, middle, bottom = figure.axes
    assert top.get_yscale() == "log"
    assert middle.get_yscale() == "log"
    assert bottom.get_yscale() == "log"
    assert bottom.get_xlabel() == "joint 5 value (rad)"
    assert top.get_xlabel() == ""


def test_residual_tail_figure_sorts_and_counts_each_campaign() -> None:
    """Each curve is non-decreasing and its label carries the unsolved count."""
    trace = _campaign_trace()
    figure = residual_tail_figure([("first", trace), ("second", trace)], solver="slow")
    (axes,) = figure.axes
    curves = [line for line in axes.lines if "unsolved" in str(line.get_label())]
    assert len(curves) == 2
    assert curves[0].get_label() == "first, 4 of 4 unsolved"
    for curve in curves:
        values = np.asarray(curve.get_ydata(), dtype=np.float64)
        assert np.all(np.diff(values) >= 0.0)
        assert values.size == 4


def test_residual_tail_figure_needs_at_least_one_campaign() -> None:
    """Drawing nothing is a mistake, not an empty figure."""
    with pytest.raises(ValueError, match="at least one"):
        residual_tail_figure([], solver="slow")


def test_residual_tail_figure_rejects_a_solver_the_trace_never_ran() -> None:
    """Asking for a missing solver is an error rather than a silent blank."""
    with pytest.raises(ValueError, match="no trials for solver"):
        residual_tail_figure([("only", _campaign_trace())], solver="absent")


# --------------------------------------------------------------------------
# Saving
# --------------------------------------------------------------------------


def test_save_figure_writes_a_png_and_creates_its_directory(tmp_path: Path) -> None:
    """The parent directory is created and the file carries the PNG signature."""
    target = tmp_path / "nested" / "deeper" / "figure.png"
    written = save_figure(Figure(figsize=(2.0, 2.0)), target, dpi=50)
    assert written == target
    assert target.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_save_figure_resolution_follows_the_requested_dpi(tmp_path: Path) -> None:
    """Doubling the dpi doubles the pixel width recorded in the PNG header."""

    def width(path: Path) -> int:
        return int.from_bytes(path.read_bytes()[16:20], "big")

    low = save_figure(Figure(figsize=(3.0, 2.0)), tmp_path / "low.png", dpi=50)
    high = save_figure(Figure(figsize=(3.0, 2.0)), tmp_path / "high.png", dpi=100)
    assert width(high) == pytest.approx(2 * width(low), abs=2)
