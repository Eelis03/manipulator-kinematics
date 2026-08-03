"""Regression against a recorded reference.

``tests/data/reference.json`` pins the behaviour of the library on a fixed
trajectory, a fixed set of analytic solutions, and a fixed inverse kinematics
campaign. The file was produced by this module and can be regenerated with

    uv run python tests/test_regression.py

which overwrites it from the current implementation. Regenerating is a deliberate
act: it should follow a decision that the behaviour ought to change, and the
difference in the file is the record of that decision.

Tolerances differ by quantity. Forward kinematics is pure arithmetic on the DH
table and is pinned tightly. Singular values come from LAPACK and are pinned
relatively. Iteration counts are allowed to move by one because a change of one
unit in the last place at the tolerance boundary can cost or save a step.

Inverse kinematics solutions are pinned differently depending on whether the
solver converged, because only one of the two cases has a reproducible answer.

A converged solution is pinned by the tolerance criterion itself: the solver
stopped because the pose error fell below 1e-6, so the configuration is
determined by the target to that accuracy on any platform. Those rows are
compared directly.

A non-converged solution is not a contract, and nothing numerical about it is
pinned. When the Jacobian transpose method exhausts its iteration budget, the
returned configuration is an arbitrary point part way down a slow descent. On a
redundant or near-singular chain it drifts through the null space while the pose
error barely moves, and two hundred iterations is long enough for a difference in
the order a BLAS kernel reduces a dot product to grow into a visible difference
in the answer.

This was measured, not assumed. Pinning the joint angles failed on both CI
runners. Pinning the final residual instead, at a relative tolerance of 1e-3,
also failed, and it failed on the same operating system the reference was
recorded on, which rules out the platform and identifies the real cause: the
iterate is chaotic, so no numerical function of it is reproducible on another
machine.

What is reproducible is which targets the solver converges on and how many
iterations it takes. Those are pinned. For a non-converged trial the test asserts
that the result is finite, that the residual is still above the tolerance, and
that the trajectory did not end above the residual it started from, so a solver
cannot report failure on a trial it actually solved, or return a NaN and have it
counted as a failure like any other.

That last assertion has a history. It was written, it failed, and it was removed,
because while the joint limits were enforced by clipping alone a clipped
minimum-norm step was not a descent direction and the pseudoinverse ended above
its starting residual on nine of two hundred PUMA 560 targets. Asserting progress
would have encoded a false claim about the method, so the limitation was recorded
in the design notes instead. The step is now solved subject to the limit box
rather than clipped afterwards, the claim became true, and the assertion is back.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from manipulator_kinematics.algorithm import (
    Tolerance,
    analytic_ik,
    conditioning,
    forward_kinematics,
    geometric_jacobian,
    numerical_solvers,
)
from manipulator_kinematics.model import chain_reach, puma560, ur5
from manipulator_kinematics.pipeline import perturbed_seeds, run_campaign, sample_targets

DATA_PATH = Path(__file__).parent / "data" / "reference.json"

_TRAJECTORY_POINTS = 25
_CAMPAIGN_TARGETS = 25
_CAMPAIGN_ITERATIONS = 200
_CAMPAIGN_SEED = 20260731
_TOOL_OFFSET = 0.05625


def _trajectory(points: int) -> np.ndarray:
    """Return a smooth, deterministic joint-space path inside the PUMA 560 limits."""
    t = np.linspace(0.0, 1.0, points)
    amplitude = np.array([1.2, 0.7, 0.9, 1.4, 0.8, 1.6], dtype=np.float64)
    centre = np.array([0.0, -0.9, 0.9, 0.0, 0.0, 0.0], dtype=np.float64)
    frequency = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
    phase = np.array([0.0, 0.6, 1.2, 1.8, 2.4, 3.0], dtype=np.float64)
    return centre + amplitude * np.sin(2.0 * np.pi * frequency * t[:, None] + phase)


def _analytic_targets() -> np.ndarray:
    """Return the configurations whose analytic solution sets are pinned."""
    return np.array(
        [
            [0.4, -1.1, 0.7, -0.5, 0.9, 0.3],
            [-0.8, -0.4, 1.5, 1.1, -0.7, -1.2],
            [1.3, -1.7, 0.2, 0.0, 0.5, 2.0],
        ],
        dtype=np.float64,
    )


def _sorted_branches(solutions: tuple[Any, ...]) -> list[list[float]]:
    """Return branch configurations in a canonical, comparison-stable order."""
    rows = [[float(value) for value in solution.q] for solution in solutions]
    return sorted(rows)


def _build_reference() -> dict[str, Any]:
    """Recompute every pinned quantity from the current implementation."""
    robot = puma560()
    length = chain_reach(robot)

    forward: list[dict[str, Any]] = []
    for q in _trajectory(_TRAJECTORY_POINTS):
        pose = forward_kinematics(robot, q)
        metrics = conditioning(geometric_jacobian(robot, q), characteristic_length=length)
        forward.append(
            {
                "q": [float(value) for value in q],
                "pose": [float(value) for value in pose.reshape(-1)],
                "manipulability": metrics.manipulability,
                "condition_number": metrics.condition_number,
                "smallest_singular_value": metrics.smallest_singular_value,
            }
        )

    wrist_robot = puma560(tool_offset=_TOOL_OFFSET)
    analytic: list[dict[str, Any]] = []
    for q in _analytic_targets():
        target = forward_kinematics(wrist_robot, q)
        analytic.append(
            {
                "q": [float(value) for value in q],
                "branches": _sorted_branches(analytic_ik(wrist_robot, target)),
            }
        )

    campaigns: list[dict[str, Any]] = []
    for factory in (puma560, ur5):
        chain = factory()
        rng = np.random.default_rng(_CAMPAIGN_SEED)
        targets = sample_targets(chain, _CAMPAIGN_TARGETS, rng, margin=0.15)
        seeds = perturbed_seeds(chain, targets, rng, spread=0.4)
        tolerance = Tolerance(position=1e-6, orientation=1e-6)
        trace = run_campaign(
            chain,
            list(numerical_solvers(max_iterations=_CAMPAIGN_ITERATIONS, tolerance=tolerance)),
            targets,
            seeds,
            tolerance=tolerance,
            max_iterations=_CAMPAIGN_ITERATIONS,
        )
        campaigns.append(
            {
                "robot": chain.name,
                "solvers": [
                    {
                        "name": name,
                        "solved": sum(
                            trial.result.converged for trial in trace.for_solver(name)
                        ),
                        "iterations": [
                            trial.result.iterations for trial in trace.for_solver(name)
                        ],
                        "converged": [
                            bool(trial.result.converged) for trial in trace.for_solver(name)
                        ],
                        "solutions": [
                            [float(value) for value in trial.result.q]
                            for trial in trace.for_solver(name)
                        ],
                    }
                    for name in trace.solvers
                ],
            }
        )

    return {
        "description": (
            "Reference values for manipulator-kinematics. Regenerate with "
            "'uv run python tests/test_regression.py' only after deciding that the "
            "behaviour should change."
        ),
        "trajectory_points": _TRAJECTORY_POINTS,
        "campaign_targets": _CAMPAIGN_TARGETS,
        "campaign_iterations": _CAMPAIGN_ITERATIONS,
        "campaign_seed": _CAMPAIGN_SEED,
        "tool_offset": _TOOL_OFFSET,
        "forward": forward,
        "analytic": analytic,
        "campaigns": campaigns,
    }


@pytest.fixture(scope="module")
def reference() -> dict[str, Any]:
    """The recorded reference, loaded once for the module."""
    if not DATA_PATH.exists():  # pragma: no cover - guards a missing data file
        pytest.fail(f"reference data is missing: {DATA_PATH}")
    loaded: dict[str, Any] = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    return loaded


def test_reference_file_matches_the_current_settings(reference: dict[str, Any]) -> None:
    """The recorded file was produced with the settings this module still uses."""
    assert reference["trajectory_points"] == _TRAJECTORY_POINTS
    assert reference["campaign_targets"] == _CAMPAIGN_TARGETS
    assert reference["campaign_iterations"] == _CAMPAIGN_ITERATIONS
    assert reference["campaign_seed"] == _CAMPAIGN_SEED
    assert reference["tool_offset"] == _TOOL_OFFSET


def test_forward_trajectory_matches_the_reference(reference: dict[str, Any]) -> None:
    """Every pose on the recorded trajectory is reproduced to 1e-12."""
    robot = puma560()
    for index, record in enumerate(reference["forward"]):
        q = np.array(record["q"], dtype=np.float64)
        pose = forward_kinematics(robot, q)
        expected = np.array(record["pose"], dtype=np.float64).reshape(4, 4)
        assert np.allclose(pose, expected, atol=1e-12), f"waypoint {index}"


def test_conditioning_trajectory_matches_the_reference(reference: dict[str, Any]) -> None:
    """Every conditioning metric on the recorded trajectory is reproduced to 1e-9 relative."""
    robot = puma560()
    length = chain_reach(robot)
    for index, record in enumerate(reference["forward"]):
        q = np.array(record["q"], dtype=np.float64)
        metrics = conditioning(geometric_jacobian(robot, q), characteristic_length=length)
        for field in ("manipulability", "condition_number", "smallest_singular_value"):
            assert getattr(metrics, field) == pytest.approx(
                record[field], rel=1e-9
            ), f"waypoint {index}, {field}"


def test_analytic_branches_match_the_reference(reference: dict[str, Any]) -> None:
    """The eight closed-form branches of each pinned pose are reproduced to 1e-9."""
    robot = puma560(tool_offset=_TOOL_OFFSET)
    for index, record in enumerate(reference["analytic"]):
        q = np.array(record["q"], dtype=np.float64)
        branches = _sorted_branches(analytic_ik(robot, forward_kinematics(robot, q)))
        expected = record["branches"]
        assert len(branches) == len(expected), f"pose {index}"
        assert np.allclose(
            np.array(branches), np.array(expected), atol=1e-9
        ), f"pose {index}"


def test_campaign_matches_the_reference(reference: dict[str, Any]) -> None:
    """Solved counts, iteration counts and solutions are reproduced within tolerance."""
    factories = {"puma560": puma560, "ur5": ur5}
    for campaign in reference["campaigns"]:
        chain = factories[campaign["robot"]]()
        rng = np.random.default_rng(_CAMPAIGN_SEED)
        targets = sample_targets(chain, _CAMPAIGN_TARGETS, rng, margin=0.15)
        seeds = perturbed_seeds(chain, targets, rng, spread=0.4)
        tolerance = Tolerance(position=1e-6, orientation=1e-6)
        trace = run_campaign(
            chain,
            list(numerical_solvers(max_iterations=_CAMPAIGN_ITERATIONS, tolerance=tolerance)),
            targets,
            seeds,
            tolerance=tolerance,
            max_iterations=_CAMPAIGN_ITERATIONS,
        )

        for record in campaign["solvers"]:
            trials = trace.for_solver(record["name"])
            label = f"{campaign['robot']}/{record['name']}"
            assert sum(trial.result.converged for trial in trials) == record["solved"], label

            iterations = np.array([trial.result.iterations for trial in trials])
            assert np.all(np.abs(iterations - np.array(record["iterations"])) <= 1), label

            converged = np.array([trial.result.converged for trial in trials], dtype=bool)
            assert converged.tolist() == record["converged"], f"{label}: convergence pattern"

            solutions = np.array([trial.result.q for trial in trials], dtype=np.float64)
            expected_solutions = np.array(record["solutions"], dtype=np.float64)
            assert np.allclose(
                solutions[converged], expected_solutions[converged], atol=1e-5
            ), f"{label}: converged solutions"

            for index, trial in enumerate(trials):
                if trial.result.converged:
                    continue
                end = trial.result.final_residual
                where = f"{label}: trial {index}"
                assert np.isfinite(end), f"{where} returned a non-finite residual"
                assert end > tolerance.position, f"{where} is flagged unconverged but met tolerance"
                history = trial.result.residuals
                assert history[-1] <= history[0], f"{where} ended above its starting residual"


if __name__ == "__main__":  # pragma: no cover - regeneration entry point
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATA_PATH.write_text(json.dumps(_build_reference(), indent=1), encoding="utf-8")
    print(f"wrote {DATA_PATH}")
