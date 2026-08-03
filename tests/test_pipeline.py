"""The pipeline layer: target generation, campaigns, and the joint sweep.

These run in process rather than through an example script, so a failure points
at the function that produced it instead of at a subprocess exit code.
"""

from __future__ import annotations

import numpy as np
import pytest

from manipulator_kinematics.algorithm import (
    PseudoinverseIK,
    SolverSettings,
    Tolerance,
    forward_kinematics,
)
from manipulator_kinematics.model import puma560, ur5, within_limits
from manipulator_kinematics.pipeline import (
    Target,
    perturbed_seeds,
    run_campaign,
    sample_targets,
    scan_joint,
)

TOLERANCE = Tolerance(position=1e-6, orientation=1e-6)


# --------------------------------------------------------------------------
# Target and seed generation
# --------------------------------------------------------------------------


def test_sampled_targets_are_reachable_by_construction(rng: np.random.Generator) -> None:
    """Each target is the forward pose of the configuration recorded beside it."""
    robot = puma560()
    targets = sample_targets(robot, 12, rng, margin=0.15)
    assert len(targets) == 12
    assert [target.index for target in targets] == list(range(12))
    for target in targets:
        assert target.reference_q is not None
        assert within_limits(target.reference_q, robot.limits)
        assert np.allclose(forward_kinematics(robot, target.reference_q), target.pose, atol=1e-12)


def test_sampling_margin_keeps_configurations_off_the_hard_stops(
    rng: np.random.Generator,
) -> None:
    """A margin of a quarter span excludes the outer quarter at each end."""
    robot = puma560()
    limits = robot.limits
    for target in sample_targets(robot, 30, rng, margin=0.25):
        assert target.reference_q is not None
        for value, limit in zip(target.reference_q, limits, strict=True):
            assert limit.lower + 0.25 * limit.span <= value <= limit.upper - 0.25 * limit.span


def test_sampling_refuses_a_negative_count(rng: np.random.Generator) -> None:
    """A negative count is a caller mistake, not an empty campaign."""
    with pytest.raises(ValueError, match="must not be negative"):
        sample_targets(puma560(), -1, rng)


def test_seeds_stay_inside_the_limits_and_move_off_the_answer(
    rng: np.random.Generator,
) -> None:
    """A perturbed seed is feasible but is not the generating configuration."""
    robot = puma560()
    targets = sample_targets(robot, 10, rng, margin=0.15)
    seeds = perturbed_seeds(robot, targets, rng, spread=0.4)
    assert len(seeds) == len(targets)
    for target, seed in zip(targets, seeds, strict=True):
        assert target.reference_q is not None
        assert within_limits(seed, robot.limits)
        assert float(np.linalg.norm(seed - target.reference_q)) > 1e-6


def test_a_target_without_a_generating_configuration_is_seeded_at_the_midpoint(
    rng: np.random.Generator,
) -> None:
    """A pose that did not come from forward kinematics starts from the limit centre."""
    robot = puma560()
    midpoint = np.array([limit.midpoint for limit in robot.limits], dtype=np.float64)
    target = Target(index=0, pose=np.eye(4, dtype=np.float64), reference_q=None)
    (seed,) = perturbed_seeds(robot, (target,), rng, spread=0.0)
    assert np.allclose(seed, midpoint, atol=1e-12)


# --------------------------------------------------------------------------
# Campaigns
# --------------------------------------------------------------------------


def test_campaign_records_one_trial_per_solver_and_target(rng: np.random.Generator) -> None:
    """The trace holds the product of solvers and targets, grouped by solver."""
    robot = puma560()
    targets = sample_targets(robot, 5, rng, margin=0.2)
    seeds = perturbed_seeds(robot, targets, rng, spread=0.1)
    solvers = [
        PseudoinverseIK(settings=SolverSettings(max_iterations=40, tolerance=TOLERANCE)),
    ]
    trace = run_campaign(
        robot, solvers, targets, seeds, tolerance=TOLERANCE, max_iterations=40
    )
    assert trace.robot == "puma560"
    assert trace.solvers == ("pseudoinverse",)
    assert len(trace.trials) == 5
    assert len(trace.for_solver("pseudoinverse")) == 5
    assert trace.for_solver("absent") == ()
    assert [trial.target_index for trial in trace.trials] == [0, 1, 2, 3, 4]
    assert all(trial.solver == "pseudoinverse" for trial in trace.trials)
    assert all(trial.conditioning.manipulability > 0.0 for trial in trace.trials)


def test_campaign_refuses_a_seed_count_that_does_not_match(
    rng: np.random.Generator,
) -> None:
    """Pairing is the point of the design, so a mismatch is refused loudly."""
    robot = puma560()
    targets = sample_targets(robot, 3, rng, margin=0.2)
    seeds = perturbed_seeds(robot, targets[:2], rng, spread=0.1)
    with pytest.raises(ValueError, match="2 seeds for 3 targets"):
        run_campaign(
            robot,
            [PseudoinverseIK()],
            targets,
            seeds,
            tolerance=TOLERANCE,
            max_iterations=10,
        )


# --------------------------------------------------------------------------
# Singularity sweep
# --------------------------------------------------------------------------


def test_sweeping_the_wrist_finds_the_singularity_at_the_centre() -> None:
    """Manipulability bottoms out at ``q5 = 0`` and the condition number peaks there."""
    robot = puma560()
    base = np.array([0.3, -0.9, 0.6, 0.4, 0.0, 0.2], dtype=np.float64)
    values = np.linspace(-0.2, 0.2, 41)
    scan = scan_joint(
        robot,
        base,
        joint_index=4,
        values=values,
        twist=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        damping=0.05,
        epsilon=0.05,
    )

    assert scan.robot == "puma560"
    assert scan.joint_index == 4
    assert len(scan.points) == 41
    assert np.allclose(scan.column("value"), values, atol=1e-12)

    centre = int(np.argmin(scan.column("manipulability")))
    assert abs(float(scan.column("value")[centre])) < 1e-12
    assert int(np.argmax(scan.column("condition_number"))) == centre
    assert scan.column("smallest_singular_value")[centre] < 1e-12


def test_the_sweep_holds_every_other_joint_fixed() -> None:
    """Only the swept joint moves, and the base configuration is recorded as given."""
    robot = puma560()
    base = np.array([0.3, -0.9, 0.6, 0.4, 0.2, 0.2], dtype=np.float64)
    scan = scan_joint(robot, base, joint_index=1, values=np.linspace(-1.0, -0.5, 6))
    assert np.allclose(scan.base_q, base, atol=1e-15)
    assert np.allclose(scan.twist, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], atol=1e-15)
    assert len(scan.points) == 6


def test_the_sweep_shows_damping_bounding_the_step_the_pseudoinverse_does_not() -> None:
    """The pseudoinverse step peaks far above the damped step across the sweep."""
    robot = puma560()
    base = np.array([0.3, -0.9, 0.6, 0.4, 0.0, 0.2], dtype=np.float64)
    scan = scan_joint(
        robot,
        base,
        joint_index=4,
        values=np.linspace(-0.3, 0.3, 121),
        twist=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        damping=0.05,
        epsilon=0.05,
    )
    plain = scan.column("pseudoinverse_step_norm")
    damped = scan.column("damped_step_norm")
    assert plain.max() > 10.0 * damped.max()
    assert damped.max() < 1.0 / (2.0 * 0.05)
    assert scan.column("damping").max() <= 0.05 + 1e-12
    assert scan.column("damping").min() == 0.0


def test_the_sweep_refuses_a_joint_index_the_chain_does_not_have() -> None:
    """An out of range index names the range it was outside."""
    robot = ur5()
    with pytest.raises(IndexError, match=r"outside 0\.\.5"):
        scan_joint(robot, robot.home(), joint_index=6, values=np.linspace(0.0, 1.0, 3))
