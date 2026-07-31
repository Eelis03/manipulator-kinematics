"""Drives solvers over a set of targets and records a structured trace.

Nothing here decides what a good result is and nothing here draws anything. The
pipeline is the layer that owns reproducibility: every source of randomness is a
:class:`numpy.random.Generator` passed in by the caller, so a campaign is fully
determined by its seed.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from manipulator_kinematics.algorithm.conditioning import conditioning
from manipulator_kinematics.algorithm.forward import forward_kinematics
from manipulator_kinematics.algorithm.jacobian import geometric_jacobian
from manipulator_kinematics.algorithm.numerical import (
    damped_step,
    pseudoinverse_step,
    transpose_step,
    variable_damping,
)
from manipulator_kinematics.algorithm.protocol import IKSolver, Tolerance
from manipulator_kinematics.model.dh import Robot, chain_reach
from manipulator_kinematics.model.joints import clamp_to_limits, sample_within_limits
from manipulator_kinematics.pipeline.trace import (
    ScanPoint,
    SingularityScan,
    Target,
    Trace,
    Trial,
)

__all__ = ["perturbed_seeds", "run_campaign", "sample_targets", "scan_joint"]

Array = NDArray[np.float64]


def sample_targets(
    robot: Robot,
    count: int,
    rng: np.random.Generator,
    *,
    margin: float = 0.15,
) -> tuple[Target, ...]:
    """Draw reachable targets by sampling configurations and running them forward.

    Sampling the configuration space rather than the task space guarantees that
    every target is reachable, so a failure is a failure of the solver rather
    than of the request.

    Args:
        robot: The chain to sample. It must declare joint limits.
        count: How many targets to produce.
        rng: Source of randomness.
        margin: Fraction of each joint span excluded at both ends.

    Returns:
        ``count`` targets, each carrying the configuration that generated it.
    """
    if count < 0:
        raise ValueError("count must not be negative")
    limits = robot.limits
    targets: list[Target] = []
    for index in range(count):
        q = sample_within_limits(limits, rng, margin=margin)
        targets.append(Target(index=index, pose=forward_kinematics(robot, q), reference_q=q))
    return tuple(targets)


def perturbed_seeds(
    robot: Robot,
    targets: Sequence[Target],
    rng: np.random.Generator,
    *,
    spread: float = 0.4,
) -> tuple[Array, ...]:
    """Return one seed per target, offset from the generating configuration.

    A seed drawn this way is a realistic starting point for a local solver: close
    enough that a solution exists nearby, far enough that the solver has work to
    do. Targets without a generating configuration are seeded at the joint limit
    midpoint.

    Args:
        robot: The chain being solved.
        targets: The targets to seed.
        rng: Source of randomness.
        spread: Standard deviation of the offset, in radians or metres.

    Returns:
        One seed vector per target, clipped into the joint limits.
    """
    limits = robot.limits
    midpoint = np.array([limit.midpoint for limit in limits], dtype=np.float64)
    seeds: list[Array] = []
    for target in targets:
        base = midpoint if target.reference_q is None else target.reference_q
        offset = rng.normal(0.0, spread, robot.n_joints)
        seeds.append(clamp_to_limits(base + offset, limits))
    return tuple(seeds)


def run_campaign(
    robot: Robot,
    solvers: Sequence[IKSolver],
    targets: Sequence[Target],
    seeds: Sequence[Array],
    *,
    tolerance: Tolerance,
    max_iterations: int,
) -> Trace:
    """Run every solver over every target and record the outcome.

    Args:
        robot: The chain to solve.
        solvers: Solvers to compare. They are run in the order given.
        targets: Poses to reach.
        seeds: One starting configuration per target, shared across solvers so
            the comparison is paired.
        tolerance: Convergence thresholds, recorded in the trace for reference.
        max_iterations: Iteration budget, recorded in the trace for reference.

    Returns:
        A trace holding one trial per solver and target pair.

    Raises:
        ValueError: If ``seeds`` and ``targets`` have different lengths.
    """
    if len(seeds) != len(targets):
        raise ValueError(f"got {len(seeds)} seeds for {len(targets)} targets")

    length = chain_reach(robot)
    trials: list[Trial] = []
    for solver in solvers:
        for target, seed in zip(targets, seeds, strict=True):
            result = solver.solve(robot, target.pose, seed)
            metrics = conditioning(
                geometric_jacobian(robot, result.q), characteristic_length=length
            )
            trials.append(
                Trial(
                    target_index=target.index,
                    seed=np.array(seed, dtype=np.float64),
                    result=result,
                    conditioning=metrics,
                )
            )

    return Trace(
        robot=robot.name,
        solvers=tuple(solver.name for solver in solvers),
        tolerance=tolerance,
        max_iterations=max_iterations,
        characteristic_length=length,
        targets=tuple(targets),
        trials=tuple(trials),
    )


def scan_joint(
    robot: Robot,
    base_q: Array,
    joint_index: int,
    values: Array,
    *,
    twist: Array | None = None,
    damping: float = 0.05,
    epsilon: float = 0.05,
) -> SingularityScan:
    """Sweep one joint and record conditioning and the step each rule requests.

    At every point the three update rules are asked to serve the same unit task
    velocity. Comparing the norms of the steps they return is the standard way to
    show what damping buys: the pseudoinverse step grows without bound as the
    Jacobian loses rank, while the damped step stays bounded by
    ``||twist|| / (2 lambda)``.

    Args:
        robot: The chain to sweep.
        base_q: The configuration held fixed outside the swept joint.
        joint_index: Which joint to sweep, counting from zero.
        values: The joint values to visit, in increasing order.
        twist: The 6-vector task velocity to serve. Defaults to a unit velocity
            along the world x axis.
        damping: Maximum damping factor for the damped rule.
        epsilon: Width of the singular region for the variable damping schedule.

    Returns:
        A scan record with one entry per value.

    Raises:
        IndexError: If ``joint_index`` is out of range.
    """
    if not 0 <= joint_index < robot.n_joints:
        raise IndexError(f"joint_index {joint_index} is outside 0..{robot.n_joints - 1}")

    reference = robot.check_configuration(base_q)
    direction = (
        np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        if twist is None
        else np.asarray(twist, dtype=np.float64).reshape(6)
    )
    length = chain_reach(robot)

    points: list[ScanPoint] = []
    for value in np.asarray(values, dtype=np.float64).reshape(-1):
        q = reference.copy()
        q[joint_index] = float(value)
        jacobian = geometric_jacobian(robot, q)
        lam = variable_damping(jacobian, damping=damping, epsilon=epsilon)
        points.append(
            ScanPoint(
                value=float(value),
                conditioning=conditioning(jacobian, characteristic_length=length),
                transpose_step_norm=float(np.linalg.norm(transpose_step(jacobian, direction))),
                pseudoinverse_step_norm=float(
                    np.linalg.norm(pseudoinverse_step(jacobian, direction))
                ),
                damped_step_norm=float(np.linalg.norm(damped_step(jacobian, direction, lam))),
                damping=lam,
            )
        )

    return SingularityScan(
        robot=robot.name,
        joint_index=joint_index,
        base_q=reference,
        twist=direction,
        characteristic_length=length,
        points=tuple(points),
    )
