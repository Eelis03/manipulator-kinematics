"""The structured record a pipeline run produces.

A trace is the only thing the analysis layer reads. It holds every input needed
to reproduce the run and every output needed to score it, so a figure or a table
can be regenerated without re-running any solver.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from manipulator_kinematics.algorithm.conditioning import Conditioning
from manipulator_kinematics.algorithm.protocol import IKResult, Tolerance

__all__ = ["ScanPoint", "SingularityScan", "Target", "Trace", "Trial"]

Array = NDArray[np.float64]


@dataclass(frozen=True, slots=True, eq=False)
class Target:
    """One pose the solvers are asked to reach.

    Attributes:
        index: Position of this target in the campaign.
        pose: The desired 4x4 tool pose in the world frame.
        reference_q: The configuration the pose was generated from, when the
            target was produced by forward kinematics. ``None`` for poses that
            came from elsewhere, which may not be reachable at all.
    """

    index: int
    pose: Array
    reference_q: Array | None = None


@dataclass(frozen=True, slots=True, eq=False)
class Trial:
    """One solver applied to one target from one seed.

    Attributes:
        target_index: Which target of the campaign was solved.
        seed: The configuration the solver started from.
        result: What the solver returned.
        conditioning: Jacobian metrics evaluated at the returned configuration.
    """

    target_index: int
    seed: Array
    result: IKResult
    conditioning: Conditioning

    @property
    def solver(self) -> str:
        """Name of the solver that produced this trial."""
        return self.result.solver


@dataclass(frozen=True, slots=True, eq=False)
class Trace:
    """Everything one inverse kinematics campaign produced.

    Attributes:
        robot: Name of the chain that was solved.
        solvers: Solver names in the order they were run.
        tolerance: Convergence thresholds every solver was held to.
        max_iterations: Iteration budget every solver was given.
        characteristic_length: Length used to homogenise the Jacobian rows.
        targets: The poses that were requested.
        trials: One entry per solver and target pair.
    """

    robot: str
    solvers: tuple[str, ...]
    tolerance: Tolerance
    max_iterations: int
    characteristic_length: float
    targets: tuple[Target, ...]
    trials: tuple[Trial, ...]

    def for_solver(self, solver: str) -> tuple[Trial, ...]:
        """Return the trials belonging to one solver, in target order."""
        return tuple(trial for trial in self.trials if trial.solver == solver)


@dataclass(frozen=True, slots=True, eq=False)
class ScanPoint:
    """Conditioning and step magnitudes at one point of a joint sweep.

    Attributes:
        value: The swept joint value, in radians or metres.
        conditioning: Jacobian metrics at this configuration.
        transpose_step_norm: Norm of the Jacobian transpose update, in radians.
        pseudoinverse_step_norm: Norm of the pseudoinverse update, in radians.
        damped_step_norm: Norm of the damped least squares update, in radians.
        damping: The variable damping factor in force at this configuration.
    """

    value: float
    conditioning: Conditioning
    transpose_step_norm: float
    pseudoinverse_step_norm: float
    damped_step_norm: float
    damping: float


@dataclass(frozen=True, slots=True, eq=False)
class SingularityScan:
    """A one-joint sweep recording how each update rule behaves near a singularity.

    Attributes:
        robot: Name of the chain that was swept.
        joint_index: Which joint was swept, counting from zero.
        base_q: The configuration held fixed outside the swept joint.
        twist: The unit task-space velocity each rule was asked to serve.
        characteristic_length: Length used to homogenise the Jacobian rows.
        points: The sweep, in increasing joint value.
    """

    robot: str
    joint_index: int
    base_q: Array
    twist: Array
    characteristic_length: float
    points: tuple[ScanPoint, ...]

    def column(self, attribute: str) -> Array:
        """Return one scalar field of every scan point as an array.

        Args:
            attribute: Name of a :class:`ScanPoint` field, or one of
                ``"manipulability"``, ``"condition_number"`` and
                ``"smallest_singular_value"`` to reach into the conditioning
                record.

        Returns:
            A one-dimensional array with one entry per scan point.
        """
        conditioning_fields = {
            "manipulability",
            "condition_number",
            "smallest_singular_value",
        }
        if attribute in conditioning_fields:
            return np.array(
                [getattr(point.conditioning, attribute) for point in self.points],
                dtype=np.float64,
            )
        return np.array([getattr(point, attribute) for point in self.points], dtype=np.float64)
