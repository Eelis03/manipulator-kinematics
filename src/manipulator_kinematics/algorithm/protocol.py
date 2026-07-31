"""The common interface every inverse kinematics solver implements.

Solvers are structurally typed against :class:`IKSolver`, so a caller can hold a
``tuple[IKSolver, ...]`` and compare methods without any of them sharing a base
class. :class:`IKResult` is the single record every solver returns, which is what
lets the pipeline layer build one trace format for all of them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from manipulator_kinematics.model.dh import Robot

__all__ = ["IKResult", "IKSolver", "Tolerance"]

Array = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Tolerance:
    """Convergence thresholds on the two halves of the pose error.

    Attributes:
        position: Maximum Euclidean position error in metres.
        orientation: Maximum rotation angle error in radians.
    """

    position: float = 1e-6
    orientation: float = 1e-6

    def __post_init__(self) -> None:
        if self.position <= 0.0 or self.orientation <= 0.0:
            raise ValueError("tolerances must be positive")

    def satisfied(self, error: Array) -> bool:
        """Return True when a 6-vector pose error meets both thresholds."""
        return bool(
            np.linalg.norm(error[:3]) <= self.position
            and np.linalg.norm(error[3:]) <= self.orientation
        )


@dataclass(frozen=True, slots=True)
class IKResult:
    """The outcome of one inverse kinematics call.

    Attributes:
        solver: Name of the method that produced the result.
        q: The configuration reached, always the best iterate seen.
        converged: True when both tolerances were met.
        iterations: Number of update steps actually taken.
        position_error: Euclidean position error at ``q``, in metres.
        orientation_error: Rotation angle error at ``q``, in radians.
        residuals: Norm of the full 6-vector pose error after each iteration,
            starting with the value at the seed configuration.
        message: Why the solver stopped.
    """

    solver: str
    q: Array
    converged: bool
    iterations: int
    position_error: float
    orientation_error: float
    residuals: tuple[float, ...]
    message: str

    @property
    def final_residual(self) -> float:
        """Norm of the pose error at ``q``."""
        return self.residuals[-1]


@runtime_checkable
class IKSolver(Protocol):
    """Structural type of an inverse kinematics solver.

    Implementations must be pure with respect to their arguments: neither
    ``robot``, ``target`` nor ``seed`` may be modified.
    """

    @property
    def name(self) -> str:
        """Short identifier used in traces, tables and figures."""
        ...

    def solve(self, robot: Robot, target: Array, seed: Array) -> IKResult:
        """Drive the tool pose of ``robot`` to ``target`` starting from ``seed``.

        Args:
            robot: The chain to solve.
            target: Desired 4x4 tool pose in the world frame.
            seed: Initial configuration, one value per joint.

        Returns:
            The best configuration found and the diagnostics of the search.
        """
        ...
