"""Random restart, wrapping a local solver so a stalled descent can begin again.

Every solver in this package is local: it returns the best iterate of one descent
from one seed. That descent can stop at a Karush-Kuhn-Tucker point of the
box-constrained problem, where no feasible direction reduces the pose error to
first order and no held joint wants to be released, so the step collapses however
it is computed. The target is reachable; it is not reachable from that seed
without leaving the limit box on the way. The remedy is a different starting
point rather than a better step, which is why it sits here, around the loop,
rather than inside it.

The rule is the multistart one of P. Beeson and B. Ames, 'TRAC-IK: an open-source
library for improved solving of generic inverse kinematics', IEEE-RAS
International Conference on Humanoid Robots 2015, pages 928 to 935,
doi:10.1109/HUMANOIDS.2015.7363472, where an inverse Jacobian solver that has
failed is restarted from configurations drawn uniformly from the joint limit box
until one attempt converges or a budget runs out.

Two things are promised and nothing else. A converged attempt is always preferred
to an unconverged one, so a target the wrapped solver already solves is never
lost, and it costs exactly one call because the search stops there. While nothing
has converged the lowest residual seen is kept, so the answer is never worse than
the single attempt the caller's own seed produces.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from manipulator_kinematics.algorithm.protocol import IKResult, IKSolver
from manipulator_kinematics.model.dh import Robot
from manipulator_kinematics.model.joints import sample_within_limits

__all__ = ["RestartingIK"]

Array = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class RestartingIK:
    """Any solver, run again from fresh configurations while it keeps failing.

    The generator that draws those configurations is built inside every call
    rather than carried on the instance, so one call is a function of its
    arguments alone and asking the same question twice returns the same answer,
    which is what the rest of the library assumes of a solver.

    Attributes:
        solver: The local solver to restart. Any implementation of
            :class:`~manipulator_kinematics.algorithm.protocol.IKSolver` will do.
        restarts: How many further starts to spend once the caller's seed has
            failed. Zero reproduces the wrapped solver exactly.
        rng_seed: Seed of the generator that draws the fresh configurations.
        margin: Fraction of each joint span excluded at both ends when drawing
            from the limit box, which keeps a restart off the hard stops.
        spread: Standard deviation of the displacement applied to the caller's
            seed instead, in radians or metres, for a chain that declares no
            limits and so offers no box to draw from.
    """

    solver: IKSolver
    restarts: int = 4
    rng_seed: int = 0
    margin: float = 0.05
    spread: float = 0.4

    def __post_init__(self) -> None:
        if self.restarts < 0:
            raise ValueError("restarts must not be negative")
        if not 0.0 <= self.margin < 0.5:
            raise ValueError("margin must be at least zero and below a half")

    @property
    def name(self) -> str:
        """Short identifier used in traces, tables and figures."""
        return f"{self.solver.name}-restart"

    def _fresh_start(self, robot: Robot, seed: Array, rng: np.random.Generator) -> Array:
        """Return a starting configuration independent of the one that failed."""
        if robot.has_limits:
            return sample_within_limits(robot.limits, rng, margin=self.margin)
        displaced: Array = seed + rng.normal(0.0, self.spread, robot.n_joints)
        return displaced

    def solve(self, robot: Robot, target: Array, seed: Array) -> IKResult:
        """Solve from ``seed``, then from fresh configurations while that fails.

        Args:
            robot: The chain to solve.
            target: Desired 4x4 tool pose in the world frame.
            seed: Initial configuration, tried before any restart is drawn.

        Returns:
            The best of the attempts made, carrying the residual history of the
            attempt that produced it rather than of all of them, because the
            record describes the descent the answer came from.
        """
        start = robot.check_configuration(seed)
        best = self.solver.solve(robot, target, start)
        rng = np.random.default_rng(self.rng_seed)
        winner, attempts = 0, 1

        while not best.converged and attempts <= self.restarts:
            attempt = self.solver.solve(robot, target, self._fresh_start(robot, start, rng))
            if attempt.converged or attempt.final_residual < best.final_residual:
                best, winner = attempt, attempts
            attempts += 1

        if attempts == 1:
            message = best.message
        elif best.converged:
            message = f"converged on start {winner + 1} of {attempts}"
        else:
            message = f"{best.message}, best of {attempts} starts"
        return replace(best, solver=self.name, message=message)
