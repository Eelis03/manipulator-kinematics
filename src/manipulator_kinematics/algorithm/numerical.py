"""Iterative inverse kinematics solvers sharing one interface.

Three first-order methods are provided. All three solve the same linear model of
one step, ``J dq = e``, and differ only in how they invert ``J``.

Jacobian transpose
    ``dq = alpha J^T e`` with the step length that minimises the residual of the
    linear model, ``alpha = <e, J J^T e> / <J J^T e, J J^T e>``. Described by
    W. A. Wolovich and H. Elliott, 'A computational technique for inverse
    kinematics', IEEE Conference on Decision and Control 1984,
    doi:10.1109/CDC.1984.272258, with the optimal step given by S. R. Buss,
    'Introduction to inverse kinematics with Jacobian transpose, pseudoinverse
    and damped least squares methods', 2009,
    https://mathweb.ucsd.edu/~sbuss/ResearchWeb/ikmethods/ikmethods.pdf .

Moore-Penrose pseudoinverse
    ``dq = J^+ e`` with ``J^+`` built from a truncated singular value
    decomposition. This is the minimum-norm least-squares solution used by
    D. E. Whitney, 'Resolved motion rate control of manipulators and human
    prostheses', IEEE Transactions on Man-Machine Systems 10(2), 1969,
    doi:10.1109/TMMS.1969.299896.

Damped least squares
    ``dq = J^T (J J^T + lambda^2 I)^-1 e``, the Levenberg-Marquardt
    regularisation of the same problem. Introduced for manipulators by
    Y. Nakamura and H. Hanafusa, 'Inverse kinematic solutions with singularity
    robustness for robot manipulator control', ASME Journal of Dynamic Systems,
    Measurement and Control 108(3), 1986, doi:10.1115/1.3143764, and independently
    by C. W. Wampler, 'Manipulator inverse kinematic solutions based on vector
    formulations and damped least-squares methods', IEEE Transactions on Systems,
    Man and Cybernetics 16(1), 1986, doi:10.1109/TSMC.1986.289285. Three damping
    schedules are offered, described on :class:`DampingSchedule`.

The three rules are also exposed as the free functions :func:`transpose_step`,
:func:`pseudoinverse_step` and :func:`damped_step`, so the same update a solver
takes can be evaluated on its own, which is what the singularity scan in
:mod:`manipulator_kinematics.pipeline.runner` does.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve, svd

from manipulator_kinematics.algorithm.forward import forward_kinematics
from manipulator_kinematics.algorithm.jacobian import geometric_jacobian
from manipulator_kinematics.algorithm.protocol import IKResult, Tolerance
from manipulator_kinematics.model.dh import Robot
from manipulator_kinematics.model.joints import JointLimit, clamp_to_limits, limit_span
from manipulator_kinematics.model.transforms import pose_error

__all__ = [
    "DampedLeastSquaresIK",
    "DampingSchedule",
    "JacobianTransposeIK",
    "PseudoinverseIK",
    "SolverSettings",
    "active_set_step",
    "blocked_joints",
    "damped_step",
    "numerical_solvers",
    "pseudoinverse_step",
    "residual_damping",
    "transpose_step",
    "variable_damping",
]

Array = NDArray[np.float64]
Mask = NDArray[np.bool_]

_STEP_FLOOR = 1e-14
_BOUND_TOLERANCE = 1e-12


def _clamp_norm(vector: Array, limit: float) -> Array:
    norm = float(np.linalg.norm(vector))
    if limit <= 0.0 or norm <= limit:
        return vector
    return vector * (limit / norm)


def transpose_step(jacobian: Array, twist: Array) -> Array:
    """Return ``alpha J^T e`` with the step length that minimises the linear residual.

    Args:
        jacobian: A 6 by n Jacobian.
        twist: The 6-vector task-space error or velocity to serve.

    Returns:
        The joint-space step, zero when the error has no component in the range
        of the Jacobian.
    """
    gradient: Array = jacobian.T @ twist
    projected = jacobian @ gradient
    denominator = float(projected @ projected)
    if denominator <= _STEP_FLOOR:
        return np.zeros_like(gradient)
    return float(twist @ projected) / denominator * gradient


def pseudoinverse_step(jacobian: Array, twist: Array, *, rcond: float = 1e-12) -> Array:
    """Return ``J^+ e`` from a singular value decomposition truncated at ``rcond``.

    Args:
        jacobian: A 6 by n Jacobian.
        twist: The 6-vector task-space error or velocity to serve.
        rcond: Singular values at or below ``rcond`` times the largest are treated
            as zero.

    Returns:
        The minimum-norm least-squares joint-space step.
    """
    left, values, right_t = svd(jacobian, full_matrices=False)
    keep = values > rcond * float(values[0])
    inverted = np.zeros_like(values)
    inverted[keep] = 1.0 / values[keep]
    result: Array = right_t.T @ (inverted * (left.T @ twist))
    return result


def damped_step(jacobian: Array, twist: Array, damping: float) -> Array:
    """Return ``J^T (J J^T + lambda^2 I)^-1 e``.

    Zero damping is answered with :func:`pseudoinverse_step`, which is the limit
    of this expression as ``lambda`` falls to zero and is defined whether or not
    ``J J^T`` is invertible. Taking the formula literally there would ask for the
    inverse of a matrix that is singular whenever the Jacobian has fewer than six
    independent columns, which happens routinely once the active set holds a
    joint at a bound.

    Args:
        jacobian: A 6 by n Jacobian.
        twist: The 6-vector task-space error or velocity to serve.
        damping: The damping factor ``lambda``.

    Returns:
        The regularised joint-space step, bounded by ``||e|| / (2 lambda)`` for
        positive ``lambda`` and unbounded without it.
    """
    if damping <= 0.0:
        return pseudoinverse_step(jacobian, twist)
    rows = jacobian.shape[0]
    gram = jacobian @ jacobian.T + (damping**2) * np.eye(rows, dtype=np.float64)
    result: Array = jacobian.T @ solve(gram, twist, assume_a="pos")
    return result


def residual_damping(twist: Array, *, bias: float = 1e-8) -> float:
    """Return the residual-scaled Levenberg-Marquardt damping factor.

    ``lambda = sqrt(||e||^2 + bias)``, so damping is strong while the linear model
    is a poor description of the problem and vanishes as the error does.

    Args:
        twist: The 6-vector task-space error being served.
        bias: Floor on ``lambda^2``.

    Returns:
        The damping factor to use at this iteration.
    """
    return float(np.sqrt(float(twist @ twist) + bias))


def variable_damping(jacobian: Array, *, damping: float, epsilon: float) -> float:
    """Return the Chiaverini variable damping factor for ``jacobian``.

    Damping is zero while the smallest singular value exceeds ``epsilon`` and
    rises to ``damping`` as that singular value reaches zero.

    Args:
        jacobian: A 6 by n Jacobian.
        damping: Maximum damping factor.
        epsilon: Width of the singular region.

    Returns:
        The damping factor to use at this configuration.
    """
    smallest = float(svd(jacobian, compute_uv=False)[-1])
    if smallest >= epsilon:
        return 0.0
    return damping * float(np.sqrt(1.0 - (smallest / epsilon) ** 2))


def blocked_joints(
    q: Array,
    delta: Array,
    limits: tuple[JointLimit, ...],
    *,
    tolerance: float = _BOUND_TOLERANCE,
) -> Mask:
    """Return the joints whose proposed motion is refused by an active bound.

    A joint is blocked when it already sits on a limit, up to ``tolerance``, and
    the proposed step would push it further outside. A joint merely heading
    towards a limit is not blocked, because the step may be short enough to stay
    inside.

    Args:
        q: The current configuration.
        delta: The proposed joint step.
        limits: Travel range of each joint.
        tolerance: How close to a bound counts as sitting on it.

    Returns:
        A boolean mask with one entry per joint.
    """
    lower, upper = limit_span(limits)
    at_lower = (q <= lower + tolerance) & (delta < 0.0)
    at_upper = (q >= upper - tolerance) & (delta > 0.0)
    blocked: Mask = at_lower | at_upper
    return blocked


def active_set_step(
    step: Callable[[Array, Array], Array],
    jacobian: Array,
    twist: Array,
    q: Array,
    limits: tuple[JointLimit, ...],
    *,
    tolerance: float = _BOUND_TOLERANCE,
) -> Array:
    """Return the update rule ``step`` solved subject to the joint limit box.

    Clipping a step back into the limit box does not generally leave a descent
    direction, so an iterate that reaches a bound can sit against it while the
    rule keeps asking for motion the joint cannot make. This is the classical
    active-set treatment of a box-constrained least-squares step, in the form
    described by C. L. Lawson and R. J. Hanson, *Solving Least Squares Problems*,
    Prentice-Hall 1974, chapter 23. Two moves alternate:

    Bind
        Every joint sitting on a bound whose proposed motion leaves the box is
        added to the active set, held at zero, and the same rule is solved again
        on the remaining columns of the Jacobian. The solver then spends the
        freedom it still has rather than the freedom it has lost.

    Release
        At the reduced solution the free components of ``J^T r`` vanish, and the
        component belonging to a held joint is its Karush-Kuhn-Tucker multiplier.
        A sign that points back into the box means holding that joint is what is
        preventing progress, so the single worst offender is released. Releasing
        one at a time is the standard guard against cycling between two sets.

    The loop is capped at ``2 n`` passes, so it terminates whatever the geometry,
    and every pass is the same rule on a narrower Jacobian, which is why the
    constrained step costs no new dependency.

    Args:
        step: The update rule, taking a Jacobian and a task-space error.
        jacobian: The full 6 by n Jacobian at ``q``.
        twist: The task-space error the step is being asked to serve.
        q: The current configuration, assumed to lie inside ``limits``.
        limits: Travel range of each joint.
        tolerance: How close to a bound counts as sitting on it.

    Returns:
        A joint step whose held components are exactly zero.
    """
    lower, upper = limit_span(limits)
    on_lower = q <= lower + tolerance
    on_upper = q >= upper - tolerance

    size = jacobian.shape[1]
    held = np.zeros(size, dtype=np.bool_)
    delta = step(jacobian, twist)

    for _ in range(2 * size):
        binding = blocked_joints(q, delta, limits, tolerance=tolerance) & ~held
        if bool(binding.any()):
            held |= binding
        else:
            multiplier = jacobian.T @ (twist - jacobian @ delta)
            escaping = held & ((on_lower & (multiplier > 0.0)) | (on_upper & (multiplier < 0.0)))
            if not bool(escaping.any()):
                break
            worst = int(np.argmax(np.where(escaping, np.abs(multiplier), -np.inf)))
            held[worst] = False

        free = ~held
        delta = np.zeros(size, dtype=np.float64)
        if not bool(free.any()):
            break
        delta[free] = step(jacobian[:, free], twist)

    delta[held] = 0.0
    return delta


@dataclass(frozen=True, slots=True)
class SolverSettings:
    """Loop parameters shared by every iterative solver.

    Attributes:
        max_iterations: Iteration budget.
        tolerance: Convergence thresholds on the two halves of the pose error.
        max_step: Largest joint step norm accepted in one iteration.
        error_clamp: Largest task-space error norm the step rule is shown, which
            keeps the linear model inside the region where it is a description of
            the problem.
        respect_limits: Whether to hold the iterate inside the joint limits.
        active_set: Whether a step refused by a bound is re-solved over the
            joints that can still move. Has no effect when ``respect_limits`` is
            cleared or the chain declares no limits.
        backtracking: How many halvings of a step that failed to reduce the
            residual are tried before the full step is taken anyway. Zero
            disables the search.
    """

    max_iterations: int = 200
    tolerance: Tolerance = field(default_factory=Tolerance)
    max_step: float = 1.0
    error_clamp: float = 0.5
    respect_limits: bool = True
    active_set: bool = True
    backtracking: int = 6


def _advance(
    robot: Robot,
    target: Array,
    q: Array,
    delta: Array,
    limits: tuple[JointLimit, ...] | None,
    residual: float,
    backtracking: int,
) -> tuple[Array, Array]:
    """Move along the projection of the ray ``q + t delta`` into the limit box.

    Clipping bends the ray into an arc, and the arc is not guaranteed to descend
    even when the ray does. The full step is taken whenever it improves on
    ``residual``, which is the common case and costs nothing extra. Otherwise the
    step is halved up to ``backtracking`` times, and the first length that does
    improve is taken. If none does, the full step is taken anyway, so the loop
    keeps the freedom to move through a worse configuration rather than stalling.

    Returns:
        The configuration reached and the pose error there.
    """

    def at(scale: float) -> tuple[Array, Array]:
        candidate = q + scale * delta
        if limits is not None:
            candidate = clamp_to_limits(candidate, limits)
        return candidate, pose_error(forward_kinematics(robot, candidate), target)

    full = at(1.0)
    if backtracking <= 0 or float(np.linalg.norm(full[1])) < residual:
        return full

    scale = 1.0
    for _ in range(backtracking):
        scale *= 0.5
        shorter = at(scale)
        if float(np.linalg.norm(shorter[1])) < residual:
            return shorter
    return full


def _iterate(
    label: str,
    settings: SolverSettings,
    robot: Robot,
    target: Array,
    seed: Array,
    step: Callable[[Array, Array], Array],
) -> IKResult:
    """Run the shared descent loop with a solver-specific ``step`` rule."""
    if settings.max_iterations < 0:
        raise ValueError("max_iterations must not be negative")

    q = robot.check_configuration(seed).copy()
    limits = robot.limits if settings.respect_limits and robot.has_limits else None
    if limits is not None:
        q = clamp_to_limits(q, limits)

    error = pose_error(forward_kinematics(robot, q), target)
    residuals = [float(np.linalg.norm(error))]
    best_q, best_residual = q.copy(), residuals[0]
    iterations = 0
    message = "iteration limit reached"

    while iterations < settings.max_iterations:
        if settings.tolerance.satisfied(error):
            message = "converged"
            break
        jacobian = geometric_jacobian(robot, q)
        served = _clamp_norm(error, settings.error_clamp)
        if limits is not None and settings.active_set:
            delta = active_set_step(step, jacobian, served, q, limits)
        else:
            delta = step(jacobian, served)
        if not np.all(np.isfinite(delta)):
            message = "step became non-finite"
            break
        delta = _clamp_norm(delta, settings.max_step)
        if float(np.linalg.norm(delta)) < _STEP_FLOOR:
            message = "step size collapsed"
            break

        q, error = _advance(
            robot, target, q, delta, limits, residuals[-1], settings.backtracking
        )
        iterations += 1
        residual = float(np.linalg.norm(error))
        residuals.append(residual)
        if residual < best_residual:
            best_q, best_residual = q.copy(), residual
    else:
        if settings.tolerance.satisfied(error):
            message = "converged"

    final_error = pose_error(forward_kinematics(robot, best_q), target)
    converged = settings.tolerance.satisfied(final_error)
    return IKResult(
        solver=label,
        q=best_q,
        converged=converged,
        iterations=iterations,
        position_error=float(np.linalg.norm(final_error[:3])),
        orientation_error=float(np.linalg.norm(final_error[3:])),
        residuals=tuple(residuals),
        message="converged" if converged else message,
    )


@dataclass(frozen=True, slots=True)
class JacobianTransposeIK:
    """Gradient descent on half the squared pose error, with an optimal step length.

    The update needs no matrix factorisation and stays bounded at a singularity,
    because it never inverts anything. It pays for that with a linear convergence
    rate that degrades as the condition number of the Jacobian grows, so it needs
    one or two orders of magnitude more iterations than the other two to reach the
    same tolerance, and often does not reach it at all within a fixed budget.

    Attributes:
        settings: Loop parameters shared with the other solvers.
    """

    settings: SolverSettings = field(default_factory=SolverSettings)

    @property
    def name(self) -> str:
        """Short identifier used in traces, tables and figures."""
        return "jacobian-transpose"

    def solve(self, robot: Robot, target: Array, seed: Array) -> IKResult:
        """Solve for a configuration whose tool pose matches ``target``."""
        return _iterate(self.name, self.settings, robot, target, seed, transpose_step)


@dataclass(frozen=True, slots=True)
class PseudoinverseIK:
    """Newton step through the Moore-Penrose pseudoinverse of the Jacobian.

    Singular values at or below ``rcond`` times the largest are discarded, which
    keeps the step finite at a singularity but leaves the corresponding task
    directions unserved. Convergence is quadratic away from singularities.

    Attributes:
        settings: Loop parameters shared with the other solvers.
        rcond: Relative singular value threshold for truncation.
    """

    settings: SolverSettings = field(default_factory=SolverSettings)
    rcond: float = 1e-12

    @property
    def name(self) -> str:
        """Short identifier used in traces, tables and figures."""
        return "pseudoinverse"

    def solve(self, robot: Robot, target: Array, seed: Array) -> IKResult:
        """Solve for a configuration whose tool pose matches ``target``."""

        def step(jacobian: Array, error: Array) -> Array:
            return pseudoinverse_step(jacobian, error, rcond=self.rcond)

        return _iterate(self.name, self.settings, robot, target, seed, step)


class DampingSchedule(Enum):
    """How the damping factor of :class:`DampedLeastSquaresIK` is chosen."""

    FIXED = "fixed"
    SINGULAR_REGION = "singular-region"
    LEVENBERG_MARQUARDT = "levenberg-marquardt"


@dataclass(frozen=True, slots=True)
class DampedLeastSquaresIK:
    """Damped least squares step, trading task accuracy for a bounded joint step.

    Three damping schedules are available.

    ``FIXED``
        ``lambda = damping`` everywhere. The simplest rule and the one originally
        proposed by Nakamura and Hanafusa and by Wampler. It bounds the step at
        every configuration but leaves a bias of order ``lambda^2`` in the
        converged pose, so it cannot reach a tight tolerance.

    ``SINGULAR_REGION``
        ``lambda = 0`` while the smallest singular value exceeds ``epsilon``, and
        ``lambda = damping sqrt(1 - (sigma_min / epsilon)^2)`` inside it. The
        variable damping factor of Chiaverini, Siciliano and Egeland, designed for
        resolved-rate velocity control, where the point is to bound the commanded
        joint rate near a singularity.

    ``LEVENBERG_MARQUARDT``
        ``lambda^2 = ||e||^2 + bias``, the residual-scaled damping applied to
        inverse kinematics by T. Sugihara, 'Solvability-unconcerned inverse
        kinematics by the Levenberg-Marquardt method', IEEE Transactions on
        Robotics 27(5), 2011, doi:10.1109/TRO.2011.2148230. Damping is strong far
        from the target, where the linear model is untrustworthy, and vanishes as
        the target is approached, so the terminal accuracy is that of the
        pseudoinverse. This is the default because it is the only one of the three
        that is both singularity robust and able to meet a tight tolerance.

    Attributes:
        settings: Loop parameters shared with the other solvers.
        schedule: Which damping rule to apply.
        damping: The damping factor for ``FIXED``, or its maximum for
            ``SINGULAR_REGION``.
        epsilon: Width of the singular region, used by ``SINGULAR_REGION``.
        bias: Floor on ``lambda^2``, used by ``LEVENBERG_MARQUARDT``, which keeps
            the normal matrix positive definite at the solution.
    """

    settings: SolverSettings = field(default_factory=SolverSettings)
    schedule: DampingSchedule = DampingSchedule.LEVENBERG_MARQUARDT
    damping: float = 0.05
    epsilon: float = 0.05
    bias: float = 1e-8

    def __post_init__(self) -> None:
        if self.damping < 0.0:
            raise ValueError("damping must not be negative")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        if self.bias <= 0.0:
            raise ValueError("bias must be positive")

    @property
    def name(self) -> str:
        """Short identifier used in traces, tables and figures."""
        return "damped-least-squares"

    def damping_for(self, jacobian: Array, error: Array) -> float:
        """Return the damping factor this solver applies at one iteration.

        Args:
            jacobian: The geometric Jacobian at the current configuration.
            error: The task-space error the step is being asked to serve.

        Returns:
            The damping factor ``lambda``.
        """
        if self.schedule is DampingSchedule.FIXED:
            return self.damping
        if self.schedule is DampingSchedule.SINGULAR_REGION:
            return variable_damping(jacobian, damping=self.damping, epsilon=self.epsilon)
        return residual_damping(error, bias=self.bias)

    def solve(self, robot: Robot, target: Array, seed: Array) -> IKResult:
        """Solve for a configuration whose tool pose matches ``target``."""

        def step(jacobian: Array, error: Array) -> Array:
            return damped_step(jacobian, error, self.damping_for(jacobian, error))

        return _iterate(self.name, self.settings, robot, target, seed, step)


def numerical_solvers(
    *,
    max_iterations: int = 200,
    tolerance: Tolerance | None = None,
) -> tuple[JacobianTransposeIK, PseudoinverseIK, DampedLeastSquaresIK]:
    """Return the three iterative solvers configured with the same loop settings.

    Args:
        max_iterations: Shared iteration budget.
        tolerance: Shared convergence thresholds, defaulting to 1e-6 on both.

    Returns:
        The solvers in the order transpose, pseudoinverse, damped least squares.
    """
    settings = SolverSettings(
        max_iterations=max_iterations,
        tolerance=tolerance if tolerance is not None else Tolerance(),
    )
    return (
        JacobianTransposeIK(settings=settings),
        PseudoinverseIK(settings=settings),
        DampedLeastSquaresIK(settings=settings),
    )
