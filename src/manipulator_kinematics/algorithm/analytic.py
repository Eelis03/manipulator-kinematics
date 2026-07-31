"""Closed-form inverse kinematics for a 6R arm with a spherical wrist.

The method is kinematic decoupling, stated in R. P. Paul, *Robot Manipulators:
Mathematics, Programming and Control*, MIT Press 1981, and in the form used here
in B. Siciliano, L. Sciavicco, L. Villani and G. Oriolo, *Robotics: Modelling,
Planning and Control*, Springer 2009, section 2.12.2. When the last three axes
intersect at a point, the position of that point depends only on the first three
joints, so the six-dimensional problem splits into two three-dimensional ones.

Step one, position. The wrist centre is ``p_wc = p - d6 R e_z``, where ``p`` and
``R`` are the desired tool position and orientation. Placing ``p_wc`` with joints
one to three gives a quadratic in the shoulder angle and a two-link planar
problem in the shoulder and elbow angles, hence four arm postures: shoulder left
or right, elbow up or down.

Step two, orientation. With ``R_0_3`` fixed by the arm posture, the wrist must
supply ``R_3_6 = R_0_3^T R``. For the standard-convention wrist used here this
product equals ``Rz(q4) Ry(-q5) Rz(q6)``, a ZYZ Euler factorisation with two
solutions that differ by the sign of ``q5``, hence eight solutions overall.

The derivation below is written for the parameter structure of the Unimation
PUMA 560, which is the canonical 6R spherical wrist arm:

    a1 = 0, alpha1 = +pi/2, alpha2 = 0, alpha3 = -pi/2,
    a4 = a5 = a6 = 0, d5 = 0, alpha4 = +pi/2, alpha5 = -pi/2, alpha6 = 0.

:func:`assert_spherical_wrist` checks that structure and explains any mismatch,
so an arm the derivation does not cover fails loudly rather than silently
returning wrong angles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np
from numpy.typing import NDArray

from manipulator_kinematics.algorithm.forward import forward_kinematics
from manipulator_kinematics.algorithm.protocol import IKResult, Tolerance
from manipulator_kinematics.model.dh import DHConvention, JointType, Robot, link_transform
from manipulator_kinematics.model.joints import JointLimit, within_limits
from manipulator_kinematics.model.transforms import pose_error, se3_inverse

__all__ = [
    "AnalyticIK",
    "AnalyticSolution",
    "StructureError",
    "analytic_ik",
    "assert_spherical_wrist",
]

Array = NDArray[np.float64]

_STRUCTURE_TOLERANCE: Final[float] = 1e-12
_WRIST_SINGULARITY: Final[float] = 1e-9


class StructureError(ValueError):
    """Raised when a robot does not have the structure the derivation assumes."""


@dataclass(frozen=True, slots=True)
class AnalyticSolution:
    """One branch of the closed-form solution.

    Attributes:
        q: The six joint values, in radians.
        shoulder: ``"right"`` or ``"left"``, the sign of the shoulder branch.
        elbow: ``"up"`` or ``"down"``, the sign of the elbow branch.
        wrist: ``"non-flip"``, ``"flip"``, or ``"degenerate"`` when the wrist is
            at a singularity and ``q4`` was fixed to zero to select one member of
            a one-parameter family.
        feasible: True when every joint value lies inside its declared limit.
        position_error: Distance in metres between the achieved and requested
            tool position.
        orientation_error: Angle in radians between the achieved and requested
            tool orientation.
    """

    q: Array
    shoulder: str
    elbow: str
    wrist: str
    feasible: bool
    position_error: float
    orientation_error: float

    @property
    def branch(self) -> str:
        """Human-readable branch label, for example ``"right/up/non-flip"``."""
        return f"{self.shoulder}/{self.elbow}/{self.wrist}"


def assert_spherical_wrist(robot: Robot) -> None:
    """Check that ``robot`` matches the structure the closed form assumes.

    Args:
        robot: The chain to check.

    Raises:
        StructureError: With a message naming the first parameter that does not
            match the required structure.
    """
    if robot.convention is not DHConvention.STANDARD:
        raise StructureError("the closed form is derived for the standard DH convention")
    if robot.n_joints != 6:
        raise StructureError(f"expected 6 joints, got {robot.n_joints}")
    if any(joint is not JointType.REVOLUTE for joint in robot.joint_types):
        raise StructureError("every joint must be revolute")

    links = robot.links
    half = np.pi / 2
    required: tuple[tuple[str, float, float], ...] = (
        ("a1", links[0].a, 0.0),
        ("alpha1", links[0].alpha, half),
        ("alpha2", links[1].alpha, 0.0),
        ("alpha3", links[2].alpha, -half),
        ("a4", links[3].a, 0.0),
        ("alpha4", links[3].alpha, half),
        ("a5", links[4].a, 0.0),
        ("d5", links[4].d, 0.0),
        ("alpha5", links[4].alpha, -half),
        ("a6", links[5].a, 0.0),
        ("alpha6", links[5].alpha, 0.0),
    )
    for label, actual, expected in required:
        if abs(actual - expected) > _STRUCTURE_TOLERANCE:
            raise StructureError(
                f"{label} must equal {expected!r} for a spherical wrist, got {actual!r}"
            )


def _representative(angle: float, limit: JointLimit | None) -> float:
    """Return ``angle`` modulo a turn, preferring a value inside ``limit``."""
    wrapped = float(np.arctan2(np.sin(angle), np.cos(angle)))
    if limit is None:
        return wrapped
    candidates = [wrapped + 2.0 * np.pi * k for k in (0, 1, -1, 2, -2)]
    inside = [c for c in candidates if limit.lower <= c <= limit.upper]
    if inside:
        return min(inside, key=abs)
    return wrapped


def _arm_branches(
    robot: Robot,
    wrist_centre: Array,
    tolerance: float,
) -> list[tuple[float, float, float, str, str]]:
    """Return the feasible ``(q1, q2, q3, shoulder, elbow)`` postures."""
    links = robot.links
    d1, a2 = links[0].d, links[1].a
    a3, d3_offset = links[2].a, links[1].d + links[2].d
    d4 = links[3].d

    px, py, pz = (float(v) for v in wrist_centre)
    planar_squared = px * px + py * py - d3_offset * d3_offset
    if planar_squared < -tolerance:
        return []
    planar = float(np.sqrt(max(planar_squared, 0.0)))

    forearm = float(np.hypot(a3, d4))
    forearm_angle = float(np.arctan2(d4, a3))
    height = pz - d1

    branches: list[tuple[float, float, float, str, str]] = []
    for shoulder_sign, shoulder in ((1.0, "right"), (-1.0, "left")):
        reach = shoulder_sign * planar
        q1 = float(np.arctan2(py, px)) - float(np.arctan2(-d3_offset, reach))

        cosine = (reach * reach + height * height - a2 * a2 - forearm * forearm) / (
            2.0 * a2 * forearm
        )
        if abs(cosine) > 1.0 + tolerance:
            continue
        interior = float(np.arccos(float(np.clip(cosine, -1.0, 1.0))))

        for elbow_sign, elbow in ((1.0, "up"), (-1.0, "down")):
            beta = elbow_sign * interior
            q3 = beta - forearm_angle
            along = a2 + forearm * float(np.cos(beta))
            across = forearm * float(np.sin(beta))
            q2 = float(np.arctan2(height, reach)) - float(np.arctan2(across, along))
            branches.append((q1, q2, q3, shoulder, elbow))
    return branches


def _wrist_branches(relative: Array) -> list[tuple[float, float, float, str]]:
    """Factor ``R_3_6`` into ``Rz(q4) Ry(-q5) Rz(q6)``, returning every branch."""
    sine = float(np.hypot(relative[0, 2], relative[1, 2]))
    if sine < _WRIST_SINGULARITY:
        if relative[2, 2] > 0.0:
            total = float(np.arctan2(relative[1, 0], relative[0, 0]))
            return [(0.0, 0.0, total, "degenerate")]
        difference = float(np.arctan2(-relative[1, 0], -relative[0, 0]))
        return [(0.0, float(np.pi), -difference, "degenerate")]

    non_flip = (
        float(np.arctan2(-relative[1, 2], -relative[0, 2])),
        float(np.arctan2(sine, relative[2, 2])),
        float(np.arctan2(-relative[2, 1], relative[2, 0])),
        "non-flip",
    )
    flip = (
        float(np.arctan2(relative[1, 2], relative[0, 2])),
        float(np.arctan2(-sine, relative[2, 2])),
        float(np.arctan2(relative[2, 1], -relative[2, 0])),
        "flip",
    )
    return [non_flip, flip]


def analytic_ik(
    robot: Robot,
    target: Array,
    *,
    tolerance: float = 1e-9,
) -> tuple[AnalyticSolution, ...]:
    """Return every closed-form solution placing the tool of ``robot`` at ``target``.

    Args:
        robot: A 6R chain with a spherical wrist in the standard DH convention.
        target: Desired 4x4 tool pose in the world frame.
        tolerance: Slack allowed when testing reachability, in metres.

    Returns:
        Up to eight solutions, ordered shoulder then elbow then wrist. The tuple
        is empty when the target lies outside the reachable workspace. Each
        solution carries the residual measured by forward kinematics, so a caller
        never has to trust the derivation on its own.

    Raises:
        StructureError: If ``robot`` is not a 6R chain with a spherical wrist.
    """
    assert_spherical_wrist(robot)
    pose = np.asarray(target, dtype=np.float64)
    if pose.shape != (4, 4):
        raise ValueError(f"target must be a 4x4 transform, got shape {pose.shape}")

    chain = se3_inverse(robot.base) @ pose @ se3_inverse(robot.tool)
    rotation = chain[:3, :3]
    wrist_centre = chain[:3, 3] - robot.links[5].d * rotation[:, 2]

    offsets = np.array([link.theta for link in robot.links], dtype=np.float64)
    solutions: list[AnalyticSolution] = []

    for theta1, theta2, theta3, shoulder, elbow in _arm_branches(robot, wrist_centre, tolerance):
        arm = np.array([theta1, theta2, theta3], dtype=np.float64) - offsets[:3]
        frame = np.eye(4, dtype=np.float64)
        for link, value in zip(robot.links[:3], arm, strict=True):
            frame = frame @ link_transform(link, float(value), robot.convention)
        relative = frame[:3, :3].T @ rotation

        for theta4, theta5, theta6, wrist in _wrist_branches(relative):
            raw = np.concatenate((arm, np.array([theta4, theta5, theta6]) - offsets[3:]))
            q = np.array(
                [
                    _representative(float(value), link.limit)
                    for value, link in zip(raw, robot.links, strict=True)
                ],
                dtype=np.float64,
            )
            residual = pose_error(forward_kinematics(robot, q), pose)
            solutions.append(
                AnalyticSolution(
                    q=q,
                    shoulder=shoulder,
                    elbow=elbow,
                    wrist=wrist,
                    feasible=(
                        within_limits(q, robot.limits) if robot.has_limits else True
                    ),
                    position_error=float(np.linalg.norm(residual[:3])),
                    orientation_error=float(np.linalg.norm(residual[3:])),
                )
            )
    return tuple(solutions)


@dataclass(frozen=True, slots=True)
class AnalyticIK:
    """The closed form wrapped in the common solver interface.

    Of the branches that meet ``tolerance``, the one nearest ``seed`` in the
    Euclidean joint metric is returned, so the solver behaves like a local method
    even though it does no searching. Branches outside the joint limits are
    discarded first unless ``require_feasible`` is cleared.

    Attributes:
        tolerance: Convergence thresholds used to accept a branch.
        require_feasible: Whether to discard branches outside the joint limits.
    """

    tolerance: Tolerance = field(default_factory=Tolerance)
    require_feasible: bool = True

    @property
    def name(self) -> str:
        """Short identifier used in traces, tables and figures."""
        return "analytic"

    def solve(self, robot: Robot, target: Array, seed: Array) -> IKResult:
        """Return the closed-form branch nearest ``seed``."""
        reference = robot.check_configuration(seed)
        candidates = analytic_ik(robot, target)
        if self.require_feasible:
            candidates = tuple(item for item in candidates if item.feasible)
        accepted = tuple(
            item
            for item in candidates
            if item.position_error <= self.tolerance.position
            and item.orientation_error <= self.tolerance.orientation
        )

        start = pose_error(forward_kinematics(robot, reference), target)
        seed_residual = float(np.linalg.norm(start))
        if not accepted:
            return IKResult(
                solver=self.name,
                q=reference.copy(),
                converged=False,
                iterations=0,
                position_error=float(np.linalg.norm(start[:3])),
                orientation_error=float(np.linalg.norm(start[3:])),
                residuals=(seed_residual,),
                message="no closed-form branch satisfied the tolerance",
            )

        best = min(accepted, key=lambda item: float(np.linalg.norm(item.q - reference)))
        residual = float(np.hypot(best.position_error, best.orientation_error))
        return IKResult(
            solver=self.name,
            q=best.q.copy(),
            converged=True,
            iterations=1,
            position_error=best.position_error,
            orientation_error=best.orientation_error,
            residuals=(seed_residual, residual),
            message=f"closed form, branch {best.branch}",
        )
