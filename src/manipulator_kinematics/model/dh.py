"""Denavit-Hartenberg parameters and the serial chain they describe.

Two conventions are supported.

Standard (also called distal or Denavit-Hartenberg 1955, in the arrangement used
by Paul 1981), where the transform from frame ``i-1`` to frame ``i`` is

    A_i = Rot_z(theta_i) Trans_z(d_i) Trans_x(a_i) Rot_x(alpha_i)

and the axis of joint ``i`` is the z axis of frame ``i-1``.

Modified (also called proximal, Craig 1986), where the transform is

    A_i = Rot_x(alpha_{i-1}) Trans_x(a_{i-1}) Rot_z(theta_i) Trans_z(d_i)

and the axis of joint ``i`` is the z axis of frame ``i``. In this convention the
``a`` and ``alpha`` entries stored on link ``i`` are the proximal values
``a_{i-1}`` and ``alpha_{i-1}``, which is how Craig tabulates them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from manipulator_kinematics.model.joints import JointLimit

__all__ = [
    "DHConvention",
    "DHParameter",
    "JointType",
    "Robot",
    "chain_reach",
    "link_transform",
]

Array = NDArray[np.float64]


class DHConvention(Enum):
    """Which Denavit-Hartenberg arrangement a parameter table follows."""

    STANDARD = "standard"
    MODIFIED = "modified"


class JointType(Enum):
    """Whether the joint variable drives ``theta`` or ``d``."""

    REVOLUTE = "revolute"
    PRISMATIC = "prismatic"


@dataclass(frozen=True, slots=True)
class DHParameter:
    """One row of a Denavit-Hartenberg table.

    Attributes:
        d: Link offset along the previous z axis, in metres. Constant for a
            revolute joint, and the constant part of the joint variable for a
            prismatic joint.
        theta: Joint angle about the previous z axis, in radians. The constant
            part of the joint variable for a revolute joint, and constant for a
            prismatic joint.
        a: Link length along the common normal, in metres.
        alpha: Link twist about the common normal, in radians.
        joint_type: Which of ``theta`` and ``d`` the joint variable adds to.
        limit: Travel range of the joint variable, or ``None`` when the source
            publication does not state one.
    """

    d: float
    theta: float
    a: float
    alpha: float
    joint_type: JointType = JointType.REVOLUTE
    limit: JointLimit | None = None

    def resolve(self, q: float) -> tuple[float, float]:
        """Return ``(theta, d)`` for joint value ``q``, applying the constant offsets."""
        if self.joint_type is JointType.REVOLUTE:
            return self.theta + q, self.d
        return self.theta, self.d + q


def link_transform(parameter: DHParameter, q: float, convention: DHConvention) -> Array:
    """Return the 4x4 transform contributed by one link at joint value ``q``."""
    theta, d = parameter.resolve(q)
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(parameter.alpha), np.sin(parameter.alpha)
    a = parameter.a

    if convention is DHConvention.STANDARD:
        return np.array(
            [
                [ct, -st * ca, st * sa, a * ct],
                [st, ct * ca, -ct * sa, a * st],
                [0.0, sa, ca, d],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    return np.array(
        [
            [ct, -st, 0.0, a],
            [st * ca, ct * ca, -sa, -sa * d],
            [st * sa, ct * sa, ca, ca * d],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True, slots=True, eq=False)
class Robot:
    """A serial chain defined by a Denavit-Hartenberg table.

    Equality is identity based because the ``base`` and ``tool`` arrays make
    element-wise comparison ambiguous. Those arrays must not be mutated after
    construction.

    Attributes:
        name: Identifier used in traces and figures.
        links: One :class:`DHParameter` per joint, ordered from the base.
        convention: Which Denavit-Hartenberg arrangement ``links`` follows.
        base: Fixed transform from the world frame to link frame 0.
        tool: Fixed transform from the last link frame to the tool frame.
        source: Citation for the parameter values.
    """

    name: str
    links: tuple[DHParameter, ...]
    convention: DHConvention = DHConvention.STANDARD
    base: Array = field(default_factory=lambda: np.eye(4, dtype=np.float64))
    tool: Array = field(default_factory=lambda: np.eye(4, dtype=np.float64))
    source: str = ""

    def __post_init__(self) -> None:
        if not self.links:
            raise ValueError("a robot needs at least one link")
        if self.base.shape != (4, 4) or self.tool.shape != (4, 4):
            raise ValueError("base and tool must be 4x4 transforms")

    @property
    def n_joints(self) -> int:
        """Number of joints in the chain."""
        return len(self.links)

    @property
    def joint_types(self) -> tuple[JointType, ...]:
        """Joint type of each joint, ordered from the base."""
        return tuple(link.joint_type for link in self.links)

    @property
    def has_limits(self) -> bool:
        """True when every joint declares a travel range."""
        return all(link.limit is not None for link in self.links)

    @property
    def limits(self) -> tuple[JointLimit, ...]:
        """Travel range of every joint.

        Raises:
            ValueError: If any joint has no declared limit.
        """
        if not self.has_limits:
            missing = [i for i, link in enumerate(self.links) if link.limit is None]
            raise ValueError(f"joints {missing} have no declared limit")
        return tuple(link.limit for link in self.links if link.limit is not None)

    def home(self) -> Array:
        """Return the zero configuration, a vector of ``n_joints`` zeros."""
        return np.zeros(self.n_joints, dtype=np.float64)

    def check_configuration(self, q: Array) -> Array:
        """Validate and normalise a configuration vector.

        Raises:
            ValueError: If ``q`` does not have ``n_joints`` entries.
        """
        values = np.asarray(q, dtype=np.float64).reshape(-1)
        if values.size != self.n_joints:
            raise ValueError(f"expected {self.n_joints} joint values, got {values.size}")
        return values


def chain_reach(robot: Robot) -> float:
    """Return an upper bound on the distance from the base to the tool, in metres.

    The bound is the sum of the absolute link lengths and offsets plus the
    translation of the tool transform. It is used as the characteristic length
    that makes the rotation rows of a geometric Jacobian dimensionally comparable
    with its translation rows.

    Args:
        robot: The chain to measure.

    Returns:
        A positive length in metres.
    """
    total = sum(abs(link.a) + abs(link.d) for link in robot.links)
    total += float(np.linalg.norm(robot.tool[:3, 3]))
    prismatic = sum(
        abs(link.limit.upper) if link.limit is not None else 0.0
        for link in robot.links
        if link.joint_type is JointType.PRISMATIC
    )
    return float(total + prismatic) or 1.0
