"""Forward kinematics for an arbitrary serial chain of revolute and prismatic joints."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from manipulator_kinematics.model.dh import Robot, link_transform

__all__ = ["forward_kinematics", "link_frames"]

Array = NDArray[np.float64]


def link_frames(robot: Robot, q: Array) -> list[Array]:
    """Return the cumulative frames ``[T_0, T_1, ..., T_n]`` of the chain.

    ``T_0`` is the robot base transform and ``T_i`` for ``i >= 1`` is the pose of
    link frame ``i`` in the world frame. The tool transform is not applied, so
    ``T_n`` is the pose of the last link frame.

    Args:
        robot: The chain to evaluate.
        q: Joint values, one per joint.

    Returns:
        A list of ``n_joints + 1`` homogeneous transforms.
    """
    values = robot.check_configuration(q)
    frames = [np.array(robot.base, dtype=np.float64)]
    for parameter, value in zip(robot.links, values, strict=True):
        frames.append(frames[-1] @ link_transform(parameter, float(value), robot.convention))
    return frames


def forward_kinematics(robot: Robot, q: Array) -> Array:
    """Return the tool pose in the world frame for configuration ``q``.

    The composition is ``base @ A_1(q_1) @ ... @ A_n(q_n) @ tool``, where each
    ``A_i`` follows the convention declared by the robot.

    Args:
        robot: The chain to evaluate.
        q: Joint values, one per joint.

    Returns:
        A 4x4 homogeneous transform.
    """
    return link_frames(robot, q)[-1] @ robot.tool
