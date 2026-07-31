"""Geometric Jacobian of a serial chain, and a finite-difference reference.

The geometric Jacobian ``J`` maps joint rates to the tool twist expressed in the
world frame, ordered as ``[vx, vy, vz, wx, wy, wz]``. Column ``i`` follows the
construction in Siciliano, Sciavicco, Villani and Oriolo, *Robotics: Modelling,
Planning and Control*, Springer 2009, section 3.1:

    revolute  ``i``:  [z_i x (p_e - p_i); z_i]
    prismatic ``i``:  [z_i; 0]

where ``z_i`` is the axis of joint ``i`` and ``p_i`` any point on it. In the
standard convention the axis of joint ``i`` is the z axis of frame ``i-1``; in
the modified convention it is the z axis of frame ``i``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from manipulator_kinematics.algorithm.forward import forward_kinematics, link_frames
from manipulator_kinematics.model.dh import DHConvention, JointType, Robot

__all__ = ["finite_difference_jacobian", "geometric_jacobian"]

Array = NDArray[np.float64]


def geometric_jacobian(robot: Robot, q: Array) -> Array:
    """Return the 6 by ``n_joints`` geometric Jacobian in the world frame.

    Args:
        robot: The chain to evaluate.
        q: Joint values, one per joint.

    Returns:
        A 6 by ``n_joints`` array whose rows are ordered
        ``[vx, vy, vz, wx, wy, wz]``.
    """
    frames = link_frames(robot, q)
    tool_position = (frames[-1] @ robot.tool)[:3, 3]

    jacobian = np.zeros((6, robot.n_joints), dtype=np.float64)
    standard = robot.convention is DHConvention.STANDARD
    for index, link in enumerate(robot.links):
        axis_frame = frames[index] if standard else frames[index + 1]
        axis = axis_frame[:3, 2]
        if link.joint_type is JointType.REVOLUTE:
            jacobian[:3, index] = np.cross(axis, tool_position - axis_frame[:3, 3])
            jacobian[3:, index] = axis
        else:
            jacobian[:3, index] = axis
    return jacobian


def finite_difference_jacobian(robot: Robot, q: Array, *, step: float = 1e-6) -> Array:
    """Return a central finite-difference approximation of the geometric Jacobian.

    The translation rows differentiate the tool position directly. The rotation
    rows use the rotation vector of ``R(q + h) @ R(q - h).T`` divided by ``2h``,
    which is the world-frame angular velocity to first order in ``h``.

    This function exists to validate :func:`geometric_jacobian`. It is roughly
    ``2n`` times more expensive and is not used by the solvers.

    Args:
        robot: The chain to evaluate.
        q: Joint values, one per joint.
        step: Half-width of the central difference.

    Returns:
        A 6 by ``n_joints`` array in the same row order as
        :func:`geometric_jacobian`.
    """
    values = robot.check_configuration(q)
    jacobian = np.zeros((6, robot.n_joints), dtype=np.float64)
    for index in range(robot.n_joints):
        forward = values.copy()
        backward = values.copy()
        forward[index] += step
        backward[index] -= step
        pose_forward = forward_kinematics(robot, forward)
        pose_backward = forward_kinematics(robot, backward)
        jacobian[:3, index] = (pose_forward[:3, 3] - pose_backward[:3, 3]) / (2.0 * step)
        relative = pose_forward[:3, :3] @ pose_backward[:3, :3].T
        jacobian[3:, index] = Rotation.from_matrix(relative).as_rotvec() / (2.0 * step)
    return jacobian
