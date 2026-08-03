"""Kinematic algorithms: forward kinematics, Jacobians, and inverse kinematics.

This package depends on :mod:`manipulator_kinematics.model` and on nothing else
in the project. It performs no input or output and draws no figures.
"""

from manipulator_kinematics.algorithm.analytic import (
    AnalyticIK,
    AnalyticSolution,
    StructureError,
    analytic_ik,
    assert_spherical_wrist,
)
from manipulator_kinematics.algorithm.conditioning import (
    Conditioning,
    conditioning,
    manipulability,
)
from manipulator_kinematics.algorithm.forward import forward_kinematics, link_frames
from manipulator_kinematics.algorithm.jacobian import (
    finite_difference_jacobian,
    geometric_jacobian,
)
from manipulator_kinematics.algorithm.numerical import (
    DampedLeastSquaresIK,
    DampingSchedule,
    JacobianTransposeIK,
    PseudoinverseIK,
    SolverSettings,
    active_set_step,
    blocked_joints,
    damped_step,
    numerical_solvers,
    pseudoinverse_step,
    residual_damping,
    transpose_step,
    variable_damping,
)
from manipulator_kinematics.algorithm.protocol import IKResult, IKSolver, Tolerance

__all__ = [
    "AnalyticIK",
    "AnalyticSolution",
    "Conditioning",
    "DampedLeastSquaresIK",
    "DampingSchedule",
    "IKResult",
    "IKSolver",
    "JacobianTransposeIK",
    "PseudoinverseIK",
    "SolverSettings",
    "StructureError",
    "Tolerance",
    "active_set_step",
    "analytic_ik",
    "assert_spherical_wrist",
    "blocked_joints",
    "conditioning",
    "damped_step",
    "finite_difference_jacobian",
    "forward_kinematics",
    "geometric_jacobian",
    "link_frames",
    "manipulability",
    "numerical_solvers",
    "pseudoinverse_step",
    "residual_damping",
    "transpose_step",
    "variable_damping",
]
