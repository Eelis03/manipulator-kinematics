"""Forward and inverse kinematics for serial manipulators.

The package is layered. ``model`` holds the Denavit-Hartenberg data and the
transform algebra, ``algorithm`` holds forward kinematics, Jacobians and the
inverse kinematics solvers, ``pipeline`` drives a solver over a set of targets
and records a trace, and ``analysis`` turns a trace into metrics and figures.
Each layer depends only on the ones before it in that list.

The names re-exported here are the ones a caller normally needs. Anything else is
reachable through the sub-packages.
"""

from manipulator_kinematics.algorithm import (
    AnalyticIK,
    AnalyticSolution,
    Conditioning,
    DampedLeastSquaresIK,
    DampingSchedule,
    IKResult,
    IKSolver,
    JacobianTransposeIK,
    PseudoinverseIK,
    SolverSettings,
    StructureError,
    Tolerance,
    analytic_ik,
    conditioning,
    finite_difference_jacobian,
    forward_kinematics,
    geometric_jacobian,
    link_frames,
    manipulability,
    numerical_solvers,
)
from manipulator_kinematics.model import (
    ROBOTS,
    DHConvention,
    DHParameter,
    JointLimit,
    JointType,
    Robot,
    chain_reach,
    puma560,
    stanford_arm,
    ur5,
)

__all__ = [
    "ROBOTS",
    "AnalyticIK",
    "AnalyticSolution",
    "Conditioning",
    "DHConvention",
    "DHParameter",
    "DampedLeastSquaresIK",
    "DampingSchedule",
    "IKResult",
    "IKSolver",
    "JacobianTransposeIK",
    "JointLimit",
    "JointType",
    "PseudoinverseIK",
    "Robot",
    "SolverSettings",
    "StructureError",
    "Tolerance",
    "__version__",
    "analytic_ik",
    "chain_reach",
    "conditioning",
    "finite_difference_jacobian",
    "forward_kinematics",
    "geometric_jacobian",
    "link_frames",
    "manipulability",
    "numerical_solvers",
    "puma560",
    "stanford_arm",
    "ur5",
]

__version__ = "0.1.0"
