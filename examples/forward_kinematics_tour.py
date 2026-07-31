"""Print the tool pose and Jacobian conditioning of each shipped robot model.

Wiring only. Every computation is a call into the library.

    uv run python examples/forward_kinematics_tour.py
"""

from __future__ import annotations

import argparse

import numpy as np

from manipulator_kinematics.algorithm import (
    conditioning,
    finite_difference_jacobian,
    forward_kinematics,
    geometric_jacobian,
)
from manipulator_kinematics.model import ROBOTS, Robot, chain_reach


def _named_configurations(robot: Robot) -> dict[str, np.ndarray]:
    limits = robot.limits
    return {
        "zero": robot.home(),
        "limit midpoint": np.array([limit.midpoint for limit in limits], dtype=np.float64),
        "quarter turn": np.full(robot.n_joints, 0.25 * np.pi, dtype=np.float64),
    }


def _report(robot: Robot) -> None:
    length = chain_reach(robot)
    print(f"=== {robot.name} ===")
    print(f"joints            : {robot.n_joints}")
    print(f"convention        : {robot.convention.value}")
    print(f"joint types       : {', '.join(t.value for t in robot.joint_types)}")
    print(f"characteristic len: {length:.4f} m")
    print(f"source            : {robot.source}")

    for label, q in _named_configurations(robot).items():
        pose = forward_kinematics(robot, q)
        jacobian = geometric_jacobian(robot, q)
        metrics = conditioning(jacobian, characteristic_length=length)
        gap = float(np.abs(jacobian - finite_difference_jacobian(robot, q)).max())
        print(f"\n  configuration: {label}")
        print(f"    tool position (m)      : {np.array2string(pose[:3, 3], precision=6)}")
        print(f"    tool z axis            : {np.array2string(pose[:3, 2], precision=6)}")
        print(f"    manipulability         : {metrics.manipulability:.6e}")
        print(f"    condition number       : {metrics.condition_number:.6e}")
        print(f"    smallest singular value: {metrics.smallest_singular_value:.6e}")
        print(f"    analytic minus finite difference (max abs): {gap:.3e}")
    print()


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--robots",
        nargs="*",
        default=sorted(ROBOTS),
        choices=sorted(ROBOTS),
        help="which shipped models to report on",
    )
    args = parser.parse_args(argv)

    for name in args.robots:
        _report(ROBOTS[name]())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
