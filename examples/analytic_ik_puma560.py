"""Enumerate the closed-form inverse kinematics branches of the PUMA 560.

Wiring only. Every computation is a call into the library.

    uv run python examples/analytic_ik_puma560.py
"""

from __future__ import annotations

import argparse

import numpy as np

from manipulator_kinematics.algorithm import analytic_ik, forward_kinematics
from manipulator_kinematics.model import puma560, sample_within_limits


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260731, help="random seed")
    parser.add_argument("--poses", type=int, default=200, help="poses in the residual sweep")
    parser.add_argument(
        "--tool-offset",
        type=float,
        default=0.05625,
        help="value of d6 in metres, which separates the wrist centre from the tool",
    )
    args = parser.parse_args(argv)

    robot = puma560(tool_offset=args.tool_offset)
    rng = np.random.default_rng(args.seed)

    reference = np.array([0.4, -1.1, 0.7, -0.5, 0.9, 0.3], dtype=np.float64)
    target = forward_kinematics(robot, reference)
    print(f"robot            : {robot.name}")
    print(f"reference q (rad): {np.array2string(reference, precision=4)}")
    print(f"target position  : {np.array2string(target[:3, 3], precision=6)}")
    print()

    solutions = analytic_ik(robot, target)
    header = (
        f"{'branch':>22}  {'feasible':>8}  {'pos err (m)':>12}  "
        f"{'rot err (rad)':>13}  q (rad)"
    )
    print(header)
    print("-" * len(header))
    for solution in solutions:
        print(
            f"{solution.branch:>22}  {solution.feasible!s:>8}  "
            f"{solution.position_error:12.4e}  {solution.orientation_error:13.4e}  "
            f"{np.array2string(solution.q, precision=4, suppress_small=True)}"
        )
    print()
    print(f"branches returned    : {len(solutions)}")
    print(f"branches within limit: {sum(s.feasible for s in solutions)}")
    print()

    counts: list[int] = []
    worst_position = 0.0
    worst_orientation = 0.0
    for _ in range(args.poses):
        q = sample_within_limits(robot.limits, rng, margin=0.05)
        pose = forward_kinematics(robot, q)
        branches = analytic_ik(robot, pose)
        exact = [
            branch
            for branch in branches
            if branch.position_error < 1e-9 and branch.orientation_error < 1e-9
        ]
        counts.append(len(exact))
        if branches:
            worst_position = max(worst_position, min(b.position_error for b in branches))
            worst_orientation = max(worst_orientation, min(b.orientation_error for b in branches))

    print(f"sweep over {args.poses} reachable poses drawn inside the joint limits")
    print(f"  exact branches per pose, minimum: {min(counts)}")
    print(f"  exact branches per pose, mean   : {np.mean(counts):.3f}")
    print(f"  worst best-branch position error: {worst_position:.4e} m")
    print(f"  worst best-branch rotation error: {worst_orientation:.4e} rad")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
