"""Compare the inverse kinematics solvers on a paired set of reachable targets.

Wiring only. Every computation is a call into the library.

    uv run python examples/compare_ik_solvers.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from manipulator_kinematics.algorithm import (
    AnalyticIK,
    DampedLeastSquaresIK,
    DampingSchedule,
    JacobianTransposeIK,
    PseudoinverseIK,
    SolverSettings,
    Tolerance,
)
from manipulator_kinematics.algorithm.protocol import IKSolver
from manipulator_kinematics.analysis import (
    convergence_figure,
    format_failure_table,
    format_summary_table,
    save_figure,
    success_figure,
    summarise,
)
from manipulator_kinematics.model import ROBOTS
from manipulator_kinematics.pipeline import perturbed_seeds, run_campaign, sample_targets


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--robots",
        nargs="*",
        default=["puma560", "ur5"],
        choices=sorted(ROBOTS),
        help="which shipped models to run",
    )
    parser.add_argument("--targets", type=int, default=200, help="targets per robot")
    parser.add_argument("--max-iterations", type=int, default=500, help="iteration budget")
    parser.add_argument("--seed", type=int, default=20260731, help="random seed")
    parser.add_argument("--spread", type=float, default=0.4, help="seed offset, in radians")
    parser.add_argument(
        "--damping-schedule",
        default=DampingSchedule.LEVENBERG_MARQUARDT.value,
        choices=[schedule.value for schedule in DampingSchedule],
        help="damping rule used by the damped least squares solver",
    )
    parser.add_argument(
        "--limit-strategy",
        default="active-set",
        choices=["clip", "active-set"],
        help="how a step refused by a joint limit is handled",
    )
    parser.add_argument(
        "--backtracking",
        type=int,
        default=6,
        help="halvings tried when a step fails to reduce the residual, 0 to disable",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="directory the figures are written to",
    )
    parser.add_argument("--dpi", type=int, default=150, help="resolution of the saved figures")
    parser.add_argument(
        "--diagnose-failures",
        action="store_true",
        help="list every unsolved trial with its active bounds and conditioning",
    )
    args = parser.parse_args(argv)

    tolerance = Tolerance(position=1e-6, orientation=1e-6)
    settings = SolverSettings(
        max_iterations=args.max_iterations,
        tolerance=tolerance,
        active_set=args.limit_strategy == "active-set",
        backtracking=args.backtracking,
    )
    schedule = DampingSchedule(args.damping_schedule)

    for name in args.robots:
        robot = ROBOTS[name]()
        rng = np.random.default_rng(args.seed)
        targets = sample_targets(robot, args.targets, rng, margin=0.15)
        seeds = perturbed_seeds(robot, targets, rng, spread=args.spread)

        solvers: list[IKSolver] = [
            JacobianTransposeIK(settings=settings),
            PseudoinverseIK(settings=settings),
            DampedLeastSquaresIK(settings=settings, schedule=schedule),
        ]
        if name == "puma560":
            solvers.append(AnalyticIK(tolerance=tolerance))

        trace = run_campaign(
            robot,
            solvers,
            targets,
            seeds,
            tolerance=tolerance,
            max_iterations=args.max_iterations,
        )
        summaries = summarise(trace)

        print(f"=== {robot.name} ===")
        print(
            f"{len(targets)} reachable targets, seeds offset by {args.spread} rad, "
            f"iteration budget {args.max_iterations}, "
            f"tolerance {tolerance.position:g} m and {tolerance.orientation:g} rad, "
            f"damping schedule {schedule.value}, "
            f"limit strategy {args.limit_strategy}, backtracking {args.backtracking}"
        )
        print(format_summary_table(summaries))
        print()
        if args.diagnose_failures:
            print(format_failure_table(trace, robot.limits))
            print()

        save_figure(
            convergence_figure(trace),
            args.output_dir / f"{name}_convergence.png",
            dpi=args.dpi,
        )
        save_figure(
            success_figure(summaries, robot=robot.name),
            args.output_dir / f"{name}_success.png",
            dpi=args.dpi,
        )

    print(f"figures written to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
