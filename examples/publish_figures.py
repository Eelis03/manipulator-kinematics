"""Regenerate the three figures committed under ``docs/figures``.

Wiring only. Every computation is a call into the library.

The other example scripts draw figures at their own defaults and write them to
``outputs/``, which is ignored by git. This script is the single command that
produces the tracked copies, at the one resolution the repository budget allows,
so the committed files always come from a documented invocation.

    uv run python examples/publish_figures.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from manipulator_kinematics.algorithm import (
    AnalyticIK,
    DampedLeastSquaresIK,
    JacobianTransposeIK,
    PseudoinverseIK,
    SolverSettings,
    Tolerance,
)
from manipulator_kinematics.algorithm.protocol import IKSolver
from manipulator_kinematics.analysis import (
    convergence_figure,
    residual_tail_figure,
    save_figure,
    singularity_figure,
)
from manipulator_kinematics.model import puma560
from manipulator_kinematics.pipeline import (
    Trace,
    perturbed_seeds,
    run_campaign,
    sample_targets,
    scan_joint,
)

SCAN_BASE = np.array([0.3, -0.9, 0.6, 0.4, 0.0, 0.2], dtype=np.float64)
SCAN_TWIST = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)


def _campaign(*, targets: int, max_iterations: int, seed: int, active_set: bool) -> Trace:
    """Run the PUMA 560 campaign under one joint limit strategy."""
    robot = puma560()
    tolerance = Tolerance(position=1e-6, orientation=1e-6)
    settings = SolverSettings(
        max_iterations=max_iterations,
        tolerance=tolerance,
        active_set=active_set,
        backtracking=6 if active_set else 0,
    )
    rng = np.random.default_rng(seed)
    poses = sample_targets(robot, targets, rng, margin=0.15)
    seeds = perturbed_seeds(robot, poses, rng, spread=0.4)
    solvers: list[IKSolver] = [
        JacobianTransposeIK(settings=settings),
        PseudoinverseIK(settings=settings),
        DampedLeastSquaresIK(settings=settings),
        AnalyticIK(tolerance=tolerance),
    ]
    return run_campaign(
        robot,
        solvers,
        poses,
        seeds,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=int, default=200, help="targets in the campaign")
    parser.add_argument("--max-iterations", type=int, default=500, help="iteration budget")
    parser.add_argument("--points", type=int, default=401, help="samples across the sweep")
    parser.add_argument("--seed", type=int, default=20260731, help="random seed")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/figures"),
        help="directory the figures are written to",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=88,
        help="resolution of the saved figures, chosen to fit the repository budget",
    )
    args = parser.parse_args(argv)

    constrained = _campaign(
        targets=args.targets,
        max_iterations=args.max_iterations,
        seed=args.seed,
        active_set=True,
    )
    clipped = _campaign(
        targets=args.targets,
        max_iterations=args.max_iterations,
        seed=args.seed,
        active_set=False,
    )

    scan = scan_joint(
        puma560(),
        SCAN_BASE,
        joint_index=4,
        values=np.linspace(-0.3, 0.3, args.points),
        twist=SCAN_TWIST,
        damping=0.05,
        epsilon=0.05,
    )

    written = [
        save_figure(
            convergence_figure(constrained),
            args.output_dir / "puma560_convergence.png",
            dpi=args.dpi,
        ),
        save_figure(
            singularity_figure(scan),
            args.output_dir / "puma560_singularity.png",
            dpi=args.dpi,
        ),
        save_figure(
            residual_tail_figure(
                [("clipping only", clipped), ("active set", constrained)],
                solver="pseudoinverse",
            ),
            args.output_dir / "puma560_limit_strategy.png",
            dpi=args.dpi,
        ),
    ]

    total = sum(path.stat().st_size for path in written)
    for path in written:
        print(f"{path.stat().st_size:>8} bytes  {path}")
    print(f"{total:>8} bytes  total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
