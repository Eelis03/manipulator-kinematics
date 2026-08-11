"""Sweep the PUMA 560 wrist through its singularity and record the response.

Wiring only. Every computation is a call into the library.

    uv run python examples/singularity_scan.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from manipulator_kinematics.analysis import format_scan_table, save_figure, singularity_figure
from manipulator_kinematics.model import puma560
from manipulator_kinematics.pipeline import scan_joint


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=401, help="samples across the sweep")
    parser.add_argument("--span", type=float, default=0.3, help="half-width of the sweep, in rad")
    parser.add_argument("--damping", type=float, default=0.05, help="maximum damping factor")
    parser.add_argument("--epsilon", type=float, default=0.05, help="width of the singular region")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="directory the figure is written to",
    )
    parser.add_argument("--dpi", type=int, default=150, help="resolution of the saved figure")
    args = parser.parse_args(argv)

    robot = puma560()
    base = np.array([0.3, -0.9, 0.6, 0.4, 0.0, 0.2], dtype=np.float64)
    values = np.linspace(-args.span, args.span, args.points)

    scan = scan_joint(
        robot,
        base,
        joint_index=4,
        values=values,
        twist=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        damping=args.damping,
        epsilon=args.epsilon,
    )

    print(f"robot            : {scan.robot}")
    print(f"swept joint      : {scan.joint_index + 1} (the wrist pitch)")
    print(f"held fixed at    : {np.array2string(base, precision=4)}")
    print(f"task velocity    : {np.array2string(scan.twist, precision=1)}")
    print(f"damping schedule : lambda_max {args.damping:g}, epsilon {args.epsilon:g}")
    print()
    print(format_scan_table(scan, stride=max(args.points // 20, 1)))
    print()

    manipulabilities = scan.column("manipulability")
    pseudoinverse = scan.column("pseudoinverse_step_norm")
    damped = scan.column("damped_step_norm")
    centre = int(np.argmin(manipulabilities))

    print(f"minimum manipulability       : {manipulabilities[centre]:.6e}")
    print(f"at joint value               : {scan.column('value')[centre]:.6f} rad")
    print(f"smallest singular value there: {scan.column('smallest_singular_value')[centre]:.6e}")
    print(f"largest condition number     : {scan.column('condition_number').max():.6e}")
    print(f"largest pseudoinverse step   : {pseudoinverse.max():.6e} rad")
    print(f"largest damped step          : {damped.max():.6e} rad")
    print(f"ratio of the peak steps      : {pseudoinverse.max() / damped.max():.3f}")
    print(f"largest per-point step ratio : {(pseudoinverse / damped).max():.3f}")

    path = save_figure(
        singularity_figure(scan), args.output_dir / "puma560_singularity.png", dpi=args.dpi
    )
    print(f"figure written to {path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
