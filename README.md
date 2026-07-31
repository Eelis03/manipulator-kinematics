# Manipulator Kinematics

Forward and inverse kinematics for serial manipulators with Jacobian conditioning and singularity detection.

[![CI](https://github.com/Eelis03/manipulator-kinematics/actions/workflows/ci.yml/badge.svg)](https://github.com/Eelis03/manipulator-kinematics/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## Overview

This library computes where the tool of a serial manipulator is, given its joint
values, and what joint values put the tool at a requested pose. It covers both
Denavit-Hartenberg conventions, chains mixing revolute and prismatic joints, a
closed-form solution for the 6R arm with a spherical wrist, three iterative
solvers behind one interface, and the singular value metrics that say how close a
configuration is to a singularity. It is aimed at anyone building or evaluating
manipulator control software who needs a reference implementation they can read
and check, rather than a black box.

Three published parameter sets ship with the library: the Unimation PUMA 560, the
Universal Robots UR5, and the Stanford manipulator. Each carries its source in a
field on the model.

## Problem

A serial manipulator is a chain of rigid links. Forward kinematics is a
composition of homogeneous transforms and is unambiguous. Inverse kinematics is
not: the map from joint values to tool poses is nonlinear, many to one, and not
onto, so a requested pose can have no solution, a finite set of solutions, or a
continuum of them. The practical questions are therefore:

1. Which arms admit a closed-form solution, and what exactly does that solution
   assume about the geometry?
2. For arms that do not, how should the linear system `J dq = e` be inverted at
   each step, given that `J` loses rank at singular configurations where the
   inverse does not exist and the pseudoinverse is unbounded in its neighbourhood?
3. How is the distance to a singularity measured, given that the Jacobian of a
   six-degree-of-freedom arm mixes rows with units of metres and rows with units
   of radians, so its singular values are not directly comparable?

This repository answers all three and measures the answers on two arms.

## Approach

Forward kinematics composes one 4x4 transform per link. Both the standard
arrangement of Denavit and Hartenberg 1955, as tabulated by Paul 1981, and the
modified arrangement of Craig 1986 are implemented, because published parameter
tables use both and mixing them silently gives wrong answers. The convention is
carried on the robot, not chosen globally, and the two are checked against each
other in the test suite by building the same chain twice.

Closed-form inverse kinematics uses kinematic decoupling, following Paul 1981 and
Siciliano et al. 2009 section 2.12.2: when the last three axes intersect, the
intersection point depends only on the first three joints, which splits the
six-dimensional problem into a three-dimensional position problem with four
postures and a three-dimensional orientation problem with two, giving eight
solutions. The derivation is written for the PUMA 560 structure and the library
checks that structure before using it, so an arm it was not derived for fails
with a message naming the offending parameter rather than returning wrong angles.

Iterative inverse kinematics is implemented behind a `Protocol`, so the three
methods are interchangeable and directly comparable: Jacobian transpose with the
optimal step of Buss 2009, the Moore-Penrose pseudoinverse from a truncated
singular value decomposition, and damped least squares with a choice of three
damping schedules. The default schedule is the residual-scaled Levenberg-Marquardt
rule of Sugihara 2011, chosen because it is the only one of the three that is both
bounded at a singularity and able to reach a tight tolerance. Singularity metrics
are the Yoshikawa manipulability index, the condition number, and the smallest
singular value, all computed from a singular value decomposition of a Jacobian
whose rotation rows have been divided by a characteristic length so the two blocks
are dimensionally comparable.

The alternatives that were considered and not chosen are recorded in
[docs/design-notes.md](docs/design-notes.md).

## Installation

Requires Python 3.12 or later.

```bash
git clone https://github.com/Eelis03/manipulator-kinematics.git
cd manipulator-kinematics
uv sync
```

Using pip instead of uv:

```bash
python -m venv .venv
.venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Usage

```python
import numpy as np

from manipulator_kinematics import (
    analytic_ik,
    chain_reach,
    conditioning,
    forward_kinematics,
    geometric_jacobian,
    numerical_solvers,
    puma560,
)

robot = puma560(tool_offset=0.05625)
q = np.array([0.4, -1.1, 0.7, -0.5, 0.9, 0.3])
target = forward_kinematics(robot, q)

jacobian = geometric_jacobian(robot, q)
metrics = conditioning(jacobian, characteristic_length=chain_reach(robot))
print(f"manipulability {metrics.manipulability:.6e}")
print(f"condition number {metrics.condition_number:.4f}")

branches = analytic_ik(robot, target)
worst = max(branch.position_error for branch in branches)
print(f"closed-form branches {len(branches)}, worst error {worst:.3e} m")

damped = numerical_solvers(max_iterations=100)[2]
result = damped.solve(robot, target, np.zeros(6))
print(f"{result.solver}: converged={result.converged} iterations={result.iterations}")
```

which prints

```text
manipulability 3.430033e-02
condition number 7.1951
closed-form branches 8, worst error 2.420e-16 m
damped-least-squares: converged=True iterations=7
```

Runnable examples live in `examples/`:

```bash
uv run python examples/forward_kinematics_tour.py
uv run python examples/analytic_ik_puma560.py
uv run python examples/compare_ik_solvers.py
uv run python examples/singularity_scan.py
```

Every script takes `--help`. The two that draw figures write them to `outputs/`
by default, changeable with `--output-dir`.

## Results

All numbers below are the output of the commands shown. They were produced on
Python 3.12 with NumPy 2.5.1 and SciPy 1.18.0.

### Solver comparison

`uv run python examples/compare_ik_solvers.py`, which solves 200 targets per arm.
Targets are generated by sampling configurations inside the joint limits with a
15 percent margin and running them through forward kinematics, so every target is
reachable. Seeds are the generating configuration displaced by a Gaussian with a
standard deviation of 0.4 rad and clipped back into the limits. All solvers see
the same targets and the same seeds. The budget is 500 iterations and the
tolerance is 1e-6 m and 1e-6 rad on the two halves of the pose error. Damped least
squares uses its default residual-scaled schedule; the other two schedules are
selectable with `--damping-schedule` and are compared in the design notes. Median
iteration counts are taken over converged trials only; median errors are taken
over all trials.

PUMA 560:

| solver | solved | median iterations | median position error (m) | median orientation error (rad) | worst residual |
| --- | --- | --- | --- | --- | --- |
| jacobian-transpose | 40/200 | 310 | 4.713e-04 | 5.533e-05 | 1.775e-02 |
| pseudoinverse | 179/200 | 5 | 1.850e-11 | 5.411e-09 | 2.796e+00 |
| damped-least-squares | 200/200 | 6 | 1.137e-09 | 4.996e-09 | 1.228e-06 |
| analytic | 200/200 | 1 | 1.360e-16 | 2.145e-16 | 5.890e-15 |

UR5, which has no spherical wrist and therefore no analytic row:

| solver | solved | median iterations | median position error (m) | median orientation error (rad) | worst residual |
| --- | --- | --- | --- | --- | --- |
| jacobian-transpose | 26/200 | 329 | 1.496e-03 | 1.828e-04 | 2.922e-02 |
| pseudoinverse | 195/200 | 5 | 2.259e-09 | 1.161e-09 | 5.214e-01 |
| damped-least-squares | 200/200 | 6 | 3.074e-09 | 1.149e-09 | 1.106e-06 |

Three findings. First, the Jacobian transpose reaches 1e-3 quickly and then
crawls, converging on 20 percent of PUMA 560 targets and 13 percent of UR5 targets
within 500 iterations, which matches the linear convergence rate the method is
known for. Second, the pseudoinverse and damped least squares are equally fast at
a median of five and six iterations, but the pseudoinverse leaves 21 PUMA 560
targets and 5 UR5 targets unsolved with residuals as large as 2.8, because
clipping an unconstrained pseudoinverse step back into the joint limit box
produces a direction that is no longer a descent direction. The residual-scaled
damping shortens the step instead of truncating it, and solved every target on
both arms. Third, the closed form is exact to floating point, and it uses its seed
only to choose among the eight branches, never to find them.

### Closed-form solution structure

`uv run python examples/analytic_ik_puma560.py`, on the PUMA 560 with `d6` set to
0.05625 m so the tool frame is separated from the wrist centre.

For the target generated by `q = [0.4, -1.1, 0.7, -0.5, 0.9, 0.3]` rad, the solver
returns all eight branches, two of which lie inside the published joint limits:

```text
                branch  feasible   pos err (m)  rot err (rad)  q (rad)
----------------------------------------------------------------------
     right/up/non-flip      True    9.2316e-17     1.1289e-16  [ 0.4 -1.1  0.7 -0.5  0.9  0.3]
         right/up/flip      True    8.8318e-17     1.7790e-16  [ 0.4    -1.1     0.7     2.6416 -0.9    -2.8416]
   right/down/non-flip     False    1.5011e-16     2.2386e-16  [ 0.4     1.1261  2.5355 -1.3658  2.7479 -1.3767]
       right/down/flip     False    1.5155e-16     3.0809e-16  [ 0.4     1.1261  2.5355  1.7758 -2.7479  1.7649]
      left/up/non-flip     False    7.4734e-17     3.2985e-16  [ 2.7943  2.0155  0.7     0.0658  2.9943 -2.4399]
          left/up/flip     False    8.0921e-17     6.4406e-16  [ 2.7943  2.0155  0.7    -3.0757 -2.9943  0.7017]
    left/down/non-flip     False    2.4120e-16     7.8961e-17  [ 2.7943 -2.0416  2.5355  3.1306  1.067   0.6419]
        left/down/flip     False    2.4197e-16     1.7893e-16  [ 2.7943 -2.0416  2.5355 -0.011  -1.067  -2.4997]
```

Over 200 poses drawn inside the joint limits, the solver returned exactly eight
exact branches for every pose. The worst position error of the best branch was
4.8660e-15 m and the worst orientation error 2.4441e-16 rad, both at the level of
floating point round-off.

### Singularity response

`uv run python examples/singularity_scan.py`, sweeping the PUMA 560 wrist pitch
`q5` from -0.3 rad to 0.3 rad in 401 steps about the configuration
`[0.3, -0.9, 0.6, 0.4, 0.0, 0.2]` rad, with each update rule asked to serve the
same unit angular velocity about the world z axis.

| quantity | value |
| --- | --- |
| minimum manipulability | 9.205714e-19 |
| smallest singular value there | 6.309454e-18 |
| largest condition number | 2.713807e+17 |
| largest pseudoinverse step | 2.573016e+02 rad |
| largest damped step | 3.742145e+00 rad |
| ratio of the peak steps | 68.758 |
| largest step ratio at a single point | 353.440 |

Manipulability falls from 1.764e-02 at the ends of the sweep to the floating point
floor at `q5 = 0`, and the condition number rises from 1.3e+01 to 2.7e+17, so
either index locates the singularity. The pseudoinverse asks for a step of 257 rad
just outside the singularity, while the damped rule with a maximum damping factor
of 0.05 never exceeds 3.74 rad, a factor of 69 on the peaks. At the singularity
itself the truncated pseudoinverse step collapses to 0.73 rad because the
unreachable direction is discarded entirely, which is why the smallest singular
value rather than the step magnitude is the reliable detector.

### Consistency checks

`uv run python examples/forward_kinematics_tour.py`. The analytic geometric
Jacobian and a central finite difference of forward kinematics with a step of
1e-6 agree to within 1.2e-10 across all three arms and all reported
configurations. The PUMA 560 zero configuration places the tool at
`(0.4521, -0.15005, 0.4318)` m, which is `(a2 + a3, -(d2 + d3), d4)` read straight
off the parameter table, and it is singular with a manipulability of 3.795e-19.

## Architecture

Five layers, each depending only on the ones above it in this table.

| Module | Responsibility |
| --- | --- |
| `src/manipulator_kinematics/model/transforms.py` | SE(3) helpers: elementary rotations, the skew matrix, the closed-form pose inverse, and the twist-valued pose error. |
| `src/manipulator_kinematics/model/joints.py` | Joint limits, limit clamping, and limit-box sampling. |
| `src/manipulator_kinematics/model/dh.py` | The `DHParameter` and `Robot` dataclasses, both DH conventions, and the reach bound used as a characteristic length. |
| `src/manipulator_kinematics/model/robots.py` | Published parameter sets for the PUMA 560, the UR5, and the Stanford arm, each with its citation. |
| `src/manipulator_kinematics/algorithm/forward.py` | Forward kinematics and the cumulative link frames. |
| `src/manipulator_kinematics/algorithm/jacobian.py` | The geometric Jacobian, and a finite-difference Jacobian used only to validate it. |
| `src/manipulator_kinematics/algorithm/conditioning.py` | Manipulability, condition number, and smallest singular value, on a dimensionally homogenised Jacobian. |
| `src/manipulator_kinematics/algorithm/protocol.py` | The `IKSolver` protocol, the `IKResult` record, and the convergence tolerance. |
| `src/manipulator_kinematics/algorithm/analytic.py` | Closed-form inverse kinematics for a 6R arm with a spherical wrist, plus the structure check that guards it. |
| `src/manipulator_kinematics/algorithm/numerical.py` | The three update rules as free functions, and the three iterative solvers that wrap them. |
| `src/manipulator_kinematics/pipeline/trace.py` | The trace dataclasses: targets, trials, campaigns, and singularity scans. |
| `src/manipulator_kinematics/pipeline/runner.py` | Target and seed generation, the solver campaign, and the joint sweep. |
| `src/manipulator_kinematics/analysis/metrics.py` | Per-solver summary statistics and the fixed-width text tables. |
| `src/manipulator_kinematics/analysis/figures.py` | Convergence, success rate, and singularity figures, drawn on the Agg canvas without `pyplot`. |
| `examples/` | Four wiring scripts, each of which parses flags, calls the library, and prints or saves. No logic. |

The model layer performs no input or output. The algorithm layer draws nothing.
The pipeline layer decides nothing about what counts as a good result. The
analysis layer calls no solver, so any figure can be redrawn from a recorded
trace.

## Testing

```bash
uv run pytest
uv run ruff check .
uv run mypy
```

The suite has three tiers: property and invariant tests covering the mathematics,
regression tests pinning recorded behaviour, and integration tests running each
example script under a reduced iteration count.

Tier one, `tests/test_properties.py`, 54 tests. Every link transform and composed
pose is orthonormal with determinant one. The analytic Jacobian matches a central
finite difference of forward kinematics to 1e-7 on all three arms. Forward
kinematics composed with the closed-form inverse returns the original pose to
below 1e-9 on every branch. Hand-computed reference poses at the zero
configuration are checked against the parameter tables for all three arms, and the
prismatic joint of the Stanford arm is checked to translate the tool by exactly
its own displacement. The two DH conventions are checked against each other by
building the same chain twice.

Tier two, `tests/test_regression.py`, 5 tests against `tests/data/reference.json`.
A 25-point joint-space trajectory pins forward kinematics to 1e-12 and the three
conditioning metrics to 1e-9 relative. Three poses pin the full eight-branch
closed-form solution sets to 1e-9. Two 25-target solver campaigns pin the solved
counts exactly, the per-trial iteration counts to within one, and the returned
configurations to 1e-5. The file is regenerated with
`uv run python tests/test_regression.py`, which is a deliberate act rather than an
automatic one.

Tier three, `tests/test_integration.py`, 7 tests. Every script in `examples/` is
launched as a subprocess with reduced target counts, iteration budgets, and sweep
resolutions, and must exit zero, print output, and write its promised figures into
a temporary directory. One test asserts that no example script has been left out
of the list, so a new example cannot silently escape coverage.

The full suite of 66 tests runs in about 9 seconds.

## References

### Algorithms

- J. Denavit and R. S. Hartenberg, "A kinematic notation for lower-pair mechanisms
  based on matrices", Journal of Applied Mechanics 22(2), 1955, pages 215 to 221.
  doi:10.1115/1.4011045. The original parameter convention.
- R. P. Paul, *Robot Manipulators: Mathematics, Programming and Control*, MIT
  Press, 1981. ISBN 978-0-262-16082-7. The standard convention as implemented
  here, and the original statement of kinematic decoupling.
- J. J. Craig, *Introduction to Robotics: Mechanics and Control*, Addison-Wesley,
  1986. ISBN 978-0-201-09528-9. The modified convention as implemented here.
- B. Siciliano, L. Sciavicco, L. Villani and G. Oriolo, *Robotics: Modelling,
  Planning and Control*, Springer, 2009. doi:10.1007/978-1-84628-642-1. Section
  3.1 for the geometric Jacobian construction and section 2.12.2 for the
  decoupled inverse kinematics used here.
- D. E. Whitney, "Resolved motion rate control of manipulators and human
  prostheses", IEEE Transactions on Man-Machine Systems 10(2), 1969, pages 47 to
  53. doi:10.1109/TMMS.1969.299896. The pseudoinverse update.
- W. A. Wolovich and H. Elliott, "A computational technique for inverse
  kinematics", IEEE Conference on Decision and Control, 1984, pages 1359 to 1363.
  doi:10.1109/CDC.1984.272258. The Jacobian transpose update.
- S. R. Buss, "Introduction to inverse kinematics with Jacobian transpose,
  pseudoinverse and damped least squares methods", 2009.
  https://mathweb.ucsd.edu/~sbuss/ResearchWeb/ikmethods/ikmethods.pdf . The
  optimal step length for the transpose update.
- Y. Nakamura and H. Hanafusa, "Inverse kinematic solutions with singularity
  robustness for robot manipulator control", ASME Journal of Dynamic Systems,
  Measurement and Control 108(3), 1986, pages 163 to 171. doi:10.1115/1.3143764.
  Damped least squares for manipulators.
- C. W. Wampler, "Manipulator inverse kinematic solutions based on vector
  formulations and damped least-squares methods", IEEE Transactions on Systems,
  Man and Cybernetics 16(1), 1986, pages 93 to 101. doi:10.1109/TSMC.1986.289285.
  The independent derivation of the same regularisation.
- S. Chiaverini, B. Siciliano and O. Egeland, "Review of the damped least-squares
  inverse kinematics with experiments on an industrial robot manipulator", IEEE
  Transactions on Control Systems Technology 2(2), 1994, pages 123 to 134.
  doi:10.1109/87.294335. The singular-region damping schedule.
- T. Sugihara, "Solvability-unconcerned inverse kinematics by the
  Levenberg-Marquardt method", IEEE Transactions on Robotics 27(5), 2011, pages
  984 to 991. doi:10.1109/TRO.2011.2148230. The residual-scaled damping schedule,
  which is the default here.
- T. Yoshikawa, "Manipulability of robotic mechanisms", International Journal of
  Robotics Research 4(2), 1985, pages 3 to 9. doi:10.1177/027836498500400201. The
  manipulability index.
- J. K. Salisbury and J. J. Craig, "Articulated hands: force control and kinematic
  issues", International Journal of Robotics Research 1(1), 1982, pages 4 to 17.
  doi:10.1177/027836498200100102. The condition number as a kinematic
  conditioning index.
- J. Angeles, *Fundamentals of Robotic Mechanical Systems*, third edition,
  Springer, 2007. doi:10.1007/978-0-387-34580-2. Section 4.9 on the
  characteristic length used to homogenise a Jacobian with mixed units.

The following are cited in [docs/design-notes.md](docs/design-notes.md) as
alternatives that were considered and not implemented.

- H. Y. Lee and C. G. Liang, "Displacement analysis of the general spatial
  7-link 7R mechanism", Mechanism and Machine Theory 23(3), 1988, pages 219 to
  226. doi:10.1016/0094-114X(88)90107-3. The degree-sixteen general solution.
- M. Raghavan and B. Roth, "Inverse kinematics of the general 6R manipulator and
  related linkages", ASME Journal of Mechanical Design 115(3), 1993, pages 502 to
  508. doi:10.1115/1.2919218. The resultant-based derivation of the same result.
- A. Liegeois, "Automatic supervisory control of the configuration and behavior of
  multibody mechanisms", IEEE Transactions on Systems, Man and Cybernetics 7(12),
  1977, pages 868 to 871. doi:10.1109/TSMC.1977.4309644. Null-space joint limit
  avoidance.
- K. P. Hawkins, "Analytic inverse kinematics for the Universal Robots UR-5 and
  UR-10 arms", Georgia Institute of Technology technical report, 2013.
  https://hdl.handle.net/1853/50782 . The arm-specific UR5 closed form.

### Robot parameters

- P. I. Corke and B. Armstrong-Helouvry, "A search for consensus among model
  parameters reported for the PUMA 560 robot", IEEE International Conference on
  Robotics and Automation, 1994, pages 1608 to 1613.
  doi:10.1109/ROBOT.1994.351360. PUMA 560 link parameters.
- P. I. Corke, "A robotics toolbox for MATLAB", IEEE Robotics and Automation
  Magazine 3(1), 1996, pages 24 to 32. doi:10.1109/100.486658. PUMA 560 joint
  limits and the Stanford arm numeric values.
- Universal Robots, "DH parameters for calculations of kinematics and dynamics".
  https://www.universal-robots.com/articles/ur/application-installation/dh-parameters-for-calculations-of-kinematics-and-dynamics/
  UR5 link parameters.
- P. M. Kebria, S. Al-Wais, H. Abdi and S. Nahavandi, "Kinematic and dynamic
  modelling of UR5 manipulator", IEEE International Conference on Systems, Man
  and Cybernetics, 2016, pages 4229 to 4234. doi:10.1109/SMC.2016.7844896. The
  cross-check on the UR5 parameters.

### Figures

- M. Okabe and K. Ito, "Color universal design: how to make figures and
  presentations that are friendly to colorblind people", 2008.
  https://jfly.uni-koeln.de/color/ . The categorical colours used in every figure.

### Dependencies

| Dependency | Purpose | License |
| --- | --- | --- |
| [NumPy](https://numpy.org/) | Array arithmetic, linear algebra, and the seeded random generators that make every campaign reproducible. | BSD 3-Clause |
| [SciPy](https://scipy.org/) | `scipy.linalg.svd` and `scipy.linalg.svdvals` for the singular value decompositions behind the conditioning metrics and the pseudoinverse, `scipy.linalg.solve` for the damped normal equations, and `scipy.spatial.transform.Rotation` for the rotation matrix logarithm used in the pose error. | BSD 3-Clause |
| [Matplotlib](https://matplotlib.org/) | Figure drawing through the Agg canvas, without `pyplot` and therefore without any global state or interactive backend. | Matplotlib License, a BSD-style license |
| [pytest](https://pytest.org/) | Test runner, development only. | MIT |
| [Ruff](https://docs.astral.sh/ruff/) | Linting and import ordering, development only. | MIT |
| [mypy](https://mypy-lang.org/) | Static type checking in strict mode over the package, the tests, and the examples, development only. | MIT |

## License

Released under the MIT license. See [LICENSE](LICENSE).
