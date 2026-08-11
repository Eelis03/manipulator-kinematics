# Manipulator Kinematics

Forward and inverse kinematics for serial manipulators, with Jacobian
conditioning and singularity detection.

[![CI](https://github.com/Eelis03/manipulator-kinematics/actions/workflows/ci.yml/badge.svg)](https://github.com/Eelis03/manipulator-kinematics/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

![Pose error against iteration for four inverse kinematics solvers on the PUMA 560 over 200 targets, with the Jacobian transpose crawling above the tolerance line for hundreds of iterations while the pseudoinverse and damped least squares drop below it within ten and the closed form reaches machine precision in one](docs/figures/puma560_convergence.png)

Four solvers, one target set, one figure. The closed form drops to 1e-16 in a
single step. The pseudoinverse and damped least squares reach 1e-6 in under ten
iterations. The Jacobian transpose reaches 1e-3 quickly and then crawls, and most
of its trials are still above the tolerance line when the budget runs out. That
shape is the reason this library exists: which method to use is not a matter of
taste, and the difference is measurable.

## Defining a robot

A robot is a Denavit-Hartenberg table, a convention, and optional joint limits.
Three published parameter sets ship with the library, each carrying its citation
on the model.

```python
import numpy as np

from manipulator_kinematics import DHParameter, JointLimit, Robot, chain_reach, puma560

robot = puma560(tool_offset=0.05625)
print(robot.n_joints, robot.convention.value, f"reach {chain_reach(robot):.4f} m")

scara = Robot(
    name="scara",
    links=(
        DHParameter(d=0.30, theta=0.0, a=0.25, alpha=0.0, limit=JointLimit(-2.6, 2.6)),
        DHParameter(d=0.00, theta=0.0, a=0.20, alpha=np.pi, limit=JointLimit(-2.4, 2.4)),
    ),
)
print(scara.name, scara.n_joints, f"reach {chain_reach(scara):.4f} m")
```

```text
6 standard reach 1.0902 m
scara 2 reach 0.7500 m
```

Both conventions are supported and the choice is a field on the `Robot`, not a
global setting, because published tables use both and evaluating a Craig table
under the Paul arrangement returns a valid pose that is silently wrong. Revolute
and prismatic joints may be mixed in one chain; the shipped Stanford arm has one
of each.

## Solving for a pose

`analytic_ik` returns every closed-form branch for a 6R arm with a spherical
wrist. The iterative solvers implement one `Protocol`, so they are
interchangeable and directly comparable.

```python
from manipulator_kinematics import analytic_ik, forward_kinematics, numerical_solvers

q = np.array([0.4, -1.1, 0.7, -0.5, 0.9, 0.3])
target = forward_kinematics(robot, q)
print(np.array2string(target[:3, 3], precision=6))

branches = analytic_ik(robot, target)
feasible = [branch for branch in branches if branch.feasible]
print(f"{len(branches)} branches, {len(feasible)} inside the joint limits")
print(f"{feasible[0].branch}: {feasible[0].position_error:.3e} m")

damped = numerical_solvers(max_iterations=100)[2]
result = damped.solve(robot, target, np.zeros(6))
print(
    f"{result.solver}: converged={result.converged} "
    f"iterations={result.iterations} residual={result.final_residual:.3e}"
)
```

```text
[0.382443 0.02172  0.052249]
8 branches, 2 inside the joint limits
right/up/non-flip: 9.232e-17 m
damped-least-squares: converged=True iterations=7 residual=1.101e-07
```

Every closed-form branch is verified by running it back through forward
kinematics before it is returned, and the residual is carried on the solution, so
a caller never has to trust the derivation on the strength of the derivation
alone. An arm the derivation was not written for raises `StructureError` naming
the offending parameter rather than returning wrong angles. Every iterative
solver returns the best iterate it saw, always inside the joint limits, together
with the residual history that drew the figure above.

## Measuring distance to a singularity

```python
from manipulator_kinematics import conditioning, geometric_jacobian

metrics = conditioning(geometric_jacobian(robot, q), characteristic_length=chain_reach(robot))
print(f"manipulability    {metrics.manipulability:.6e}")
print(f"condition number  {metrics.condition_number:.4f}")
print(f"smallest sigma    {metrics.smallest_singular_value:.6e}")
print(f"is_singular       {metrics.is_singular}")

wrist_lock = q.copy()
wrist_lock[4] = 0.0
locked = conditioning(
    geometric_jacobian(robot, wrist_lock), characteristic_length=chain_reach(robot)
)
print(f"at q5 = 0         {locked.manipulability:.6e}  {locked.is_singular}")
```

```text
manipulability    3.430033e-02
condition number  7.1951
smallest sigma    2.281307e-01
is_singular       False
at q5 = 0         2.229981e-18  True
```

The `characteristic_length` is not optional decoration. The translation rows of a
geometric Jacobian carry metres and the rotation rows do not, so without dividing
the rotation rows by a length the condition number depends on whether the arm was
measured in metres or millimetres. `chain_reach` is the length used everywhere
here, following Angeles 2007 section 4.9.

## Installation

Requires Python 3.12 or later. CI runs the whole suite on 3.12 and 3.13, on
Linux and on Windows, so the version floor in `pyproject.toml` is a tested claim
rather than a declared one.

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

The package ships a `py.typed` marker, so an installing project gets the
annotations rather than `Any`.

## Results

All numbers below are the output of the commands shown, on Python 3.12 with
NumPy 2.5.1 and SciPy 1.18.0.

### Four solvers on the same 200 targets

`uv run python examples/compare_ik_solvers.py`. Targets are generated by sampling
configurations inside the joint limits with a 15 percent margin and running them
through forward kinematics, so every target is reachable. Seeds are the
generating configuration displaced by a Gaussian with a standard deviation of
0.4 rad and clipped back into the limits. All solvers see the same targets and
the same seeds. The budget is 500 iterations and the tolerance is 1e-6 m and
1e-6 rad on the two halves of the pose error. Median iteration counts are taken
over converged trials only; median errors over all trials; the worst residual is
measured at the configuration actually returned.

PUMA 560:

| solver | solved | median iterations | median position error (m) | median orientation error (rad) | worst residual |
| --- | --- | --- | --- | --- | --- |
| jacobian-transpose | 40/200 | 310 | 4.713e-04 | 5.533e-05 | 1.775e-02 |
| pseudoinverse | 191/200 | 5 | 2.301e-11 | 2.377e-09 | 5.666e-01 |
| damped-least-squares | 200/200 | 6 | 1.219e-09 | 5.838e-09 | 1.156e-06 |
| analytic | 200/200 | 1 | 1.360e-16 | 2.145e-16 | 5.890e-15 |

UR5, which has no spherical wrist and therefore no analytic row:

| solver | solved | median iterations | median position error (m) | median orientation error (rad) | worst residual |
| --- | --- | --- | --- | --- | --- |
| jacobian-transpose | 26/200 | 329 | 1.496e-03 | 1.828e-04 | 2.922e-02 |
| pseudoinverse | 197/200 | 5 | 1.821e-09 | 5.277e-10 | 2.568e-02 |
| damped-least-squares | 200/200 | 6 | 3.058e-09 | 8.913e-10 | 1.106e-06 |

Three findings. The Jacobian transpose converges on 20 percent of PUMA 560
targets and 13 percent of UR5 targets within 500 iterations, which is the linear
convergence rate the method is known for behaving exactly as documented. The
pseudoinverse and damped least squares are equally fast at a median of five and
six iterations, but the pseudoinverse still misses 9 PUMA 560 and 3 UR5 targets,
for reasons taken apart in the next section. The closed form is exact to floating
point and uses its seed only to choose among the eight branches, never to find
them.

### The joint limit box, and what enforcing it properly bought

![Sorted final residual of the pseudoinverse solver over 200 PUMA 560 targets under two joint limit strategies, showing the two curves indistinguishable below the tolerance line and the clipping curve breaking away into a tail of 21 failures where the active set curve holds to 191 targets before breaking into 9](docs/figures/puma560_limit_strategy.png)

The iterate has to stay inside the joint limits. The obvious way to arrange that
is to take the unconstrained step and clip the result back into the box, and it
was how this library started. A clipped minimum-norm step is not generally a
descent direction, so an iterate that reached a bound could sit against it while
the rule kept asking for motion the joint could not make.

The step is now solved subject to the box instead, by the active-set method of
Lawson and Hanson 1974: a joint on a bound whose motion leaves the box is held at
zero and the rule is solved again on the remaining columns of the Jacobian, and a
held joint whose multiplier points back into the box is released. Underneath it,
a step that fails to reduce the residual is halved up to six times before it is
accepted anyway. Both are on by default and both can be turned off:

```bash
uv run python examples/compare_ik_solvers.py --limit-strategy clip --backtracking 0
uv run python examples/compare_ik_solvers.py --limit-strategy active-set --backtracking 6
```

| strategy | line search | PUMA 560 solved | UR5 solved |
| --- | --- | --- | --- |
| clip | no | 179/200 | 195/200 |
| clip | yes | 181/200 | 196/200 |
| active set | no | 189/200 | 196/200 |
| active set | yes | 191/200 | 197/200 |

The figure is what the counts cannot show. Below the tolerance line the two
curves lie on top of each other, so the constrained step costs nothing on the
targets that were never blocked. Above it, clipping breaks away into a tail
spanning six decades while the active set holds twelve more targets under the
line, and the failures that remain fail by less.

Only the pseudoinverse row moves in the table. The Jacobian transpose takes steps
far too short to reach a bound, and damped least squares was already at 200/200
on both arms. A separate consequence is that the recorded trajectory now never
ends above the residual it started from, on any solver, on either arm. That
assertion was written earlier, failed, and was deleted; it is back in
`tests/test_regression.py` and in `tests/test_properties.py`.

The nine PUMA 560 targets the pseudoinverse still misses are not the same kind of
failure. `uv run python examples/compare_ik_solvers.py --diagnose-failures` shows
that eight of the twelve remaining failures across both arms stop with a joint on
a bound at a well conditioned Jacobian, manipulability between 3.7336e-04 and
8.2048e-03: these are genuine constrained local minima, reachable targets that
cannot be reached from that seed without leaving the box on the way. The other
four stop near a singularity, below 6.6887e-06. Escaping a constrained local
minimum is a restart question rather than a step question, so it is answered
outside the update loop. `RestartingIK` wraps any solver and runs it again from
configurations drawn inside the limit box until one attempt converges or the
budget is spent. It cannot lose a target the wrapped solver already solves,
because the first attempt is that call and the search stops at the first attempt
that converges, and until one does it keeps the lowest residual it has seen. What
that leaves, and why the update loop itself was left alone, is recorded in
[docs/design-notes.md](docs/design-notes.md) rather than papered over.

### Eight closed-form branches, exact to round-off

`uv run python examples/analytic_ik_puma560.py`, on the PUMA 560 with `d6` set to
0.05625 m so the tool frame is separated from the wrist centre. For the target
generated by `q = [0.4, -1.1, 0.7, -0.5, 0.9, 0.3]` rad the solver returns all
eight branches, two of which lie inside the published joint limits:

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

Over 200 poses drawn inside the joint limits the solver returned exactly eight
exact branches for every pose. The worst position error of the best branch was
4.8660e-15 m and the worst orientation error 2.4441e-16 rad, both at the level of
floating point round-off.

### What a singularity does to each update rule

![Three stacked panels sweeping the PUMA 560 wrist pitch through zero, showing manipulability and the smallest singular value collapsing to the floating point floor, the condition number spiking to 1e17, and the pseudoinverse step rising to 257 rad while the damped step stays under 3.8 rad and the Jacobian transpose step stays flat](docs/figures/puma560_singularity.png)

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

Manipulability falls from 1.764e-02 at the ends of the sweep to the floating
point floor at `q5 = 0`, and the condition number rises from 1.3e+01 to 2.7e+17,
so either index locates the singularity. The pseudoinverse asks for a step of
257 rad just outside the singularity while the damped rule with a maximum damping
factor of 0.05 never exceeds 3.74 rad. The bottom panel also shows why the step
magnitude is the wrong detector: at the singularity itself the truncated
pseudoinverse step collapses to 0.73 rad, because the unreachable direction is
discarded entirely, so the spike has a notch in it. The smallest singular value
has no such notch.

### Consistency checks

`uv run python examples/forward_kinematics_tour.py`. The analytic geometric
Jacobian and a central finite difference of forward kinematics with a step of
1e-6 agree to within 1.2e-10 across all three arms and all reported
configurations. The PUMA 560 zero configuration places the tool at
`(0.4521, -0.15005, 0.4318)` m, which is `(a2 + a3, -(d2 + d3), d4)` read
straight off the parameter table, and it is singular with a manipulability of
3.795e-19.

## Reproducing this page

Every number above comes from one of these commands.

```bash
uv run python examples/forward_kinematics_tour.py
uv run python examples/analytic_ik_puma560.py
uv run python examples/compare_ik_solvers.py
uv run python examples/compare_ik_solvers.py --diagnose-failures
uv run python examples/singularity_scan.py
```

Every script takes `--help`. `compare_ik_solvers.py` and `singularity_scan.py`
draw as they go and write into `outputs/`, which is not tracked.

The three figures on this page are tracked, under `docs/figures`. They are
snapshots, regenerated by one command:

```bash
uv run python examples/publish_figures.py
```

which writes them at 88 dpi, the resolution chosen so the three together stay
inside the 250 KB the repository budgets for tracked images. CI does not compare
the committed files against freshly drawn ones, byte for byte or otherwise,
because matplotlib output is not byte reproducible across platforms or across
library versions. What CI does check is that every figure function still runs and
still produces the artists it claims, which is what `tests/test_analysis.py`
asserts.

The checks:

```bash
uv run pytest --cov=src/manipulator_kinematics --cov-report=term-missing
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

142 tests in about 20 seconds, at 100 percent statement coverage of the package.
CI runs the same command on Linux and Windows with `--cov-fail-under=98`.

The suite is one file per layer, plus two that check the repository rather than
the mathematics.

| File | Tests | What it establishes |
| --- | --- | --- |
| `tests/test_properties.py` | 93 | Mathematical facts. Every transform is orthonormal with determinant one. The analytic Jacobian matches a central finite difference on all three arms. Forward kinematics composed with the closed-form inverse returns the original pose on every branch. The two DH conventions are checked against each other by building the same chain twice. The constrained step serves the linear model better than clipping does, solves more targets than clipping does, and never ends a run above where it began. A restart never loses a solve the wrapped solver made and never returns a worse residual than the seed it was given. |
| `tests/test_pipeline.py` | 11 | Target generation, campaigns, and the joint sweep, in process rather than through a script, so a failure names the function. |
| `tests/test_analysis.py` | 23 | Summaries against hand computations, and figures read as artist trees rather than pixels, because the artists are reproducible across platforms and the pixels are not. |
| `tests/test_regression.py` | 5 | Recorded behaviour pinned against `tests/data/reference.json`: a 25-point trajectory to 1e-12, three eight-branch solution sets to 1e-9, and two 25-target campaigns by solved count, iteration count and returned configuration. Regenerated deliberately with `uv run python tests/test_regression.py`. |
| `tests/test_integration.py` | 8 | Every script in `examples/` runs as a subprocess under reduced settings, exits zero, prints, and writes its promised figures into a temporary directory. One test fails if a new example is added without being listed. |
| `tests/test_packaging.py` | 2 | The `py.typed` marker exists inside the package directory and the wheel is configured to ship it. |

## Inside the package

Five layers, each depending only on the ones above it.

| Module | Responsibility |
| --- | --- |
| `model/transforms.py` | SE(3) helpers: elementary rotations, the skew matrix, the closed-form pose inverse, and the twist-valued pose error. |
| `model/joints.py` | Joint limits, limit clamping, and limit-box sampling. |
| `model/dh.py` | The `DHParameter` and `Robot` dataclasses, both DH conventions, and the reach bound used as a characteristic length. |
| `model/robots.py` | Published parameter sets for the PUMA 560, the UR5, and the Stanford arm, each with its citation. |
| `algorithm/forward.py` | Forward kinematics and the cumulative link frames. |
| `algorithm/jacobian.py` | The geometric Jacobian, and a finite-difference Jacobian used only to validate it. |
| `algorithm/conditioning.py` | Manipulability, condition number, and smallest singular value, on a dimensionally homogenised Jacobian. |
| `algorithm/protocol.py` | The `IKSolver` protocol, the `IKResult` record, and the convergence tolerance. |
| `algorithm/analytic.py` | Closed-form inverse kinematics for a 6R arm with a spherical wrist, plus the structure check that guards it. |
| `algorithm/numerical.py` | The three update rules as free functions, the box-constrained active set that wraps any of them, and the three iterative solvers. |
| `algorithm/restart.py` | The multistart wrapper: any solver run again from fresh configurations while it keeps failing. |
| `pipeline/trace.py` | The trace dataclasses: targets, trials, campaigns, and singularity scans. |
| `pipeline/runner.py` | Target and seed generation, the solver campaign, and the joint sweep. |
| `analysis/metrics.py` | Per-solver summaries, the failure diagnosis, and the fixed-width text tables. |
| `analysis/figures.py` | Convergence, success, residual tail, and singularity figures, drawn on the Agg canvas without `pyplot`. |
| `examples/` | Five wiring scripts. Each parses flags, calls the library, and prints or saves. No logic. |

The model layer performs no input or output. The algorithm layer draws nothing.
The pipeline layer decides nothing about what counts as a good result. The
analysis layer calls no solver, so any figure can be redrawn from a recorded
trace.

The alternatives that were considered and rejected, and the limitations that
remain, are in [docs/design-notes.md](docs/design-notes.md).

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
- C. L. Lawson and R. J. Hanson, *Solving Least Squares Problems*, Prentice-Hall,
  1974. ISBN 978-0-13-822585-0. Chapter 23, the active-set method for a
  least-squares problem with bounds, which is how the joint limit box is imposed
  on every iterative step here.
- P. Beeson and B. Ames, "TRAC-IK: an open-source library for improved solving of
  generic inverse kinematics", IEEE-RAS International Conference on Humanoid
  Robots, 2015, pages 928 to 935. doi:10.1109/HUMANOIDS.2015.7363472. Restarting
  a failed solve from configurations drawn inside the joint limit box.
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
  avoidance, which is empty on a non-redundant arm.
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
| [pytest-cov](https://pytest-cov.readthedocs.io/) | Coverage measurement, development only. | MIT |
| [Ruff](https://docs.astral.sh/ruff/) | Linting and import ordering, development only. | MIT |
| [mypy](https://mypy-lang.org/) | Static type checking in strict mode over the package, the tests, and the examples, development only. | MIT |

## License

Released under the MIT license. See [LICENSE](LICENSE).
