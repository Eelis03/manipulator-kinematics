# Design notes for Manipulator Kinematics

## Method selection

### Both Denavit-Hartenberg conventions, carried on the robot

The standard convention (Denavit and Hartenberg 1955, in the arrangement of Paul
1981) and the modified convention (Craig 1986) describe the same geometry with
the link length and twist attached to different frames. The two produce different
transforms from the same four numbers, so a table transcribed from one source and
evaluated under the other convention is silently wrong: forward kinematics still
returns a valid pose, the Jacobian is still orthonormal in its rotation rows, and
nothing raises. Published parameter sets use both, so supporting only one would
have forced every user to convert tables by hand, which is exactly where the
mistake happens.

The convention is therefore a field on the `Robot` rather than a module-level
setting, and the Jacobian construction branches on it: in the standard convention
the axis of joint `i` is the z axis of frame `i-1`, in the modified convention it
is the z axis of frame `i`. The test suite builds the same four-link chain under
both conventions and asserts that the tool poses agree over twenty random
configurations, which is the only check that catches a sign error present in one
convention and not the other.

The assumption is that the parameter table itself is correct and complete. The
library validates the structural facts a closed form needs, but it cannot detect
that a transcribed link length is wrong.

### Kinematic decoupling for the closed form

Paul 1981, and Siciliano et al. 2009 section 2.12.2. When the last three joint
axes intersect at a point, that point is fixed by the first three joints alone,
because the last three contribute only rotations about axes through it. The
position of the wrist centre is `p - d6 R e_z`, which is computable from the
request, so the position problem is solved first for four arm postures, and the
orientation problem afterwards as a ZYZ Euler factorisation with two solutions
each, giving eight.

The derivation is specific: it assumes `a1 = 0`, `alpha1 = +pi/2`, `alpha2 = 0`,
`alpha3 = -pi/2`, `a4 = a5 = a6 = 0`, `d5 = 0`, `alpha4 = +pi/2`,
`alpha5 = -pi/2` and `alpha6 = 0`, which is the PUMA 560 structure and the
canonical form of a 6R arm with a spherical wrist. Rather than leave that
assumption in a comment, `assert_spherical_wrist` checks all eleven parameters and
raises a `StructureError` naming the first that does not match. This is why the
UR5 is rejected by name: its wrist axes are mutually perpendicular but do not
intersect, its `alpha3` is zero rather than `-pi/2`, and its closed form is a
different derivation.

Two implementation details are worth stating. First, the arm and wrist angles are
computed as raw values and then mapped to the representative modulo a full turn
that lies inside the declared joint limit, so a branch that is physically
reachable at `q + 2 pi` is not reported as infeasible. Second, every branch is
verified by running it back through forward kinematics before it is returned, and
the residual is carried on the solution. A caller never has to trust the
derivation on the strength of the derivation alone.

### The joint limit box is a constraint on the step, not a clip afterwards

Every iterative solver is asked for a configuration inside the declared joint
limits. The first implementation obtained one the cheapest way available: take
the unconstrained step, add it, and clip the result back into the box. That is
correct by construction and it was wrong as a method, for the reason recorded in
the limitation this section replaces. A clipped minimum-norm step is not
generally a descent direction, so an iterate that reached a bound could sit
against it while the rule kept asking for motion the joint could not make.

The step is now solved subject to the box, by the classical active-set treatment
of a box-constrained least-squares problem (Lawson and Hanson 1974, chapter 23).
Two moves alternate. A joint sitting on a bound whose proposed motion leaves the
box joins the active set: it is held at zero and the same update rule is solved
again on the remaining columns of the Jacobian. At the reduced solution the
component of
`J^T r` belonging to a held joint is its Karush-Kuhn-Tucker multiplier, and a
sign pointing back into the box means holding that joint is what is preventing
progress, so the single worst offender is released. Releasing one at a time is
the standard guard against cycling, and the pass count is capped at twice the
joint count so the routine terminates whatever the geometry.

A second guard sits underneath it. Clipping bends the ray `q + t dq` into an arc,
and the arc can climb even where the ray descends, so a step that fails to reduce
the residual is halved up to six times and the first length that does reduce it
is taken. The full step is taken when none does, which keeps the freedom to pass
through a worse configuration rather than stalling there.

Both are on by default and both are separately selectable, because the ablation
is the evidence. Over the 200 PUMA 560 and 200 UR5 targets of the results table:

```bash
uv run python examples/compare_ik_solvers.py --limit-strategy clip --backtracking 0
uv run python examples/compare_ik_solvers.py --limit-strategy active-set --backtracking 0
uv run python examples/compare_ik_solvers.py --limit-strategy clip --backtracking 6
uv run python examples/compare_ik_solvers.py --limit-strategy active-set --backtracking 6
```

| strategy | line search | PUMA 560 solved | UR5 solved |
| --- | --- | --- | --- |
| clip | no | 179/200 | 195/200 |
| clip | yes | 181/200 | 196/200 |
| active set | no | 189/200 | 196/200 |
| active set | yes | 191/200 | 197/200 |

Only the pseudoinverse row moves. The Jacobian transpose is unchanged at 40/200
and 26/200 because its steps are far too short to reach a bound in the first
place, and damped least squares is unchanged at 200/200 and 200/200 because the
residual-scaled damping already keeps it inside the box. That is the honest
reading: the constrained step buys nothing for a method that was never blocked,
and for the one that was it recovers 12 of 21 failures on the PUMA 560 and 2 of 5
on the UR5.

The cost is three things. The reduced solve runs at most twice per joint per
iteration, which is bounded but not free, and the line search costs up to six
extra forward kinematics evaluations on the iterations where a step fails. The
loop gained two settings and three functions. And `damped_step` had to learn
that zero damping means the pseudoinverse rather than the literal formula,
because `J J^T` is singular the moment the active set removes a column, which
made the Chiaverini schedule raise `LinAlgError` the first time it met a held
joint. That defect existed only because the constrained step created the
rank-deficient case, and it is now covered by a test.

### Residual-scaled damping as the default for the iterative solver

Three damping schedules are implemented and the choice is explicit on the solver.
Fixed damping (Nakamura and Hanafusa 1986, Wampler 1986) bounds the step
everywhere but leaves a bias of order `lambda^2` at the solution, so it cannot
meet a tight tolerance. Singular-region damping (Chiaverini, Siciliano and
Egeland 1994) switches damping on only when the smallest singular value falls
below a threshold. It was designed for resolved-rate velocity control, where the
goal is to bound the commanded joint rate, and it inherits the same terminal bias
whenever a solution happens to sit inside the singular region. Residual-scaled
damping (Sugihara 2011) sets `lambda^2 = ||e||^2 + bias`, so damping is strong
exactly when the linear model is untrustworthy and vanishes as the target is
approached.

The measured difference is large. Running

```bash
uv run python examples/compare_ik_solvers.py --robots puma560 --damping-schedule fixed
uv run python examples/compare_ik_solvers.py --robots puma560 --damping-schedule singular-region
uv run python examples/compare_ik_solvers.py --robots puma560 --damping-schedule levenberg-marquardt
```

over the same 200 PUMA 560 targets gives

| schedule | solved | median iterations | median position error (m) |
| --- | --- | --- | --- |
| fixed | 178/200 | 10 | 7.600e-07 |
| singular-region | 177/200 | 5 | 2.931e-08 |
| levenberg-marquardt | 200/200 | 6 | 1.219e-09 |

All three bound the step near a singularity. Only the third reaches 1e-6 on every
target, and it does so without paying for it in iterations. The singular-region
rule is kept because it is the right rule for the velocity-control question the
singularity scan asks, and the scan uses it directly.

### Singular values rather than a determinant

The Yoshikawa index is defined as `sqrt(det(J J^T))`, but forming `J J^T` squares
the condition number and loses roughly half the available precision, which is
worst precisely where the index matters. The implementation takes the product of
the singular values instead, which is mathematically identical and numerically far
better conditioned. A test asserts the two agree to 1e-9 relative on well
conditioned configurations, where the determinant form is still trustworthy.

Three metrics are reported rather than one because they answer different
questions. Manipulability is a volume and vanishes when any single direction is
lost, so it is the natural scalar alarm, but it says nothing about which direction
was lost. The condition number is the amplification a unit task error suffers,
which is what governs the step size an inverse-based rule requests. The smallest
singular value is the distance in the spectral norm to the nearest rank deficient
Jacobian, which is the quantity a damping schedule should key on, and the only one
of the three with a direct physical reading.

All three are computed after dividing the rotation rows by a characteristic
length, following Angeles 2007 section 4.9, because the translation rows of a
geometric Jacobian have units of metres while the rotation rows are dimensionless
per radian. Without that step the ratio of the largest to the smallest singular
value depends on whether the arm is measured in metres or millimetres. The default
characteristic length is the reach bound of the chain, the sum of the absolute
link lengths and offsets.

## Rejected alternatives

### A single Denavit-Hartenberg convention

Supporting only the standard convention would have removed one branch from the
Jacobian, one field from the `Robot`, and one test. It would have cost the ability
to use a Craig-style table without converting it, and conversion between the two
is a well-known source of sign errors. The branch is three lines and the test that
the two agree is one function. That was judged a good trade.

### A general closed form through resultants or Groebner bases

A general 6R arm has at most sixteen inverse kinematics solutions (Lee and Liang
1988, Raghavan and Roth 1993), obtainable as the roots of a degree-sixteen
univariate polynomial. Implementing that would have covered the UR5 and every
other 6R arm with one routine. It was rejected on three grounds. It needs a
polynomial eigenvalue solve whose conditioning is delicate and hard to check
against a hand computation. Its failure mode is a wrong root rather than a raised
error. And it would have obscured the point of the decoupling argument, which is
the reason a spherical wrist is designed in the first place. The structure check
makes the narrow scope explicit instead of hiding it.

### An arm-specific closed form for the UR5

The UR5 does have a published closed form (Hawkins 2013) that exploits the fact
that joints two, three and four are parallel. Adding it would have given a second
exact solver and
a second exact row in the results table. It was rejected because it is a second
derivation with no structure shared with the first, so it would have doubled the
closed-form surface area to cover one more arm, and because the UR5 is more useful
in this repository as the counterexample the structure check rejects. The library
still solves the UR5 numerically, at 200 out of 200 with the default solver.

### The pseudoinverse as the default iterative solver

The pseudoinverse converges quadratically and reaches the tolerance in a median of
five iterations, one fewer than damped least squares. It is not the default
because it still fails 9 of 200 PUMA 560 targets and 3 of 200 UR5 targets even
with the box-constrained step described above, while damped least squares solves
every target on both arms. One extra iteration for the last five percent of the
success rate is a clear trade.

### Null-space joint limit avoidance

Liegeois 1977 projects a limit-avoidance gradient into the null space of the
Jacobian, which is the standard treatment when the chain has joints to spare. It
was considered and does not apply here. Both shipped six-joint arms are
non-redundant, so `I - J^+ J` is the zero matrix wherever the Jacobian has full
rank, and the null-space term is identically zero exactly where it would be
needed. The active-set step was chosen instead because it works on a square
Jacobian: it does not need spare freedom, only the freedom that remains after a
bound has taken some away. The null-space term stays worth having for a
seven-joint arm, which this library does not ship.

### An analytic Jacobian with a Euler-angle orientation error

The orientation half of the task error could be expressed as the difference of
three Euler angles, which pairs with the analytic Jacobian rather than the
geometric one. That representation has coordinate singularities of its own, at
configurations that have nothing to do with the arm, and it makes the error a
function of a chart rather than of the geometry. The rotation vector of
`R_target R_current^T` was chosen instead: it is chart free, it is the correct
first-order pairing with the geometric Jacobian, and it is what the SciPy routine
`Rotation.as_rotvec` computes. Its own limitation, discontinuity at a half turn,
is stated below.

### A bespoke binary format for traces

Traces are plain frozen dataclasses held in memory, and the regression reference
is JSON. A binary format would have been smaller and faster to load. It was
rejected because a regression file a reviewer cannot read in a diff is not serving
its purpose: the point of pinning behaviour is that a change shows up as a
readable difference attached to a decision.

## Known limitations

### The closed form covers one arm structure

`analytic_ik` solves 6R chains with the PUMA 560 wrist structure and nothing else.
Arms with a non-intersecting wrist, a prismatic joint, more or fewer than six
joints, or a different twist pattern raise `StructureError`. Removing this would
mean implementing the general degree-sixteen solution, or one closed form per arm
family. The numerical solvers have no such restriction and are the fallback.

### An iterate can stop at a constrained local minimum

This entry replaces the one it grew out of. That entry read "joint limits are
enforced by clipping", and said that a clipped step is not generally a descent
direction and that this accounted for every pseudoinverse failure in the results
table. The step is now solved subject to the limit box rather than clipped after
the fact, by the active set described under method selection above, so the first
half of that claim no longer holds. The second half turned out to be wrong, and
what is left is the correct diagnosis.

Run

```bash
uv run python examples/compare_ik_solvers.py --diagnose-failures
```

which lists every unsolved trial with the number of joints pressed against a
bound and the conditioning at the answer. The 12 pseudoinverse failures, 9 on the
PUMA 560 and 3 on the UR5, fall into two groups.

Eight of them, 7 on the PUMA 560 and 1 on the UR5, stop with a joint on a bound at
a Jacobian that is nowhere near singular: manipulability between 3.7336e-04 and
8.2048e-03, condition number between 17.8 and 906. These are genuine
Karush-Kuhn-Tucker points of the box-constrained problem. No feasible direction
reduces the pose error to first order, the multiplier test agrees that no held
joint wants to be released, and the step therefore collapses. The target is
reachable, but not from that seed without leaving the box on the way.

The other four, 2 on each arm, are the singular case rather than the constrained
one: manipulability between 2.3758e-07 and 6.6887e-06 and condition number above
2.0e+04, where the truncated pseudoinverse discards the direction it cannot serve.
One of the four sits on a bound as well, which is a coincidence of that target
rather than a cause.

Escaping a constrained local minimum is a restart question, not a step question:
a second seed, a homotopy in the target, or a global method. None of those is a
better step rule, so none of them belongs in the update loop, and the library
offers the damped solver instead, which solved every target on both arms.

### The orientation error is discontinuous at a half turn

`pose_error` returns the rotation vector of `R_target R_current^T`, whose norm is
bounded by pi. When the current and target orientations are close to a half turn
apart, an arbitrarily small change in either can flip the direction of the
returned axis. Iterations are unaffected in practice because the error shrinks
below that regime within a step or two, but a seed exactly a half turn from the
target may take an arbitrary first step. Seeding closer, or solving position
first, avoids it.

### The Jacobian transpose solver is included for comparison, not for use

It converges on 20 percent of PUMA 560 targets and 13 percent of UR5 targets
within 500 iterations. This is the method behaving as documented, not a defect:
its convergence is linear and degrades with the condition number, and the mixed
units of the geometric Jacobian mean the gradient is dominated by the orientation
rows, because a position error of a few centimetres is numerically far smaller
than an orientation error of a few tenths of a radian. Weighting the two halves of
the error would improve it. It is left unweighted so the comparison is against the
method as published.

### Singularity metrics depend on a chosen length scale

Manipulability and the condition number are reported after dividing the rotation
rows by a characteristic length, and their numerical values move with that choice.
Comparisons across arms are therefore only meaningful when the same rule generates
the length, which is why `chain_reach` is used everywhere rather than a per-arm
constant. The smallest singular value is affected in the same way. There is no
canonical resolution to this in the literature; the alternative is to report the
translation and rotation blocks separately, which loses the coupling that makes
the metric interesting in the first place.

### Reachability is asserted by construction, not tested at the boundary

Campaign targets are generated by sampling configurations inside the joint limits
and running forward kinematics, so every target is reachable by construction. That
makes solver failures attributable to the solver, but it also means the results
say nothing about behaviour on unreachable or boundary requests, which is the case
that matters most in a real controller. `analytic_ik` returns an empty tuple for
an out-of-workspace target and the iterative solvers return their best iterate
with `converged` false, but neither behaviour is characterised statistically here.

### Joint limits for the Stanford arm are not published values

The PUMA 560 and UR5 limits are transcribed from the cited sources. The cited
source for the Stanford arm does not publish a limit table in a form that can be
transcribed without interpretation, so its limits are working ranges chosen for
this repository: 0.0 m to 0.5 m on the prismatic joint and plus or minus 170
degrees on the revolute joints. This is stated in the model docstring as well as
here. Any result that depends on Stanford arm limits, which at present means only
the sampling margin used in tests, should be read with that in mind.

### No dynamics, no collision checking, no trajectory

The library answers a kinematic question only. It says nothing about torques,
joint velocity or acceleration limits, self-collision, or the path taken between
two configurations. A branch reported as feasible satisfies the joint limits at
that configuration, and nothing more.
