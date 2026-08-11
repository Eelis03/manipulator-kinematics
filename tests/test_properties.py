"""Properties and invariants of the kinematics.

Each test states a mathematical fact that must hold for any correct
implementation, and checks it either against a hand computation or against an
independent numerical route to the same quantity.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from manipulator_kinematics.algorithm import (
    AnalyticIK,
    DampedLeastSquaresIK,
    DampingSchedule,
    IKResult,
    PseudoinverseIK,
    RestartingIK,
    SolverSettings,
    Tolerance,
    active_set_step,
    analytic_ik,
    assert_spherical_wrist,
    blocked_joints,
    conditioning,
    damped_step,
    finite_difference_jacobian,
    forward_kinematics,
    geometric_jacobian,
    link_frames,
    manipulability,
    numerical_solvers,
    pseudoinverse_step,
    residual_damping,
    transpose_step,
    variable_damping,
)
from manipulator_kinematics.algorithm.analytic import StructureError
from manipulator_kinematics.model import (
    DHConvention,
    DHParameter,
    JointLimit,
    JointType,
    Robot,
    chain_reach,
    clamp_to_limits,
    identity_pose,
    is_rotation,
    link_transform,
    pose_error,
    puma560,
    rotx,
    rotz,
    sample_within_limits,
    se3_inverse,
    skew,
    stanford_arm,
    ur5,
    within_limits,
)

TIGHT = 1e-9


def _random_configurations(robot: Robot, rng: np.random.Generator, count: int) -> list[np.ndarray]:
    return [sample_within_limits(robot.limits, rng, margin=0.05) for _ in range(count)]


# --------------------------------------------------------------------------
# Transform algebra
# --------------------------------------------------------------------------


def test_link_transforms_are_rigid(robot: Robot, rng: np.random.Generator) -> None:
    """Every link transform is orthonormal with determinant one."""
    for q in _random_configurations(robot, rng, 5):
        for link, value in zip(robot.links, q, strict=True):
            transform = link_transform(link, float(value), robot.convention)
            assert is_rotation(transform[:3, :3], tolerance=TIGHT)
            assert transform[3, :].tolist() == [0.0, 0.0, 0.0, 1.0]


def test_forward_kinematics_pose_is_rigid(robot: Robot, rng: np.random.Generator) -> None:
    """The composed pose stays in SE(3) over the whole configuration space."""
    for q in _random_configurations(robot, rng, 20):
        pose = forward_kinematics(robot, q)
        assert is_rotation(pose[:3, :3], tolerance=TIGHT)
        assert abs(float(np.linalg.det(pose)) - 1.0) < TIGHT


def test_se3_inverse_matches_matrix_inverse(robot: Robot, rng: np.random.Generator) -> None:
    """The closed-form SE(3) inverse agrees with a general matrix inverse."""
    for q in _random_configurations(robot, rng, 5):
        pose = forward_kinematics(robot, q)
        assert np.allclose(se3_inverse(pose), np.linalg.inv(pose), atol=TIGHT)
        assert np.allclose(se3_inverse(pose) @ pose, np.eye(4), atol=TIGHT)


def test_skew_reproduces_the_cross_product(rng: np.random.Generator) -> None:
    """``skew(a) @ b`` equals ``cross(a, b)``."""
    for _ in range(10):
        a, b = rng.normal(size=3), rng.normal(size=3)
        assert np.allclose(skew(a) @ b, np.cross(a, b), atol=TIGHT)


def test_pose_error_vanishes_only_at_the_target(rng: np.random.Generator) -> None:
    """The pose error is zero for identical poses and non-zero otherwise."""
    robot = puma560()
    q = sample_within_limits(robot.limits, rng, margin=0.05)
    pose = forward_kinematics(robot, q)
    assert np.allclose(pose_error(pose, pose), np.zeros(6), atol=1e-12)

    other = forward_kinematics(robot, q + 0.1)
    assert float(np.linalg.norm(pose_error(pose, other))) > 1e-3


def test_rotation_helpers_compose_as_expected() -> None:
    """``rotz`` and ``rotx`` are the textbook elementary rotations."""
    assert np.allclose(rotz(np.pi / 2) @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=TIGHT)
    assert np.allclose(rotx(np.pi / 2) @ np.array([0.0, 1.0, 0.0]), [0.0, 0.0, 1.0], atol=TIGHT)
    assert np.allclose(rotz(0.3) @ rotz(0.4), rotz(0.7), atol=TIGHT)


# --------------------------------------------------------------------------
# Hand-computed reference poses
# --------------------------------------------------------------------------


def test_puma560_zero_configuration_matches_hand_computation() -> None:
    """At q = 0 the PUMA 560 tool sits at a position readable from the DH table.

    With every joint angle zero the chain reduces to
    ``x = a2 + a3``, ``y = -(d2 + d3)``, ``z = d4``, and the tool frame is
    aligned with the base frame.
    """
    robot = puma560()
    pose = forward_kinematics(robot, robot.home())
    assert np.allclose(pose[:3, 3], [0.4318 + 0.0203, -0.15005, 0.4318], atol=1e-12)
    assert np.allclose(pose[:3, :3], np.eye(3), atol=1e-12)


def test_puma560_first_joint_rotates_the_whole_arm() -> None:
    """Turning joint one by 90 degrees maps the zero-pose x offset onto y."""
    robot = puma560()
    q = robot.home()
    q[0] = np.pi / 2
    pose = forward_kinematics(robot, q)
    assert np.allclose(pose[:3, 3], [0.15005, 0.4318 + 0.0203, 0.4318], atol=1e-12)


def test_stanford_zero_configuration_matches_hand_computation() -> None:
    """At q = 0 the Stanford tool sits at ``(0, d2, d1 + d6)``."""
    robot = stanford_arm()
    pose = forward_kinematics(robot, robot.home())
    assert np.allclose(pose[:3, 3], [0.0, 0.154, 0.412 + 0.263], atol=1e-12)


def test_stanford_prismatic_joint_extends_along_the_tool_axis() -> None:
    """Extending the prismatic joint by 0.2 m moves the tool 0.2 m along z."""
    robot = stanford_arm()
    q = robot.home()
    extended = q.copy()
    extended[2] = 0.2
    base = forward_kinematics(robot, q)[:3, 3]
    moved = forward_kinematics(robot, extended)[:3, 3]
    assert np.allclose(moved - base, [0.0, 0.0, 0.2], atol=1e-12)


def test_ur5_zero_configuration_matches_hand_computation() -> None:
    """At q = 0 the UR5 tool position follows directly from its DH table.

    Composing the table by hand at zero gives ``x = a2 + a3`` along the base x
    axis, ``y = -(d4 + d6)`` because the twist at joint one turns the base z axis
    into the negative y direction of the later frames, and ``z = d1 - d5``
    because the twist at joint five reverses the sense of the wrist offset.
    """
    robot = ur5()
    pose = forward_kinematics(robot, robot.home())
    expected_x = -0.425 - 0.39225
    expected_y = -(0.10915 + 0.0823)
    expected_z = 0.089159 - 0.09465
    assert np.allclose(pose[:3, 3], [expected_x, expected_y, expected_z], atol=1e-12)


def test_fully_extended_arm_reaches_the_expected_radius() -> None:
    """A planar two-link arm folded straight out reaches ``a2 + a3`` from its axis."""
    robot = puma560()
    q = robot.home()
    pose = forward_kinematics(robot, q)
    radius = float(np.hypot(pose[0, 3], pose[1, 3]))
    expected = float(np.hypot(0.4318 + 0.0203, 0.15005))
    assert abs(radius - expected) < 1e-12


def test_chain_reach_bounds_every_reachable_position(
    robot: Robot, rng: np.random.Generator
) -> None:
    """No configuration puts the tool further from the base than ``chain_reach``."""
    bound = chain_reach(robot)
    for q in _random_configurations(robot, rng, 30):
        assert float(np.linalg.norm(forward_kinematics(robot, q)[:3, 3])) <= bound


# --------------------------------------------------------------------------
# Both DH conventions
# --------------------------------------------------------------------------


def test_modified_convention_reproduces_the_standard_chain() -> None:
    """A chain written in modified DH matches the same chain in standard DH.

    The two conventions describe the same geometry with the parameters shifted by
    one link, so a standard table ``(d_i, theta_i, a_i, alpha_i)`` becomes a
    modified table whose row ``i`` carries ``a_{i-1}`` and ``alpha_{i-1}``, with
    one extra frame at the end holding the last pair. Building both and comparing
    the tool poses checks each convention against the other.
    """
    standard_rows = (
        (0.10, 0.0, 0.00, np.pi / 2),
        (0.00, 0.0, 0.35, 0.0),
        (0.05, 0.0, 0.20, -np.pi / 2),
        (0.30, 0.0, 0.00, np.pi / 2),
    )
    standard = Robot(
        name="standard-chain",
        links=tuple(
            DHParameter(d=d, theta=theta, a=a, alpha=alpha) for d, theta, a, alpha in standard_rows
        ),
        convention=DHConvention.STANDARD,
    )

    modified_links = [DHParameter(d=standard_rows[0][0], theta=0.0, a=0.0, alpha=0.0)]
    for index in range(1, len(standard_rows)):
        modified_links.append(
            DHParameter(
                d=standard_rows[index][0],
                theta=0.0,
                a=standard_rows[index - 1][2],
                alpha=standard_rows[index - 1][3],
            )
        )
    tool = np.eye(4, dtype=np.float64)
    tool[:3, :3] = rotx(standard_rows[-1][3])
    tool[0, 3] = standard_rows[-1][2]
    modified = Robot(
        name="modified-chain",
        links=tuple(modified_links),
        convention=DHConvention.MODIFIED,
        tool=tool,
    )

    rng = np.random.default_rng(11)
    for _ in range(20):
        q = rng.uniform(-2.0, 2.0, 4)
        assert np.allclose(
            forward_kinematics(standard, q), forward_kinematics(modified, q), atol=1e-12
        )


def test_link_frames_compose_to_the_tool_pose(robot: Robot, rng: np.random.Generator) -> None:
    """The last cumulative frame times the tool transform is the forward pose."""
    for q in _random_configurations(robot, rng, 5):
        frames = link_frames(robot, q)
        assert len(frames) == robot.n_joints + 1
        assert np.allclose(frames[-1] @ robot.tool, forward_kinematics(robot, q), atol=1e-15)


# --------------------------------------------------------------------------
# Jacobian
# --------------------------------------------------------------------------


def test_analytic_jacobian_matches_central_differences(
    robot: Robot, rng: np.random.Generator
) -> None:
    """The geometric Jacobian agrees with a central difference of forward kinematics."""
    for q in _random_configurations(robot, rng, 15):
        analytic = geometric_jacobian(robot, q)
        numeric = finite_difference_jacobian(robot, q, step=1e-6)
        assert np.allclose(analytic, numeric, atol=1e-7)


def test_jacobian_columns_follow_the_joint_type(robot: Robot, rng: np.random.Generator) -> None:
    """A prismatic column has no rotation part and a unit translation part."""
    q = sample_within_limits(robot.limits, rng, margin=0.05)
    jacobian = geometric_jacobian(robot, q)
    for index, joint_type in enumerate(robot.joint_types):
        if joint_type is JointType.PRISMATIC:
            assert np.allclose(jacobian[3:, index], np.zeros(3), atol=TIGHT)
            assert abs(float(np.linalg.norm(jacobian[:3, index])) - 1.0) < TIGHT
        else:
            assert abs(float(np.linalg.norm(jacobian[3:, index])) - 1.0) < TIGHT


def test_jacobian_predicts_the_pose_change(robot: Robot, rng: np.random.Generator) -> None:
    """A small joint step moves the tool by ``J dq`` to first order."""
    for q in _random_configurations(robot, rng, 5):
        jacobian = geometric_jacobian(robot, q)
        delta = 1e-6 * rng.normal(size=robot.n_joints)
        predicted = jacobian @ delta
        actual = pose_error(forward_kinematics(robot, q), forward_kinematics(robot, q + delta))
        assert np.allclose(predicted, actual, atol=1e-9)


# --------------------------------------------------------------------------
# Conditioning
# --------------------------------------------------------------------------


def test_manipulability_equals_the_determinant_form(robot: Robot, rng: np.random.Generator) -> None:
    """The product of singular values equals ``sqrt(det(J J^T))``."""
    for q in _random_configurations(robot, rng, 5):
        jacobian = geometric_jacobian(robot, q)
        direct = float(np.sqrt(max(float(np.linalg.det(jacobian @ jacobian.T)), 0.0)))
        assert abs(manipulability(jacobian) - direct) < 1e-9 * max(direct, 1e-6)


def test_conditioning_detects_the_puma_wrist_singularity() -> None:
    """Manipulability collapses and the condition number diverges at ``q5 = 0``."""
    robot = puma560()
    length = chain_reach(robot)
    regular = np.array([0.3, -0.9, 0.6, 0.4, 0.4, 0.2], dtype=np.float64)
    singular = np.array([0.3, -0.9, 0.6, 0.4, 0.0, 0.2], dtype=np.float64)

    good = conditioning(geometric_jacobian(robot, regular), characteristic_length=length)
    bad = conditioning(geometric_jacobian(robot, singular), characteristic_length=length)

    assert not good.is_singular
    assert bad.is_singular
    assert bad.manipulability < 1e-12 < good.manipulability
    assert bad.condition_number > 1e6 * good.condition_number
    assert len(bad.singular_values) == 6


def test_conditioning_rejects_a_non_positive_length() -> None:
    """A characteristic length must be positive."""
    jacobian = geometric_jacobian(puma560(), np.full(6, 0.4))
    with pytest.raises(ValueError, match="characteristic_length"):
        conditioning(jacobian, characteristic_length=0.0)


# --------------------------------------------------------------------------
# Analytic inverse kinematics
# --------------------------------------------------------------------------


@pytest.mark.parametrize("tool_offset", [0.0, 0.05625])
def test_analytic_ik_inverts_forward_kinematics(tool_offset: float) -> None:
    """Forward kinematics of every returned branch reproduces the requested pose."""
    robot = puma560(tool_offset=tool_offset)
    rng = np.random.default_rng(5)
    for _ in range(40):
        q = sample_within_limits(robot.limits, rng, margin=0.05)
        target = forward_kinematics(robot, q)
        branches = analytic_ik(robot, target)
        assert len(branches) == 8
        for branch in branches:
            assert branch.position_error < 1e-9
            assert branch.orientation_error < 1e-9


def test_analytic_ik_recovers_the_generating_configuration() -> None:
    """One branch equals the configuration the target was generated from."""
    robot = puma560(tool_offset=0.05625)
    rng = np.random.default_rng(6)
    for _ in range(20):
        q = sample_within_limits(robot.limits, rng, margin=0.2)
        branches = analytic_ik(robot, forward_kinematics(robot, q))
        gaps = [float(np.abs(branch.q - q).max()) for branch in branches]
        assert min(gaps) < 1e-8


def test_analytic_ik_branch_labels_are_distinct() -> None:
    """The eight branches carry the eight distinct posture labels."""
    robot = puma560()
    target = forward_kinematics(robot, np.array([0.4, -1.1, 0.7, -0.5, 0.9, 0.3]))
    labels = {branch.branch for branch in analytic_ik(robot, target)}
    assert len(labels) == 8


def test_analytic_ik_returns_nothing_outside_the_workspace() -> None:
    """A target beyond the reach of the arm yields no branch."""
    robot = puma560()
    target = np.eye(4, dtype=np.float64)
    target[:3, 3] = [5.0, 5.0, 5.0]
    assert analytic_ik(robot, target) == ()


def test_analytic_ik_handles_the_wrist_singularity() -> None:
    """A target on the wrist singularity still solves, through a degenerate branch.

    Only the arm posture that generated the pose sees a singular wrist, because a
    different posture presents a different ``R_0_3`` to the wrist. That posture
    contributes one branch rather than two, and the free choice of ``q4`` is
    resolved by fixing it at zero.
    """
    robot = puma560(tool_offset=0.05625)
    q = np.array([0.3, -0.9, 0.6, 0.4, 0.0, 0.2], dtype=np.float64)
    branches = analytic_ik(robot, forward_kinematics(robot, q))
    degenerate = [branch for branch in branches if branch.wrist == "degenerate"]
    assert degenerate
    assert len(branches) < 8
    for branch in branches:
        assert branch.position_error < 1e-9
        assert branch.orientation_error < 1e-9
    assert all(abs(branch.q[3]) < 1e-12 for branch in degenerate)


def test_analytic_ik_rejects_arms_without_a_spherical_wrist() -> None:
    """The closed form refuses a chain whose structure it was not derived for."""
    with pytest.raises(StructureError, match="alpha3"):
        assert_spherical_wrist(ur5())
    with pytest.raises(StructureError, match="revolute"):
        assert_spherical_wrist(stanford_arm())


def test_analytic_ik_rejects_the_wrong_convention_and_the_wrong_joint_count() -> None:
    """The structure check names the convention and the joint count before the table."""
    standard = puma560()
    modified = Robot(name="craig", links=standard.links, convention=DHConvention.MODIFIED)
    with pytest.raises(StructureError, match="standard DH convention"):
        assert_spherical_wrist(modified)

    short = Robot(name="five", links=standard.links[:5])
    with pytest.raises(StructureError, match="expected 6 joints, got 5"):
        assert_spherical_wrist(short)


def test_analytic_ik_rejects_a_target_that_is_not_a_pose() -> None:
    """A target of the wrong shape is an error, not a silent reshape."""
    with pytest.raises(ValueError, match="4x4 transform"):
        analytic_ik(puma560(), np.eye(3, dtype=np.float64))


def test_analytic_ik_returns_nothing_when_the_wrist_centre_is_on_the_shoulder_axis() -> None:
    """A wrist centre inside the shoulder offset cylinder is unreachable."""
    robot = puma560()
    target = np.eye(4, dtype=np.float64)
    assert analytic_ik(robot, target) == ()


def test_analytic_ik_solves_a_chain_that_declares_no_joint_limits() -> None:
    """Without limits every branch is feasible and no angle is remapped."""
    limited = puma560(tool_offset=0.05625)
    unlimited = Robot(
        name="puma560-unlimited",
        links=tuple(
            DHParameter(d=link.d, theta=link.theta, a=link.a, alpha=link.alpha)
            for link in limited.links
        ),
        convention=limited.convention,
    )
    assert not unlimited.has_limits
    q = np.array([0.4, -1.1, 0.7, -0.5, 0.9, 0.3], dtype=np.float64)
    branches = analytic_ik(unlimited, forward_kinematics(unlimited, q))
    assert len(branches) == 8
    assert all(branch.feasible for branch in branches)
    assert all(abs(value) <= np.pi + 1e-12 for branch in branches for value in branch.q)


def test_analytic_ik_handles_the_reversed_wrist_singularity() -> None:
    """At ``q5 = pi`` the wrist is degenerate with the two z axes opposed."""
    robot = puma560(tool_offset=0.05625)
    q = np.array([0.3, -0.9, 0.6, 0.4, np.pi, 0.2], dtype=np.float64)
    branches = analytic_ik(robot, forward_kinematics(robot, q))
    degenerate = [branch for branch in branches if branch.wrist == "degenerate"]
    assert degenerate
    for branch in branches:
        assert branch.position_error < 1e-9
        assert branch.orientation_error < 1e-9


def test_analytic_solver_reports_failure_when_no_branch_qualifies() -> None:
    """An unreachable target returns the seed, unconverged, with a stated reason."""
    robot = puma560(tool_offset=0.05625)
    target = np.eye(4, dtype=np.float64)
    target[:3, 3] = [5.0, 5.0, 5.0]
    seed = np.array([0.2, -0.8, 0.5, 0.3, 0.6, -0.1], dtype=np.float64)
    result = AnalyticIK().solve(robot, target, seed)
    assert not result.converged
    assert result.iterations == 0
    assert np.allclose(result.q, seed, atol=0.0)
    assert result.message == "no closed-form branch satisfied the tolerance"


def test_analytic_solver_picks_the_branch_nearest_the_seed() -> None:
    """The solver interface returns the feasible branch closest to the seed."""
    robot = puma560(tool_offset=0.05625)
    q = np.array([0.4, -1.1, 0.7, -0.5, 0.9, 0.3], dtype=np.float64)
    target = forward_kinematics(robot, q)
    result = AnalyticIK().solve(robot, target, q + 0.05)
    assert result.converged
    assert result.iterations == 1
    assert np.allclose(result.q, q, atol=1e-8)


# --------------------------------------------------------------------------
# Numerical inverse kinematics
# --------------------------------------------------------------------------


def test_numerical_solvers_satisfy_the_protocol() -> None:
    """Each solver exposes the name and solve members the protocol requires."""
    for solver in (*numerical_solvers(), AnalyticIK()):
        assert isinstance(solver.name, str)
        assert callable(solver.solve)


def test_numerical_solvers_recover_a_nearby_pose(rng: np.random.Generator) -> None:
    """From a seed close to the answer, the second-order methods converge."""
    robot = puma560()
    tolerance = Tolerance(position=1e-8, orientation=1e-8)
    _, pseudo, damped = numerical_solvers(max_iterations=200, tolerance=tolerance)
    for _ in range(10):
        q = sample_within_limits(robot.limits, rng, margin=0.2)
        target = forward_kinematics(robot, q)
        seed = q + 0.05 * rng.normal(size=6)
        for solver in (pseudo, damped):
            result = solver.solve(robot, target, seed)
            assert result.converged, f"{solver.name}: {result.message}"
            assert result.position_error <= tolerance.position
            assert result.orientation_error <= tolerance.orientation
            assert 0 < result.iterations <= 200
            assert len(result.residuals) == result.iterations + 1


def test_residuals_decrease_overall(rng: np.random.Generator) -> None:
    """The recorded residual at the end is far below the residual at the seed."""
    robot = ur5()
    for solver in numerical_solvers(max_iterations=200):
        q = sample_within_limits(robot.limits, rng, margin=0.2)
        result = solver.solve(robot, forward_kinematics(robot, q), q + 0.1)
        assert result.residuals[-1] < 0.5 * result.residuals[0]


def test_solver_reports_a_zero_iteration_solve() -> None:
    """Seeding at the answer converges without taking a step."""
    robot = puma560()
    q = np.array([0.2, -0.8, 0.5, 0.3, 0.6, -0.1], dtype=np.float64)
    result = numerical_solvers()[1].solve(robot, forward_kinematics(robot, q), q)
    assert result.converged
    assert result.iterations == 0
    assert result.message == "converged"


def test_solver_respects_joint_limits(rng: np.random.Generator) -> None:
    """No returned configuration leaves the limit box."""
    robot = puma560()
    lower = np.array([limit.lower for limit in robot.limits])
    upper = np.array([limit.upper for limit in robot.limits])
    for solver in numerical_solvers(max_iterations=100):
        for _ in range(5):
            q = sample_within_limits(robot.limits, rng, margin=0.05)
            result = solver.solve(robot, forward_kinematics(robot, q), q + 0.3)
            assert np.all(result.q >= lower - 1e-12)
            assert np.all(result.q <= upper + 1e-12)


def test_solver_rejects_a_wrong_length_seed() -> None:
    """A seed of the wrong length is an error, not a silent reshape."""
    robot = puma560()
    with pytest.raises(ValueError, match="expected 6 joint values"):
        numerical_solvers()[1].solve(robot, np.eye(4), np.zeros(3))


def test_damping_schedules_bound_the_step_differently() -> None:
    """Near a singularity the damped step is far smaller than the pseudoinverse step."""
    robot = puma560()
    near = np.array([0.3, -0.9, 0.6, 0.4, 1e-3, 0.2], dtype=np.float64)
    jacobian = geometric_jacobian(robot, near)
    twist = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    plain = float(np.linalg.norm(pseudoinverse_step(jacobian, twist)))
    regularised = float(np.linalg.norm(damped_step(jacobian, twist, 0.05)))
    assert plain > 100.0 * regularised


def test_undamped_damped_least_squares_is_the_pseudoinverse() -> None:
    """Zero damping is the limit of the damped rule, and stays defined at rank loss."""
    robot = puma560()
    jacobian = geometric_jacobian(robot, np.array([0.3, -0.9, 0.6, 0.4, 0.5, 0.2]))
    twist = np.array([0.02, -0.01, 0.03, 0.05, -0.02, 0.01], dtype=np.float64)
    assert np.allclose(
        damped_step(jacobian, twist, 0.0), pseudoinverse_step(jacobian, twist), atol=1e-9
    )

    narrow = jacobian[:, :3]
    assert np.allclose(
        damped_step(narrow, twist, 0.0), pseudoinverse_step(narrow, twist), atol=1e-12
    )


def test_the_singular_region_schedule_survives_the_active_set(
    rng: np.random.Generator,
) -> None:
    """Undamped steps on a held-back Jacobian must not ask for a singular inverse.

    The Chiaverini schedule returns zero damping outside the singular region. Once
    the active set holds a joint, the reduced Jacobian has fewer than six columns
    and ``J J^T`` is singular, so the undamped normal equations have no solution.
    """
    robot = puma560()
    solver = DampedLeastSquaresIK(
        settings=SolverSettings(max_iterations=80, tolerance=Tolerance(1e-6, 1e-6)),
        schedule=DampingSchedule.SINGULAR_REGION,
    )
    lower = np.array([limit.lower for limit in robot.limits], dtype=np.float64)
    for _ in range(20):
        q = sample_within_limits(robot.limits, rng, margin=0.15)
        result = solver.solve(robot, forward_kinematics(robot, q), lower)
        assert np.all(np.isfinite(result.q))


def test_transpose_step_is_zero_when_the_error_is_outside_the_range() -> None:
    """A task error the Jacobian cannot see produces no step at all."""
    jacobian = np.zeros((6, 6), dtype=np.float64)
    twist = np.ones(6, dtype=np.float64)
    assert np.allclose(transpose_step(jacobian, twist), np.zeros(6), atol=0.0)


def test_variable_damping_switches_on_only_inside_the_singular_region() -> None:
    """Damping is zero away from a singularity and reaches its maximum at one."""
    robot = puma560()
    regular = geometric_jacobian(robot, np.array([0.3, -0.9, 0.6, 0.4, 0.6, 0.2]))
    singular = geometric_jacobian(robot, np.array([0.3, -0.9, 0.6, 0.4, 0.0, 0.2]))
    assert variable_damping(regular, damping=0.05, epsilon=0.05) == 0.0
    assert variable_damping(singular, damping=0.05, epsilon=0.05) == pytest.approx(0.05)


def test_residual_damping_falls_to_the_bias_as_the_error_vanishes() -> None:
    """``lambda^2`` is the squared error plus the bias, so it never reaches zero."""
    assert residual_damping(np.zeros(6), bias=1e-8) == pytest.approx(1e-4)
    assert residual_damping(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.3]), bias=0.0) == pytest.approx(
        0.3
    )


def test_each_damping_schedule_picks_the_factor_it_documents() -> None:
    """The three schedules return the fixed, the Chiaverini and the Sugihara value."""
    robot = puma560()
    jacobian = geometric_jacobian(robot, np.array([0.3, -0.9, 0.6, 0.4, 0.0, 0.2]))
    error = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    fixed = DampedLeastSquaresIK(schedule=DampingSchedule.FIXED, damping=0.07)
    region = DampedLeastSquaresIK(schedule=DampingSchedule.SINGULAR_REGION, damping=0.07)
    marquardt = DampedLeastSquaresIK(schedule=DampingSchedule.LEVENBERG_MARQUARDT, bias=1e-8)

    assert fixed.damping_for(jacobian, error) == pytest.approx(0.07)
    assert region.damping_for(jacobian, error) == pytest.approx(
        variable_damping(jacobian, damping=0.07, epsilon=region.epsilon)
    )
    assert marquardt.damping_for(jacobian, error) == pytest.approx(
        residual_damping(error, bias=1e-8)
    )


def test_damped_solver_rejects_settings_it_cannot_honour() -> None:
    """Each damping parameter is validated where it is declared."""
    with pytest.raises(ValueError, match="damping must not be negative"):
        DampedLeastSquaresIK(damping=-0.1)
    with pytest.raises(ValueError, match="epsilon must be positive"):
        DampedLeastSquaresIK(epsilon=0.0)
    with pytest.raises(ValueError, match="bias must be positive"):
        DampedLeastSquaresIK(bias=0.0)


def test_solver_rejects_a_negative_iteration_budget() -> None:
    """A negative budget is refused rather than silently treated as zero."""
    robot = puma560()
    solver = PseudoinverseIK(settings=SolverSettings(max_iterations=-1))
    with pytest.raises(ValueError, match="max_iterations"):
        solver.solve(robot, forward_kinematics(robot, robot.home()), robot.home())


def test_solver_stops_when_the_step_stops_being_finite() -> None:
    """A target carrying a non-finite entry ends the search with a stated reason."""
    robot = puma560()
    target = forward_kinematics(robot, np.array([0.2, -0.8, 0.5, 0.3, 0.6, -0.1]))
    target[0, 3] = np.nan
    result = numerical_solvers(max_iterations=20)[1].solve(robot, target, robot.home())
    assert not result.converged
    assert result.message == "step became non-finite"
    assert result.iterations == 0


def test_solver_reports_convergence_reached_on_the_last_permitted_step() -> None:
    """Meeting the tolerance exactly as the budget runs out still counts as converged."""
    robot = puma560()
    q = np.array([0.2, -0.8, 0.5, 0.3, 0.6, -0.1], dtype=np.float64)
    settings = SolverSettings(max_iterations=1, tolerance=Tolerance(1e-6, 1e-6))
    result = PseudoinverseIK(settings=settings).solve(robot, forward_kinematics(robot, q), q + 1e-4)
    assert result.iterations == 1
    assert result.converged
    assert result.message == "converged"


def test_a_solver_ignoring_the_limits_leaves_the_box() -> None:
    """Clearing ``respect_limits`` is the documented way to solve without a box."""
    robot = puma560()
    q = np.array([0.2, -0.8, 0.5, 0.3, 0.6, -0.1], dtype=np.float64)
    settings = SolverSettings(max_iterations=60, respect_limits=False)
    result = PseudoinverseIK(settings=settings).solve(robot, forward_kinematics(robot, q), q + 0.2)
    assert result.converged


# --------------------------------------------------------------------------
# The joint limit box as a constraint rather than a clip
# --------------------------------------------------------------------------


def _cornered(robot: Robot) -> np.ndarray:
    """Return a configuration sitting on the lower bound of every joint."""
    return np.array([limit.lower for limit in robot.limits], dtype=np.float64)


def test_blocked_joints_names_only_the_bounds_being_pushed_through() -> None:
    """A joint on a bound moving outward is blocked; moving inward it is not."""
    limits = (JointLimit(-1.0, 1.0), JointLimit(-1.0, 1.0), JointLimit(-1.0, 1.0))
    q = np.array([-1.0, 1.0, 0.0], dtype=np.float64)
    delta = np.array([-0.5, 0.5, -0.5], dtype=np.float64)
    assert blocked_joints(q, delta, limits).tolist() == [True, True, False]
    assert blocked_joints(q, -delta, limits).tolist() == [False, False, False]


def test_the_constrained_step_zeroes_exactly_the_joints_it_holds() -> None:
    """Held components are exactly zero and free components are not clipped."""
    robot = puma560()
    q = _cornered(robot)
    jacobian = geometric_jacobian(robot, q)
    twist = np.array([0.0, 0.0, -0.4, 0.0, 0.0, 0.0], dtype=np.float64)
    delta = active_set_step(pseudoinverse_step, jacobian, twist, q, robot.limits)
    held = delta == 0.0
    assert bool(held.any()), "this configuration was chosen because a bound binds"
    assert np.all(clamp_to_limits(q + delta, robot.limits) == q + delta)


def test_the_constrained_step_matches_the_plain_step_away_from_every_bound() -> None:
    """With no bound active the active set changes nothing at all."""
    robot = puma560()
    q = np.array([0.3, -0.9, 0.6, 0.4, 0.5, 0.2], dtype=np.float64)
    jacobian = geometric_jacobian(robot, q)
    twist = np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    assert np.allclose(
        active_set_step(pseudoinverse_step, jacobian, twist, q, robot.limits),
        pseudoinverse_step(jacobian, twist),
        atol=1e-15,
    )


def test_the_constrained_step_serves_the_task_better_than_clipping_does() -> None:
    """On the linear model, the held solve beats clipping the unconstrained one."""
    robot = puma560()
    q = _cornered(robot)
    jacobian = geometric_jacobian(robot, q)
    twist = np.array([0.05, -0.05, 0.05, 0.1, -0.1, 0.1], dtype=np.float64)

    plain = pseudoinverse_step(jacobian, twist)
    clipped = clamp_to_limits(q + plain, robot.limits) - q
    constrained = active_set_step(pseudoinverse_step, jacobian, twist, q, robot.limits)

    clipped_miss = float(np.linalg.norm(jacobian @ clipped - twist))
    constrained_miss = float(np.linalg.norm(jacobian @ constrained - twist))
    assert constrained_miss < clipped_miss


def test_the_constrained_step_is_zero_when_no_joint_may_move() -> None:
    """A corner that blocks every joint yields no step rather than an illegal one."""
    limits = (JointLimit(-1.0, 1.0), JointLimit(-1.0, 1.0))
    q = np.array([-1.0, -1.0], dtype=np.float64)
    jacobian = -np.ones((6, 2), dtype=np.float64)
    twist = np.ones(6, dtype=np.float64)
    delta = active_set_step(pseudoinverse_step, jacobian, twist, q, limits)
    assert np.allclose(delta, np.zeros(2), atol=0.0)


def test_the_active_set_solves_more_targets_than_clipping_alone(
    rng: np.random.Generator,
) -> None:
    """The constrained step is a strict improvement on the recorded campaign."""
    from manipulator_kinematics.pipeline import perturbed_seeds, sample_targets

    robot = puma560()
    targets = sample_targets(robot, 60, rng, margin=0.15)
    seeds = perturbed_seeds(robot, targets, rng, spread=0.4)
    tolerance = Tolerance(position=1e-6, orientation=1e-6)

    def solved(active: bool) -> int:
        solver = PseudoinverseIK(
            settings=SolverSettings(
                max_iterations=200,
                tolerance=tolerance,
                active_set=active,
                backtracking=6 if active else 0,
            )
        )
        return sum(
            solver.solve(robot, target.pose, seed).converged
            for target, seed in zip(targets, seeds, strict=True)
        )

    assert solved(True) > solved(False)


def test_no_run_ends_worse_than_it_started(rng: np.random.Generator) -> None:
    """The recorded trajectory never finishes above the residual it began with.

    This assertion was tried while the limits were enforced by clipping alone,
    and it failed: a clipped minimum-norm step is not a descent direction, so the
    pseudoinverse ended above its starting residual on nine of two hundred PUMA
    560 targets, by as much as a factor of ten. It holds now, which is the
    property the constrained step was added to obtain.
    """
    from manipulator_kinematics.pipeline import perturbed_seeds, sample_targets

    for robot in (puma560(), ur5()):
        targets = sample_targets(robot, 25, rng, margin=0.15)
        seeds = perturbed_seeds(robot, targets, rng, spread=0.4)
        for solver in numerical_solvers(max_iterations=200):
            for target, seed in zip(targets, seeds, strict=True):
                result = solver.solve(robot, target.pose, seed)
                assert result.residuals[-1] <= result.residuals[0], (
                    f"{robot.name}/{solver.name} target {target.index}"
                )


# --------------------------------------------------------------------------
# Random restart
# --------------------------------------------------------------------------


@dataclass
class _CountingSolver:
    """A solver that records the seed of every call before delegating it."""

    inner: PseudoinverseIK
    calls: list[np.ndarray] = field(default_factory=list)

    @property
    def name(self) -> str:
        """Short identifier used in traces, tables and figures."""
        return self.inner.name

    def solve(self, robot: Robot, target: np.ndarray, seed: np.ndarray) -> IKResult:
        """Record the starting configuration and hand the call on."""
        self.calls.append(np.array(seed, dtype=np.float64))
        return self.inner.solve(robot, target, seed)


def _unreachable() -> np.ndarray:
    """Return a pose far outside the workspace, so every start is bound to fail."""
    target = np.eye(4, dtype=np.float64)
    target[:3, 3] = [5.0, 5.0, 5.0]
    return target


def test_restarting_without_a_restart_budget_is_the_solver_it_wraps() -> None:
    """A zero budget changes the label and nothing about the answer."""
    robot = puma560()
    q = np.array([0.2, -0.8, 0.5, 0.3, 0.6, -0.1], dtype=np.float64)
    target = forward_kinematics(robot, q)
    inner = PseudoinverseIK(settings=SolverSettings(max_iterations=8))

    direct = inner.solve(robot, target, _cornered(robot))
    wrapped = RestartingIK(solver=inner, restarts=0).solve(robot, target, _cornered(robot))

    assert wrapped.solver == "pseudoinverse-restart"
    assert wrapped.converged == direct.converged
    assert wrapped.iterations == direct.iterations
    assert wrapped.message == direct.message
    assert wrapped.residuals == direct.residuals
    assert np.allclose(wrapped.q, direct.q, atol=0.0)


def test_restarting_spends_one_call_on_a_target_the_seed_already_solves() -> None:
    """A seed that converges is never followed by a restart."""
    robot = puma560()
    q = np.array([0.2, -0.8, 0.5, 0.3, 0.6, -0.1], dtype=np.float64)
    counted = _CountingSolver(inner=PseudoinverseIK(settings=SolverSettings(max_iterations=50)))

    result = RestartingIK(solver=counted, restarts=8).solve(
        robot, forward_kinematics(robot, q), q + 0.05
    )

    assert result.converged
    assert result.message == "converged"
    assert len(counted.calls) == 1


def test_restarting_draws_every_fresh_start_from_inside_the_limit_box() -> None:
    """A restart begins where the arm may stand, not where the last one stopped."""
    robot = puma560()
    counted = _CountingSolver(inner=PseudoinverseIK(settings=SolverSettings(max_iterations=6)))

    result = RestartingIK(solver=counted, restarts=3, margin=0.1).solve(
        robot, _unreachable(), _cornered(robot)
    )

    assert len(counted.calls) == 4
    assert np.allclose(counted.calls[0], _cornered(robot), atol=0.0)
    for start in counted.calls[1:]:
        assert within_limits(start, robot.limits)
    assert not result.converged
    assert result.message == "iteration limit reached, best of 4 starts"


def test_restarting_never_returns_a_worse_result_than_the_start_it_was_given(
    rng: np.random.Generator,
) -> None:
    """Restarting cannot lose a solve, and until one is found it only lowers the residual.

    Both halves are exact rather than statistical. The first attempt is the
    wrapped call itself, and a later attempt replaces it only by converging or by
    beating its residual, so the comparison holds target by target.
    """
    from manipulator_kinematics.pipeline import sample_targets

    robot = puma560()
    targets = sample_targets(robot, 8, rng, margin=0.15)
    inner = PseudoinverseIK(settings=SolverSettings(max_iterations=100))
    restarting = RestartingIK(solver=inner, restarts=3)
    start = _cornered(robot)

    for target in targets:
        direct = inner.solve(robot, target.pose, start)
        best = restarting.solve(robot, target.pose, start)
        assert best.converged or not direct.converged, f"target {target.index}"
        if not best.converged:
            assert best.final_residual <= direct.final_residual, f"target {target.index}"


def test_restarting_recovers_targets_a_single_start_cannot(rng: np.random.Generator) -> None:
    """From the corner of the limit box, a fresh start solves what a better step cannot.

    The corner is the case the design notes describe. The descent stops at a
    Karush-Kuhn-Tucker point of the box-constrained problem, where no feasible
    direction reduces the pose error and no held joint wants to be released, so
    the step collapses however it is computed. Measured on this campaign, a
    single start solved none of the 12 targets and four starts solved 11. The
    margin asserted below sits well inside that gap, because the count is
    evidence that restarting works and not a value any machine can promise
    another.
    """
    from manipulator_kinematics.pipeline import sample_targets

    robot = puma560()
    targets = sample_targets(robot, 12, rng, margin=0.15)
    inner = PseudoinverseIK(settings=SolverSettings(max_iterations=120))
    restarting = RestartingIK(solver=inner, restarts=3)
    start = _cornered(robot)

    single = sum(inner.solve(robot, target.pose, start).converged for target in targets)
    results = [restarting.solve(robot, target.pose, start) for target in targets]

    assert sum(result.converged for result in results) >= single + 4
    assert all(
        result.message == "converged" or result.message.startswith("converged on start ")
        for result in results
        if result.converged
    )
    for result in results:
        assert within_limits(result.q, robot.limits, tolerance=1e-12)


def test_restarting_repeats_itself_and_a_new_generator_seed_starts_elsewhere() -> None:
    """The generator is built inside the call, so one call is a function of its arguments."""
    robot = puma560()
    inner = PseudoinverseIK(settings=SolverSettings(max_iterations=4))
    first, again, other = (_CountingSolver(inner=inner) for _ in range(3))

    RestartingIK(solver=first, restarts=2).solve(robot, _unreachable(), _cornered(robot))
    RestartingIK(solver=again, restarts=2).solve(robot, _unreachable(), _cornered(robot))
    RestartingIK(solver=other, restarts=2, rng_seed=7).solve(
        robot, _unreachable(), _cornered(robot)
    )

    assert np.allclose(np.array(first.calls), np.array(again.calls), atol=0.0)
    assert not np.allclose(np.array(first.calls), np.array(other.calls), atol=1e-9)


def test_restarting_displaces_the_seed_when_the_chain_declares_no_limits() -> None:
    """Without a box to draw from, a restart is the given seed plus a Gaussian offset."""
    limited = puma560()
    unlimited = Robot(
        name="puma560-unlimited",
        links=tuple(
            DHParameter(d=link.d, theta=link.theta, a=link.a, alpha=link.alpha)
            for link in limited.links
        ),
    )
    q = np.array([0.4, -1.1, 0.7, -0.5, 0.9, 0.3], dtype=np.float64)
    inner = PseudoinverseIK(settings=SolverSettings(max_iterations=4))

    still = _CountingSolver(inner=inner)
    RestartingIK(solver=still, restarts=2, spread=0.0).solve(unlimited, _unreachable(), q)
    assert len(still.calls) == 3
    assert all(np.allclose(start, q, atol=0.0) for start in still.calls)

    spread = _CountingSolver(inner=inner)
    RestartingIK(solver=spread, restarts=2, spread=0.4).solve(unlimited, _unreachable(), q)
    assert all(float(np.linalg.norm(start - q)) > 1e-6 for start in spread.calls[1:])


def test_restarting_rejects_settings_it_cannot_honour() -> None:
    """A negative budget and a margin that empties the box are refused where declared."""
    with pytest.raises(ValueError, match="restarts must not be negative"):
        RestartingIK(solver=PseudoinverseIK(), restarts=-1)
    with pytest.raises(ValueError, match="margin must be at least zero and below a half"):
        RestartingIK(solver=PseudoinverseIK(), margin=0.5)


# --------------------------------------------------------------------------
# Declared data is validated where it is declared
# --------------------------------------------------------------------------


def test_a_joint_limit_must_be_finite_and_ordered() -> None:
    """An unbounded or inverted range is refused at construction."""
    with pytest.raises(ValueError, match="finite"):
        JointLimit(-np.inf, 1.0)
    with pytest.raises(ValueError, match="exceeds upper limit"):
        JointLimit(1.0, -1.0)


def test_a_joint_limit_reports_its_centre_and_width() -> None:
    """Midpoint and span are the two derived numbers the sampler needs."""
    limit = JointLimit(-0.5, 1.5)
    assert limit.midpoint == pytest.approx(0.5)
    assert limit.span == pytest.approx(2.0)


def test_a_robot_needs_links_and_four_by_four_frames() -> None:
    """An empty chain or a malformed base or tool transform is refused."""
    link = DHParameter(d=0.0, theta=0.0, a=0.1, alpha=0.0)
    with pytest.raises(ValueError, match="at least one link"):
        Robot(name="empty", links=())
    with pytest.raises(ValueError, match="4x4"):
        Robot(name="bad-base", links=(link,), base=np.eye(3, dtype=np.float64))
    with pytest.raises(ValueError, match="4x4"):
        Robot(name="bad-tool", links=(link,), tool=np.eye(3, dtype=np.float64))


def test_asking_for_limits_a_chain_never_declared_names_the_joints() -> None:
    """A chain without limits reports which joints are missing them."""
    links = (
        DHParameter(d=0.0, theta=0.0, a=0.1, alpha=0.0, limit=JointLimit(-1.0, 1.0)),
        DHParameter(d=0.0, theta=0.0, a=0.1, alpha=0.0),
    )
    robot = Robot(name="partial", links=links)
    assert not robot.has_limits
    with pytest.raises(ValueError, match=r"joints \[1\] have no declared limit"):
        _ = robot.limits


def test_a_tolerance_must_be_positive() -> None:
    """A zero tolerance would be met by nothing, so it is refused."""
    with pytest.raises(ValueError, match="tolerances must be positive"):
        Tolerance(position=0.0, orientation=1e-6)
    with pytest.raises(ValueError, match="tolerances must be positive"):
        Tolerance(position=1e-6, orientation=-1.0)


def test_identity_pose_is_the_neutral_element() -> None:
    """Composing with the identity pose changes nothing."""
    robot = puma560()
    pose = forward_kinematics(robot, np.full(6, 0.3))
    assert np.allclose(identity_pose() @ pose, pose, atol=0.0)
    assert np.allclose(pose @ identity_pose(), pose, atol=0.0)


def test_is_rotation_rejects_a_matrix_of_the_wrong_shape() -> None:
    """A 4x4 pose is not a rotation matrix, whatever its upper block contains."""
    assert not is_rotation(np.eye(4, dtype=np.float64))
    assert not is_rotation(2.0 * np.eye(3, dtype=np.float64))
    assert is_rotation(rotz(0.3) @ rotx(0.4))
