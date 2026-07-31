"""Tier one: properties and invariants of the kinematics.

Each test states a mathematical fact that must hold for any correct
implementation, and checks it either against a hand computation or against an
independent numerical route to the same quantity.
"""

from __future__ import annotations

import numpy as np
import pytest

from manipulator_kinematics.algorithm import (
    AnalyticIK,
    Tolerance,
    analytic_ik,
    assert_spherical_wrist,
    conditioning,
    finite_difference_jacobian,
    forward_kinematics,
    geometric_jacobian,
    link_frames,
    manipulability,
    numerical_solvers,
)
from manipulator_kinematics.algorithm.analytic import StructureError
from manipulator_kinematics.model import (
    DHConvention,
    DHParameter,
    JointType,
    Robot,
    chain_reach,
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
)

TIGHT = 1e-9


def _random_configurations(
    robot: Robot, rng: np.random.Generator, count: int
) -> list[np.ndarray]:
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
            DHParameter(d=d, theta=theta, a=a, alpha=alpha)
            for d, theta, a, alpha in standard_rows
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


def test_manipulability_equals_the_determinant_form(
    robot: Robot, rng: np.random.Generator
) -> None:
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
    from manipulator_kinematics.algorithm import damped_step, pseudoinverse_step

    robot = puma560()
    near = np.array([0.3, -0.9, 0.6, 0.4, 1e-3, 0.2], dtype=np.float64)
    jacobian = geometric_jacobian(robot, near)
    twist = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    plain = float(np.linalg.norm(pseudoinverse_step(jacobian, twist)))
    regularised = float(np.linalg.norm(damped_step(jacobian, twist, 0.05)))
    assert plain > 100.0 * regularised
