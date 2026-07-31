"""Published Denavit-Hartenberg parameter sets for real manipulators.

Every table below is transcribed from the cited publication. Lengths are metres
and angles are radians. Where a source states limits in degrees they are
converted here with :func:`numpy.deg2rad`, and the degree values are repeated in
the docstring so the transcription can be checked against the source.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final

import numpy as np

from manipulator_kinematics.model.dh import DHConvention, DHParameter, JointType, Robot
from manipulator_kinematics.model.joints import JointLimit

__all__ = ["ROBOTS", "puma560", "stanford_arm", "ur5"]

_PUMA_SOURCE: Final[str] = (
    "P. I. Corke and B. Armstrong-Helouvry, 'A search for consensus among model "
    "parameters reported for the PUMA 560 robot', IEEE ICRA 1994, "
    "doi:10.1109/ROBOT.1994.351360. Joint limits from the Unimation PUMA 560 "
    "specification as tabulated in P. I. Corke, 'A robotics toolbox for MATLAB', "
    "IEEE Robotics and Automation Magazine 3(1), 1996, doi:10.1109/100.486658."
)

_UR5_SOURCE: Final[str] = (
    "Universal Robots, 'DH parameters for calculations of kinematics and dynamics', "
    "https://www.universal-robots.com/articles/ur/application-installation/"
    "dh-parameters-for-calculations-of-kinematics-and-dynamics/ , cross-checked "
    "against P. M. Kebria, S. Al-Wais, H. Abdi and S. Nahavandi, 'Kinematic and "
    "dynamic modelling of UR5 manipulator', IEEE SMC 2016, doi:10.1109/SMC.2016.7844896."
)

_STANFORD_SOURCE: Final[str] = (
    "R. P. Paul, 'Robot Manipulators: Mathematics, Programming and Control', "
    "MIT Press, 1981, ISBN 978-0-262-16082-7. Numeric values as tabulated by "
    "P. I. Corke, 'A robotics toolbox for MATLAB', IEEE Robotics and Automation "
    "Magazine 3(1), 1996, doi:10.1109/100.486658."
)


def _degree_limits(pairs: tuple[tuple[float, float], ...]) -> tuple[JointLimit, ...]:
    return tuple(
        JointLimit(float(np.deg2rad(low)), float(np.deg2rad(high))) for low, high in pairs
    )


def puma560(*, tool_offset: float = 0.0) -> Robot:
    """Return the Unimation PUMA 560 as a six revolute joint chain in standard DH.

    The parameter table is

    ==== ========= ========= ========= =========
    Link theta     d (m)     a (m)     alpha
    ==== ========= ========= ========= =========
    1    q1        0.0       0.0       +pi/2
    2    q2        0.0       0.4318    0
    3    q3        0.15005   0.0203    -pi/2
    4    q4        0.4318    0.0       +pi/2
    5    q5        0.0       0.0       -pi/2
    6    q6        0.0       0.0       0
    ==== ========= ========= ========= =========

    Joint limits in degrees are (-160, 160), (-225, 45), (-45, 225), (-110, 170),
    (-100, 100) and (-266, 266).

    The published table gives ``d6 = 0``, which places the tool frame at the wrist
    centre. ``tool_offset`` sets ``d6`` instead, which is the usual way to attach
    an end effector along the final z axis and makes the position and orientation
    parts of the inverse kinematics genuinely distinct.

    Args:
        tool_offset: Value assigned to ``d6``, in metres.

    Returns:
        The configured robot.
    """
    limits = _degree_limits(
        (
            (-160.0, 160.0),
            (-225.0, 45.0),
            (-45.0, 225.0),
            (-110.0, 170.0),
            (-100.0, 100.0),
            (-266.0, 266.0),
        )
    )
    links = (
        DHParameter(d=0.0, theta=0.0, a=0.0, alpha=np.pi / 2, limit=limits[0]),
        DHParameter(d=0.0, theta=0.0, a=0.4318, alpha=0.0, limit=limits[1]),
        DHParameter(d=0.15005, theta=0.0, a=0.0203, alpha=-np.pi / 2, limit=limits[2]),
        DHParameter(d=0.4318, theta=0.0, a=0.0, alpha=np.pi / 2, limit=limits[3]),
        DHParameter(d=0.0, theta=0.0, a=0.0, alpha=-np.pi / 2, limit=limits[4]),
        DHParameter(d=float(tool_offset), theta=0.0, a=0.0, alpha=0.0, limit=limits[5]),
    )
    return Robot(
        name="puma560" if tool_offset == 0.0 else f"puma560+tool{tool_offset:g}",
        links=links,
        convention=DHConvention.STANDARD,
        source=_PUMA_SOURCE,
    )


def ur5() -> Robot:
    """Return the Universal Robots UR5 as a six revolute joint chain in standard DH.

    The parameter table is

    ==== ========= ========== ========== =========
    Link theta     d (m)      a (m)      alpha
    ==== ========= ========== ========== =========
    1    q1        0.089159   0.0        +pi/2
    2    q2        0.0        -0.425     0
    3    q3        0.0        -0.39225   0
    4    q4        0.10915    0.0        +pi/2
    5    q5        0.09465    0.0        -pi/2
    6    q6        0.0823     0.0        0
    ==== ========= ========== ========== =========

    Every joint travels plus or minus 360 degrees. The wrist axes of the UR5 are
    mutually perpendicular but do not intersect, so this arm has no spherical
    wrist and the analytic solver in
    :mod:`manipulator_kinematics.algorithm.analytic` does not apply to it. It is
    included to exercise the numerical solvers and the conditioning metrics on a
    second, structurally different chain.

    Returns:
        The configured robot.
    """
    limit = JointLimit(-2.0 * np.pi, 2.0 * np.pi)
    links = (
        DHParameter(d=0.089159, theta=0.0, a=0.0, alpha=np.pi / 2, limit=limit),
        DHParameter(d=0.0, theta=0.0, a=-0.425, alpha=0.0, limit=limit),
        DHParameter(d=0.0, theta=0.0, a=-0.39225, alpha=0.0, limit=limit),
        DHParameter(d=0.10915, theta=0.0, a=0.0, alpha=np.pi / 2, limit=limit),
        DHParameter(d=0.09465, theta=0.0, a=0.0, alpha=-np.pi / 2, limit=limit),
        DHParameter(d=0.0823, theta=0.0, a=0.0, alpha=0.0, limit=limit),
    )
    return Robot(
        name="ur5",
        links=links,
        convention=DHConvention.STANDARD,
        source=_UR5_SOURCE,
    )


def stanford_arm() -> Robot:
    """Return the Stanford manipulator in standard DH.

    The arm has five revolute joints and one prismatic joint, so it exercises the
    prismatic branch of the forward kinematics and of the Jacobian. The parameter
    table is

    ==== ============ ========= ========= =========
    Link theta        d (m)     a (m)     alpha
    ==== ============ ========= ========= =========
    1    q1           0.412     0.0       -pi/2
    2    q2           0.154     0.0       +pi/2
    3    -pi/2        q3        0.0       0
    4    q4           0.0       0.0       -pi/2
    5    q5           0.0       0.0       +pi/2
    6    q6           0.263     0.0       0
    ==== ============ ========= ========= =========

    The cited source does not publish a joint limit table in a form that can be
    transcribed without interpretation, so the prismatic joint is given the travel
    range 0.0 m to 0.5 m and the revolute joints plus or minus 170 degrees. These
    are working ranges chosen for this repository, not published values, and they
    are marked as such wherever they are reported.

    Returns:
        The configured robot.
    """
    revolute_limit = JointLimit(float(np.deg2rad(-170.0)), float(np.deg2rad(170.0)))
    links = (
        DHParameter(d=0.412, theta=0.0, a=0.0, alpha=-np.pi / 2, limit=revolute_limit),
        DHParameter(d=0.154, theta=0.0, a=0.0, alpha=np.pi / 2, limit=revolute_limit),
        DHParameter(
            d=0.0,
            theta=-np.pi / 2,
            a=0.0,
            alpha=0.0,
            joint_type=JointType.PRISMATIC,
            limit=JointLimit(0.0, 0.5),
        ),
        DHParameter(d=0.0, theta=0.0, a=0.0, alpha=-np.pi / 2, limit=revolute_limit),
        DHParameter(d=0.0, theta=0.0, a=0.0, alpha=np.pi / 2, limit=revolute_limit),
        DHParameter(d=0.263, theta=0.0, a=0.0, alpha=0.0, limit=revolute_limit),
    )
    return Robot(
        name="stanford",
        links=links,
        convention=DHConvention.STANDARD,
        source=_STANFORD_SOURCE,
    )


ROBOTS: Final[dict[str, Callable[[], Robot]]] = {
    "puma560": puma560,
    "ur5": ur5,
    "stanford": stanford_arm,
}
