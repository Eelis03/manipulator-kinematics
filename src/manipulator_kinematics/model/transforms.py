"""Homogeneous transform helpers for SE(3).

A pose is a 4x4 array of dtype ``float64`` whose upper-left 3x3 block is a
rotation matrix and whose upper-right 3x1 block is a translation. Every function
here is pure and allocates a new array rather than mutating its argument.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

__all__ = [
    "identity_pose",
    "is_rotation",
    "pose_error",
    "pose_from_rotation_translation",
    "rotation_of",
    "rotx",
    "rotz",
    "se3_inverse",
    "skew",
    "translation_of",
]

Array = NDArray[np.float64]


def identity_pose() -> Array:
    """Return the identity element of SE(3)."""
    return np.eye(4, dtype=np.float64)


def rotz(angle: float) -> Array:
    """Return the 3x3 rotation about the z axis by ``angle`` radians."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def rotx(angle: float) -> Array:
    """Return the 3x3 rotation about the x axis by ``angle`` radians."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def skew(vector: Array) -> Array:
    """Return the 3x3 skew-symmetric matrix ``S`` with ``S @ y == cross(vector, y)``."""
    v = np.asarray(vector, dtype=np.float64).reshape(3)
    return np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
        dtype=np.float64,
    )


def rotation_of(pose: Array) -> Array:
    """Return the rotation block of a 4x4 pose."""
    return np.array(pose[:3, :3], dtype=np.float64)


def translation_of(pose: Array) -> Array:
    """Return the translation block of a 4x4 pose."""
    return np.array(pose[:3, 3], dtype=np.float64)


def pose_from_rotation_translation(rotation: Array, translation: Array) -> Array:
    """Assemble a 4x4 pose from a 3x3 rotation and a 3-vector translation."""
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    pose[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return pose


def se3_inverse(pose: Array) -> Array:
    """Return the inverse of a 4x4 pose without a general matrix inversion."""
    rotation = rotation_of(pose)
    translation = translation_of(pose)
    return pose_from_rotation_translation(rotation.T, -rotation.T @ translation)


def is_rotation(matrix: Array, *, tolerance: float = 1e-9) -> bool:
    """Return True when ``matrix`` is orthonormal with determinant plus one."""
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape != (3, 3):
        return False
    orthonormal = bool(np.allclose(m.T @ m, np.eye(3), atol=tolerance))
    proper = bool(abs(float(np.linalg.det(m)) - 1.0) <= tolerance)
    return orthonormal and proper


def pose_error(current: Array, target: Array) -> Array:
    """Return the 6-vector twist error taking ``current`` towards ``target``.

    The first three components are the translation difference in the base frame.
    The last three are the rotation vector of ``R_target @ R_current.T``, which is
    the base-frame angular displacement consistent with the geometric Jacobian
    produced by :func:`manipulator_kinematics.algorithm.jacobian.geometric_jacobian`.
    """
    position = translation_of(target) - translation_of(current)
    relative = rotation_of(target) @ rotation_of(current).T
    orientation: Array = Rotation.from_matrix(relative).as_rotvec().astype(np.float64)
    return np.concatenate((position, orientation))
