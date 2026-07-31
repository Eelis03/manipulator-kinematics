"""Singularity metrics derived from the singular values of the Jacobian.

Three scalars are computed from the singular values ``s_1 >= ... >= s_m`` of the
geometric Jacobian.

Manipulability
    ``w = sqrt(det(J J^T))``, which equals the product of the singular values.
    Introduced by T. Yoshikawa, 'Manipulability of robotic mechanisms',
    International Journal of Robotics Research 4(2), 1985, doi:10.1177/027836498500400201.
    It vanishes exactly at a singularity.

Condition number
    ``s_1 / s_m``. Introduced as a kinematic conditioning index by J. K. Salisbury
    and J. J. Craig, 'Articulated hands: force control and kinematic issues',
    International Journal of Robotics Research 1(1), 1982,
    doi:10.1177/027836498200100102. It diverges at a singularity and is
    dimensionally inhomogeneous for a Jacobian mixing translation and rotation
    rows, so it is reported alongside the two scale-free alternatives rather than
    on its own.

Smallest singular value
    ``s_m``, the distance in the spectral norm from ``J`` to the nearest rank
    deficient matrix. It is the metric used to schedule damping in the damped
    least squares solver.

The rotation rows of a geometric Jacobian have units of ``1/rad`` and the
translation rows ``m``, so a length scale is needed before the two blocks can be
compared. :func:`conditioning` therefore takes a ``characteristic_length`` and
divides the rotation rows by it, following the homogenisation discussed in
J. Angeles, *Fundamentals of Robotic Mechanical Systems*, Springer 2007, section
4.9. The default is the maximum reach of the chain, which the caller supplies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import svdvals

__all__ = ["Conditioning", "conditioning", "manipulability"]

Array = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Conditioning:
    """Singularity metrics of one Jacobian evaluation."""

    manipulability: float
    condition_number: float
    smallest_singular_value: float
    singular_values: tuple[float, ...]

    @property
    def is_singular(self) -> bool:
        """True when the smallest singular value is at or below 1e-9."""
        return self.smallest_singular_value <= 1e-9


def _homogenise(jacobian: Array, characteristic_length: float) -> Array:
    if characteristic_length <= 0.0:
        raise ValueError("characteristic_length must be positive")
    scaled = np.array(jacobian, dtype=np.float64)
    scaled[3:, :] /= characteristic_length
    return scaled


def manipulability(jacobian: Array) -> float:
    """Return the Yoshikawa manipulability index of ``jacobian``.

    Computed as the product of the singular values rather than as
    ``sqrt(det(J J^T))``, because forming ``J J^T`` squares the condition number
    and loses roughly half the available precision near a singularity.

    Args:
        jacobian: A 6 by n array.

    Returns:
        The manipulability index, zero at a singular configuration.
    """
    values = svdvals(np.asarray(jacobian, dtype=np.float64))
    return float(np.prod(values))


def conditioning(jacobian: Array, *, characteristic_length: float = 1.0) -> Conditioning:
    """Return the manipulability, condition number and smallest singular value.

    Args:
        jacobian: A 6 by n array, ordered ``[vx, vy, vz, wx, wy, wz]``.
        characteristic_length: Length in metres used to make the rotation rows
            dimensionally comparable with the translation rows.

    Returns:
        The three metrics together with the full singular value spectrum.
    """
    scaled = _homogenise(np.asarray(jacobian, dtype=np.float64), characteristic_length)
    values = np.asarray(svdvals(scaled), dtype=np.float64)
    smallest = float(values[-1])
    largest = float(values[0])
    ratio = float(np.inf) if smallest <= 0.0 else largest / smallest
    return Conditioning(
        manipulability=float(np.prod(values)),
        condition_number=ratio,
        smallest_singular_value=smallest,
        singular_values=tuple(float(v) for v in values),
    )
