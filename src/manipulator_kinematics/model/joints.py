"""Joint limits and configuration sampling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "JointLimit",
    "clamp_to_limits",
    "limit_span",
    "sample_within_limits",
    "within_limits",
]

Array = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class JointLimit:
    """Inclusive travel range of one joint.

    Units are radians for a revolute joint and metres for a prismatic joint.
    """

    lower: float
    upper: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.lower) or not np.isfinite(self.upper):
            raise ValueError("joint limits must be finite")
        if self.lower > self.upper:
            raise ValueError(f"lower limit {self.lower} exceeds upper limit {self.upper}")

    @property
    def midpoint(self) -> float:
        """Return the centre of the travel range."""
        return 0.5 * (self.lower + self.upper)

    @property
    def span(self) -> float:
        """Return the width of the travel range."""
        return self.upper - self.lower


def limit_span(limits: tuple[JointLimit, ...]) -> tuple[Array, Array]:
    """Return the lower and upper bound vectors of a limit tuple."""
    lower = np.array([limit.lower for limit in limits], dtype=np.float64)
    upper = np.array([limit.upper for limit in limits], dtype=np.float64)
    return lower, upper


def within_limits(q: Array, limits: tuple[JointLimit, ...], *, tolerance: float = 0.0) -> bool:
    """Return True when every joint value lies inside its limit, up to ``tolerance``."""
    lower, upper = limit_span(limits)
    values = np.asarray(q, dtype=np.float64)
    return bool(np.all(values >= lower - tolerance) and np.all(values <= upper + tolerance))


def clamp_to_limits(q: Array, limits: tuple[JointLimit, ...]) -> Array:
    """Return ``q`` clipped element-wise into the limit box."""
    lower, upper = limit_span(limits)
    return np.clip(np.asarray(q, dtype=np.float64), lower, upper)


def sample_within_limits(
    limits: tuple[JointLimit, ...],
    rng: np.random.Generator,
    *,
    margin: float = 0.0,
) -> Array:
    """Draw one configuration uniformly from the limit box, shrunk by ``margin``.

    ``margin`` is a fraction of each joint span removed from both ends, which keeps
    sampled configurations away from the hard stops.
    """
    lower, upper = limit_span(limits)
    inset = margin * (upper - lower)
    return rng.uniform(lower + inset, upper - inset).astype(np.float64)
