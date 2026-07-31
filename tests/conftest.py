"""Shared fixtures."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from manipulator_kinematics.model import Robot, puma560, stanford_arm, ur5

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"

ROBOT_FACTORIES: tuple[Callable[[], Robot], ...] = (puma560, ur5, stanford_arm)


@pytest.fixture(params=ROBOT_FACTORIES, ids=lambda factory: factory().name)
def robot(request: pytest.FixtureRequest) -> Robot:
    """Every shipped robot model, one per test invocation."""
    factory: Callable[[], Robot] = request.param
    return factory()


@pytest.fixture
def rng() -> np.random.Generator:
    """A generator with a fixed seed, so every test is reproducible."""
    return np.random.default_rng(20260731)
