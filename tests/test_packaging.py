"""What the installed distribution carries.

A package that passes mypy in strict mode still delivers nothing to its callers
unless it says so. PEP 561 makes that claim with a marker file inside the package
directory, and a marker outside it, or missing from the wheel configuration, is
silently ignored by every type checker.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import manipulator_kinematics
from tests.conftest import REPO_ROOT

PACKAGE_ROOT = Path(manipulator_kinematics.__file__).resolve().parent


def test_py_typed_marker_sits_inside_the_package_directory() -> None:
    """The marker is next to ``__init__.py``, which is where PEP 561 looks."""
    marker = PACKAGE_ROOT / "py.typed"
    assert marker.is_file(), f"no py.typed marker in {PACKAGE_ROOT}"
    assert (PACKAGE_ROOT / "__init__.py").is_file()
    assert marker.read_bytes() == b""


def test_the_wheel_is_configured_to_ship_the_package_directory() -> None:
    """The build backend packages the directory the marker lives in."""
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    packaged = config["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]
    assert "src/manipulator_kinematics" in packaged
    assert (REPO_ROOT / "src" / "manipulator_kinematics" / "py.typed").is_file()
