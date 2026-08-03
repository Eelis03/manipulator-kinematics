"""Every example script runs to completion.

Each script is launched as a subprocess with the interpreter running the tests,
so the test exercises the same entry point a reader would type. Iteration counts,
target counts and sweep resolutions are reduced through the command line flags
the scripts already expose, which keeps the whole tier inside a few seconds while
still running every code path the full run uses.

Figures are written to a temporary directory, so a test run never touches the
repository working tree. In particular the tracked figures under ``docs/figures``
are never overwritten by a test run, only by the documented regeneration command.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import EXAMPLES_DIR, REPO_ROOT

EXAMPLE_INVOCATIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("forward_kinematics_tour.py", ("--robots", "puma560", "ur5", "stanford")),
    ("analytic_ik_puma560.py", ("--poses", "5")),
    (
        "compare_ik_solvers.py",
        ("--robots", "puma560", "ur5", "--targets", "4", "--max-iterations", "12"),
    ),
    ("singularity_scan.py", ("--points", "21")),
    (
        "publish_figures.py",
        ("--targets", "4", "--max-iterations", "12", "--points", "21"),
    ),
)


def _run(
    script: str, arguments: tuple[str, ...], output_dir: Path
) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, str(EXAMPLES_DIR / script), *arguments]
    if "--output-dir" not in arguments and _accepts_output_dir(script):
        command += ["--output-dir", str(output_dir)]
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def _accepts_output_dir(script: str) -> bool:
    return "--output-dir" in (EXAMPLES_DIR / script).read_text(encoding="utf-8")


def test_every_example_is_covered() -> None:
    """No example script is left out of this tier."""
    scripts = {path.name for path in EXAMPLES_DIR.glob("*.py")}
    assert scripts == {script for script, _ in EXAMPLE_INVOCATIONS}


@pytest.mark.parametrize(("script", "arguments"), EXAMPLE_INVOCATIONS, ids=lambda v: str(v)[:40])
def test_example_runs_to_completion(
    script: str, arguments: tuple[str, ...], tmp_path: Path
) -> None:
    """The script exits zero and prints something."""
    completed = _run(script, arguments, tmp_path)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip()


def test_examples_write_the_figures_they_promise(tmp_path: Path) -> None:
    """The two plotting scripts leave readable PNG files behind."""
    for script, arguments in EXAMPLE_INVOCATIONS:
        if not _accepts_output_dir(script):
            continue
        target = tmp_path / script.removesuffix(".py")
        completed = _run(script, arguments, target)
        assert completed.returncode == 0, completed.stderr
        written = sorted(target.glob("*.png"))
        assert written, f"{script} wrote no figure"
        for path in written:
            assert path.stat().st_size > 1000


def test_example_help_is_available() -> None:
    """Every script documents its own flags."""
    for script, _ in EXAMPLE_INVOCATIONS:
        completed = subprocess.run(
            [sys.executable, str(EXAMPLES_DIR / script), "--help"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        assert "usage:" in completed.stdout
