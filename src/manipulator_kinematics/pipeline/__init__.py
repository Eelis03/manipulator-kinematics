"""Campaign drivers that turn solvers and targets into a structured trace.

This package depends on the model and algorithm layers. It performs no input or
output beyond returning trace objects, and it draws nothing.
"""

from manipulator_kinematics.pipeline.runner import (
    perturbed_seeds,
    run_campaign,
    sample_targets,
    scan_joint,
)
from manipulator_kinematics.pipeline.trace import (
    ScanPoint,
    SingularityScan,
    Target,
    Trace,
    Trial,
)

__all__ = [
    "ScanPoint",
    "SingularityScan",
    "Target",
    "Trace",
    "Trial",
    "perturbed_seeds",
    "run_campaign",
    "sample_targets",
    "scan_joint",
]
