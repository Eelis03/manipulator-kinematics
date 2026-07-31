"""Metrics and figures derived from a pipeline trace.

This package reads traces and writes summaries and figures. It never calls a
solver, so a figure can always be redrawn from a recorded trace.
"""

from manipulator_kinematics.analysis.figures import (
    PALETTE,
    convergence_figure,
    save_figure,
    singularity_figure,
    success_figure,
)
from manipulator_kinematics.analysis.metrics import (
    SolverSummary,
    format_scan_table,
    format_summary_table,
    summarise,
)

__all__ = [
    "PALETTE",
    "SolverSummary",
    "convergence_figure",
    "format_scan_table",
    "format_summary_table",
    "save_figure",
    "singularity_figure",
    "success_figure",
    "summarise",
]
