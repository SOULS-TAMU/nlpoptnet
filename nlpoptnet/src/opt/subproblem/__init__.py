"""Subproblem extraction utilities for jaxmodel-backed optimization."""

from ..model_bridge import (
    CPSolveResult,
    extract_projection_subproblem,
    extract_sqp_subproblem,
    solve_projection_subproblem_with_cp,
    solve_sqp_subproblem_with_cp,
)

__all__ = [
    "CPSolveResult",
    "extract_projection_subproblem",
    "solve_projection_subproblem_with_cp",
    "extract_sqp_subproblem",
    "solve_sqp_subproblem_with_cp",
]
