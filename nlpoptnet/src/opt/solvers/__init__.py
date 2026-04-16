"""Solver entrypoints for the opt module."""

from ..CP_jax import CP_accelerated, CP_accelerated_jit, CP_fixed, CP_fixed_jit
from ..CP_jax_implicit import (
    CP_accelerated_implicit,
    CP_accelerated_implicit_jit,
    CP_fixed_implicit,
    CP_fixed_implicit_jit,
)

__all__ = [
    "CP_fixed",
    "CP_fixed_jit",
    "CP_accelerated",
    "CP_accelerated_jit",
    "CP_fixed_implicit",
    "CP_fixed_implicit_jit",
    "CP_accelerated_implicit",
    "CP_accelerated_implicit_jit",
]
