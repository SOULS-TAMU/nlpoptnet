"""Bounds resources for the NLPOptNet frontend."""

from __future__ import annotations
from typing import Callable
from typing import Optional
from dataclasses import dataclass
import jax.numpy as jnp

Array = jnp.ndarray


@dataclass
class BoundSpec:
    """Hold callable lower and upper bound evaluators."""

    lower_fun: Optional[Callable[[dict], Array]] = None
    upper_fun: Optional[Callable[[dict], Array]] = None
