"""Parameters resources for the NLPOptNet frontend."""

from __future__ import annotations
from typing import Callable
from typing import Dict
from typing import Tuple
from dataclasses import dataclass
import jax.numpy as jnp

Array = jnp.ndarray


ParamsDict = Dict[str, Array]


VarsDict = Dict[str, Array]


ScalarModelFun = Callable[[ParamsDict, VarsDict], Array]


VectorModelFun = Callable[[ParamsDict, VarsDict], Array]


FlatScalarFun = Callable[[ParamsDict, Array], Array]


FlatVectorFun = Callable[[ParamsDict, Array], Array]


Array = jnp.ndarray


@dataclass
class ParameterSpec:
    """Describe named parameter blocks and their expected shapes."""

    names: list[str]
    shapes: Dict[str, Tuple[int, ...]]

    @staticmethod
    def from_example(params: Dict[str, Array]) -> "ParameterSpec":
        """Infer a parameter specification from example parameter arrays."""
        return ParameterSpec(
            names=list(params.keys()),
            shapes={k: tuple(jnp.asarray(v).shape) for k, v in params.items()},
        )

    def validate(self, params: Dict[str, Array]) -> None:
        """Validate that runtime parameters match the recorded specification."""
        for name in self.names:
            if name not in params:
                raise KeyError(f"Missing parameter '{name}'")

            got_shape = tuple(jnp.asarray(params[name]).shape)
            expected_shape = self.shapes[name]
            if got_shape != expected_shape:
                raise ValueError(
                    f"Parameter '{name}' expected shape {expected_shape}, got {got_shape}"
                )
