from __future__ import annotations

from typing import Tuple

import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state

Array = jnp.ndarray


class Backbone(nn.Module):
    p: int
    n: int
    me: int
    mi: int
    hidden_size: int
    hidden_dim: int

    @nn.compact
    def __call__(self, x: Array) -> Tuple[Array, Array, Array]:
        h = x
        for _ in range(self.hidden_dim):
            h = nn.Dense(self.hidden_size)(h)
            h = nn.tanh(h)
        y = nn.Dense(self.n)(h)
        lam = nn.Dense(self.me)(h) if self.me > 0 else jnp.zeros((x.shape[0], 0), dtype=x.dtype)
        mu = nn.Dense(self.mi)(h) if self.mi > 0 else jnp.zeros((x.shape[0], 0), dtype=x.dtype)
        return y, lam, mu


class State(train_state.TrainState):
    pass
