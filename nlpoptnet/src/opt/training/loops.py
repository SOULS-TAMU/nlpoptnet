from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

import jax
import jax.numpy as jnp

from ..models import State
from .config import TrainConfig

Array = jnp.ndarray
_METRIC_KEYS = ("loss", "obj", "mse_y", "mse_lam", "mse_mu")


def build_epoch_fns(train_step, eval_step):
    @jax.jit
    def train_epoch(state, batches):
        def body(carry, xb):
            st = carry
            st, metrics = train_step(st, xb)
            return st, tuple(metrics[k] for k in _METRIC_KEYS)

        state_out, per_batch = jax.lax.scan(body, state, batches)
        means = tuple(jnp.mean(v) for v in per_batch)
        return state_out, means

    @jax.jit
    def eval_epoch(params, batches):
        def one_batch(xb):
            metrics = eval_step(params, xb)
            return tuple(metrics[k] for k in _METRIC_KEYS)

        per_batch = jax.vmap(one_batch)(batches)
        means = tuple(jnp.mean(v) for v in per_batch)
        return means

    return train_epoch, eval_epoch


def warmup_compile(
    *,
    cfg: TrainConfig,
    state: State,
    train_step_fn: Callable[[State, Array], Tuple[State, Dict[str, Array]]],
    eval_step_fn: Callable[[Any, Array], Dict[str, Array]],
    p: int,
    dtype=jnp.float64,
    device=None,
):
    key = jax.random.PRNGKey(cfg.seed)
    xb = jax.random.normal(key, (cfg.batch_size, p), dtype=dtype)
    if device is not None:
        xb = jax.device_put(xb, device)
        state = jax.device_put(state, device)
    state2, _ = train_step_fn(state, xb)
    _ = eval_step_fn(state2.params, xb)
    jax.block_until_ready(state2.params)
    return state2


def make_fixed_batches(X: Array, batch_size: int) -> Array:
    n_samples = X.shape[0]
    n_trimmed = (n_samples // batch_size) * batch_size
    X = X[:n_trimmed]
    return X.reshape((n_trimmed // batch_size, batch_size, X.shape[1]))
