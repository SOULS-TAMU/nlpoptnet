"""Projection resources for the NLPOptNet frontend."""

from __future__ import annotations
from typing import Callable
from purin.projections.adapters import CallableCorrection
from ohagi.nn.adapters.flax import FlaxLinenBackbone
from typing import Optional
from ohagi.models import ProjectedModel
from nlpoptnet.api.configuration import TrainConfig
from typing import Tuple
import jax.numpy as jnp
from purin.solvers.chambolle_pock.dense_layer import solve_dense_implicit


def projection(sub_layer, cfg):
    """Supply local nonlinear QP coefficients to PURIN through a correction adapter."""

    def step(context, target, equality, inequality):
        coefficients = sub_layer(context, target)
        return solve_cp_implicit(
            *coefficients, cfg=cfg, y0=target, lam0=equality, mu0=inequality
        )

    return CallableCorrection(step, int(cfg.k_layer))


def projected_model(backbone, sub_layer, cfg):
    """Compose the Flax backbone and PURIN corrections using OHAGI's model contract."""
    return ProjectedModel(
        FlaxLinenBackbone(backbone, mode_keyword=None), projection(sub_layer, cfg)
    )


Array = jnp.ndarray


def solve_cp_implicit(
    Q_diag: Array,
    c: Array,
    A: Array,
    b: Array,
    C: Array,
    d: Array,
    l: Array,
    u: Array,
    *,
    cfg: TrainConfig,
    y0: Optional[Array] = None,
    lam0: Optional[Array] = None,
    mu0: Optional[Array] = None,
    return_iterations: bool = False,
):
    """Map NLPOPTNet configuration to PURIN's shared implicit CP solver.

    Args:
        Q_diag: Batched local objective diagonal, shape ``(batch, n)``.
        c: Batched local linear objective coefficients.
        A: Linearized equality matrices.
        b: Linearized equality right-hand sides.
        C: Linearized inequality matrices.
        d: Linearized inequality right-hand sides.
        l: Lower bounds in the current optimization coordinates.
        u: Upper bounds in the current optimization coordinates.
        cfg: Frontend options. ``IS_FIXED`` selects fixed-step CP when True
            and accelerated CP when False.
        y0: Optional primal warm start.
        lam0: Optional equality-multiplier warm start.
        mu0: Optional inequality-multiplier warm start.
        return_iterations: Request PURIN iteration counts for inference.

    Returns:
        PURIN's tuple of primal, equality-multiplier and inequality-multiplier
        arrays. Scaling and implicit derivatives are handled entirely by PURIN.

    Notes:
        This function translates names and configuration only; it contains no
        local solver iteration or differentiation implementation.
    """
    sparse_options = {}
    if getattr(cfg, "inequality_support", None) is not None:
        sparse_options["inequality_support"] = cfg.inequality_support
    if return_iterations:
        sparse_options["return_iterations"] = True
    return solve_dense_implicit(
        Q_diag,
        c,
        A,
        b,
        C,
        d,
        l,
        u,
        accelerated=not cfg.IS_FIXED,
        safety=cfg.safety,
        norm_iterations=cfg.knorm_iters,
        norm_seed=cfg.knorm_seed,
        max_iterations=cfg.cp_iters,
        tolerance=cfg.cp_tol,
        adjoint_iterations=cfg.adjoint_iters,
        equilibrate=cfg.use_ruiz,
        equilibration_iterations=cfg.ruiz_iters,
        y0=y0,
        lambda0=lam0,
        mu0=mu0,
        **sparse_options,
    )


def apply_projection_layers(
    *,
    sub_layer: Callable[
        [Array, Array], Tuple[Array, Array, Array, Array, Array, Array, Array, Array]
    ],
    x_batch: Array,
    y0: Array,
    lam0: Array,
    mu0: Array,
    cfg: TrainConfig,
):
    """Apply the learned projection layer repeatedly according to ``cfg.k_layer``."""
    from nlpoptnet.api.projection import projection

    result = projection(sub_layer, cfg).project_training(
        y0, x_batch, multipliers_eq=lam0, multipliers_ineq=mu0
    )
    return result.y, result.multipliers_eq, result.multipliers_ineq
