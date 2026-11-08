"""Configuration resources for the NLPOptNet frontend."""

from __future__ import annotations
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from dataclasses import asdict
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class TrainConfig:
    r"""Typed training and projection hyperparameters.

    Attributes
    ----------
    batch_size:
        Mini-batch size used for train and validation updates.
    epochs:
        Number of optimization epochs.
    learning_rate:
        Adam learning rate for the neural backbone.
    alpha_consistency:
        Weight applied to the projection-consistency penalty.
    train_frac, val_frac:
        Fractions used when splitting parameter samples.
    hidden_size, hidden_dim:
        Backbone width and number of hidden layers.
    cp_iters, cp_tol, cp_mode:
        Projection-operator iteration budget, tolerance, and mode.
    safety, knorm_iters, knorm_seed:
        Step-size and operator-norm estimation controls.
    adjoint_iters:
        Iteration budget used in implicit differentiation.
    use_ruiz, ruiz_iters:
        Controls for Ruiz equilibration.
    k_layer:
        Number of projection layers to apply after the backbone.
    inequality_support:
        Optional structural column indices per inequality row. All possible
        nonzeros must be declared; omitted entries are structurally zero.
        Enables packed sparse products in both CP and its implicit pullback.
    dtype, device, jit_warmup:
        Numeric precision and execution settings.

    Notes
    -----
    The projection-oriented part of the training loop can be viewed as

    .. math::

        (\hat y, \hat \lambda, \hat \mu) = \Phi_\theta(x), \qquad
        (y^\star, \lambda^\star, \mu^\star) = \Pi(\hat y, \hat \lambda, \hat \mu; x),

    where ``k_layer``, ``cp_mode``, ``cp_iters``, ``cp_tol``, ``safety``, and
    the Ruiz-scaling options control the projection operator :math:`\Pi`.
    """

    batch_size: int = 500
    epochs: int = 5000
    learning_rate: float = 1e-3
    alpha_consistency: float = 10.0
    train_frac: float = 0.8
    val_frac: float = 0.2
    hidden_size: int = 128
    hidden_dim: int = 2
    cp_iters: int = 500
    cp_tol: float = 1e-9
    cp_mode: str = "fixed"
    IS_FIXED: bool = True
    stepsize: str = "auto"
    safety: float = 0.95
    knorm_iters: int = 25
    knorm_seed: int = 42
    seed: int = 42
    adjoint_iters: int = 25
    use_ruiz: bool = True
    ruiz_iters: int = 10
    k_layer: int = 1
    dtype: str = "float64"
    device: str = "auto"
    jit_warmup: bool = True
    inequality_support: tuple[tuple[int, ...], ...] | None = None


def cfg_from_dict(d: Dict[str, Any]) -> TrainConfig:
    """Normalize a config dictionary into a validated :class:`TrainConfig`."""
    base = asdict(TrainConfig())
    base.update(d)
    if "cp_mode" in d:
        mode = str(d["cp_mode"]).strip().lower()
        if mode not in {"fixed", "accelerated"}:
            raise ValueError("cp_mode must be either 'fixed' or 'accelerated'.")
        base["cp_mode"] = mode
        base["IS_FIXED"] = mode == "fixed"
    elif "IS_FIXED" in d:
        base["IS_FIXED"] = bool(d["IS_FIXED"])
        base["cp_mode"] = "fixed" if bool(d["IS_FIXED"]) else "accelerated"
    if "use_ruiz" in d:
        base["use_ruiz"] = bool(d["use_ruiz"])
    if "k_layer" in d:
        base["k_layer"] = int(d["k_layer"])
    if int(base["k_layer"]) < 1:
        raise ValueError("k_layer must be at least 1.")
    if base["inequality_support"] is not None:
        base["inequality_support"] = tuple(
            tuple(row) for row in base["inequality_support"]
        )
    return TrainConfig(**base)


_METRIC_KEYS = (
    "loss",
    "obj",
    "consistency",
    "eq_violation",
    "ineq_violation",
    "mse_y",
    "mse_lam",
    "mse_mu",
)


_TYPE_ALIASES = {"nonconvx": "nonconvex"}


_REQUIRED_CONFIG_KEYS = (
    "epochs",
    "batch_size",
    "learning_rate",
    "hidden_size",
    "hidden_layers",
)


_DEFAULT_CONFIG = {
    "train_frac": 0.8,
    "seed": 42,
    "alpha_consistency": 10.0,
    "cp_mode": "fixed",
    "cp_iters": 500,
    "cp_tol": 1e-9,
    "safety": 0.95,
    "knorm_iters": 25,
    "knorm_seed": 42,
    "adjoint_iters": 25,
    "k_layer": 1,
    "use_ruiz": True,
    "ruiz_iters": 10,
    "dtype": "float64",
    "print_every": 50,
    "device": "auto",
    "verbose": True,
    "stepsize": "auto",
    "jit_warmup": True,
    "num_samples": 1000,
    "y_bound": 10.0,
}


def _normalize_problem_type(problem_type: str | None) -> str | None:
    if problem_type is None:
        return None
    normalized = _TYPE_ALIASES.get(
        str(problem_type).strip().lower(), str(problem_type).strip().lower()
    )
    return normalized


def _normalize_cp_mode(value: Any) -> str:
    mode = str(value).strip().lower()
    if mode not in {"fixed", "accelerated"}:
        raise ValueError("cp_mode must be either 'fixed' or 'accelerated'.")
    return mode


def _apply_config_aliases(config: dict[str, Any]) -> None:
    if "cp_mode" in config:
        mode = _normalize_cp_mode(config["cp_mode"])
        config["cp_mode"] = mode
        config["IS_FIXED"] = mode == "fixed"
    elif "IS_FIXED" in config:
        config["IS_FIXED"] = bool(config["IS_FIXED"])
        config["cp_mode"] = "fixed" if bool(config["IS_FIXED"]) else "accelerated"
    else:
        config["cp_mode"] = "fixed"
        config["IS_FIXED"] = True


def _coerce_names(names: str | Iterable[str]) -> list[str]:
    if isinstance(names, str):
        out = [names]
    else:
        out = [str(name) for name in names]
    if not out:
        raise ValueError("At least one name is required.")
    if len(set(out)) != len(out):
        raise ValueError(f"Names must be unique, got {out}.")
    return out


def _metrics_tuple_to_dict(values) -> dict[str, float]:
    return {key: float(value) for key, value in zip(_METRIC_KEYS, values)}


@dataclass
class _BuildState:
    model: Any
    cfg: Any
    train_config: dict[str, Any]
    dtype: Any
    X: np.ndarray
    X_original: np.ndarray
    train_idx: np.ndarray
    val_idx: np.ndarray
    train_batches: Any
    val_batches: Any
    backbone: Any
    init_state_fn: Callable
    train_step_fn: Callable
    eval_step_fn: Callable
    train_epoch_fn: Callable
    eval_epoch_fn: Callable
    violation_fn: Callable
    objective_fn: Callable
    sub_layer: Callable
    lower_M: np.ndarray
    lower_c: np.ndarray
    upper_M: np.ndarray
    upper_c: np.ndarray


@dataclass
class _ScalingState:
    enabled: bool = False

    D_p: np.ndarray | None = None
    D_v: np.ndarray | None = None
    D_obj: float = 1.0
    D_eq: np.ndarray | None = None
    D_ineq: np.ndarray | None = None

    eps: float = 1e-12
    objective_percentile: float = 95.0
    constraint_percentile: float = 95.0
    probe_points: int = 8
    mode: str = "fast"
    use_jacobian_scaling: bool = False

    manual_D_p: np.ndarray | None = None
    manual_D_v: np.ndarray | None = None
    manual_D_obj: float | None = None
    manual_D_eq: np.ndarray | None = None
    manual_D_ineq: np.ndarray | None = None


@dataclass
class _InferenceState:
    model: Any
    cfg: Any
    train_config: dict[str, Any]
    params: Any
    backbone: Any
    sub_layer: Callable
    n_x: int
    output_dir: str | None


def _train_cfg_dict(self, batch_size: int) -> dict[str, Any]:
    _apply_config_aliases(self.config)
    return {
        "inequality_support": self.config.get("inequality_support"),
        "batch_size": int(batch_size),
        "epochs": int(self.config["epochs"]),
        "learning_rate": float(self.config["learning_rate"]),
        "alpha_consistency": float(
            self.config.get("alpha_consistency", _DEFAULT_CONFIG["alpha_consistency"])
        ),
        "train_frac": float(self.config["train_frac"]),
        "hidden_size": int(self.config["hidden_size"]),
        "hidden_dim": int(self.config["hidden_layers"]),
        "cp_mode": str(self.config.get("cp_mode", _DEFAULT_CONFIG["cp_mode"])),
        "cp_iters": int(self.config.get("cp_iters", _DEFAULT_CONFIG["cp_iters"])),
        "cp_tol": float(self.config.get("cp_tol", _DEFAULT_CONFIG["cp_tol"])),
        "IS_FIXED": bool(self.config.get("IS_FIXED", True)),
        "stepsize": str(self.config.get("stepsize", "auto")),
        "safety": float(self.config.get("safety", _DEFAULT_CONFIG["safety"])),
        "knorm_iters": int(
            self.config.get("knorm_iters", _DEFAULT_CONFIG["knorm_iters"])
        ),
        "knorm_seed": int(self.config.get("knorm_seed", _DEFAULT_CONFIG["knorm_seed"])),
        "seed": int(self.config["seed"]),
        "adjoint_iters": int(
            self.config.get("adjoint_iters", _DEFAULT_CONFIG["adjoint_iters"])
        ),
        "use_ruiz": bool(self.config.get("use_ruiz", _DEFAULT_CONFIG["use_ruiz"])),
        "ruiz_iters": int(self.config.get("ruiz_iters", _DEFAULT_CONFIG["ruiz_iters"])),
        "k_layer": int(self.config.get("k_layer", _DEFAULT_CONFIG["k_layer"])),
        "dtype": str(self.config["dtype"]),
        "device": str(self.config.get("device", _DEFAULT_CONFIG["device"])),
        "jit_warmup": bool(
            self.config.get("jit_warmup", _DEFAULT_CONFIG["jit_warmup"])
        ),
    }
