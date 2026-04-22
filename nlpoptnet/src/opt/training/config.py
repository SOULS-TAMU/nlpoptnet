from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class TrainConfig:
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


def cfg_from_dict(d: Dict[str, Any]) -> TrainConfig:
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
    return TrainConfig(**base)
