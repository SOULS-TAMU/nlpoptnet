from __future__ import annotations

import copy
from typing import Any, Mapping


_OVERRIDES = {
    "p": None,
    "n": None,
    "me": None,
    "mi": None,
    "train_frac": None,
}


def set_cli_overrides(
    *,
    p: int | None = None,
    n: int | None = None,
    me: int | None = None,
    mi: int | None = None,
    train_frac: float | None = None,
) -> None:
    _OVERRIDES["p"] = None if p is None else int(p)
    _OVERRIDES["n"] = None if n is None else int(n)
    _OVERRIDES["me"] = None if me is None else int(me)
    _OVERRIDES["mi"] = None if mi is None else int(mi)
    _OVERRIDES["train_frac"] = None if train_frac is None else float(train_frac)


def get_cli_overrides() -> dict[str, Any]:
    return copy.deepcopy(_OVERRIDES)


def _broadcast_or_validate_vector(values: list[Any], target_len: int, *, key: str) -> list[Any]:
    if target_len < 0:
        raise ValueError("target_len must be nonnegative.")
    if len(values) == target_len:
        return copy.deepcopy(values)
    if target_len == 0:
        return []
    if len(values) == 1:
        return copy.deepcopy(values * target_len)
    raise ValueError(
        f"{key} must either have length 1 or match the parameter dimension {target_len}; "
        f"got length {len(values)}."
    )


def apply_cli_overrides(
    data_cfg: Mapping[str, Any],
    cfg_dict: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    data = copy.deepcopy(dict(data_cfg))
    cfg = None if cfg_dict is None else copy.deepcopy(dict(cfg_dict))

    param_dim: int | None = None
    if _OVERRIDES["p"] is not None:
        if "p" in data:
            data["p"] = int(_OVERRIDES["p"])
            param_dim = int(_OVERRIDES["p"])
        elif "n_x" in data:
            data["n_x"] = int(_OVERRIDES["p"])
            param_dim = int(_OVERRIDES["p"])

    if _OVERRIDES["n"] is not None:
        if "n" in data:
            data["n"] = int(_OVERRIDES["n"])
        elif "n_y" in data:
            data["n_y"] = int(_OVERRIDES["n"])

    if _OVERRIDES["me"] is not None:
        if "me" in data:
            data["me"] = int(_OVERRIDES["me"])
        elif "n_eq" in data:
            data["n_eq"] = int(_OVERRIDES["me"])

    if _OVERRIDES["mi"] is not None:
        if "mi" in data:
            data["mi"] = int(_OVERRIDES["mi"])
        elif "n_ineq" in data:
            data["n_ineq"] = int(_OVERRIDES["mi"])

    if param_dim is not None:
        for key in ("x_L", "x_U"):
            if key in data and isinstance(data[key], (list, tuple)):
                data[key] = _broadcast_or_validate_vector(list(data[key]), param_dim, key=key)

    if cfg is not None and _OVERRIDES["train_frac"] is not None:
        cfg["train_frac"] = float(_OVERRIDES["train_frac"])

    return data, cfg
