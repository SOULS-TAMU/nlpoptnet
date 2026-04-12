from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp


def resolve_dtype(name: str):
    normalized = str(name).strip().lower()
    mapping = {
        "float32": jnp.float32,
        "fp32": jnp.float32,
        "float64": jnp.float64,
        "fp64": jnp.float64,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported dtype '{name}'. Use float32 or float64.")
    return mapping[normalized]


def select_device(name: str):
    requested = str(name).strip().lower()
    if requested in {"", "auto"}:
        for backend in ("gpu", "cpu"):
            try:
                devices = jax.devices(backend)
            except RuntimeError:
                devices = []
            if devices:
                return devices[0]
        raise RuntimeError("No JAX devices available.")

    try:
        devices = jax.devices(requested)
    except RuntimeError as exc:
        raise RuntimeError(f"Requested device backend '{requested}' is unavailable.") from exc
    if not devices:
        raise RuntimeError(f"Requested device backend '{requested}' has no devices.")
    return devices[0]


def runtime_summary(device: Any, dtype) -> str:
    return f"device={device.platform}:{device.id} dtype={jnp.dtype(dtype).name} backend={jax.default_backend()}"
