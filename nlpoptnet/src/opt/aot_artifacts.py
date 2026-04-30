"""Artifact helpers for exporting backbone weights and derivative evaluators."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


def _dense_key(index: int) -> str:
    """Return the Flax parameter-tree key for a dense layer index."""
    return f"Dense_{index}"


def save_backbone_npz(
    path: str | Path,
    params: dict[str, Any],
    *,
    p: int,
    n: int,
    me: int,
    mi: int,
    hidden_size: int,
    hidden_dim: int,
    dtype: str,
) -> dict[str, Any]:
    """Save Flax dense weights as a plain, pickle-free ``.npz`` artifact."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    dense_index = 0

    for layer in range(int(hidden_dim)):
        dense = params[_dense_key(dense_index)]
        arrays[f"hidden_{layer}_kernel"] = np.asarray(dense["kernel"])
        arrays[f"hidden_{layer}_bias"] = np.asarray(dense["bias"])
        dense_index += 1

    y_dense = params[_dense_key(dense_index)]
    arrays["y_kernel"] = np.asarray(y_dense["kernel"])
    arrays["y_bias"] = np.asarray(y_dense["bias"])
    dense_index += 1

    if int(me) > 0:
        lam_dense = params[_dense_key(dense_index)]
        arrays["lam_kernel"] = np.asarray(lam_dense["kernel"])
        arrays["lam_bias"] = np.asarray(lam_dense["bias"])
        dense_index += 1
    else:
        arrays["lam_kernel"] = np.zeros((int(hidden_size), 0), dtype=np.float64)
        arrays["lam_bias"] = np.zeros((0,), dtype=np.float64)

    if int(mi) > 0:
        mu_dense = params[_dense_key(dense_index)]
        arrays["mu_kernel"] = np.asarray(mu_dense["kernel"])
        arrays["mu_bias"] = np.asarray(mu_dense["bias"])
    else:
        arrays["mu_kernel"] = np.zeros((int(hidden_size), 0), dtype=np.float64)
        arrays["mu_bias"] = np.zeros((0,), dtype=np.float64)

    metadata = {
        "format": "nlpoptnet-backbone-v1",
        "p": int(p),
        "n": int(n),
        "me": int(me),
        "mi": int(mi),
        "hidden_size": int(hidden_size),
        "hidden_dim": int(hidden_dim),
        "dtype": str(dtype),
        "activation": "tanh",
    }
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    np.savez(target, **arrays)
    return metadata


def load_backbone_npz(path: str | Path) -> dict[str, Any]:
    """Load a backbone artifact created by :func:`save_backbone_npz`."""
    source = Path(path)
    loaded = np.load(source, allow_pickle=False)
    metadata = json.loads(str(loaded["metadata_json"].item()))
    arrays = {key: np.asarray(loaded[key]) for key in loaded.files if key != "metadata_json"}
    return {"metadata": metadata, "arrays": arrays}


def backbone_forward_numpy(artifact: dict[str, Any], x_values: np.ndarray):
    """Run the saved backbone in NumPy without requiring Flax at inference time."""
    metadata = artifact["metadata"]
    arrays = artifact["arrays"]
    x = np.asarray(x_values, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    h = x
    for layer in range(int(metadata["hidden_dim"])):
        h = np.tanh(h @ arrays[f"hidden_{layer}_kernel"] + arrays[f"hidden_{layer}_bias"])
    y = h @ arrays["y_kernel"] + arrays["y_bias"]
    lam = h @ arrays["lam_kernel"] + arrays["lam_bias"] if int(metadata["me"]) > 0 else np.zeros((x.shape[0], 0))
    mu = h @ arrays["mu_kernel"] + arrays["mu_bias"] if int(metadata["mi"]) > 0 else np.zeros((x.shape[0], 0))
    return y, lam, mu


def _export_one(name: str, fn, specs: tuple[Any, ...], out_dir: Path) -> dict[str, Any]:
    """Export one JAX function to a serialized StableHLO artifact."""
    from jax import export

    exported = export.export(jax.jit(fn))(*specs)
    serialization = "jax_exported_flatbuffer"
    serialization_error = None
    try:
        blob = bytes(exported.serialize())
    except ImportError as exc:
        # jax.export creates StableHLO portable artifact bytes before the
        # optional flatbuffer wrapper is needed. Persist those bytes rather
        # than falling back to compiler_ir/debug text.
        blob = bytes(exported.mlir_module_serialized)
        serialization = "stablehlo_portable_artifact"
        serialization_error = f"{type(exc).__name__}: {exc}"
    filename = f"{name}.stablehlo"
    (out_dir / filename).write_bytes(blob)
    payload = {
        "file": filename,
        "function_name": exported.fun_name,
        "in_avals": [str(aval) for aval in exported.in_avals],
        "out_avals": [str(aval) for aval in exported.out_avals],
        "calling_convention_version": int(exported.calling_convention_version),
        "serialization": serialization,
    }
    if serialization_error is not None:
        payload["serialization_error"] = serialization_error
    return payload


def export_projection_derivatives(
    model,
    out_dir: str | Path,
    *,
    p: int,
    n: int,
    dtype,
    param_name: str = "x",
) -> dict[str, Any]:
    """Persist StableHLO derivative evaluators for archival and reuse."""
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    x_spec = jax.ShapeDtypeStruct((int(p),), dtype)
    y_spec = jax.ShapeDtypeStruct((int(n),), dtype)

    def params(x):
        return {param_name: x}

    exports = {
        "objective_gradient": _export_one(
            "objective_gradient",
            lambda x, y: model.grad_y_objective(params(x), y),
            (x_spec, y_spec),
            target,
        ),
        "objective_diag_hessian": _export_one(
            "objective_diag_hessian",
            lambda x, y: model.diag_hess_y_objective(params(x), y),
            (x_spec, y_spec),
            target,
        ),
        "equality_jacobian": _export_one(
            "equality_jacobian",
            lambda x, y: model.jac_y_eq(params(x), y),
            (x_spec, y_spec),
            target,
        ),
        "inequality_jacobian": _export_one(
            "inequality_jacobian",
            lambda x, y: model.jac_y_ineq(params(x), y),
            (x_spec, y_spec),
            target,
        ),
        "equality_residual": _export_one(
            "equality_residual",
            lambda x, y: model.eq_residual(params(x), y),
            (x_spec, y_spec),
            target,
        ),
        "inequality_residual": _export_one(
            "inequality_residual",
            lambda x, y: model.ineq_residual(params(x), y),
            (x_spec, y_spec),
            target,
        ),
        "lower_bounds": _export_one(
            "lower_bounds",
            lambda x: model.lower_bounds(params(x)),
            (x_spec,),
            target,
        ),
        "upper_bounds": _export_one(
            "upper_bounds",
            lambda x: model.upper_bounds(params(x)),
            (x_spec,),
            target,
        ),
    }
    manifest = {
        "format": "nlpoptnet-stablehlo-derivatives-v1",
        "jax_version": jax.__version__,
        "dtype": str(jnp.dtype(dtype)),
        "parameter_shape": [int(p)],
        "variable_shape": [int(n)],
        "exports": exports,
        "note": "These files are serialized jax.export artifacts, not compiler_ir/debug text.",
    }
    manifest_path = target / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest
