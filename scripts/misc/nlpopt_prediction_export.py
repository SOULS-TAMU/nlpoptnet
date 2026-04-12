from __future__ import annotations

import csv
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from opt.training import apply_projection_layers


def _write_csv(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(arr, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if data.ndim == 2 and data.shape[1] == 0:
            for _ in range(int(data.shape[0])):
                writer.writerow([])
        else:
            writer.writerows(data.tolist())


def export_ordered_backbone_logits(
    *,
    model,
    params,
    X: np.ndarray,
    dtype,
    device,
    batch_size: int,
    output_dir: Path | str,
) -> tuple[Path, Path]:
    """Export raw backbone y/mu predictions for the full cached dataset order."""
    X_np = np.asarray(X, dtype=np.float64)
    if X_np.ndim != 2:
        raise ValueError(f"Expected a 2D parameter matrix, received shape {X_np.shape}.")

    batch_size = max(1, int(batch_size))
    save_dir = Path(output_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    @jax.jit
    def _forward_logits(x_batch):
        y_hat, _lam_hat, mu_hat = model.apply({"params": params}, x_batch)
        return y_hat, mu_hat

    y_chunks: list[np.ndarray] = []
    mu_chunks: list[np.ndarray] = []
    for start in range(0, int(X_np.shape[0]), batch_size):
        stop = min(start + batch_size, int(X_np.shape[0]))
        x_batch = jax.device_put(jnp.asarray(X_np[start:stop], dtype=dtype), device)
        y_hat, mu_hat = _forward_logits(x_batch)
        y_chunks.append(np.asarray(jax.device_get(y_hat), dtype=np.float64))
        mu_chunks.append(np.asarray(jax.device_get(mu_hat), dtype=np.float64))

    y_full = np.concatenate(y_chunks, axis=0) if y_chunks else np.zeros((0, 0), dtype=np.float64)
    if mu_chunks:
        mu_full = np.concatenate(mu_chunks, axis=0)
    else:
        mu_full = np.zeros((int(X_np.shape[0]), 0), dtype=np.float64)

    logit_var_path = save_dir / "logit_var.csv"
    logit_mu_path = save_dir / "logit_mu.csv"
    _write_csv(logit_var_path, y_full)
    _write_csv(logit_mu_path, mu_full)
    return logit_var_path, logit_mu_path


def summarize_multiplier_activity(
    *,
    optimizer_multipliers: np.ndarray,
    predicted_multipliers: np.ndarray,
    active_tol: float = 1e-6,
) -> dict[str, float | int | None]:
    opt_mu = np.asarray(optimizer_multipliers, dtype=np.float64)
    pred_mu = np.asarray(predicted_multipliers, dtype=np.float64)

    if opt_mu.ndim == 1:
        opt_mu = opt_mu.reshape(-1, 1)
    if pred_mu.ndim == 1:
        pred_mu = pred_mu.reshape(-1, 1)
    if opt_mu.shape != pred_mu.shape:
        raise ValueError(
            f"Optimizer and predicted multipliers must share a shape, got {opt_mu.shape} and {pred_mu.shape}."
        )

    total_entries = int(opt_mu.size)
    if total_entries == 0:
        return {
            "active_tol": float(active_tol),
            "total_entries": 0,
            "optimizer_active_count": 0,
            "optimizer_inactive_count": 0,
            "optimizer_active_fraction": 0.0,
            "predicted_mean_on_optimizer_active": None,
            "predicted_mean_on_optimizer_inactive": None,
            "predicted_median_on_optimizer_active": None,
            "predicted_median_on_optimizer_inactive": None,
            "predicted_fraction_gt_tol_on_optimizer_active": None,
            "predicted_fraction_gt_tol_on_optimizer_inactive": None,
            "activity_agreement_rate_at_tol": None,
        }

    optimizer_active = opt_mu > float(active_tol)
    predicted_active = pred_mu > float(active_tol)

    active_vals = pred_mu[optimizer_active]
    inactive_vals = pred_mu[~optimizer_active]

    def _maybe_mean(values: np.ndarray) -> float | None:
        return None if values.size == 0 else float(np.mean(values))

    def _maybe_median(values: np.ndarray) -> float | None:
        return None if values.size == 0 else float(np.median(values))

    def _maybe_fraction(mask: np.ndarray) -> float | None:
        return None if mask.size == 0 else float(np.mean(mask))

    active_count = int(np.sum(optimizer_active))
    inactive_count = int(total_entries - active_count)
    return {
        "active_tol": float(active_tol),
        "total_entries": total_entries,
        "optimizer_active_count": active_count,
        "optimizer_inactive_count": inactive_count,
        "optimizer_active_fraction": float(active_count / total_entries),
        "predicted_mean_on_optimizer_active": _maybe_mean(active_vals),
        "predicted_mean_on_optimizer_inactive": _maybe_mean(inactive_vals),
        "predicted_median_on_optimizer_active": _maybe_median(active_vals),
        "predicted_median_on_optimizer_inactive": _maybe_median(inactive_vals),
        "predicted_fraction_gt_tol_on_optimizer_active": _maybe_fraction(predicted_active[optimizer_active]),
        "predicted_fraction_gt_tol_on_optimizer_inactive": _maybe_fraction(predicted_active[~optimizer_active]),
        "activity_agreement_rate_at_tol": float(np.mean(predicted_active == optimizer_active)),
    }


def export_ordered_projected_predictions(
    *,
    model,
    params,
    sub_layer,
    cfg,
    X: np.ndarray,
    dtype,
    device,
    batch_size: int,
    output_dir: Path | str,
    optimizer_multipliers: np.ndarray | None = None,
    active_tol: float = 1e-6,
) -> tuple[Path, Path, dict[str, float | int | None] | None]:
    X_np = np.asarray(X, dtype=np.float64)
    if X_np.ndim != 2:
        raise ValueError(f"Expected a 2D parameter matrix, received shape {X_np.shape}.")

    batch_size = max(1, int(batch_size))
    save_dir = Path(output_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    @jax.jit
    def _forward_projected(x_batch):
        y_hat, lam_hat, mu_hat = model.apply({"params": params}, x_batch)
        y_tilde, _lam_tilde, mu_tilde = apply_projection_layers(
            sub_layer=sub_layer,
            x_batch=x_batch,
            y0=y_hat,
            lam0=lam_hat,
            mu0=mu_hat,
            cfg=cfg,
        )
        return y_tilde, mu_tilde

    y_chunks: list[np.ndarray] = []
    mu_chunks: list[np.ndarray] = []
    for start in range(0, int(X_np.shape[0]), batch_size):
        stop = min(start + batch_size, int(X_np.shape[0]))
        x_batch = jax.device_put(jnp.asarray(X_np[start:stop], dtype=dtype), device)
        y_tilde, mu_tilde = _forward_projected(x_batch)
        y_chunks.append(np.asarray(jax.device_get(y_tilde), dtype=np.float64))
        mu_chunks.append(np.asarray(jax.device_get(mu_tilde), dtype=np.float64))

    y_full = np.concatenate(y_chunks, axis=0) if y_chunks else np.zeros((0, 0), dtype=np.float64)
    mu_full = np.concatenate(mu_chunks, axis=0) if mu_chunks else np.zeros((int(X_np.shape[0]), 0), dtype=np.float64)

    predicted_var_path = save_dir / "predicted_variables.csv"
    predicted_mu_path = save_dir / "predicted_multipliers.csv"
    _write_csv(predicted_var_path, y_full)
    _write_csv(predicted_mu_path, mu_full)

    activity_summary = None
    if optimizer_multipliers is not None:
        activity_summary = summarize_multiplier_activity(
            optimizer_multipliers=optimizer_multipliers,
            predicted_multipliers=mu_full,
            active_tol=active_tol,
        )

    return predicted_var_path, predicted_mu_path, activity_summary
