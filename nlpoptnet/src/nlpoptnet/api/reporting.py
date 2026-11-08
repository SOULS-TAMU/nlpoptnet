"""Reporting resources for the NLPOptNet frontend."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nlpoptnet.api.model import NLPOptNet

from typing import Any
from pathlib import Path
import json
import numpy as np
from ohagi._utils.serialization import resolve_path
import time
from ohagi.data.sampling import write_csv_matrix


def _fmt_metric(x, COL_W=12):
    return f"{x:>{COL_W}.4e}"


def _fmt_summary_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6g}"
    return str(value)


def _fmt_time_msec(value):
    if value is None:
        return "N/A"
    return f"{value * 1000:.2f} ms"


def _fmt_time_sec(value):
    if value is None:
        return "N/A"
    return f"{value * 1:.2f} s"


def summary(self) -> dict[str, Any]:
    """Print a compact summary of the current completed or loaded run."""
    run_dir = self._run_dir()
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise RuntimeError(
            f"summary.json not found in {run_dir}. Train or load a completed run first."
        )
    with open(summary_path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)

    rows = [
        ("Model Name", payload.get("model_name", self.name)),
        ("No. of Parameters", payload.get("num_parameters", len(self.parameter_names))),
        ("No. of Variables", payload.get("num_variables", len(self.variable_names))),
        ("No. of Equalities", payload.get("num_equalities")),
        ("No. of Inequalities", payload.get("num_inequalities")),
        ("No. of Train Samples", payload.get("train_samples")),
        ("No. of Validation Samples", payload.get("val_samples")),
        ("Maximum Constraint Violation", payload.get("max_violation")),
        ("Training Time", _fmt_time_sec(payload.get("training_wall_time_sec"))),
        (
            "Est. JAX Single Inference Time",
            _fmt_time_msec(payload.get("estimated_jax_single_inference_time_sec")),
        ),
        (
            "Est. JAX Batch Inference Time",
            _fmt_time_msec(payload.get("estimated_jax_batch_inference_time_sec")),
        ),
    ]
    print("📊 NLPOptNet :: Training Summary")
    print("-" * 60)
    # compute max width of keys
    max_key_len = max(len(key) for key, _ in rows)

    for key, value in rows:
        print(f"{key:<{max_key_len}} : {_fmt_summary_value(value)}")
    print("-" * 60)

    note_msg = (
        "Note: Inference time estimations are based on\n"
        "microbenchmarking on the hardware used during\n"
        "training and may vary across different hardware\n"
        "and runtime conditions."
    )
    print(note_msg)


def plot_history(
    self,
    *,
    show: bool = True,
    save_dir: str | Path | None = None,
    bg: str = "grey",  # "grey" (default) or "white"
):
    """Plot and save objective and violation history curves."""
    history = self._read_history()

    epochs = history["epoch"]
    tr_obj = history["train_objective"]
    val_obj = history["val_objective"]

    tr_violation = np.maximum(
        self._history_column(history, "train_eq_violation"),
        self._history_column(history, "train_ineq_violation"),
    )
    val_violation = np.maximum(
        self._history_column(history, "val_eq_violation"),
        self._history_column(history, "val_ineq_violation"),
    )

    tr_violation = np.maximum(tr_violation, 1e-16)
    val_violation = np.maximum(val_violation, 1e-16)

    import os

    os.environ.setdefault("MPLCONFIGDIR", "/tmp")

    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    # -------------------------------
    # Font handling
    # -------------------------------
    font_names = {font.name for font in font_manager.fontManager.ttflist}
    font_family = "Arial" if "Arial" in font_names else "DejaVu Sans"

    # -------------------------------
    # Background + style selection
    # -------------------------------
    bg_colors = {
        "grey": "#E5E5E5",
        "white": "#FFFFFF",
    }
    fig_bg = bg_colors.get(bg, "#E5E5E5")

    style = "default" if bg == "white" else "ggplot"

    # -------------------------------
    # Plotting
    # -------------------------------
    with plt.style.context(style):
        plt.rcParams.update(
            {
                "font.family": font_family,
                "font.size": 32,
                "axes.titlesize": 32,
                "axes.labelsize": 24,
                "legend.fontsize": 24,
                "xtick.labelsize": 20,
                "ytick.labelsize": 20,
            }
        )

        fig, axes = plt.subplots(1, 2, figsize=(20, 6), facecolor=fig_bg)

        # Ensure axes match background
        for ax in axes:
            ax.set_facecolor(fig_bg)

        # -------------------------------
        # Objective plot
        # -------------------------------
        axes[0].plot(epochs, tr_obj, linewidth=2, label="Train Objective")
        axes[0].plot(
            epochs, val_obj, linewidth=2, linestyle="--", label="Validation Objective"
        )
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Objective")
        axes[0].set_title("Objective Evolution")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # -------------------------------
        # Constraint violation plot
        # -------------------------------
        axes[1].plot(epochs, tr_violation, linewidth=2, label="Train Violation")
        axes[1].plot(
            epochs,
            val_violation,
            linewidth=2,
            linestyle="--",
            label="Validation Violation",
        )
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Max. Violation")
        axes[1].set_title("Constraint Violation")
        axes[1].set_yscale("log")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()

        # -------------------------------
        # Save
        # -------------------------------
        target = self._run_dir() if save_dir is None else resolve_path(save_dir)
        target.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            target / "training_history.png",
            dpi=600,
            bbox_inches="tight",
            facecolor=fig_bg,  # ensure saved background matches
        )

        if show:
            plt.show()

    return fig, axes


def _run_dir(self) -> Path:
    if (
        self._inference_state is not None
        and self._inference_state.output_dir is not None
    ):
        return resolve_path(self._inference_state.output_dir)
    if self._metadata_path is not None:
        return resolve_path(self._metadata_path).parent
    raise RuntimeError("No completed or loaded run is available.")


def _read_history(self) -> dict[str, np.ndarray]:
    run_dir = self._run_dir()
    history_path = run_dir / "history.csv"
    if not history_path.exists():
        raise RuntimeError(f"history.csv not found in {run_dir}.")
    data = np.genfromtxt(
        history_path, delimiter=",", names=True, dtype=np.float64, encoding="utf-8"
    )
    if data.dtype.names is None:
        raise RuntimeError(f"history.csv has no header: {history_path}")
    return {
        name: np.atleast_1d(data[name]).astype(np.float64) for name in data.dtype.names
    }


def _history_column(history: dict[str, np.ndarray], name: str) -> np.ndarray:
    if name in history:
        return history[name]
    aliases = {
        "train_eq_violation": ("train_eq_l2", "train_eq", "train_eq_inf"),
        "val_eq_violation": ("val_eq_l2", "val_eq", "val_eq_inf"),
        "train_ineq_violation": ("train_ineq_l2", "train_ineq", "train_ineq_inf"),
        "val_ineq_violation": ("val_ineq_l2", "val_ineq", "val_ineq_inf"),
    }
    for alias in aliases.get(name, ()):
        if alias in history:
            return history[alias]
    if "epoch" in history:
        return np.zeros_like(history["epoch"], dtype=np.float64)
    raise KeyError(name)


def _estimate_inference_times(
    self,
    *,
    metadata_path: Path,
    X_train: np.ndarray,
    batch_size: int,
    repetitions: int = 50,
) -> dict[str, Any]:
    """Measure synchronized predictions from the reconstructed trained frontend model."""
    samples = np.asarray(X_train, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[0] == 0:
        return {
            "estimated_inference_samples": 0,
            "estimated_jax_single_inference_time_sec": None,
            "estimated_jax_batch_inference_time_sec": None,
            "estimated_jax_batch_size": int(batch_size),
        }
    n_single = int(min(int(repetitions), samples.shape[0]))
    single_samples = samples[:n_single]
    loaded = NLPOptNet().load(metadata_path, verbose=False)

    def time_single_backend(backend: str) -> dict[str, Any]:
        try:
            loaded.predict(single_samples[0], projection_backend=backend)
            start = time.perf_counter()
            for row in single_samples:
                loaded.predict(row, projection_backend=backend)
            elapsed = time.perf_counter() - start
            return {
                f"estimated_{backend}_single_inference_time_sec": float(
                    elapsed / n_single
                ),
                f"estimated_{backend}_single_total_time_sec": float(elapsed),
                f"estimated_{backend}_single_error": None,
            }
        except Exception as exc:
            return {
                f"estimated_{backend}_single_inference_time_sec": None,
                f"estimated_{backend}_single_total_time_sec": None,
                f"estimated_{backend}_single_error": f"{type(exc).__name__}: {exc}",
            }

    out: dict[str, Any] = {
        "estimated_inference_samples": n_single,
        "estimated_jax_batch_size": int(batch_size),
    }
    out.update(time_single_backend("jax"))

    try:
        batch_size = int(max(1, batch_size))
        if samples.shape[0] < batch_size:
            reps = int(np.ceil(batch_size / samples.shape[0]))
            batch = np.tile(samples, (reps, 1))[:batch_size]
        else:
            batch = samples[:batch_size]
        loaded.predict(batch, projection_backend="jax")
        start = time.perf_counter()
        for _ in range(int(repetitions)):
            loaded.predict(batch, projection_backend="jax")
        elapsed = time.perf_counter() - start
        out["estimated_jax_batch_inference_time_sec"] = float(
            elapsed / int(repetitions)
        )
        out["estimated_jax_batch_total_time_sec"] = float(elapsed)
        out["estimated_jax_batch_error"] = None
    except Exception as exc:
        out["estimated_jax_batch_inference_time_sec"] = None
        out["estimated_jax_batch_total_time_sec"] = None
        out["estimated_jax_batch_error"] = f"{type(exc).__name__}: {exc}"
    return out


def _write_history_csv(self, path: Path, history: list[dict[str, Any]]) -> None:
    if not history:
        return
    write_csv_matrix(
        path,
        np.asarray(
            [[row[key] for key in history[0].keys()] for row in history],
            dtype=np.float64,
        ),
        headers=list(history[0].keys()),
    )
