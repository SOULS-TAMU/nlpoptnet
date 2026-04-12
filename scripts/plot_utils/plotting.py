from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


_PLOT_FONT_SIZE = 18
_GRID_ALPHA = 0.18
_LABEL_FONT_SIZE = _PLOT_FONT_SIZE - 2
_LEGEND_FONT_SIZE = _PLOT_FONT_SIZE - 2
_VIOLATION_FLOOR = 1e-10
_VIOLATION_AXIS_FLOOR = 1e-12
_GAP_FLOOR = 1e-2
_GAP_CEIL = 1e4
_GAP_TICKS = [1e-2, 1e0, 1e2, 1e4]


def _model_color(series_label: str) -> str:
    normalized = str(series_label).strip().lower()
    if normalized == "nlpopt":
        return "#E67E22"
    return "#4B5563"


def _add_stacked_legends(fig, *, model_specs: Sequence[tuple[str, str]]):
    from matplotlib.lines import Line2D

    model_handles = [Line2D([0], [0], color=color, linewidth=2.4) for label, color in model_specs]
    model_labels = [label for label, _ in model_specs]
    split_handles = [
        Line2D([0], [0], color="black", linewidth=2.4, linestyle="-"),
        Line2D([0], [0], color="black", linewidth=2.4, linestyle=":"),
    ]

    model_legend = fig.legend(
        model_handles,
        model_labels,
        loc="lower center",
        ncol=max(1, len(model_specs)),
        frameon=False,
        bbox_to_anchor=(0.5, -0.04),
        fontsize=_LEGEND_FONT_SIZE,
    )
    fig.add_artist(model_legend)
    fig.legend(
        split_handles,
        ["Train", "Validation"],
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.10),
        fontsize=_LEGEND_FONT_SIZE,
    )


def _safe_log_series(values: Sequence[float], *, eps: float) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    out = np.full(arr.shape, np.nan, dtype=float)
    finite_mask = np.isfinite(arr)
    out[finite_mask] = np.maximum(arr[finite_mask], eps)
    if not np.any(np.isfinite(out)):
        out.fill(eps)
    return out


def _clamp_log_series(values: Sequence[float], *, floor: float, ceil: float | None = None) -> np.ndarray:
    arr = _safe_log_series(values, eps=floor)
    if ceil is not None:
        finite_mask = np.isfinite(arr)
        arr[finite_mask] = np.minimum(arr[finite_mask], ceil)
    return arr


def _log_axis_limits(*series: Sequence[float], eps: float) -> tuple[float, float]:
    finite_chunks = []
    for values in series:
        arr = np.asarray(values, dtype=float)
        mask = np.isfinite(arr) & (arr > 0.0)
        if np.any(mask):
            finite_chunks.append(arr[mask])
    if not finite_chunks:
        return eps, eps * 10.0
    merged = np.concatenate(finite_chunks)
    lower = max(float(np.min(merged)), eps)
    upper = max(float(np.max(merged)), lower)
    if upper <= lower:
        upper = lower * 10.0
    return lower, upper


def _sparse_log_ticks(lower: float, upper: float, *, max_ticks: int = 5) -> list[float]:
    lower_power = int(math.floor(math.log10(lower)))
    upper_power = int(math.ceil(math.log10(upper)))
    tick_powers = list(range(lower_power, upper_power + 1))
    if len(tick_powers) <= max_ticks:
        return [10.0 ** power for power in tick_powers]

    step = max(1, int(math.ceil((len(tick_powers) - 1) / float(max_ticks - 1))))
    sparse_powers = tick_powers[::step]
    if sparse_powers[-1] != upper_power:
        sparse_powers.append(upper_power)
    return [10.0 ** power for power in sparse_powers]


def _configure_log_metric_axis(
    ax,
    *series: Sequence[float],
    eps: float,
    min_lower: float | None = None,
    min_upper: float | None = None,
    max_ticks: int = 5,
) -> None:
    from matplotlib.ticker import FixedLocator, LogFormatterMathtext, NullFormatter, NullLocator

    has_positive_violation = False
    finite_chunks = []
    for values in series:
        arr = np.asarray(values, dtype=float)
        mask = np.isfinite(arr) & (arr > 0.0)
        if np.any(mask):
            finite_chunks.append(arr[mask])
            has_positive_violation = True

    if not has_positive_violation:
        ax.set_ylim(0.0, 1.0)
        return

    lower_value, upper_value = _log_axis_limits(*series, eps=eps)
    if min_lower is not None:
        lower_value = min(lower_value, min_lower)
    if min_upper is not None:
        upper_value = max(upper_value, min_upper)

    lower_tick = 10.0 ** math.floor(math.log10(max(lower_value, eps)))
    upper_tick = 10.0 ** math.ceil(math.log10(max(upper_value, lower_tick)))
    major_yticks = _sparse_log_ticks(lower_tick, upper_tick, max_ticks=max_ticks)

    ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator(major_yticks))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(lower_tick, max(upper_tick, upper_value * 1.15))


def _configure_violation_axis(ax, *series: Sequence[float], eps: float) -> None:
    _configure_log_metric_axis(ax, *series, eps=eps, min_lower=_VIOLATION_AXIS_FLOOR, min_upper=1e-6, max_ticks=4)


def _configure_gap_axis(ax) -> None:
    from matplotlib.ticker import FixedLocator, LogFormatterMathtext, NullFormatter, NullLocator

    ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator(_GAP_TICKS))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(_GAP_FLOOR, _GAP_CEIL)


def _as_epoch_run_matrix(values: Sequence[Sequence[float]], epochs: Sequence[int], *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != len(epochs):
        raise ValueError(f"{name} must have shape (n_runs, n_epochs).")
    return arr


def save_objective_violation_plot(
    output_path: Path,
    *,
    epochs: Sequence[int],
    train_gap_pct: Sequence[float],
    val_gap_pct: Sequence[float],
    train_violation: Sequence[float],
    val_violation: Sequence[float],
    title: str,
    series_label: str = "NLPOpt",
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter

    if len(epochs) == 0:
        raise ValueError("Expected at least one history point to plot.")

    plt.rcParams.update({
        "font.size": _PLOT_FONT_SIZE,
        "axes.linewidth": 1.5,
        "xtick.major.size": 6,
        "xtick.major.width": 1.5,
        "ytick.major.size": 6,
        "ytick.major.width": 1.5,
        "legend.frameon": False,
    })

    output_path = Path(output_path)
    out_dir = output_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    f1_path = os.path.join(out_dir, output_path.name)
    plot_epochs = [max(int(ep) + 1, 1) for ep in epochs]

    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.8))
    fig1.patch.set_facecolor("white")

    model_color = _model_color(series_label)
    eps = 1e-16

    train_gap_safe = _clamp_log_series(train_gap_pct, floor=_GAP_FLOOR, ceil=_GAP_CEIL)
    val_gap_safe = _clamp_log_series(val_gap_pct, floor=_GAP_FLOOR, ceil=_GAP_CEIL)

    ax1.plot(
        plot_epochs,
        train_gap_safe,
        color=model_color,
        linewidth=2.2,
        linestyle="-",
    )
    ax1.plot(
        plot_epochs,
        val_gap_safe,
        color=model_color,
        linewidth=2.2,
        linestyle=":",
    )
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Epoch", fontsize=_LABEL_FONT_SIZE)
    ax1.set_ylabel("Worst Optimality Gap (%)", fontsize=_LABEL_FONT_SIZE)
    ax1.grid(True, linestyle="--", alpha=_GRID_ALPHA)
    ax1.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0,)))
    ax1.xaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax1.xaxis.set_minor_formatter(NullFormatter())
    _configure_gap_axis(ax1)

    train_viol_safe = _safe_log_series(train_violation, eps=_VIOLATION_FLOOR)
    val_viol_safe = _safe_log_series(val_violation, eps=_VIOLATION_FLOOR)

    ax2.plot(
        plot_epochs,
        train_viol_safe,
        color=model_color,
        linewidth=2.2,
        linestyle="-",
    )
    ax2.plot(
        plot_epochs,
        val_viol_safe,
        color=model_color,
        linewidth=2.2,
        linestyle=":",
    )
    ax2.set_xscale("log")
    ax2.set_xlabel("Epoch", fontsize=_LABEL_FONT_SIZE)
    ax2.set_ylabel("Worst Violation", fontsize=_LABEL_FONT_SIZE)
    ax2.grid(True, linestyle="--", alpha=_GRID_ALPHA)
    ax2.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0,)))
    ax2.xaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax2.xaxis.set_minor_formatter(NullFormatter())
    _configure_violation_axis(ax2, train_viol_safe, val_viol_safe, eps=eps)

    _add_stacked_legends(fig1, model_specs=[(series_label, model_color)])

    fig1.subplots_adjust(bottom=0.22, wspace=0.25)
    fig1.tight_layout(rect=[0, 0.09, 1, 1])
    fig1.savefig(f1_path, bbox_inches="tight", dpi=600)
    plt.close(fig1)
    print(f"[plot] Saved: {f1_path}")
    return Path(f1_path)


def save_objective_value_violation_plot(
    output_path: Path,
    *,
    epochs: Sequence[int],
    train_objective: Sequence[float],
    val_objective: Sequence[float],
    train_violation: Sequence[float],
    val_violation: Sequence[float],
    title: str,
    series_label: str = "NLPOpt",
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter

    if len(epochs) == 0:
        raise ValueError("Expected at least one history point to plot.")

    plt.rcParams.update({
        "font.size": _PLOT_FONT_SIZE,
        "axes.linewidth": 1.5,
        "xtick.major.size": 6,
        "xtick.major.width": 1.5,
        "ytick.major.size": 6,
        "ytick.major.width": 1.5,
        "legend.frameon": False,
    })

    output_path = Path(output_path)
    out_dir = output_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    f1_path = os.path.join(out_dir, output_path.name)
    plot_epochs = [max(int(ep) + 1, 1) for ep in epochs]

    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.8))
    fig1.patch.set_facecolor("white")

    model_color = _model_color(series_label)
    eps = 1e-16

    train_obj = np.asarray(train_objective, dtype=float)
    val_obj = np.asarray(val_objective, dtype=float)
    ax1.plot(plot_epochs, train_obj, color=model_color, linewidth=2.2, linestyle="-")
    ax1.plot(plot_epochs, val_obj, color=model_color, linewidth=2.2, linestyle=":")
    ax1.set_xscale("log")
    ax1.set_xlabel("Epoch", fontsize=_LABEL_FONT_SIZE)
    ax1.set_ylabel("Projected Objective", fontsize=_LABEL_FONT_SIZE)
    ax1.grid(True, linestyle="--", alpha=_GRID_ALPHA)
    ax1.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0,)))
    ax1.xaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax1.xaxis.set_minor_formatter(NullFormatter())

    finite_objective = np.concatenate(
        [arr[np.isfinite(arr)] for arr in (train_obj, val_obj) if np.any(np.isfinite(arr))]
    ) if (np.any(np.isfinite(train_obj)) or np.any(np.isfinite(val_obj))) else np.asarray([], dtype=float)
    if finite_objective.size > 0:
        obj_min = float(np.min(finite_objective))
        obj_max = float(np.max(finite_objective))
        if obj_min == obj_max:
            pad = max(1.0, abs(obj_min)) * 0.05
        else:
            pad = 0.05 * (obj_max - obj_min)
        ax1.set_ylim(obj_min - pad, obj_max + pad)

    train_viol_safe = _safe_log_series(train_violation, eps=_VIOLATION_FLOOR)
    val_viol_safe = _safe_log_series(val_violation, eps=_VIOLATION_FLOOR)
    ax2.plot(plot_epochs, train_viol_safe, color=model_color, linewidth=2.2, linestyle="-")
    ax2.plot(plot_epochs, val_viol_safe, color=model_color, linewidth=2.2, linestyle=":")
    ax2.set_xscale("log")
    ax2.set_xlabel("Epoch", fontsize=_LABEL_FONT_SIZE)
    ax2.set_ylabel("Worst Violation", fontsize=_LABEL_FONT_SIZE)
    ax2.grid(True, linestyle="--", alpha=_GRID_ALPHA)
    ax2.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0,)))
    ax2.xaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax2.xaxis.set_minor_formatter(NullFormatter())
    _configure_violation_axis(ax2, train_viol_safe, val_viol_safe, eps=eps)

    _add_stacked_legends(fig1, model_specs=[(series_label, model_color)])

    fig1.subplots_adjust(bottom=0.22, wspace=0.25)
    fig1.tight_layout(rect=[0, 0.09, 1, 1])
    fig1.savefig(f1_path, bbox_inches="tight", dpi=600)
    plt.close(fig1)
    print(f"[plot] Saved: {f1_path}")
    return Path(f1_path)


def save_shadow_objective_violation_plot(
    output_path: Path,
    *,
    epochs: Sequence[int],
    train_gap_pct_runs: Sequence[Sequence[float]],
    val_gap_pct_runs: Sequence[Sequence[float]],
    train_violation_runs: Sequence[Sequence[float]],
    val_violation_runs: Sequence[Sequence[float]],
    series_label: str = "NLPOpt",
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter

    if len(epochs) == 0:
        raise ValueError("Expected at least one history point to plot.")

    plt.rcParams.update({
        "font.size": _PLOT_FONT_SIZE,
        "axes.linewidth": 1.5,
        "xtick.major.size": 6,
        "xtick.major.width": 1.5,
        "ytick.major.size": 6,
        "ytick.major.width": 1.5,
        "legend.frameon": False,
    })

    output_path = Path(output_path)
    out_dir = output_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    f1_path = os.path.join(out_dir, output_path.name)
    plot_epochs = np.asarray([max(int(ep) + 1, 1) for ep in epochs], dtype=float)

    train_gap_arr = _as_epoch_run_matrix(train_gap_pct_runs, epochs, name="train_gap_pct_runs")
    val_gap_arr = _as_epoch_run_matrix(val_gap_pct_runs, epochs, name="val_gap_pct_runs")
    train_violation_arr = _as_epoch_run_matrix(train_violation_runs, epochs, name="train_violation_runs")
    val_violation_arr = _as_epoch_run_matrix(val_violation_runs, epochs, name="val_violation_runs")

    if val_gap_arr.shape != train_gap_arr.shape:
        raise ValueError("val_gap_pct_runs must match train_gap_pct_runs shape.")
    if train_violation_arr.shape != train_gap_arr.shape:
        raise ValueError("train_violation_runs must match train_gap_pct_runs shape.")
    if val_violation_arr.shape != train_gap_arr.shape:
        raise ValueError("val_violation_runs must match train_gap_pct_runs shape.")

    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.8))
    fig1.patch.set_facecolor("white")

    model_color = _model_color(series_label)
    eps = 1e-16
    shadow_alpha_train = 0.14
    shadow_alpha_val = 0.07

    train_gap_safe = _clamp_log_series(train_gap_arr, floor=_GAP_FLOOR, ceil=_GAP_CEIL)
    val_gap_safe = _clamp_log_series(val_gap_arr, floor=_GAP_FLOOR, ceil=_GAP_CEIL)
    train_gap_mean = np.nanmean(train_gap_safe, axis=0)
    train_gap_lo = np.nanmin(train_gap_safe, axis=0)
    train_gap_hi = np.nanmax(train_gap_safe, axis=0)
    val_gap_mean = np.nanmean(val_gap_safe, axis=0)
    val_gap_lo = np.nanmin(val_gap_safe, axis=0)
    val_gap_hi = np.nanmax(val_gap_safe, axis=0)

    ax1.fill_between(plot_epochs, train_gap_lo, train_gap_hi, color=model_color, alpha=shadow_alpha_train)
    ax1.fill_between(plot_epochs, val_gap_lo, val_gap_hi, color=model_color, alpha=shadow_alpha_val)
    ax1.plot(plot_epochs, train_gap_mean, color=model_color, linewidth=2.2, linestyle="-")
    ax1.plot(plot_epochs, val_gap_mean, color=model_color, linewidth=2.2, linestyle=":")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Epoch", fontsize=_LABEL_FONT_SIZE)
    ax1.set_ylabel("Worst Optimality Gap (%)", fontsize=_LABEL_FONT_SIZE)
    ax1.grid(True, linestyle="--", alpha=_GRID_ALPHA)
    ax1.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0,)))
    ax1.xaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax1.xaxis.set_minor_formatter(NullFormatter())
    _configure_gap_axis(ax1)

    train_viol_safe = _safe_log_series(train_violation_arr, eps=_VIOLATION_FLOOR)
    val_viol_safe = _safe_log_series(val_violation_arr, eps=_VIOLATION_FLOOR)
    train_viol_mean = np.nanmean(train_viol_safe, axis=0)
    train_viol_lo = np.nanmin(train_viol_safe, axis=0)
    train_viol_hi = np.nanmax(train_viol_safe, axis=0)
    val_viol_mean = np.nanmean(val_viol_safe, axis=0)
    val_viol_lo = np.nanmin(val_viol_safe, axis=0)
    val_viol_hi = np.nanmax(val_viol_safe, axis=0)

    ax2.fill_between(plot_epochs, train_viol_lo, train_viol_hi, color=model_color, alpha=shadow_alpha_train)
    ax2.fill_between(plot_epochs, val_viol_lo, val_viol_hi, color=model_color, alpha=shadow_alpha_val)
    ax2.plot(plot_epochs, train_viol_mean, color=model_color, linewidth=2.2, linestyle="-")
    ax2.plot(plot_epochs, val_viol_mean, color=model_color, linewidth=2.2, linestyle=":")
    ax2.set_xscale("log")
    ax2.set_xlabel("Epoch", fontsize=_LABEL_FONT_SIZE)
    ax2.set_ylabel("Worst Violation", fontsize=_LABEL_FONT_SIZE)
    ax2.grid(True, linestyle="--", alpha=_GRID_ALPHA)
    ax2.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0,)))
    ax2.xaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax2.xaxis.set_minor_formatter(NullFormatter())
    _configure_violation_axis(ax2, train_viol_safe, val_viol_safe, eps=eps)

    _add_stacked_legends(fig1, model_specs=[(series_label, model_color)])

    fig1.subplots_adjust(bottom=0.22, wspace=0.25)
    fig1.tight_layout(rect=[0, 0.09, 1, 1])
    fig1.savefig(f1_path, bbox_inches="tight", dpi=600)
    plt.close(fig1)
    print(f"[plot] Saved: {f1_path}")
    return Path(f1_path)


def save_shadow_objective_value_violation_plot(
    output_path: Path,
    *,
    epochs: Sequence[int],
    train_objective_runs: Sequence[Sequence[float]],
    val_objective_runs: Sequence[Sequence[float]],
    train_violation_runs: Sequence[Sequence[float]],
    val_violation_runs: Sequence[Sequence[float]],
    series_label: str = "NLPOpt",
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter

    if len(epochs) == 0:
        raise ValueError("Expected at least one history point to plot.")

    plt.rcParams.update({
        "font.size": _PLOT_FONT_SIZE,
        "axes.linewidth": 1.5,
        "xtick.major.size": 6,
        "xtick.major.width": 1.5,
        "ytick.major.size": 6,
        "ytick.major.width": 1.5,
        "legend.frameon": False,
    })

    output_path = Path(output_path)
    out_dir = output_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    f1_path = os.path.join(out_dir, output_path.name)
    plot_epochs = np.asarray([max(int(ep) + 1, 1) for ep in epochs], dtype=float)

    train_obj_arr = _as_epoch_run_matrix(train_objective_runs, epochs, name="train_objective_runs")
    val_obj_arr = _as_epoch_run_matrix(val_objective_runs, epochs, name="val_objective_runs")
    train_violation_arr = _as_epoch_run_matrix(train_violation_runs, epochs, name="train_violation_runs")
    val_violation_arr = _as_epoch_run_matrix(val_violation_runs, epochs, name="val_violation_runs")

    if val_obj_arr.shape != train_obj_arr.shape:
        raise ValueError("val_objective_runs must match train_objective_runs shape.")
    if train_violation_arr.shape != train_obj_arr.shape:
        raise ValueError("train_violation_runs must match train_objective_runs shape.")
    if val_violation_arr.shape != train_obj_arr.shape:
        raise ValueError("val_violation_runs must match train_objective_runs shape.")

    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.8))
    fig1.patch.set_facecolor("white")

    model_color = _model_color(series_label)
    eps = 1e-16
    shadow_alpha_train = 0.14
    shadow_alpha_val = 0.07

    train_obj_mean = np.nanmean(train_obj_arr, axis=0)
    train_obj_lo = np.nanmin(train_obj_arr, axis=0)
    train_obj_hi = np.nanmax(train_obj_arr, axis=0)
    val_obj_mean = np.nanmean(val_obj_arr, axis=0)
    val_obj_lo = np.nanmin(val_obj_arr, axis=0)
    val_obj_hi = np.nanmax(val_obj_arr, axis=0)

    ax1.fill_between(plot_epochs, train_obj_lo, train_obj_hi, color=model_color, alpha=shadow_alpha_train)
    ax1.fill_between(plot_epochs, val_obj_lo, val_obj_hi, color=model_color, alpha=shadow_alpha_val)
    ax1.plot(plot_epochs, train_obj_mean, color=model_color, linewidth=2.2, linestyle="-")
    ax1.plot(plot_epochs, val_obj_mean, color=model_color, linewidth=2.2, linestyle=":")
    ax1.set_xscale("log")
    ax1.set_xlabel("Epoch", fontsize=_LABEL_FONT_SIZE)
    ax1.set_ylabel("Projected Objective", fontsize=_LABEL_FONT_SIZE)
    ax1.grid(True, linestyle="--", alpha=_GRID_ALPHA)
    ax1.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0,)))
    ax1.xaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax1.xaxis.set_minor_formatter(NullFormatter())

    finite_objective = np.concatenate(
        [arr[np.isfinite(arr)] for arr in (train_obj_arr, val_obj_arr) if np.any(np.isfinite(arr))]
    ) if (np.any(np.isfinite(train_obj_arr)) or np.any(np.isfinite(val_obj_arr))) else np.asarray([], dtype=float)
    if finite_objective.size > 0:
        obj_min = float(np.min(finite_objective))
        obj_max = float(np.max(finite_objective))
        if obj_min == obj_max:
            pad = max(1.0, abs(obj_min)) * 0.05
        else:
            pad = 0.05 * (obj_max - obj_min)
        ax1.set_ylim(obj_min - pad, obj_max + pad)

    train_viol_safe = _safe_log_series(train_violation_arr, eps=_VIOLATION_FLOOR)
    val_viol_safe = _safe_log_series(val_violation_arr, eps=_VIOLATION_FLOOR)
    train_viol_mean = np.nanmean(train_viol_safe, axis=0)
    train_viol_lo = np.nanmin(train_viol_safe, axis=0)
    train_viol_hi = np.nanmax(train_viol_safe, axis=0)
    val_viol_mean = np.nanmean(val_viol_safe, axis=0)
    val_viol_lo = np.nanmin(val_viol_safe, axis=0)
    val_viol_hi = np.nanmax(val_viol_safe, axis=0)

    ax2.fill_between(plot_epochs, train_viol_lo, train_viol_hi, color=model_color, alpha=shadow_alpha_train)
    ax2.fill_between(plot_epochs, val_viol_lo, val_viol_hi, color=model_color, alpha=shadow_alpha_val)
    ax2.plot(plot_epochs, train_viol_mean, color=model_color, linewidth=2.2, linestyle="-")
    ax2.plot(plot_epochs, val_viol_mean, color=model_color, linewidth=2.2, linestyle=":")
    ax2.set_xscale("log")
    ax2.set_xlabel("Epoch", fontsize=_LABEL_FONT_SIZE)
    ax2.set_ylabel("Worst Violation", fontsize=_LABEL_FONT_SIZE)
    ax2.grid(True, linestyle="--", alpha=_GRID_ALPHA)
    ax2.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0,)))
    ax2.xaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax2.xaxis.set_minor_formatter(NullFormatter())
    _configure_violation_axis(ax2, train_viol_safe, val_viol_safe, eps=eps)

    _add_stacked_legends(fig1, model_specs=[(series_label, model_color)])

    fig1.subplots_adjust(bottom=0.22, wspace=0.25)
    fig1.tight_layout(rect=[0, 0.09, 1, 1])
    fig1.savefig(f1_path, bbox_inches="tight", dpi=600)
    plt.close(fig1)
    print(f"[plot] Saved: {f1_path}")
    return Path(f1_path)


def save_multi_series_shadow_plot(
    output_path: Path,
    *,
    epochs: Sequence[int],
    model_runs: Mapping[str, Mapping[str, Sequence[Sequence[float]]]],
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter

    if len(epochs) == 0:
        raise ValueError("Expected at least one history point to plot.")
    if not model_runs:
        raise ValueError("Expected at least one model entry for plotting.")

    plt.rcParams.update({
        "font.size": _PLOT_FONT_SIZE,
        "axes.linewidth": 1.5,
        "xtick.major.size": 6,
        "xtick.major.width": 1.5,
        "ytick.major.size": 6,
        "ytick.major.width": 1.5,
        "legend.frameon": False,
    })

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_epochs = np.asarray([max(int(ep) + 1, 1) for ep in epochs], dtype=float)

    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.8))
    fig1.patch.set_facecolor("white")

    eps = 1e-16
    shadow_alpha_train = 0.14
    shadow_alpha_val = 0.07
    violation_series = []
    model_specs = []

    for series_label, payload in model_runs.items():
        model_color = _model_color(series_label)
        model_specs.append((series_label, model_color))

        train_gap_arr = _as_epoch_run_matrix(payload["train_gap_pct_runs"], epochs, name=f"{series_label}.train_gap_pct_runs")
        val_gap_arr = _as_epoch_run_matrix(payload["val_gap_pct_runs"], epochs, name=f"{series_label}.val_gap_pct_runs")
        train_violation_arr = _as_epoch_run_matrix(payload["train_violation_runs"], epochs, name=f"{series_label}.train_violation_runs")
        val_violation_arr = _as_epoch_run_matrix(payload["val_violation_runs"], epochs, name=f"{series_label}.val_violation_runs")

        if val_gap_arr.shape != train_gap_arr.shape:
            raise ValueError(f"{series_label} validation gap runs must match training gap shape.")
        if train_violation_arr.shape != train_gap_arr.shape:
            raise ValueError(f"{series_label} training violation runs must match training gap shape.")
        if val_violation_arr.shape != train_gap_arr.shape:
            raise ValueError(f"{series_label} validation violation runs must match training gap shape.")

        train_gap_safe = _clamp_log_series(train_gap_arr, floor=_GAP_FLOOR, ceil=_GAP_CEIL)
        val_gap_safe = _clamp_log_series(val_gap_arr, floor=_GAP_FLOOR, ceil=_GAP_CEIL)
        train_gap_mean = np.nanmean(train_gap_safe, axis=0)
        train_gap_lo = np.nanmin(train_gap_safe, axis=0)
        train_gap_hi = np.nanmax(train_gap_safe, axis=0)
        val_gap_mean = np.nanmean(val_gap_safe, axis=0)
        val_gap_lo = np.nanmin(val_gap_safe, axis=0)
        val_gap_hi = np.nanmax(val_gap_safe, axis=0)

        ax1.fill_between(plot_epochs, train_gap_lo, train_gap_hi, color=model_color, alpha=shadow_alpha_train)
        ax1.fill_between(plot_epochs, val_gap_lo, val_gap_hi, color=model_color, alpha=shadow_alpha_val)
        ax1.plot(plot_epochs, train_gap_mean, color=model_color, linewidth=2.2, linestyle="-")
        ax1.plot(plot_epochs, val_gap_mean, color=model_color, linewidth=2.2, linestyle=":")

        train_viol_safe = _safe_log_series(train_violation_arr, eps=_VIOLATION_FLOOR)
        val_viol_safe = _safe_log_series(val_violation_arr, eps=_VIOLATION_FLOOR)
        violation_series.extend([train_viol_safe, val_viol_safe])

        train_viol_mean = np.nanmean(train_viol_safe, axis=0)
        train_viol_lo = np.nanmin(train_viol_safe, axis=0)
        train_viol_hi = np.nanmax(train_viol_safe, axis=0)
        val_viol_mean = np.nanmean(val_viol_safe, axis=0)
        val_viol_lo = np.nanmin(val_viol_safe, axis=0)
        val_viol_hi = np.nanmax(val_viol_safe, axis=0)

        ax2.fill_between(plot_epochs, train_viol_lo, train_viol_hi, color=model_color, alpha=shadow_alpha_train)
        ax2.fill_between(plot_epochs, val_viol_lo, val_viol_hi, color=model_color, alpha=shadow_alpha_val)
        ax2.plot(plot_epochs, train_viol_mean, color=model_color, linewidth=2.2, linestyle="-")
        ax2.plot(plot_epochs, val_viol_mean, color=model_color, linewidth=2.2, linestyle=":")

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Epoch", fontsize=_LABEL_FONT_SIZE)
    ax1.set_ylabel("Worst Optimality Gap (%)", fontsize=_LABEL_FONT_SIZE)
    ax1.grid(True, linestyle="--", alpha=_GRID_ALPHA)
    ax1.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0,)))
    ax1.xaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax1.xaxis.set_minor_formatter(NullFormatter())
    _configure_gap_axis(ax1)

    ax2.set_xscale("log")
    ax2.set_xlabel("Epoch", fontsize=_LABEL_FONT_SIZE)
    ax2.set_ylabel("Worst Violation", fontsize=_LABEL_FONT_SIZE)
    ax2.grid(True, linestyle="--", alpha=_GRID_ALPHA)
    ax2.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0,)))
    ax2.xaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax2.xaxis.set_minor_formatter(NullFormatter())
    _configure_violation_axis(ax2, *violation_series, eps=eps)

    _add_stacked_legends(fig1, model_specs=model_specs)

    fig1.subplots_adjust(bottom=0.22, wspace=0.25)
    fig1.tight_layout(rect=[0, 0.09, 1, 1])
    fig1.savefig(output_path, bbox_inches="tight", dpi=600)
    plt.close(fig1)
    print(f"[plot] Saved: {output_path}")
    return output_path
