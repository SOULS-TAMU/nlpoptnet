#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "matplotlib is required for plotting. Install it with `pip install matplotlib` "
        "or install the package with `pip install -e nlpopt`."
    ) from exc


NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
EPOCH_RE = re.compile(
    rf"^ep\s+(?P<epoch>\d+)\s+\|\s+"
    rf"train\s+loss\s+(?P<train_loss>{NUMBER})\s+obj\s+(?P<train_obj>{NUMBER})\s+"
    rf"cons\s+(?P<train_cons>{NUMBER})\s+viol\s+(?P<train_viol>{NUMBER})\s+\|\|\s+"
    rf"val\s+loss\s+(?P<val_loss>{NUMBER})\s+obj\s+(?P<val_obj>{NUMBER})\s+"
    rf"cons\s+(?P<val_cons>{NUMBER})\s+viol\s+(?P<val_viol>{NUMBER})\s*$"
)
SAVED_METRICS_RE = re.compile(r"^Saved metrics:\s+(?P<path>.+?)\s*$")


@dataclass(frozen=True)
class EpochRecord:
    epoch: int
    train_loss: float
    train_obj: float
    train_cons: float
    train_viol: float
    val_loss: float
    val_obj: float
    val_cons: float
    val_viol: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a testcase externally and/or plot NLPOpt train/validation metrics "
            "from the epoch lines printed by the main.py testcase dispatcher."
        )
    )
    parser.add_argument(
        "case_dir",
        nargs="?",
        default=".",
        help="Path to the testcase directory. Default: current directory.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run the selected testcase through main.py first, save the console log, then plot it.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Existing log file to parse. If omitted, the latest log in <case_dir>/plots is used.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Directory for logs, CSV, and plots. Default: the dataset folder that contains "
            "run_metrics_*.json; fallback is <case_dir>/plots."
        ),
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use with --run. Default: current interpreter.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional custom plot title.",
    )
    return parser.parse_args()


def _latest_log(output_dir: Path) -> Optional[Path]:
    logs = sorted(output_dir.glob("*.log"), key=lambda path: path.stat().st_mtime)
    return logs[-1] if logs else None


def _latest_case_log(case_dir: Path) -> Optional[Path]:
    candidates = list((case_dir / "plots").glob("*.log"))
    candidates.extend((case_dir / "problem_data").glob("**/*.log"))
    if not candidates:
        return None
    candidates.sort(key=lambda path: path.stat().st_mtime)
    return candidates[-1]


def _ensure_case_dir(case_dir: Path) -> Path:
    case_dir = case_dir.resolve()
    if not case_dir.exists():
        raise FileNotFoundError(f"Case directory does not exist: {case_dir}")
    return case_dir


def _run_testcase(case_dir: Path, output_dir: Path, python_exe: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = output_dir / f"{case_dir.name}_{stamp}.log"

    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    main_py = case_dir.parents[1] / "main.py"
    case_type_map = {
        "qp": "qp",
        "qcqp": "qcqp",
        "nlp": "nlp",
        "nonconvx": "nonconvx",
        "custom": "custom",
    }
    case_type = case_type_map.get(case_dir.name)
    if case_type is None:
        raise FileNotFoundError(
            "Could not determine which main.py --type to use for "
            f"{case_dir}. Supported case directories are: {', '.join(sorted(case_type_map))}."
        )
    if not main_py.exists():
        raise FileNotFoundError(f"Could not find main.py at {main_py}")

    cmd = [python_exe, str(main_py), "--type", case_type, "--action", "run"]
    print(f"Running: {' '.join(cmd)}")
    print(f"Saving log to: {log_path}")

    with open(log_path, "w", encoding="utf-8") as log_fh:
        proc = subprocess.Popen(
            cmd,
            cwd=str(case_dir.parent),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log_fh.write(line)
        return_code = proc.wait()

    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)
    return log_path


def _read_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def _parse_epoch_records(text: str) -> list[EpochRecord]:
    records: list[EpochRecord] = []
    for line in text.splitlines():
        match = EPOCH_RE.match(line.strip())
        if not match:
            continue
        values = match.groupdict()
        records.append(
            EpochRecord(
                epoch=int(values["epoch"]),
                train_loss=float(values["train_loss"]),
                train_obj=float(values["train_obj"]),
                train_cons=float(values["train_cons"]),
                train_viol=float(values["train_viol"]),
                val_loss=float(values["val_loss"]),
                val_obj=float(values["val_obj"]),
                val_cons=float(values["val_cons"]),
                val_viol=float(values["val_viol"]),
            )
        )
    return records


def _saved_metrics_path(text: str) -> Optional[Path]:
    for line in text.splitlines():
        match = SAVED_METRICS_RE.match(line.strip())
        if match:
            return Path(match.group("path")).expanduser().resolve()
    return None


def _fallback_metrics_path(case_dir: Path) -> Optional[Path]:
    metric_files = sorted(
        case_dir.glob("problem_data/**/run_metrics_*.json"),
        key=lambda path: path.stat().st_mtime,
    )
    return metric_files[-1] if metric_files else None


def _load_summary_metrics(case_dir: Path, text: str) -> tuple[Optional[Path], Optional[dict]]:
    metrics_path = _saved_metrics_path(text)
    if metrics_path is None or not metrics_path.exists():
        metrics_path = _fallback_metrics_path(case_dir)
    if metrics_path is None or not metrics_path.exists():
        return None, None
    with open(metrics_path, "r", encoding="utf-8") as fh:
        return metrics_path, json.load(fh)


def _default_output_dir(
    case_dir: Path,
    requested_output_dir: Optional[Path],
    metrics_path: Optional[Path],
) -> Path:
    if requested_output_dir is not None:
        return requested_output_dir.resolve()
    if metrics_path is not None:
        return metrics_path.parent
    return (case_dir / "plots").resolve()


def _move_log_into_output_dir(log_path: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / log_path.name
    if log_path.resolve() == target.resolve():
        return log_path
    log_path.replace(target)
    return target


def _write_epoch_csv(records: Iterable[EpochRecord], path: Path) -> None:
    fieldnames = [
        "epoch",
        "train_loss",
        "train_obj",
        "train_cons",
        "train_viol",
        "val_loss",
        "val_obj",
        "val_cons",
        "val_viol",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow(row.__dict__)


def _style_axis(ax, ylabel: str) -> None:
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.35)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))


def _plot_metric(ax, epochs, train_vals, val_vals, title: str, ylabel: str) -> None:
    ax.plot(epochs, train_vals, color="#0B4F6C", linewidth=2.2, marker="o", markersize=4, label="Train")
    ax.plot(epochs, val_vals, color="#C84C09", linewidth=2.2, marker="s", markersize=4, label="Val")
    ax.set_title(title, fontsize=12, fontweight="bold")
    _style_axis(ax, ylabel)


def _summary_text(summary_metrics: Optional[dict], metrics_path: Optional[Path]) -> str:
    if not summary_metrics:
        return "Final summary metrics not found."

    pieces = []
    mapping = [
        ("relative_objective_gap", "rel gap"),
        ("mse_y_tilde_vs_label", "mse"),
        ("eq_inf", "eq inf"),
        ("ineq_inf", "ineq inf"),
        ("bound_inf", "bound inf"),
        ("training_wall_time_sec", "wall time (s)"),
    ]
    for key, label in mapping:
        if key in summary_metrics:
            pieces.append(f"{label}: {summary_metrics[key]:.3e}")
    if metrics_path is not None:
        pieces.append(f"summary: {metrics_path.name}")
    return "\n".join(pieces)


def _plot_records(
    records: list[EpochRecord],
    output_png: Path,
    output_svg: Path,
    title: str,
    summary_metrics: Optional[dict],
    metrics_path: Optional[Path],
) -> None:
    epochs = [r.epoch for r in records]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9), constrained_layout=True)
    fig.patch.set_facecolor("white")

    _plot_metric(
        axes[0, 0],
        epochs,
        [r.train_loss for r in records],
        [r.val_loss for r in records],
        "Loss",
        "Loss",
    )
    _plot_metric(
        axes[0, 1],
        epochs,
        [r.train_obj for r in records],
        [r.val_obj for r in records],
        "Objective",
        "Objective",
    )
    _plot_metric(
        axes[1, 0],
        epochs,
        [r.train_cons for r in records],
        [r.val_cons for r in records],
        "Consistency",
        "Consistency",
    )
    _plot_metric(
        axes[1, 1],
        epochs,
        [r.train_viol for r in records],
        [r.val_viol for r in records],
        "Constraint Violation",
        "Violation",
    )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(title, fontsize=16, fontweight="bold")
    fig.text(
        0.985,
        0.02,
        _summary_text(summary_metrics, metrics_path),
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#F7F4EA", "edgecolor": "#D3C7A1"},
    )

    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    fig.savefig(output_svg, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = _parse_args()
    case_dir = _ensure_case_dir(Path(args.case_dir))
    requested_output_dir = args.output_dir

    if args.run:
        staging_output_dir = (
            requested_output_dir.resolve()
            if requested_output_dir is not None
            else (case_dir / "plots").resolve()
        )
        staging_output_dir.mkdir(parents=True, exist_ok=True)
        log_path = _run_testcase(case_dir, staging_output_dir, args.python)
    elif args.log_file is not None:
        log_path = args.log_file.expanduser().resolve()
    else:
        if requested_output_dir is not None:
            search_dir = requested_output_dir.resolve()
            log_path = _latest_log(search_dir)
        else:
            search_dir = case_dir
            log_path = _latest_case_log(case_dir)
        if log_path is None:
            raise FileNotFoundError(
                f"No log file found in {search_dir}. Use --run or pass --log-file."
            )

    if not log_path.exists():
        raise FileNotFoundError(f"Log file does not exist: {log_path}")

    log_text = _read_text(log_path)
    records = _parse_epoch_records(log_text)
    if not records:
        raise ValueError(
            "No epoch metrics found in the log. Expected lines like:\n"
            "ep 00000 | train loss ... obj ... cons ... viol ... || val loss ... obj ... cons ... viol ..."
        )

    metrics_path, summary_metrics = _load_summary_metrics(case_dir, log_text)
    output_dir = _default_output_dir(case_dir, requested_output_dir, metrics_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.run and requested_output_dir is None:
        log_path = _move_log_into_output_dir(log_path, output_dir)
    stem = log_path.stem
    csv_path = output_dir / f"{stem}_epoch_metrics.csv"
    png_path = output_dir / f"{stem}_metrics.png"
    svg_path = output_dir / f"{stem}_metrics.svg"

    _write_epoch_csv(records, csv_path)
    plot_title = args.title or f"NLPOpt Training Curves: {case_dir.name}"
    _plot_records(records, png_path, svg_path, plot_title, summary_metrics, metrics_path)

    print(f"Parsed {len(records)} epoch checkpoints from: {log_path}")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved PNG: {png_path}")
    print(f"Saved SVG: {svg_path}")
    if metrics_path is not None:
        print(f"Loaded final summary: {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
