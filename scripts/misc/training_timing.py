from __future__ import annotations

from typing import Any, Mapping


def timing_window(total_epochs: int) -> tuple[int, int]:
    epochs = max(1, int(total_epochs))
    start_epoch = 1 if epochs > 1 else 0
    return start_epoch, max(1, epochs - start_epoch)


def should_track_epoch(epoch_idx: int, total_epochs: int) -> bool:
    start_epoch, _tracked_epochs = timing_window(total_epochs)
    return int(epoch_idx) >= start_epoch


def timing_window_label(total_epochs: int) -> str:
    epochs = max(1, int(total_epochs))
    start_epoch, _tracked_epochs = timing_window(epochs)
    if start_epoch <= 0:
        return "epoch 1"
    if epochs == 2:
        return "epoch 2"
    return f"epochs {start_epoch + 1}-{epochs}"


def summarize_timing_profile(profile: Mapping[str, Any]) -> dict[str, float | int]:
    epochs = max(1, int(profile["epochs"]))
    start_epoch, tracked_epochs = timing_window(epochs)
    start_epoch = int(profile.get("timing_start_epoch", start_epoch))
    tracked_epochs = max(1, int(profile.get("timing_epochs_recorded", tracked_epochs)))
    train_batches = max(1, int(profile["train_batches_per_epoch"]))
    val_batches = max(1, int(profile["val_batches_per_epoch"]))

    train_total = float(profile.get("train_epoch_time_tracked_sec", profile.get("train_epoch_time_total_sec", 0.0)))
    val_total = float(profile.get("val_epoch_time_tracked_sec", profile.get("val_epoch_time_total_sec", 0.0)))
    backbone_total = float(profile.get("backbone_tracked_total_sec", profile.get("backbone_total_sec", 0.0)))
    projection_total = float(profile.get("projection_tracked_total_sec", profile.get("projection_total_sec", 0.0)))
    backward_total = float(profile.get("backward_tracked_total_sec", profile.get("backward_total_sec", 0.0)))
    optimizer_total = float(profile.get("optimizer_tracked_total_sec", profile.get("optimizer_total_sec", 0.0)))
    profiled_total = backbone_total + projection_total + backward_total + optimizer_total

    return {
        "timing_start_epoch": start_epoch,
        "timing_epochs_recorded": tracked_epochs,
        "avg_train_epoch_time_sec": train_total / tracked_epochs,
        "avg_val_epoch_time_sec": val_total / tracked_epochs,
        "avg_total_epoch_time_sec": (train_total + val_total) / tracked_epochs,
        "avg_train_batch_time_sec": train_total / (tracked_epochs * train_batches),
        "avg_val_batch_time_sec": val_total / (tracked_epochs * val_batches),
        "avg_total_batch_time_sec": (train_total + val_total) / (tracked_epochs * (train_batches + val_batches)),
        "backbone_total_sec": backbone_total,
        "projection_total_sec": projection_total,
        "backward_total_sec": backward_total,
        "optimizer_total_sec": optimizer_total,
        "time_backbone_percent": 100.0 * backbone_total / profiled_total if profiled_total > 0.0 else 0.0,
        "time_projection_percent": 100.0 * projection_total / profiled_total if profiled_total > 0.0 else 0.0,
        "time_backward_percent": 100.0 * backward_total / profiled_total if profiled_total > 0.0 else 0.0,
        "time_optimizer_percent": 100.0 * optimizer_total / profiled_total if profiled_total > 0.0 else 0.0,
    }
