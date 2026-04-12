from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping


def artifact_size_mb(paths: Iterable[Path]) -> float:
    total_bytes = 0
    for path in paths:
        if path.exists() and path.is_file():
            total_bytes += int(path.stat().st_size)
    return float(total_bytes) / (1024.0 * 1024.0)


def history_optimizer_timing_fields(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    if not metadata:
        return {}
    keys = (
        "optimizer_generation_num_points",
        "optimizer_generation_wall_time_sec",
        "optimizer_generation_time_per_sample_sec",
        "optimizer_generation_total_wall_time_sec",
    )
    return {key: metadata[key] for key in keys if key in metadata}


def enrich_optimizer_generation_metadata(
    metadata: Mapping[str, Any],
    *,
    num_points: int,
    wall_time_sec: float | None = None,
    artifact_paths: Iterable[Path] | None = None,
) -> dict[str, Any]:
    enriched = dict(metadata)
    num_points = int(num_points)
    enriched["optimizer_generation_num_points"] = num_points

    if wall_time_sec is not None:
        wall_time_sec = float(wall_time_sec)
        enriched["optimizer_generation_wall_time_sec"] = wall_time_sec
        if num_points > 0:
            enriched["optimizer_generation_time_per_sample_sec"] = wall_time_sec / num_points

    if artifact_paths is not None:
        space_mb = artifact_size_mb(artifact_paths)
        enriched["optimizer_generation_space_mb"] = space_mb
        if num_points > 0:
            enriched["optimizer_generation_space_mb_per_sample"] = space_mb / num_points

    return enriched
