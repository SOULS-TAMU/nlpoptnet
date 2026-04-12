#!/usr/bin/env python3
from __future__ import annotations

import copy
from pathlib import Path

from scripts.testcase import nonconvx_run as base_runner


def _load_local_configs(case_dir: Path) -> tuple[dict, dict, dict]:
    return (
        base_runner._load_json(case_dir / "data.json"),
        base_runner._load_json(case_dir / "config.json"),
        base_runner._load_json(case_dir / "proj.json"),
    )


def _normalize_legacy_data_cfg(data_cfg: dict) -> dict:
    data = copy.deepcopy(dict(data_cfg))
    if "n_y" not in data and "n" in data:
        data["n_y"] = int(data["n"])
    if "n_eq" not in data and "me" in data:
        data["n_eq"] = int(data["me"])
    if "n_ineq" not in data and "mi" in data:
        data["n_ineq"] = int(data["mi"])
    data["type"] = "nonconvex"
    return base_runner._normalize_local_data_cfg(data)


def run_case(
    case_dir: Path,
    _path_arg: str | None = None,
    *,
    data_cfg_override: dict | None = None,
    cfg_dict_override: dict | None = None,
    proj_cfg_override: dict | None = None,
    output_dir_override: Path | None = None,
) -> int:
    del _path_arg
    raw_data_cfg, cfg_dict, proj_cfg = _load_local_configs(case_dir)
    data_cfg = _normalize_legacy_data_cfg(data_cfg_override if data_cfg_override is not None else raw_data_cfg)
    cfg = copy.deepcopy(cfg_dict_override if cfg_dict_override is not None else cfg_dict)
    proj = copy.deepcopy(proj_cfg_override if proj_cfg_override is not None else proj_cfg)
    cfg["model"] = "nlpopt"

    artifacts = base_runner._run_single_case(
        case_dir,
        data_cfg,
        cfg,
        proj,
        output_dir_override=output_dir_override,
    )
    metadata_path = base_runner.unified._append_family_metadata(
        artifacts.dataset_dir,
        mode="single_model",
        output_dir=artifacts.run_dir,
        data_cfg=data_cfg,
        cfg_dict=cfg,
        proj_cfg=proj,
        framework=artifacts.framework,
        seeds=[int(cfg.get("seed", 42))],
        extra={"summary_path": str(artifacts.metrics_path), "history_path": str(artifacts.history_path)},
    )
    print(f"[run] Updated family metadata: {metadata_path}")
    return 0
