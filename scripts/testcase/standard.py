#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from scripts.misc.cli_overrides import apply_cli_overrides
from scripts.misc.json_io import load_json
from scripts.testcase import nlp_run, poly_run


def load_local_configs(case_dir: Path) -> tuple[dict, dict, dict]:
    data_cfg = load_json(case_dir / "data.json")
    cfg_dict = load_json(case_dir / "config.json")
    proj_cfg = load_json(case_dir / "proj.json")
    data_cfg, cfg_dict = apply_cli_overrides(data_cfg, cfg_dict)
    cfg_dict["model"] = "nlpopt"
    return data_cfg, cfg_dict, proj_cfg


def run_case(case_dir: Path, _path_arg: str | None = None) -> int:
    del _path_arg
    data_cfg, cfg_dict, proj_cfg = load_local_configs(case_dir)
    if str(data_cfg.get("type", "")).strip().lower() == "nlp":
        return nlp_run.run_case(
            case_dir,
            data_cfg_override=data_cfg,
            cfg_dict_override=cfg_dict,
            proj_cfg_override=proj_cfg,
        )
    return poly_run.run_case(
        case_dir,
        data_cfg_override=data_cfg,
        cfg_dict_override=cfg_dict,
        proj_cfg_override=proj_cfg,
    )
