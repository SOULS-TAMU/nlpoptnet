#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "nlpopt" / "src"

for candidate in (ROOT, SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))


_TYPE_TO_DIR = {
    "qp": ROOT / "case" / "qp",
    "qcqp": ROOT / "case" / "qcqp",
    "nlp": ROOT / "case" / "nlp",
    "nonconvex": ROOT / "case" / "nonconvx",
    "nonconvx": ROOT / "case" / "nonconvx",
    "general": ROOT / "case" / "general",
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NLPOptNet run-only dispatcher.")
    parser.add_argument("--type", required=True, choices=("qp", "qcqp", "nlp", "nonconvex", "nonconvx", "general"))
    parser.add_argument("--action", default="run", choices=("run",), help="Only run mode is supported.")
    parser.add_argument("--p", type=int, default=None, help="Override p or n_x.")
    parser.add_argument("--n", type=int, default=None, help="Override n or n_y.")
    parser.add_argument("--me", type=int, default=None, help="Override me or n_eq.")
    parser.add_argument("--mi", type=int, default=None, help="Override mi or n_ineq.")
    parser.add_argument("--samples", type=int, default=None, help="Override num_samples/N_samples/N_points.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--train_frac", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--solver", default=None)
    parser.add_argument("--output_dir", default=None, help="Optional run-artifact directory.")
    return parser.parse_args(argv)


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_case_configs(case_dir: Path) -> tuple[dict, dict, dict]:
    return _load_json(case_dir / "data.json"), _load_json(case_dir / "config.json"), _load_json(case_dir / "proj.json")


def _set_first(data: dict, value, *keys: str) -> None:
    if value is None:
        return
    for key in keys:
        if key in data:
            data[key] = int(value)
            return


def _broadcast_bounds(data: dict) -> None:
    dim = data.get("p", data.get("n_x"))
    if dim is None:
        return
    dim = int(dim)
    for key in ("x_L", "x_U"):
        if key not in data or not isinstance(data[key], list):
            continue
        values = list(data[key])
        if len(values) == 1 and dim != 1:
            data[key] = values * dim
        elif len(values) not in {1, dim}:
            raise ValueError(f"{key} must have length 1 or {dim}.")


def _apply_overrides(data_cfg: dict, cfg_dict: dict, args: argparse.Namespace) -> tuple[dict, dict]:
    data = copy.deepcopy(data_cfg)
    cfg = copy.deepcopy(cfg_dict)

    _set_first(data, args.p, "p", "n_x")
    _set_first(data, args.n, "n", "n_y")
    _set_first(data, args.me, "me", "n_eq")
    _set_first(data, args.mi, "mi", "n_ineq")
    if args.samples is not None:
        if "num_samples" in data:
            data["num_samples"] = int(args.samples)
        if "N_samples" in data:
            data["N_samples"] = int(args.samples)
        if "N_points" in data:
            data["N_points"] = int(args.samples)
    if args.seed is not None:
        data["seed"] = int(args.seed)
        cfg["seed"] = int(args.seed)
    if args.solver is not None:
        data["solver"] = str(args.solver)

    if args.epochs is not None:
        cfg["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        cfg["batch_size"] = int(args.batch_size)
    if args.learning_rate is not None:
        cfg["learning_rate"] = float(args.learning_rate)
    if args.train_frac is not None:
        cfg["train_frac"] = float(args.train_frac)
    cfg["model"] = "nlpopt"

    _broadcast_bounds(data)
    return data, cfg


def _dispatch(problem_type: str):
    if problem_type in {"qp", "qcqp"}:
        from scripts.testcase import poly_run

        return poly_run.run_case
    if problem_type == "nlp":
        from scripts.testcase import nlp_run

        return nlp_run.run_case
    if problem_type in {"nonconvex", "nonconvx"}:
        from scripts.testcase import nonconvx

        return nonconvx.run_case
    if problem_type == "general":
        from scripts.testcase import general_run

        return general_run.run_case
    raise ValueError(f"Unsupported type: {problem_type}")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    case_dir = _TYPE_TO_DIR[args.type]
    if not case_dir.exists():
        raise FileNotFoundError(f"Expected case directory at {case_dir}")

    data_cfg, cfg_dict, proj_cfg = _load_case_configs(case_dir)
    data_cfg, cfg_dict = _apply_overrides(data_cfg, cfg_dict, args)
    output_dir = None if args.output_dir is None else Path(args.output_dir)

    print("")
    print("=" * 80)
    print(f"NLPOptNet runner | type={args.type} action=run")
    print(f"Case directory: {case_dir}")
    print(f"Data config: {data_cfg}")
    print(f"Train config: {cfg_dict}")
    print(f"Projection config: {proj_cfg}")
    if output_dir is not None:
        print(f"Output directory: {output_dir}")
    print("=" * 80)

    runner = _dispatch(args.type)
    return int(
        runner(
            case_dir,
            data_cfg_override=data_cfg,
            cfg_dict_override=cfg_dict,
            proj_cfg_override=proj_cfg,
            output_dir_override=output_dir,
        )
        or 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
