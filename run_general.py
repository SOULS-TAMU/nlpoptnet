#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "nlpopt" / "src"

for candidate in (ROOT, SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from nlpopt import ProblemBuilder  # noqa: E402
from scripts.testcase import general_run  # noqa: E402


DATA = {
    "type": "general",
    "num_samples": 12,
    "seed": 42,
    "force_regenerate": True,
    "x_L": [-1.0, -1.0],
    "x_U": [1.0, 1.0],
}

CONFIG = {
    "model": "nlpopt",
    "epochs": 3,
    "batch_size": 4,
    "learning_rate": 1e-3,
    "train_frac": 0.5,
    "hidden_size": 32,
    "hidden_dim": 2,
    "seed": 42,
    "dtype": "float64",
    "print_every": 1,
}

PROJ = {
    "cp_mode": "fixed",
    "safety": 0.95,
    "knorm_iters": 20,
    "adjoint_iters": 30,
    "use_ruiz": True,
    "ruiz_iters": 4,
    "k_layer": 1,
}


def build_problem() -> ProblemBuilder:
    builder = ProblemBuilder(y_bound=4.0)
    x = builder.add_parameter(["x1", "x2"])
    y = builder.add_variable(["y1", "y2", "y3"])

    builder.objective = 0.5 * (y.y1**2 + y.y2**2 + y.y3**2)
    builder.constraints.add(
        y.y1 + y.y2 - x.x1 == 0,
        y.y2 - y.y3 - x.x2 == 0,
        y.y1**2 + y.y3**2 <= 2.0,
    )
    builder.bounds.set(lower=-4.0, upper=4.0)
    return builder


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a builder-defined simple general NLPOptNet problem.")
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--train_frac", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--workspace", default=None, help="Generated case workspace. Default: case/general_simple.")
    parser.add_argument("--output_dir", default=None)
    return parser.parse_args(argv)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def _write_model_definition(workspace: Path) -> None:
    source_hash = hashlib.sha1(Path(__file__).read_bytes()).hexdigest()[:12]
    text = f'''from __future__ import annotations

import jax.numpy as jnp

from run_general import build_problem

PARAM_NAME = "x"
SOURCE_HASH = "{source_hash}"


def build_model(*, dtype=jnp.float64):
    builder = build_problem()
    model, _metadata = builder.build_model(dtype=dtype, train_inverse=False)
    return model
'''
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "model_definition.py").write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    data = dict(DATA)
    config = dict(CONFIG)
    proj = dict(PROJ)

    if args.samples is not None:
        data["num_samples"] = int(args.samples)
    if args.epochs is not None:
        config["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        config["batch_size"] = int(args.batch_size)
    if args.train_frac is not None:
        config["train_frac"] = float(args.train_frac)
    if args.seed is not None:
        data["seed"] = int(args.seed)
        config["seed"] = int(args.seed)

    workspace = Path(args.workspace) if args.workspace else ROOT / "case" / "general_simple"
    output_dir = None if args.output_dir is None else Path(args.output_dir)

    _write_model_definition(workspace)
    _write_json(workspace / "data.json", data)
    _write_json(workspace / "config.json", config)
    _write_json(workspace / "proj.json", proj)

    print("=" * 80)
    print("NLPOptNet general builder runner")
    print(f"Workspace: {workspace}")
    print(f"Data config: {data}")
    print(f"Train config: {config}")
    print(f"Projection config: {proj}")
    print("=" * 80)

    return general_run.run_case(
        workspace,
        data_cfg_override=data,
        cfg_dict_override=config,
        proj_cfg_override=proj,
        output_dir_override=output_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())

