# NLPOptNet

NLPOptNet trains a JAX-based optimization surrogate with a differentiable
projection layer for parametric constrained optimization problems.

The codespace is run-only: `main.py` supports the NLPOptNet model for one
problem configuration at a time.

## Install

```bash
python -m pip install --upgrade pip
python nlpopt/install_info.py
pip install -e nlpopt
```

See [docs/INSTALL.md](docs/INSTALL.md) for Windows, macOS, Linux, CPU, and GPU
setup notes.

## Runner

```bash
python main.py --type <problem> --action run [options]
```

Supported problem types:

- `qp`
- `qcqp`
- `nlp`
- `nonconvex`
- `general`

Arguments:

- `--type`: problem family.
- `--action`: only `run` is supported.
- `--p`: override parameter dimension `p` or `n_x`.
- `--n`: override decision dimension `n` or `n_y`.
- `--me`: override equality count `me` or `n_eq`.
- `--mi`: override inequality count `mi` or `n_ineq`.
- `--samples`: override generated sample count.
- `--epochs`: override training epochs.
- `--batch_size`: override training batch size.
- `--learning_rate`: override optimizer learning rate.
- `--train_frac`: override train split fraction.
- `--seed`: override data and training seed.
- `--solver`: override optimizer/data-generation solver.
- `--output_dir`: write run artifacts to a specific directory.

## Examples

QP:

```bash
python main.py --type qp --action run --p 2 --n 4 --me 1 --mi 1 --samples 12 --epochs 3 --batch_size 4 --train_frac 0.5 --solver OSQP
```

QCQP:

```bash
python main.py --type qcqp --action run --p 2 --n 4 --me 1 --mi 1 --samples 12 --epochs 3 --batch_size 4 --train_frac 0.5 --solver SCS
```

NLP:

```bash
python main.py --type nlp --action run --p 2 --n 4 --me 1 --mi 1 --samples 12 --epochs 3 --batch_size 4 --train_frac 0.5 --solver SCS
```

Nonconvex:

```bash
python main.py --type nonconvex --action run --p 1 --n 3 --me 1 --mi 1 --samples 12 --epochs 3 --batch_size 4 --train_frac 0.5
```

General block problem from `case/general/model_definition.py`:

```bash
python main.py --type general --action run --samples 12 --epochs 3 --batch_size 4 --train_frac 0.5
```

Simple builder-defined general problem:

```bash
python run_general.py --samples 12 --epochs 3 --batch_size 4 --train_frac 0.5
```

## Problem Definition

Structured QP/QCQP/NLP/nonconvex problems are configured under `case/<type>/`.

General block problems are defined in `case/general/model_definition.py`.
Simple general problems can be defined directly in `run_general.py` with
`nlpopt.ProblemBuilder`.

See [docs/PROBLEM.md](docs/PROBLEM.md) for details.

## Notebooks

Example notebooks are in `notebooks/`. They are intentionally small and write
local notebook artifacts under `notebooks/_runs/`.
