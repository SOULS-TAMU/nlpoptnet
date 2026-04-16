# NLPOptNet

`NLPOptNet` is a JAX-based package for learning optimization mappings with a
projection layer. The repo now focuses on the optimization model code itself:
users provide `parameters.csv` or a bounded parameter region, then define the
objective and constraints with the `NLPOptNet` API.

The installable package lives in [`nlpoptnet/`](./nlpoptnet). Example
workflows live in [`notebooks/`](./notebooks), and the API notes are in
[`docs/`](./docs).

## Install

For local development from this repo:

```bash
python -m pip install --upgrade pip
pip install -e nlpoptnet
```

For a published release:

```bash
pip install nlpoptnet
```

See [docs/INSTALL.md](./docs/INSTALL.md) for Python version notes and local
setup details.

## Quick Start

```python
from nlpoptnet import NLPOptNet

CONFIG = {
    "epochs": 1000,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "train_frac": 0.8,
    "hidden_size": 64,
    "hidden_layers": 2,
    "seed": 42,
    "dtype": "float64",
}

model = NLPOptNet(config=CONFIG, type="qp", name="demo_qp")
x = model.add_parameter(["x1", "x2"])
y = model.add_variable(["y1", "y2", "y3", "y4"])

model.extract("notebooks/data/qp/problem.npz")
model.objective(0.5 * model.quad(model.Q, y) + model.lin(model.c, y))
model.constraints.equality.add(
    model.lin(model.A, y) == model.b + model.lin(model.B, x),
)
model.constraints.inequality.add(
    model.lin(model.C, y) <= model.d + model.lin(model.D, x),
)
model.constraints.box.add(
    var=y,
    lower=model.l + model.lin(model.L, x),
    upper=model.u + model.lin(model.U, x),
)
model.dataset(parameters="notebooks/data/qp/parameters.csv")
model.build()
result = model.optimize()
run_dir = result["output_dir"]
```

Each optimization run writes a timestamped output directory in the working
directory:

```text
<model_name>_<timestamp>/
```

That directory includes `metadata.json`, `model_weights.npz`,
`predicted_variables.csv`, `parameters.csv`, and the saved problem constants
needed for `model.load(...)` and `model.predict(...)`.

## Core Ideas

- `NLPOptNet(config=..., type=...)` is the user-facing entry point.
- `model.extract(problem.npz)` loads structured matrices and exposes them as
  model attributes like `model.Q`, `model.A`, `model.b`, and so on.
- `model.dataset(...)`, `model.simplex(...)`, and `model.box(...)` define the
  parameter region.
- `model.constraints.box.add(...)` is separate from general inequalities so the
  bound constraints stay on the dedicated projection path.
- `model.optimize()` trains and returns a result dictionary with `output_dir`,
  `metadata_path`, `summary`, and `history`.
- `model.load(metadata_path)` restores a trained model, and
  `model.predict(x_value)` returns projected variable predictions.

## Repository Layout

```text
docs/
nlpoptnet/
notebooks/
.gitignore
README.md
```

## Documentation

- [docs/INSTALL.md](./docs/INSTALL.md)
- [docs/PROBLEM.md](./docs/PROBLEM.md)
- [docs/PUBLISH.md](./docs/PUBLISH.md)
- [docs/VERSION.md](./docs/VERSION.md)
