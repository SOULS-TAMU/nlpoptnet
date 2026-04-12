# Problem Definitions

NLPOptNet supports five run types from `main.py`:

- `qp`
- `qcqp`
- `nlp`
- `nonconvex`
- `general`

The structured cases use the JSON files under `case/<type>/`:

- `data.json`: problem dimensions, sampling, solver, and data generation.
- `config.json`: training configuration.
- `proj.json`: projection/subproblem configuration.

## Block General Problems

The block general workflow uses:

```text
case/general/model_definition.py
case/general/data.json
case/general/config.json
case/general/proj.json
```

`model_definition.py` must define:

```python
def build_model(*, dtype):
    ...
    return jax_nlp_model
```

The returned model should use the parameter name `x`. The existing
`case/general/model_definition.py` is the reference block-style example.

Run it with:

```bash
python main.py --type general --action run
```

## Simple Builder General Problems

Use `run_general.py` for a Pyomo-style builder workflow:

```python
from nlpopt import ProblemBuilder

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
```

Run it with:

```bash
python run_general.py --samples 12 --epochs 3 --batch_size 4 --train_frac 0.5
```
