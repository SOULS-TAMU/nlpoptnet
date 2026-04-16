# nlpoptnet

`nlpoptnet` is an installable JAX package for learning projected solutions to
parametric optimization problems.

## Install

```bash
pip install nlpoptnet
```

For local development:

```bash
pip install -e .
```

## Minimal Example

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

model.extract("problem.npz")
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
model.dataset(parameters="parameters.csv")
model.build()
result = model.optimize()
print(result["output_dir"])
```

## Loading a Trained Run

```python
from nlpoptnet import NLPOptNet

model = NLPOptNet()
model.load("demo_qp_20260416_120000/metadata.json")
prediction = model.predict([1.0, 2.0])
```
