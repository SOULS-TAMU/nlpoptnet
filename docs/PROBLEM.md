# Problem Workflow

`NLPOptNet` is for optimization problems where the user defines the objective,
constraints, and parameter region, then trains the network to predict projected
solutions.

## Main API

```python
from nlpoptnet import NLPOptNet

model = NLPOptNet(config=CONFIG, type="qp", name="my_model")
```

The `type` argument is optional for general problems. Typical structured values
are `qp`, `qcqp`, `nlp`, and `nonconvex`.

## Variables and Parameters

```python
x = model.add_parameter(["x1", "x2"])
y = model.add_variable(["y1", "y2", "y3"])
```

Parameters are the network inputs. Variables are the optimization outputs.

## Loading Structured Problem Data

If the problem matrices are already saved in `problem.npz`, load them with:

```python
model.extract("notebooks/data/qp/problem.npz")
```

Each key becomes a model attribute, so `Q`, `c`, `A`, `b`, `B`, `C`, `d`, `D`,
`l`, `L`, `u`, `U`, `M`, and other saved arrays can be accessed as
`model.Q`, `model.c`, and so on.

You can also define constants directly:

```python
model.Q = model.matrix([[1.0, 0.0], [0.0, 1.0]])
model.c = model.vector([1.0, 2.0])
```

## Objective

Examples:

```python
model.objective(0.5 * (y.y1**2 + y.y2**2 + y.y3**2))
```

```python
model.objective(0.5 * model.quad(model.Q, y) + model.lin(model.c, y))
```

```python
model.objective(0.5 * model.quad(model.Q, y) + model.lin(model.c, model.sin(y)))
```

Callable objectives are also supported:

```python
def objective(params, vars):
    del params
    y_value = vars["y"]
    return 0.5 * y_value @ model.Q @ y_value + model.c @ model.sin(y_value)

model.objective(objective)
```

## Constraints

### Equality

```python
model.constraints.equality.add(
    y.y1 + y.y2 - x.x1 == 0,
    y.y2 - y.y3 - x.x2 == 0,
)
```

Structured affine equalities:

```python
model.constraints.equality.add(
    model.lin(model.A, y) == model.b + model.lin(model.B, x),
)
```

### Inequality

```python
model.constraints.inequality.add(
    y.y1**2 + y.y3**2 <= 2.0,
)
```

QP:

```python
model.constraints.inequality.add(
    model.lin(model.C, y) <= model.d + model.lin(model.D, x),
)
```

QCQP:

```python
model.constraints.inequality.add(
    model.batch_quad(model.C, y) + model.batch_lin(model.d, y)
    <= model.e + model.batch_lin(model.E, x),
)
```

NLP:

```python
model.constraints.inequality.add(
    model.batch_lin(model.a, model.batch_exp(y)) + model.batch_quad(model.W, y)
    <= model.beta + model.batch_lin(model.E, x),
)
```

Callable blocks are supported for nonlinear systems:

```python
def equality_block(params, vars):
    x_value = params["x"]
    y_value = vars["y"]
    return y_value[:2] - x_value

model.constraints.equality.add(equality_block)
```

### Box Constraints

Box bounds are handled separately from general inequalities:

```python
model.constraints.box.add(
    var=y,
    lower=model.l + model.lin(model.L, x),
    upper=model.u + model.lin(model.U, x),
)
```

Scalar-style bounds are also accepted:

```python
model.constraints.box.add(y.y1 >= 0.0, y.y2 <= 1.0)
```

## Parameter Data

### From CSV

```python
model.dataset(parameters="notebooks/data/qp/parameters.csv")
```

### From a Simplex or Polytope Region

```python
model.simplex(
    x.x1 >= 0.0,
    x.x2 >= 0.0,
    x.x1 + x.x2 <= 1.0,
    num_samples=1000,
)
```

You can also load a saved matrix region:

```python
model.simplex(M="simplex.npz", num_samples=1000)
```

### From a Box Region

```python
model.box(lower=[-1.0, -1.0], upper=[1.0, 1.0], num_samples=1000)
```

## Build and Optimize

```python
model.build()
result = model.optimize()
run_dir = result["output_dir"]
```

`build()` prepares the symbolic problem, loads or samples parameters, creates
the train and validation splits, and runs the warmup compilation path.

`optimize()` trains the model and writes a timestamped run directory in the
working directory.

## Loading and Predicting

```python
model = NLPOptNet()
model.load("demo_qp_20260416_120000/metadata.json")
prediction = model.predict([1.0, 2.0])
```

Automatic reload works for the symbolic and structured problems created from
expressions and extracted constants. If a run was built from raw Python
callables for the objective or constraint blocks, `load()` raises a clear error
because that callable code is not reconstructed from metadata alone.

If a model has not been trained or loaded, `predict()` raises:

```text
Please train or load the model before calling predict().
```

## Saved Run Artifacts

Each run directory contains the training metadata and the files needed to load
the model again:

- `metadata.json`
- `model_weights.npz`
- `parameters.csv`
- `predicted_variables.csv`
- `history.csv`
- `summary.json`
- `problem_constants.npz`
- copied `problem.npz` when `extract(...)` was used

## Example Data

The repository includes notebook-ready structured data here:

```text
notebooks/data/qp/
notebooks/data/qcqp/
notebooks/data/nlp/
notebooks/data/nonconvex/
```
