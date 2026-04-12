# case/general

This folder is a general JAX-model-based entrypoint for NLPOpt.

What makes it different:
- you define the optimization model in `model_definition.py`
- you keep sampling and training settings in JSON files
- you do not need to classify the problem as `qp`, `qcqp`, or `nlp`
- the same workflow runs through `python main.py --type general --action run`

Files:
- `model_definition.py`
- `data.json`
- `config.json`
- `proj.json`
- `factory.py`

The included template already shows how to build:
- a nonlinear objective with `set_objective(...)`
- affine equality blocks
- affine inequality blocks
- single affine equalities
- single affine inequalities
- quadratic inequalities
- optional nonlinear inequality callbacks
- box bounds

## Required contract

`model_definition.py` must define:

```python
def build_model(*, dtype):
    ...
    return jax_nlp_model
```

The parameter name should stay `x`, because the existing NLPOpt pipeline expects that parameter name.

`data.json` stores:
- dataset size
- parameter sampling bounds

`config.json` stores training settings.

`proj.json` stores projection-layer settings.

## Optional hooks

You can also define these in `model_definition.py`:

```python
def solve_instance(*, model, params, solver):
    ...
```

You do not need this to train NLPOpt in the general case.

Use it only if you also want to generate reference solutions for your own analysis.

```python
def sample_parameters(*, num_samples, rng, data_cfg):
    ...
```

Use this if you want something other than uniform sampling over `x_L` / `x_U`.

## Training behavior

The general case does not require any external solver to train NLPOpt.

The default workflow is:
- sample parameter vectors `x` from `data.json`
- build the optimization model from `model_definition.py`
- train the backbone plus projection stack directly on projected objective and feasibility

That means there is no solver-based label set, no supervised MSE term against an optimizer solution, and no optimality-gap report against a reference solver.

If you want reference solutions anyway, you can still add your own `solve_instance(...)` hook.

## How to create a new general model

1. Edit `model_definition.py` to build the model you want with `jaxmodel`.
2. Use `HighLevelNLPBuilder` methods such as `add_affine_equality`, `add_affine_inequality`, `add_nonlinear_equality`, `add_nonlinear_inequality`, `add_quadratic_inequality`, `set_affine_lower_bound`, and `set_affine_upper_bound`.
3. Keep the parameter block named `x`.
4. Set sampling bounds in `data.json` so they match the dimension of `x`.
5. Set training options in `config.json`.
6. Set CP projection options in `proj.json`.
7. Run:

```bash
python main.py --type general --action run
```

Outputs are written under:

```text
case/general/problem_data/general/
```

For objectives, you are not limited to quadratic forms.

You can either use:

```python
builder.set_quadratic_objective(Q=..., c=...)
```

or define any differentiable objective callable:

```python
def objective_fun(params, vars):
    x = params["x"]
    y = vars["y"]
    return jnp.sum(jnp.log1p(jnp.exp(y))) + 0.1 * jnp.sin(x[0] + y[0])

builder.set_objective(objective_fun)
```

The NLPOpt projection pipeline will quadraticize that objective locally inside each projection subproblem.

## Nonlinear Block Constraints

For nonlinear block constraints, pass a callable that takes:

```python
def my_constraint_block(params, vars):
    ...
    return residual_vector
```

Where:
- `params["x"]` is your parameter vector
- `vars["y"]` is your decision-variable vector
- the return value must be a 1D array

The sign convention is:
- equality block: returned residual must be exactly `0`
- inequality block: returned residual must be `<= 0`

Example nonlinear equality block:

```python
def nonlinear_eq_block(params, vars):
    x = params["x"]
    y = vars["y"]
    return jnp.asarray(
        [
            y[0] ** 2 + y[1] - x[0],
            jnp.sin(y[2]) + y[0] * y[1] - 0.25 * x[1],
        ],
        dtype=y.dtype,
    )

builder = builder.add_nonlinear_equality(
    nonlinear_eq_block,
    name="nonlinear_eq_block",
)
```

This defines two equality constraints:
- `y[0]^2 + y[1] - x[0] = 0`
- `sin(y[2]) + y[0] y[1] - 0.25 x[1] = 0`

Example nonlinear inequality block:

```python
def nonlinear_ineq_block(params, vars):
    x = params["x"]
    y = vars["y"]
    return jnp.asarray(
        [
            jnp.exp(0.2 * y[0]) + y[1] ** 2 - (1.0 + 0.1 * x[0]),
            y[0] * y[2] + jnp.cos(y[1]) - (0.8 + 0.05 * x[1]),
            jnp.sum(jnp.square(y)) - 2.0,
        ],
        dtype=y.dtype,
    )

builder = builder.add_nonlinear_inequality(
    nonlinear_ineq_block,
    name="nonlinear_ineq_block",
)
```

This defines three inequality constraints:
- `exp(0.2 y[0]) + y[1]^2 - (1.0 + 0.1 x[0]) <= 0`
- `y[0] y[2] + cos(y[1]) - (0.8 + 0.05 x[1]) <= 0`
- `||y||_2^2 - 2.0 <= 0`

You can combine nonlinear blocks with affine blocks, quadratic constraints, and bounds in the same model.

## Structured Block Constraints With Matrix Operations

If several constraints share the same algebraic structure, it is often cleaner to define the whole block with matrix operations instead of writing each residual separately.

This is useful for expressions like:

- `0.5 y^T A_i y + b_i^T sin(y) = rhs_i`
- `0.5 y^T A_i y + b_i^T sin(y) <= rhs_i + E_i x`

where only the data `(A_i, b_i, rhs_i, E_i)` changes from one constraint to the next.

The example `case/general/model_definition.py` includes helper patterns for this:
- [make_structured_nonlinear_eq_block](/home/grads/b/bimolnathroy/bimolnathroy/Work/NLPOpt/case/general/model_definition.py#L12)
- [make_structured_nonlinear_ineq_block](/home/grads/b/bimolnathroy/bimolnathroy/Work/NLPOpt/case/general/model_definition.py#L30)

Example structured nonlinear equality block:

```python
def make_structured_nonlinear_eq_block(A_stack, b_stack, rhs, *, dtype=jnp.float64):
    A_stack = jnp.asarray(A_stack, dtype=dtype)   # shape (m, n, n)
    b_stack = jnp.asarray(b_stack, dtype=dtype)   # shape (m, n)
    rhs = jnp.asarray(rhs, dtype=dtype)           # shape (m,)

    def block(params, vars):
        del params
        y = vars["y"]
        quad = 0.5 * jnp.einsum("mij,i,j->m", A_stack, y, y)
        trig = b_stack @ jnp.sin(y)
        return quad + trig - rhs

    return block
```

Usage:

```python
eq_block = make_structured_nonlinear_eq_block(
    A_stack=[
        [[1.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.0]],
        [[0.0, 0.2, 0.0], [0.2, 0.0, 0.0], [0.0, 0.0, 0.8]],
    ],
    b_stack=[
        [1.0, -0.5, 0.2],
        [0.3, 0.4, -0.6],
    ],
    rhs=[0.2, -0.1],
    dtype=dtype,
)

builder = builder.add_nonlinear_equality(eq_block, name="structured_nonlinear_eq")
```

This creates a 2-constraint equality block:
- `0.5 y^T A_1 y + b_1^T sin(y) - rhs_1 = 0`
- `0.5 y^T A_2 y + b_2^T sin(y) - rhs_2 = 0`

Example structured nonlinear inequality block:

```python
def make_structured_nonlinear_ineq_block(A_stack, b_stack, rhs, E=None, *, dtype=jnp.float64):
    A_stack = jnp.asarray(A_stack, dtype=dtype)   # shape (m, n, n)
    b_stack = jnp.asarray(b_stack, dtype=dtype)   # shape (m, n)
    rhs = jnp.asarray(rhs, dtype=dtype)           # shape (m,)
    E = None if E is None else jnp.asarray(E, dtype=dtype)  # shape (m, p)

    def block(params, vars):
        y = vars["y"]
        x = params["x"]
        quad = 0.5 * jnp.einsum("mij,i,j->m", A_stack, y, y)
        trig = b_stack @ jnp.sin(y)
        param_part = jnp.zeros_like(rhs) if E is None else E @ x
        return quad + trig - rhs - param_part

    return block
```

Usage:

```python
ineq_block = make_structured_nonlinear_ineq_block(
    A_stack=[
        [[0.8, 0.0, 0.0], [0.0, 0.3, 0.0], [0.0, 0.0, 0.0]],
        [[0.0, 0.1, 0.0], [0.1, 0.0, 0.0], [0.0, 0.0, 0.5]],
    ],
    b_stack=[
        [0.6, 0.0, 0.3],
        [0.2, -0.4, 0.5],
    ],
    rhs=[1.0, 0.7],
    E=[
        [0.1, 0.0],
        [0.0, 0.2],
    ],
    dtype=dtype,
)

builder = builder.add_nonlinear_inequality(ineq_block, name="structured_nonlinear_ineq")
```

This creates a block of residuals that must each satisfy `<= 0`.

In practice:
- use `A_stack` with shape `(m, n, n)` for `m` constraints over `n` variables
- use `b_stack` with shape `(m, n)` when each constraint has its own vector multiplying `sin(y)` or another elementwise transform
- use `E` with shape `(m, p)` when the block depends affinely on parameters `x` of size `p`

This pattern works for many other shared-structure blocks too. For example you could replace `sin(y)` with `exp(y)`, `tanh(y)`, `softplus(y)`, or any other differentiable elementwise transform.
