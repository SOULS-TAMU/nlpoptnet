# NLPOptNet

Parametric nonlinear optimization through a neural warm start and differentiable
local projection in JAX.

NLPOptNet supplies the nonlinear problem, local approximations, and task policy
to the shared **OHAGI** learning engine and **PURIN** projection library. It has
no independent neural backbone implementation, optimizer step, epoch engine,
CP solver, or deployment runtime.

## Install

From the package directory in a checkout of the development branch:

```shell
python -m pip install .
```

For Linux with an NVIDIA CUDA 12 GPU, use `python -m pip install ".[cuda12]"`.

Python 3.10 or newer is required. Dependency availability also depends on the
Python version and platform. The existing training backend uses Flax and Optax
through OHAGI. Deployment through OHAGI requires neither Flax nor NLPOptNet.

## Define and train a problem

```python
from nlpoptnet import NLPOptNet

model = NLPOptNet(name="example", config={
    "epochs": 100, "batch_size": 32, "learning_rate": 1e-3,
    "hidden_size": 64, "hidden_layers": 2,
})
p = model.add_parameter(["p"])
y = model.add_variable(["u", "v"])
model.objective((y.u - p.p)**2 + 0.5*y.v**2)
model.constraints.equality.add(y.u + y.v == p.p)
model.constraints.inequality.add(y.u <= 1.5)
model.constraints.box.add(y.u >= 0, y.v >= 0, y.u <= 2, y.v <= 2)
model.box(lower=0.4, upper=1.4, num_samples=1024)
model.build()
result = model.optimize()
```

The existing `NLPOptNet` and `ProblemBuilder` interfaces, configuration aliases,
sampling options, result dictionary, and output filenames remain unchanged.
CSV datasets, simplex/polytope sampling, matrix/vector expressions, and manual
or automatic problem scaling remain supported.

## Shared workflow

1. Define the objective, equalities, inequalities, bounds, and parameter domain.
2. OHAGI constructs the configured backbone, which predicts primal and dual
   warm starts using its shared Linen MLP head construction.
3. NLPOptNet forms the local diagonal quadratic objective and linearized
   constraints. PURIN solves the local projection and differentiates it.
4. NLPOptNet supplies its modified-Lagrangian/consistency loss and reporting
   policy; OHAGI performs optimizer transitions and compiled epoch execution.
5. OHAGI saves the shared JAX model bundle and prepares repeated inference.

`cp_mode`, Ruiz equilibration, adjoint settings, and `k_layer` retain their
existing meanings. Warm-up update behavior is preserved explicitly by the
frontend policy. The underlying numerical operations and default initialization
remain unchanged.

## Save, load, and infer

Full-model metadata is returned in `result["metadata_path"]`. Symbolic models
can be reopened with the existing frontend API:

```python
restored = NLPOptNet().load(result["metadata_path"])
predictions = restored.predict(parameter_batch)
```

For deployment, export once and load using only the metadata path:

```python
model.export_inference("deployment/metadata.json")

from ohagi.inference import load
predictor = load("deployment/metadata.json")
prediction = predictor(single_input)
```

Keep the referenced `.jax_export` file alongside `metadata.json`. Loading warms
one JAX signature; reuse the predictor across requests. `predict_jax` returns a
JAX array, while `predict` returns synchronized NumPy output.

The only model format is the shared `ohagi_model_v1` JAX bundle. Its parameter
archive uses `ohagi_pytree_npz_v1`; inference uses `ohagi_jax_inference_v1` and
`ohagi_jax_program_v1`. Deployment loads directly through OHAGI without
reconstructing the nonlinear problem or importing frontends, Flax, or Orbax.
There are no legacy model readers or native export backends.

## Source organization

```text
src/nlpoptnet/
├── __init__.py
├── api/
│   ├── __init__.py
│   ├── root.py
│   ├── model.py
│   ├── configuration.py
│   ├── network.py
│   ├── data.py
│   ├── problem.py
│   ├── projection.py
│   ├── objectives.py
│   ├── fit.py
│   ├── predict.py
│   ├── reporting.py
│   ├── deployment.py
│   └── formulation/
│       ├── __init__.py
│       ├── variables.py
│       ├── parameters.py
│       ├── expressions.py
│       ├── constraints.py
│       ├── bounds.py
│       ├── builder.py
│       ├── compiler.py
│       ├── approximation.py
│       └── scaling.py
└── serialization/
    ├── __init__.py
    ├── definition.py
    └── artifacts.py
```

`model.py` binds the existing public methods to small task modules.
`formulation/` owns expression handling, variable packing, objective/residual
evaluation, derivatives, scaling, and local subproblem construction. These are
mathematical problem resources, not a second neural training framework.

The flat API adapters supply those resources to OHAGI/PURIN. Serialization
records the nonlinear definition and captures the prediction graph for OHAGI's
common bundle writer. Standard Flax training state and optimizer transport are
provided by OHAGI, rather than a package-specific training-state class.

The old `optimization/`, `problems/`, empty `adapters/`, CP forwarding modules,
and `_aliases.py` catalog are removed. Obsolete private import paths are retired;
the benchmark-facing model interface is preserved.

## Development and validation

Packaging uses the `src` layout and `pyproject.toml`. See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines, and
[CITATION.cff](CITATION.cff) for citation information.

See [CHANGELOG.md](CHANGELOG.md), [LICENSE](LICENSE), and [NOTICE](NOTICE) for
changes, licensing, and attribution.

## License

This project is licensed under the [Apache License, Version 2.0](LICENSE).

Original authorship and applicable upstream notices are retained in [NOTICE](NOTICE).

## Optional sparse inequality projection

Set `config["inequality_support"]` to one list of structural variable-column
indices per inequality row before building the model. Include every entry that
can be nonzero anywhere in the problem domain. Omitted entries are assumed to
be structurally zero; a sparsity pattern sampled at one point is insufficient.

PURIN packs those entries and uses sparse products in CP and its implicit
adjoint. Equality matrices stay dense, and quadratic diagonals stay vectors.
The dense path works with the published PURIN 1.0.0 wheel. To enable
`inequality_support`, install a PURIN checkout that contains sparse
Chambolle–Pock support. With both checkouts available locally, install them
together from the NLPOptNet package directory.

```shell
python -m pip install /path/to/purin .
```

The option is preserved by model
serialization; loading inference still takes only `metadata.json`.


## Paper and experiments

See the [NLPOpt-Net paper](https://arxiv.org/abs/2605.00260) and the
[article experiments](https://github.com/SOULS-TAMU/NLPOpt-Net-Experiments).
The existing [Read the Docs site](https://nlpoptnet.readthedocs.io/) describes the
main-branch release. Use the training and inference examples above for this
development branch.
