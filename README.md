# NLPOptNet

Parametric nonlinear optimization through a neural warm start and differentiable
local projection in JAX.

[![PyPI](https://img.shields.io/pypi/v/nlpoptnet)](https://pypi.org/project/nlpoptnet/)
[![Documentation](https://img.shields.io/readthedocs/nlpoptnet/latest?logo=readthedocs&label=docs)](https://nlpoptnet.readthedocs.io/)

[![CI](https://github.com/SOULS-TAMU/nlpoptnet/actions/workflows/ci.yml/badge.svg?branch=development)](https://github.com/SOULS-TAMU/nlpoptnet/actions/workflows/ci.yml?query=branch%3Adevelopment)
[![Validation](https://github.com/SOULS-TAMU/nlpoptnet/actions/workflows/validation.yml/badge.svg?branch=development)](https://github.com/SOULS-TAMU/nlpoptnet/actions/workflows/validation.yml?query=branch%3Adevelopment)
[![CodeQL](https://github.com/SOULS-TAMU/nlpoptnet/actions/workflows/codeql.yml/badge.svg?branch=development)](https://github.com/SOULS-TAMU/nlpoptnet/actions/workflows/codeql.yml?query=branch%3Adevelopment)
[![Dependency review](https://github.com/SOULS-TAMU/nlpoptnet/actions/workflows/dependency-review.yml/badge.svg?branch=development)](https://github.com/SOULS-TAMU/nlpoptnet/actions/workflows/dependency-review.yml?query=branch%3Adevelopment)
[![Coverage](https://raw.githubusercontent.com/SOULS-TAMU/nlpoptnet/codex/coverage-development/coverage.svg)](https://github.com/SOULS-TAMU/nlpoptnet/tree/codex/coverage-development)

[![Python versions](https://img.shields.io/badge/Python-3.10%E2%80%933.14-3776AB?logo=python&logoColor=white)](https://github.com/SOULS-TAMU/nlpoptnet/blob/development/.github/workflows/ci.yml)
[![CPU platforms](https://img.shields.io/badge/CPU-Linux%20%7C%20Windows%20%7C%20macOS-555555)](https://github.com/SOULS-TAMU/nlpoptnet/blob/development/.github/workflows/ci.yml)
[![JAX](https://img.shields.io/badge/Powered_by-JAX-8A2BE2)](https://docs.jax.dev/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue)](https://github.com/SOULS-TAMU/nlpoptnet/blob/development/LICENSE)
[![Citation: BibTeX](https://img.shields.io/badge/Cite-BibTeX-6F4E37)](https://github.com/SOULS-TAMU/nlpoptnet/blob/development/CITATION.bib)


NLPOptNet supplies the nonlinear problem, local approximations, and task policy
to the shared **OHAGI** learning engine and **PURIN** projection library. It has
no independent neural backbone implementation, optimizer step, epoch engine,
CP solver, or deployment runtime.

## Install

From a checkout of the development branch:

```shell
python -m pip install ./nlpoptnet
```

For Linux with an NVIDIA CUDA 12 GPU, use `python -m pip install "./nlpoptnet[cuda12]"`.

CPU CI targets Python 3.10–3.14 on Linux, Windows and macOS. Python 3.15 is pending compatible wheels for the Flax dependency chain. The existing training backend uses Flax and Optax
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

Use the loaded model to inspect original-problem KKT residuals at inference:

```python
info = restored.predict_with_info(
    parameter_batch, adaptive=True, max_layers=10, primal_tolerance=1e-6
)
print(info["primal_feasible"], info["kkt_satisfied"], info["layers_used"])
print(info["residuals"], info["cp_iterations"])
```

Pass a single parameter vector or a batch as the first argument. Adaptive depth
stops each sample on primal feasibility and can compile new JAX batch shapes
as samples finish. The KKT check also tests dual feasibility, complementarity
and stationarity in original problem units. Use PURIN from its GitHub main
branch for CP iteration counts. With PURIN 1.0.1, that field returns `None`.
Regular prediction and training use fixed depth.

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

Packaging uses the `src` layout and `pyproject.toml`. See [CONTRIBUTING.md](nlpoptnet/CONTRIBUTING.md) for development guidelines, and
[CITATION.cff](CITATION.cff) for citation information.

See [CHANGELOG.md](nlpoptnet/CHANGELOG.md), [LICENSE](LICENSE), and [NOTICE](nlpoptnet/NOTICE) for
changes, licensing, and attribution.

## License

This project is licensed under the [Apache License, Version 2.0](LICENSE).

Original authorship and applicable upstream notices are retained in [NOTICE](nlpoptnet/NOTICE).

## Optional sparse inequality projection

Set `config["inequality_support"]` to one list of structural variable-column
indices per inequality row before building the model. Include every entry that
can be nonzero anywhere in the problem domain. Omitted entries are assumed to
be structurally zero; a sparsity pattern sampled at one point is insufficient.

PURIN packs those entries and uses sparse products in CP and its implicit
adjoint. Equality matrices stay dense, and quadratic diagonals stay vectors.
NLPOptNet 2.0 uses PURIN 1.0.1 or newer for both paths. Omit this option to retain dense behavior. The option is preserved by model
serialization; loading inference still takes only `metadata.json`.


## Paper and experiments

See the [NLPOpt-Net paper](https://arxiv.org/abs/2605.00260) and the
[article experiments](https://github.com/SOULS-TAMU/NLPOpt-Net-Experiments).
The existing [Read the Docs site](https://nlpoptnet.readthedocs.io/) describes the
main-branch release. Use the training and inference examples above for this
development branch.
