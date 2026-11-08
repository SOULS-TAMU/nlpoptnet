# Contributing

From the repository root, install the development package with
`python -m pip install -e "./nlpoptnet[dev]"`. To exercise sparse projection,
install a local PURIN checkout with sparse Chambolle–Pock support in the same
environment.

Keep mathematical problem formulation and task policies in NLPOptNet. Use
OHAGI for training and deployment and PURIN for projection algorithms.
Preserve the public model and benchmark interfaces when changing internals.

Before submitting changes, compare outputs, objective values, constraint
residuals, and gradients under the same data, seed, precision, and device.
For deployment changes, export a trained model, load it from `metadata.json`,
and verify repeated single-instance predictions. Keep test results,
environments, model checkpoints, and generated builds out of commits.

From the package directory, check source formatting with
`python -m ruff format --check src` and build distributions with
`python -m build`.
