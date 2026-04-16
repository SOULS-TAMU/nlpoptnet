# NLPOptNet Version Tracker

## 0.2.0 - 2026-04-16

PyPI-oriented optimization package refactor.

- Added the new `nlpoptnet` package layout under `nlpoptnet/src/`.
- Promoted the public API to `from nlpoptnet import NLPOptNet`.
- Reworked the user workflow around `extract`, `dataset`, `simplex`, `box`,
  `build`, `optimize`, `load`, and `predict`.
- Kept bound constraints on the dedicated box-constraint path.
- Removed the old in-pipeline problem generation flow from the packaged module.
- Added package docs for installation, problem definition, and publishing.
- Updated packaging metadata for Python `3.13` compatible dependency markers.
- Refocused notebooks on loading data from `notebooks/data/<problem_type>/`.

## 0.1.1 - 2026-04-12

Console/log formatting and workspace cleanup.

- Rounded run-time console metrics to three decimals for cleaner training logs.
- Added shared console formatting helpers for scientific, decimal, time, and
  percent values.
- Preserved full precision in saved JSON and CSV artifacts.

## 0.1.0 - 2026-04-12

Run-only packaged codespace.

- Simplified the root runner to support only `--action run`.
- Added editable package dependency selection for `pip install -e nlpopt`.
- Added notebook examples under `notebooks/`.
