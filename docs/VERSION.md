# NLPOptNet Version Tracker

## 0.1.1 - 2026-04-12

Console/log formatting and workspace cleanup.

- Rounded run-time console metrics to three decimals for cleaner training logs.
- Added shared console formatting helpers for scientific, decimal, time, and percent values.
- Preserved full precision in saved JSON/CSV artifacts.
- Removed the old root Grace GPU requirements file from the saved workspace.
- Updated the root quick-run script and default QP print frequency in the saved workspace.

## 0.1.0 - 2026-04-12

Run-only packaged codespace.

- Simplified the root runner to support only `--action run`.
- Removed the legacy extra runner modes from the active command-line workflow.
- Kept the active root workflow to one configured run at a time.
- Added editable package dependency selection for `pip install -e nlpopt`.
- Added CPU/GPU requirement files inside `nlpopt/`.
- Added `import nlpopt` with a public `ProblemBuilder`.
- Added `run_general.py` for simple builder-defined general problems.
- Added documentation under `docs/`.
- Added notebook examples under `notebooks/`.
