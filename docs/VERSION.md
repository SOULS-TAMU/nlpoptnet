# NLPOptNet Version Tracker

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
