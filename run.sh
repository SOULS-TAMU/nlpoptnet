#!/usr/bin/env bash
set -euo pipefail

# Windows PowerShell activation:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#   D:\Projects\virtual_envs\env\Scripts\Activate.ps1
#
# macOS/Linux/WSL activation:
#   source env/bin/activate

python -m pip install --upgrade pip
python nlpopt/install_info.py
pip install -e nlpopt

python - <<'PY'
import importlib.metadata as md
import nlpopt

print("nlpopt module version:", nlpopt.__version__)
print("installed package version:", md.version("nlpopt"))
PY

python main.py \
  --type qp \
  --action run \
  --p 2 \
  --n 4 \
  --me 1 \
  --mi 1 \
  --samples 12 \
  --epochs 3 \
  --batch_size 4 \
  --train_frac 0.5 \
  --solver OSQP
