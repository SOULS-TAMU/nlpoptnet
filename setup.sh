#!/usr/bin/env bash
set -euo pipefail

# Run this from inside the nlpopt directory
if [[ "$(basename "$PWD")" != "nlpopt" ]]; then
  echo "Please run this from inside the nlpopt directory."
  exit 1
fi

SCRATCH_ROOT="/scratch/user/$USER"
ENV_ROOT="$SCRATCH_ROOT/virtual_envs/env"
MODULE_DIR="$SCRATCH_ROOT/modulefiles"
REPO_ROOT="$SCRATCH_ROOT/NLPOptNet"
PKG_SRC="$PWD"

mkdir -p "$SCRATCH_ROOT/virtual_envs" "$MODULE_DIR" "$REPO_ROOT"

# Make sure the expected repo path exists for the modulefile.
# If your current nlpopt folder is elsewhere, expose it at $REPO_ROOT/nlpopt.
if [[ "$PKG_SRC" != "$REPO_ROOT/nlpopt" ]]; then
  ln -sfn "$PKG_SRC" "$REPO_ROOT/nlpopt"
fi

module purge
module load GCCcore/13.3.0
module load Python/3.12.3

python3 -m venv "$ENV_ROOT"
source "$ENV_ROOT/bin/activate"

python -m pip install --upgrade pip setuptools wheel

# Core package deps with GPU JAX stack
python -m pip install \
  "jax[cuda12]==0.4.30" \
  "jaxlib==0.4.30" \
  "flax==0.8.5" \
  "optax==0.2.4" \
  "numpy==1.26.4" \
  "scipy==1.13.1" \
  "cvxpy==1.7.5"

# Install nlpopt from the current source tree
python -m pip install -e "$REPO_ROOT/nlpopt"

cat > "$MODULE_DIR/nlpopt" <<'EOF'
#%Module1.0
module-whatis "NLPOpt GPU environment"
module load GCCcore/13.3.0
module load Python/3.12.3

set root /scratch/user/$env(USER)/virtual_envs/env
set repo /scratch/user/$env(USER)/NLPOptNet

setenv VIRTUAL_ENV $root
setenv NLP_OPT_ROOT $repo

prepend-path PATH $root/bin
prepend-path PYTHONPATH $repo
EOF

echo
echo "Setup complete."
echo "To use it later:"
echo "  module purge"
echo "  module use /scratch/user/\$USER/modulefiles"
echo "  module load nlpopt"
echo
echo "Quick test:"
echo '  python -c "import jax, nlpopt; print(jax.devices())"'