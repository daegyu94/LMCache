#!/bin/bash
# Create a venv with vLLM 0.33.0 and build LMCache's native extensions in REPO.
# usage: setup_venv.sh <venv_dir>      env: REPO [git toplevel], VLLM_VERSION [0.33.0]
set -eu
VENV=$1
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=${REPO:-$(git -C "$HERE" rev-parse --show-toplevel)}
VLLM_VERSION=${VLLM_VERSION:-0.33.0}
uv venv --python 3.12 "$VENV"
uv pip install --python "$VENV/bin/python" "vllm==$VLLM_VERSION"
TORCH=$("$VENV/bin/python" -c 'import torch; print(torch.__version__.split("+")[0])')
# vLLM pins torch; keep it while installing LMCache's runtime/test deps.
echo "torch==$TORCH" > "$VENV/torch-constraint.txt"
(cd "$REPO" && uv pip install --python "$VENV/bin/python" -c "$VENV/torch-constraint.txt" \
  -r requirements/common.txt -r requirements/cuda13_core.txt \
  -r requirements/build.txt -r requirements/test.txt)
# In-place build against the venv's torch (overwrites lmcache/*.so in REPO).
(cd "$REPO" && SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE=0.0.0 MAX_JOBS=${MAX_JOBS:-32} \
  "$VENV/bin/python" setup.py build_ext --inplace)
(cd /tmp && PYTHONPATH=$REPO "$VENV/bin/python" -c \
  'import torch, vllm; from lmcache.integration.vllm import lmcache_mp_connector; print("ok", vllm.__version__, torch.__version__)')
