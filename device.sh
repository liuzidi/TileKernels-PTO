#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
export PATH="$PTOAS_ROOT/build/tools/ptoas:$PATH"
export PTO_FLAGS='--enable-vecscope-mem-bar=false'
export TORCH_DEVICE_BACKEND_AUTOLOAD=1
export TILELANG_CACHE_DIR="$PWD/bootstrap-cache" TMPDIR="$PWD/bootstrap-tmp"
python3 run.py device
