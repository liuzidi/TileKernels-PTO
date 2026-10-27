#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
: "${PTOAS_ROOT:?set PTOAS_ROOT to frozen 105e48b04 build}"
: "${TK_BRANCH_ROOT:?set TK_BRANCH_ROOT to pinned TileKernels checkout}"
export PATH="$PTOAS_ROOT/build/tools/ptoas:$PATH"
export PTO_FLAGS='--enable-vecscope-mem-bar=false'
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export TILELANG_CACHE_DIR="$PWD/bootstrap-cache" TMPDIR="$PWD/bootstrap-tmp"
mkdir -p "$TMPDIR" bin
ln -sf ../inject.py bin/ptoas
sha256sum -c SHA256SUMS
python3 patch.py
for i in 0 10 20; do
 ptoas --pto-arch=a5 --pto-backend=vpto --enable-vecscope-mem-bar=false "$i.layout.vpto" -o "$i.layout.fatobj.o"
done
python3 run.py compile
