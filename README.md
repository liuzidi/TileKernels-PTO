# BF16 norm_backward whole-chain layout reproduction

Baseline: PTOAS 105e48b046c9ca30fda8c978e1a00f030df0633d, TileLang 3d70ede9643e3e9e5d182f6ae0d569374f403471, TileKernels-PTO vmi-impl 5617945a8ee7c854720b7a9052617a070d19736f plus two signed16 row-bound casts (PR30). CANN9.2.0, A5 36 SM/72 vector cores, N8001. Auto vecscope barriers disabled. Signed16 is included in the supplied VMI source and in all supplied PTO/VPTO.

## Files
0/10/20 mean H4096/5120/7168. `*.pto` are original VMI input, `*.vpto` are baseline final VPTO, `*.layout.vpto` are patched final VPTO, `*.diff` are full diffs. patch.py regenerates all variants with structural assertions. JSON files are original measurements. No prebuilt binary is required.

## Environment
Use a coherent TileLang/PTOAS build at the exact commits, activate the Python environment with torch_npu and source CANN set_env.sh. Set PTOAS_ROOT, TILELANG_ROOT, TK_BRANCH_ROOT to your builds/checkout. PYTHONPATH must import TileLang from TILELANG_ROOT and ptodsl/ptoas from PTOAS_ROOT/build/python (remove conflicting editable finders if present). LD_LIBRARY_PATH must prioritize TILELANG_ROOT/build/lib and matching LLVM libraries. Confirm module.__file__ and `which ptoas`. The source snapshots in this archive are the authority; support.py asserts ASC and forward helper hashes match the checkout.

Example after environment activation:
```
export PYTHONPATH="$TILELANG_ROOT:$PTOAS_ROOT/build/python:$PTOAS_ROOT/install:$TK_BRANCH_ROOT"
export LD_LIBRARY_PATH="$TILELANG_ROOT/build/lib:/opt/ptoas/llvm/19.1.7/lib:$LD_LIBRARY_PATH"
bash compile.sh
```
Host compilation builds all 9 kernels in isolated arm caches. It injects ONLY the layout arm's fatobj after matching the input PTO SHA256. Both factory and dispatch memory caches are cleared on arm switches. Unknown input hashes fail closed. If a different frontend changes the PTO spelling, investigate instead of bypassing the hash gate.

On device1, using the same environment (queue job must source your activation/env):
```
task-submit --device 1 --max-time 0 --run 'bash /absolute/path/to/env-and-device-wrapper.sh'
```
The wrapper must source your environment then run `bash /absolute/path/to/reproducer/device.sh`. Ensure root queue environment has Git safe.directory as needed. Backend autoload must be enabled during device execution. This runner uses device1 explicitly.

Checks: 3 deterministic seeds; reference tolerance for dX and dW; exact element mismatches against ASC. Five rotated arm orders, msprof warmup0/rep20; times in microseconds. Expected baseline gaps around12–15%, whole-chain gaps around0–3.3%. Timing varies by hardware/software. Results are written to results.json; preserve the original JSON before rerunning. Layout is expected to have zero unequal dX/dW elements for the tested seeds. This is NOT a signed-zero/NaN bytewise assertion.

For compiler-only reproduction (no TileLang/NPU), run patch.py and ptoas on before/after VPTO as in compile.sh. Compare hot VF: six BF16 VINTLV sites disappear, 12 BF16-to-F32 native VCVTs appear, two broadcast selects remain, dX uses OR and one store; final dW canonicalization is outside per-token work. Instruction count alone does not establish timing benefit.

The prototype changes reduction pairing under existing reassoc permission and scratch dW element order until its GM boundary. It is not a generic safe compiler transformation. Handle aliasing, masks/tails, all consumers and memory boundaries before productionizing.
