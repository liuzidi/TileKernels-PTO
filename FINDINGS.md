# Backward layout A/B, 2026-10-27

Current baseline: PTOAS 105e48b046c9ca30fda8c978e1a00f030df0633d; TileLang 3d70ede9643e3e9e5d182f6ae0d569374f403471; CANN9.2; --enable-vecscope-mem-bar=false; signed16 dynamic row loops. Full baseline/source/build identity: identity.json. Cases 0/10/20: N8001 BF16 H4096/5120/7168, weight, no residual, scale1, no prior dW accumulation.

## Accumulation semantics

weight_grad_initialized means a caller supplied an existing dW output. False writes this call's dW. True adds the existing dW after the cross-core sum. Both modes perform the same token reduction. Existing level0 cases confound accumulation with residual, scale and padding; they do not establish an accumulation-induced gap.

## Why the new planner did not implement the historical prototype

The current output optimization works: the lowered VF uses two PK_B32 BF16 stores instead of VDINTLV. BF16 widening still uses six vintlv(zero,bf16)+bitcast recipes in the arithmetic VF (static sites, not runtime instruction counts).

Source evidence in PTOAS lib/PTO/Transforms/VMI:
- VMILayoutSupportCastTables.inc:72-77 and 142-145 enumerate/prefer same-layout contiguous BF16/F32 relations. Native contiguous-to-deinterleaved2 is also a legal alternative (157); it is not an ISA limitation.
- VMILayoutSupportSolverCosts.inc:52-73 explicitly charges same-layout BF16 widening interleaves; the planner does not simply forget their cost.
- VMILayoutSupportSolverQueries.inc:235-242 requires contiguous reduction input/mask/output. The historical prototype permutes the reduction pairing (floating addition reassociation); that is beyond selecting an equivalent lane layout at each SSA edge.
- VMILayoutComponentGraph.cpp:28-68 connects VMI vector/mask values and structural SSA edges. It does not change the element order of the dW scratch buffer shared by loads/stores over successive tiles, or insert a final canonicalization before GM copy.

Historical whole-chain layout is more than a local cast optimization: keep EVEN/ODD arithmetic and scratch dW over all tiles, merge dX correctly, canonicalize dW once per core. Current experiment patches final VPTO only, keeps current scheduling, signed16, barriers and the two remaining broadcast selects unchanged. This isolates whole-chain layout from no-group broadcast optimization. It is a compiler prototype, not a production source change or generic safe pass.

## Generated-code proof

patch.py generates three variants; inject.py rejects unknown input PTO hashes and records exact fatobj hashes in injections.jsonl. Separate caches for ASC/baseline/layout. JIT factory caches and dispatch caches are both cleared when switching arms.

H5120 arithmetic VF: VINTLV.b16 6->0, VCVTFF.bf162f32 0->12, VSELR.b32 2->2, BF16 narrowing 2->2, one OR merge, one fewer dX store. One extra final VF restores dW with FP32 VINTLV outside per-token work. No instruction-count-to-time inference is made.

object_evidence.json verifies the captured full-flag cc1 replay .text equals the injected fatobj's embedded device .text and occurs in the cached loaded library. Fatobj outer metadata differs across capture paths; comparing device .text avoids that false difference.

## Hardware results (completed)

Both jobs completed successfully: layout task_20261027_154100_2271101; accumulation task_20261027_155311_2363105. Current pins and disabled auto barriers unchanged.

| H | ASC us | baseline us | layout us | baseline overhead | layout overhead | latency reduction |
|---|---:|---:|---:|---:|---:|---:|
| 4096 | 64.75 | 73.61 | 66.86 | 13.68% | 3.25% | 9.18% |
| 5120 | 75.26 | 86.33 | 75.97 | 14.70% | 0.93% | 12.00% |
| 7168 | 103.84 | 116.29 | 104.15 | 11.98% | 0.30% | 10.44% |

Five rotated-order timing rounds, msprof rep20. Each layout variant passes reference tolerance and has zero unequal dX/dW elements against ASC for all three seeds. This is elementwise exact equality as measured, not a bytewise signed-zero/NaN comparison. Baseline dW is exact; baseline dX has 187–345 unequal elements across the tested cases/seeds while passing reference tolerance.

Accumulation single-variable control (same residual=false, scale1, padding0 and BF16 output):

| H | VMI no accumulate us | VMI accumulate us |
|---|---:|---:|
| 4096 | 73.64 | 73.59 |
| 5120 | 86.24 | 86.41 |
| 7168 | 116.25 | 116.32 |

The VMI accumulation toggle changes latency by at most 0.20%; no material performance benefit. The original matrix association with accumulation is confounded. This experiment establishes whole-chain layout as a major contributor to the three current gaps, without attributing all of the gain to individual opcodes. Remaining H4096 overhead is 3.25%; no further attribution yet. Prototype edits final VPTO and is not a production compiler/operator fix.
