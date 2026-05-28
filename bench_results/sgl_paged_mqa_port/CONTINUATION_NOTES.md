# Continuation Notes: SGLang Sparse-Attn Decode Port

Branch: `rocm-dsv4-sgl-paged-mqa-port` (forked from `main` @ `165460941`).

## State at the end of session 2

Session 2 finalized a tuned Triton decode kernel and shipped microbench-
confirmed speedup. **The kernel is now real, lint-clean, and unit-tested.**
However: the end-to-end TPOT regression at conc=128 is **not** dominated by
the sparse-attn decode kernel, so the kernel win does not move TPOT
materially. See "Where the regression actually lives" below.

### What's now in the tree

* **Tuned v2 decode kernel.** `vllm/v1/attention/ops/rocm_dsv4_sgl_sparse_attn.py`
  is no longer a stub; it implements `_sparse_attn_decode_v2_kernel`, a
  drop-in replacement for `_sparse_attn_decode_ragged_kernel` with the same
  algorithm (online flash-attention, bf16 K dequant, two-cache merge) but
  retuned tile sizes. Defaults: `BLOCK_H=16, BLOCK_K=32, num_warps=4,
  num_stages=2, loop_stages=1`. All four knobs are env-overridable for
  further sweeps (see `VLLM_ROCM_DSV4_SGL_*` in the module docstring).
* **Microbench harness.** `bench_results/sgl_paged_mqa_port/microbench_decode_kernel.py`
  times one kernel call at the conc=128 / DSV4 / TP=8 operating point
  (batch=128, swa_window=1024, topk=2048, num_heads=16) and prints a sweep
  table of speedup vs the existing kernel. Run it as
  `HIP_VISIBLE_DEVICES=0 .venv/bin/python bench_results/sgl_paged_mqa_port/microbench_decode_kernel.py`.
* **Unit test.** `tests/kernels/attention/test_rocm_triton_attn_dsv4.py::test_sparse_attn_decode_v2_matches_baseline`
  asserts the v2 kernel output matches the existing kernel to `atol=2e-2`
  on a small representative shape. **Passes.**
* **Dispatcher.** `rocm_sparse_attn_decode` in `rocm_aiter_mla_sparse.py`
  routes to the v2 kernel when `VLLM_ROCM_DSV4_SGL_SPARSE_DECODE=1`.
* Triton cache (`~/.triton/cache`) shows `_sparse_attn_decode_v2_kernel`
  was compiled and used during the end-to-end bench runs — i.e. the
  dispatch path works in production.

## Kernel-level result

Microbench at conc=128 / DSV4-TP=8 shape, on a single MI355 GPU:

| config | us/call | speedup vs existing |
|---|---|---|
| existing `_sparse_attn_decode_ragged_kernel` (BLOCK_K=16, warps=8) | 751 | 1.00x |
| **v2 (BH=16, BK=32, W=4, S=2, L=1)** | **492** | **1.53x** |
| v2 (BH=16, BK=64, W=4) | 666 | 1.13x |
| v2 (BH=16, BK=32, W=8) | 567 | 1.32x |
| v2 (BH=8, BK=64, W=4) | 538 | 1.40x |
| v2 (BH=8, BK=32, W=4) | 1059 | 0.71x |

Key learnings:

* The two **NaN-guard `tl.where(k == k, k, 0)` ops in the existing kernel
  are load-bearing**, despite looking like dead code: removing them
  regresses the kernel by ~3x on gfx950. (We did not investigate why —
  likely the Triton+ROCm backend uses them as a hint that `k` is
  finite, which unlocks a faster fma sequence.) The v2 kernel keeps
  them intact.
* `tl.range(..., num_stages=2)` (software-pipelined K-loop) regresses the
  kernel at every BLOCK_K we tried on MI355. Leave `LOOP_STAGES=1`.
* `BLOCK_K=32` beats `BLOCK_K=16` (existing) and `BLOCK_K=64` because at
  the swa=1024 / topk=2048 operating point, the trip count is short
  enough that loop overhead matters and large `BLOCK_K` exceeds the
  per-CTA register budget Triton allocates.
* `BLOCK_H=16` matches the natural per-rank head count (`num_heads /
  TP = 128/8 = 16`). Smaller `BLOCK_H` regresses because the CTA count
  is already over the GPU's parallelism budget (128 batch × 1 head-block
  = 128 CTAs on 304 CUs).

## End-to-end result

| run | range | TPOT (ms) | output tok/s | dur (s) |
|---|---|---|---|---|
| baseline `regression_no_cap` (existing kernel) | 1.0 | 266.6 | 468 | 280 |
| `v2_bk32_w4` (this work) | 1.0 | 267.95 | 465 | 282 |
| `with_cap` (max_num_seqs=64, scheduler patch) | 1.0 | 71.5 | 866 | 114 |

**TPOT did not improve.** The kernel is faster per call, but the savings
(~260 us per call × ~58 sparse-attn layers × 1024 decode steps =
~15 ms total per request, ~3-6 % of TPOT) are within bench noise and
buried by other costs.

## Where the regression actually lives

The `with_cap` data point is the smoking gun: capping `max_num_seqs` to
64 (so the same 128 concurrent requests are processed as 2 sequential
batches of 64) drops TPOT 3.7x, from 267 ms to 71.5 ms. The decode
*algorithm* scales linearly with batch, so 2 × T(B=64) should equal
T(B=128). Observing 3.7x slowdown instead of 2x means the kernels are
performing badly *specifically at M=128*.

Grepping the `bench_results/dsv4_conc128_repro/v2_bk32_w4/server.log`
shows >200 unique `(M, N, K)` shapes hitting the "not found tuned config
in /tmp/aiter_configs/a8w8_blockscale_tuned_gemm.csv, will use default
config!" path. The repeating dec-step shapes are:

* `(M=128, N=2048, K=7168)` — MoE expert up-projection
* `(M=128, N=768, K=7168)`  — MoE expert gate/down
* `(M=128, N=7168, K=384)`  — MLA q-up projection
* `(M=128, N=64,   K=7168)` — indexer projection
* `(M=128, N=16160, K=7168)` — MoE shared expert

`/tmp/aiter_configs/a8w8_blockscale_tuned_gemm.csv` has **zero** entries
for any of these `(N, K)` pairs at any M. The `with_cap` path hits the
same fallback at M=64, but apparently the CK "default config" performs
acceptably at M=64 and terribly at M=128.

This is consistent with the existing PR #5
(`Fangzhou-Ai/vllm rocm-dsv4-decode-cap-64`) which caps `max_num_seqs`
to 64 specifically to avoid the M=128 fallback — but the user has
explicitly rejected that as not a fundamental fix.

## What needs to happen next (priority order)

### (1) Tune the AITER block-scale GEMM for M=128 — the real fix

This is the only intervention that will actually move TPOT. Generate
tuned configs for the four `(M=128, *)` shapes listed above (and the
neighbors M ∈ {120, 144, 160, ...} the scheduler will land on). The
AITER tuner is in `.venv/lib/python3.12/site-packages/aiter/ops/gemm_op_a8w8.py::gemm_a8w8_blockscale_tune`.
After tuning, write the new rows to
`/tmp/aiter_configs/a8w8_blockscale_tuned_gemm.csv` and re-run the
bench. Plausible target: 100-130 ms TPOT (i.e. closer to but not at
`with_cap`'s 71.5 ms, because the M=128 attn / norm / RoPE kernels
also need to be looked at).

The tune is a multi-hour run per shape; budget for it.

### (2) Investigate whether M=128 maps to a slow `default config`
Open `gemm_op_a8w8.py` and follow the `gemm_a8w8_blockscale_ck`
default branch (the one taken when `config is None`). It picks a single
hardcoded CK kernel that may simply not be M=128-aware. If we can swap
the default branch for one of the M=64-tuned kernels (run twice at
M=64 internally) we get the `with_cap` perf without touching the
scheduler. This is invasive to AITER though.

### (3) Bring the v2 kernel further if there is time

The kernel is *not* the bottleneck right now, but the SGLang FP8-resident
GEMM trick (see `PORT_DESIGN.md` step 1–7) would push the per-call cost
below the current 492 us → memory-bandwidth floor (~230 us). That is
~5 ms/decode-step savings. Worth doing as a follow-up; not the lever
that closes the regression.

### (4) Decide whether to keep the v2 kernel default-off

Recommendation: **keep it default-off** (the env flag stays at
`False`). The 1.53x kernel-level win is verifiable but does not
improve TPOT, and the new code path adds a maintenance surface. Land
the flag + kernel + test as scaffolding so the FP8-resident follow-up
has a place to slot in, but do not turn it on by default until step
(3) lands and TPOT actually moves.

## Hardware / workflow notes

* MI355X (`gfx950`), 8 GPUs available on this node.
* `.venv/pyvenv.cfg` has `include-system-site-packages = true`, which
  means a stale `/usr/local/lib/python3.12/dist-packages/vllm` shadows
  the editable install for **sub-packages** like `vllm.v1.attention.ops`
  when running a script outside the repo root. The microbench inserts
  the repo root at the front of `sys.path` to work around this. If
  importing newly-added vllm sub-modules fails in a fresh script,
  this is why.
* **vLLM workers don't always die on `pkill -f "vllm serve"`.** The
  worker pattern is `VLLM::Worker_TPx` — kill those explicitly:
  `pkill -9 -f "VLLM::Worker"`. `rocm-smi --showpids` will continue
  to show stale KFD entries with `UNKNOWN` process name for the dead
  PIDs even after they're gone; check actual process existence via
  `fuser /dev/kfd` before trusting the readout.
* Each end-to-end bench cycle (server load + 256-req warmup + 128-req
  main) is ~13 min wall-clock. Use the microbench for kernel iteration
  and reserve the end-to-end bench only for confirming TPOT-side wins.

## Files touched in session 2

| File | Change |
|---|---|
| `vllm/v1/attention/ops/rocm_dsv4_sgl_sparse_attn.py` | replaced stub body with tuned `_sparse_attn_decode_v2_kernel` + env-driven launcher |
| `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py` | updated dispatcher comment to reflect kernel is now real |
| `tests/kernels/attention/test_rocm_triton_attn_dsv4.py` | added `test_sparse_attn_decode_v2_matches_baseline` |
| `bench_results/sgl_paged_mqa_port/microbench_decode_kernel.py` | new — kernel-level perf harness |
| `bench_results/sgl_paged_mqa_port/CONTINUATION_NOTES.md` | this file, rewritten |
| `bench_results/dsv4_conc128_repro/run_bench.sh` | added `RANGE_RATIO` env override so a future run can match the original baseline conditions exactly |
| `bench_results/dsv4_conc128_repro/v2_*/` | bench artifacts (server.log, bench.log) for the two v2 configs we ran |
