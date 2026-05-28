# Continuation Notes: SGLang Sparse-Attn Decode Port

Branch: `rocm-dsv4-sgl-paged-mqa-port` (forked from `main` @ `165460941`).
Session 1 commit: <will be filled by `git log -1` after the commit lands>.

## State at the end of session 1

* **Design doc written.** `bench_results/sgl_paged_mqa_port/PORT_DESIGN.md` —
  read this **before** touching anything; it contains the full
  SGLang↔vLLM data-structure mapping, the exact algorithm we are porting, and
  the list of out-of-scope items.
* **Env flag added.** `VLLM_ROCM_DSV4_SGL_SPARSE_DECODE` (default off) in
  `vllm/envs.py`. Setting it to `1` flips the dispatcher in
  `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py::rocm_sparse_attn_decode`
  to the new entry point.
* **Stub kernel file created.** `vllm/v1/attention/ops/rocm_dsv4_sgl_sparse_attn.py`
  — public signature locked in (`rocm_sparse_attn_decode_fp8_resident`), the
  implementation raises `NotImplementedError` deliberately so accidentally
  setting the env flag fails loudly instead of producing garbage tokens.
* **Dispatcher wired.** The flag-gated branch in `rocm_sparse_attn_decode`
  also handles the ragged-CSR conversion the SGLang algorithm wants, so the
  kernel sees `main_ragged_indices/main_ragged_indptr` etc. with no extra
  Python work at call time.
* **Lint clean.** `ruff-check` and `ruff-format` pass on the three touched
  files.

Nothing in the production path changes when the flag is off — verified by
import smoke test.

## What needs to happen next

### Step 1 (mandatory): implement the kernel
File: `vllm/v1/attention/ops/rocm_dsv4_sgl_sparse_attn.py`.

Algorithm — copy from SGLang TileLang
(`/shared/amdgpu/home/fai_qle/sglang-ref/python/sglang/srt/layers/attention/dsa/tilelang_kernel.py:1085`)
into a Triton kernel. The key ingredients are documented in
`PORT_DESIGN.md` section 2b; the most important details:

1. The Q NoPE region must be cast to FP8 once at kernel prologue
   (per-query scale = `max(|q|) / fp8_max`).
2. The K NoPE region stays FP8 throughout. Do **not** dequantize K to bf16.
3. The QK GEMM is FP8 × FP8 → FP32; multiply the F32 accumulator by the
   per-(BI block) K-scale fragment after the GEMM.
4. Split the 448-element NoPE dim into 4 × 112 sub-tiles. Each sub-tile is its
   own FP8 GEMM that accumulates into a private `acc_tile`, then is added
   into `acc_s`. This breaks the MFMA accumulator dependency chain (this is
   the single biggest perf trick in the TileLang kernel).
5. Keep the 64-element RoPE region in bf16 with its own small GEMM at the end
   of each BI iteration. (DSV4's KV layout stores RoPE as bf16; we are
   intentionally **not** changing the cache layout in this port.)
6. After softmax, scale `s` up by `fp8_max_val`, cast to FP8, do the SV GEMM
   in FP8 × FP8 → FP32, then fold the inverse scale into the running
   accumulator. (SGLang `s_inv_scale_const` / `s_scale_const`.)
7. If `extra_cache` is given (compress_ratio>1 path), repeat the same loop
   on the extra cache after the main loop and merge LSEs.

The existing `_sparse_attn_decode_ragged_kernel`
(`vllm/v1/attention/ops/rocm_aiter_mla_sparse.py:1143`) is a structurally
correct reference — copy its IO and ragged-CSR handling verbatim and only
change the GEMM/dequant logic.

### Step 2: unit test
File: `tests/kernels/attention/test_dsv4_sgl_decode.py` (does not yet exist).

Minimum cases:
* `sq=1, num_heads=16, swa_only=True`, random topk small. Compare against
  the existing `_rocm_sparse_attn_decode_ragged_triton`, atol=5e-2 bf16.
* `sq=4, num_heads=16, swa_only=False, extra_cache=` with random
  topk+swa indices. Same tolerance.
* `sq=128, num_heads=16, sequence_length=1024, topk=2048` (representative of
  conc=128 real workload). Use timed comparison just for visibility, not as
  an assertion.

Skip the test on non-ROCm via `pytest.importorskip` / `platform` check.

### Step 3: GSM8K eval
Run with `VLLM_ROCM_DSV4_SGL_SPARSE_DECODE=1` and verify GSM8K exact_match
matches the existing baseline (`bench_results/gsm8k_eval_20260523_025642`
recorded ~0.959). Anything below 0.85 means a kernel bug.

### Step 4: bench
Re-use `bench_results/dsv4_conc128_repro/run_bench.sh`:

```bash
cd /shared/amdgpu/home/fai_qle/vllm
TAG=sgl_resident DISABLE_DSV4_CAP=1 PORT=8001 \
  VLLM_ROCM_DSV4_SGL_SPARSE_DECODE=1 \
  bash bench_results/dsv4_conc128_repro/run_bench.sh
```

The baseline rows to beat are in
`bench_results/dsv4_conc128_repro/{regression_no_cap,multistream_v2}/bench.log`:
* output tok/s = 468–475
* TPOT = 263–267 ms

The acceptance threshold from the PORT_DESIGN doc is **TPOT < 200 ms** (i.e.
> 25% improvement); if it does not clear that, write a NEGATIVE_RESULT.md in
the same dir and keep the env flag off-by-default.

### Step 5 (only after steps 1–4 pass): PR
Open against `Fangzhou-Ai/vllm` `main`. PR template:

* Title: `[ROCm][DSV4] FP8-resident sparse-attn decode kernel (SGLang port)`
* Body should include:
  * Link to InferenceX baseline:
    https://github.com/SemiAnalysisAI/InferenceX/actions/runs/26207210484/attempts/1
  * Local bench numbers (before/after) at conc=128.
  * GSM8K eval delta.
  * A "Why this is not duplicate" paragraph: link to PR #5 (the max_num_seqs
    cap) and explain why this is independent (it improves the per-step
    decode kernel; PR #5 only changes scheduler admission).
  * AI assistance statement per `AGENTS.md`.

**Do not open the PR until step 4 numbers actually improve.** The dir
`bench_results/dsv4_conc128_repro/{indexer_fix,cg_cap64,multistream_v2,batched_tokens_64}/`
is full of well-intentioned ROCm changes that did not move TPOT; do not become
that.

## Gotchas / non-obvious things found in session 1

1. **The Preshuffle gate is closed on DSV4 today.** The vLLM lightning-indexer
   already calls `aiter.ops.triton.pa_mqa_logits.deepgemm_fp8_paged_mqa_logits`
   on gfx950 (good), but `Preshuffle=block_size == 64` (line 436 of
   `rocm_aiter_mla_sparse.py`), and `DeepseekV4IndexerBackend` reports
   `kernel_block_sizes=[256]` (`vllm/v1/attention/backends/mla/indexer.py:158`).
   So Preshuffle is currently **off** for DSV4 — the indexer is not on
   AITER's gluon fast path. This is a separate optimization track from the
   decode kernel port; do **not** conflate them in one PR.
2. **TileLang is not a vLLM build dep, and should not be added.** It pulls in
   LLVM + a JIT runtime. The port is Triton-only.
3. **SGLang itself isn't importable in the local venv** (`orjson` missing
   when you `import sglang`). For algorithm reference, read the source files
   directly — they don't need to import.
4. **AITER's `deepgemm_fp8_paged_mqa_logits` is already wrapped** in
   `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py:375
   (rocm_fp8_paged_mqa_logits)`. No need to re-wire the indexer path.
5. **DSV4 KV layout packs RoPE as bf16 inline** with FP8 NoPE and FP8 scales
   in 576-byte rows (see `_sparse_attn_decode_ragged_kernel` lines
   1216-1247). SGLang TileLang assumes FP8 RoPE in a separate region. We
   keep vLLM's layout; do not change the cache spec without a separate
   review.
6. **Decode batch and indexer warning correlation.** Server logs from prior
   runs show AITER falls back to `torch solution:0` for BF16 GEMM shapes
   `M={136..512}, N=64, K=7168` (the indexer 7168→64 head proj) and not for
   `M=128`. If you change scheduler admission such that the decode batch
   stops landing exactly on a captured cudagraph size of 128, the indexer
   may fall onto the torch.matmul path. Keep an eye on it during step 4.

## Hardware notes

* MI355X (`gfx950`), 8 GPUs available on this node.
* When iterating, **kill any lingering vLLM servers first** —
  `pkill -f "vllm serve deepseek-ai/DeepSeek-V4-Pro"` — session 1 saw two
  stale servers still resident on ports 8001/8003.
* Each bench cycle (server startup + warmup + bench) is ~15 min wall-clock.
  Plan for at least 5–10 cycles across step 3+step 4.

## Files touched in session 1

| File | Status |
|---|---|
| `bench_results/sgl_paged_mqa_port/PORT_DESIGN.md` | created |
| `bench_results/sgl_paged_mqa_port/CONTINUATION_NOTES.md` | created (this file) |
| `vllm/envs.py` | added `VLLM_ROCM_DSV4_SGL_SPARSE_DECODE` |
| `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py` | added flag-gated dispatch branch in `rocm_sparse_attn_decode` |
| `vllm/v1/attention/ops/rocm_dsv4_sgl_sparse_attn.py` | created stub with `NotImplementedError` body |
