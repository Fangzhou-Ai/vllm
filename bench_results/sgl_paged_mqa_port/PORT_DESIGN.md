# DSV4 ROCm Sparse-Attn Decode: SGLang -> vLLM Port Design

Branch: `rocm-dsv4-sgl-paged-mqa-port` (from `main` @ 165460941)
Scope: multi-session engineering port. No PR until perf gain is validated.

## 1. Problem Statement

On MI355X (gfx950), TP=8, DeepSeek-V4-Pro with `compilation-config={mode:3, cudagraph_mode:FULL_AND_PIECEWISE}`,
1k/1k random, conc=128, output throughput is **468 tok/s, TPOT 266 ms** (see
`bench_results/dsv4_conc128_repro/regression_no_cap/bench.log` and the InferenceX CI artifact
at https://github.com/SemiAnalysisAI/InferenceX/actions/runs/26207210484/attempts/1).

Capping `max_num_seqs=64` (PR Fangzhou-Ai/vllm#5) moves the operating point to
**866 tok/s, TPOT 71 ms** by halving the decode batch. That is *not* a kernel speedup;
it just picks a different point on the throughput/concurrency curve. The CI benchmark
posts requests with concurrency=128, so the active decode batch at steady state is ~128;
the cap forces queueing instead of fixing per-step latency.

Prior optimizations on `dsv4-rocm-multi-stream-decode` (multistream CSA overlap, AITER
Triton GEMM routing for q-proj / norm / down, sparse-attn in-place writes, indexer
topk fast paths, WO_A dequant caching, pre-allocated `paged_mqa_logits` buffer, sparse-attn
on local heads only) move TPOT by <2% at batch=128. Confirmed by
`bench_results/dsv4_conc128_repro/{multistream,multistream_v2,indexer_fix,cg_cap64,batched_tokens_64}/bench.log`.

So the remaining bottleneck is structural in the per-step decode work, and the
candidate algorithmic win is to replace vLLM's existing decode-side Triton kernel
`_sparse_attn_decode_ragged_kernel` (in `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py`)
with the FP8-resident two-cache scheme SGLang uses in
`sglang/srt/layers/attention/dsa/tilelang_kernel.py::sparse_mla_fwd_decode_partial_fp8`.

## 2. What SGLang Actually Does on HIP for DSV4 Decode

SGLang has **two** independent sparse-attn decode implementations on HIP:

### 2a. `_forward_aiter` (`dsa_backend.py:1928`)
Used when `self.dsa_decode_impl == "aiter"`. Reuses upstream
`aiter.mla.mla_decode_fwd` with the topk indices flattened into a page table:
```python
page_table_1 = topk_indices   # if SGLANG_DSA_FUSE_TOPK
get_valid_kv_indices(page_table_1, kv_indptr, kv_indices, bs)  # build CSR
mla_decode_fwd(
    q_kernel,                                 # [B, H, D_qk]
    kv_cache.view(-1, 1, 1, layer.head_dim),  # page_size=1 view
    o_kernel,                                 # [B, H, D_v]
    cu_seqlens_q, kv_indptr, kv_indices,
    cu_seqlens_q, max_seq_len_q,
    sm_scale=layer.scaling,
    logit_cap=layer.logit_cap,
)
```
This works because in SGLang's DSV3.2-style DSA there is a **single** KV cache
(no compress, no SWA split). The topk indices are slot indices into that flat cache,
and `mla_decode_fwd` accepts them as a CSR-flattened page_table with page_size=1.

### 2b. `_forward_tilelang` (`dsa_backend.py:1910`)
Used when `self.dsa_decode_impl == "tilelang"`. Calls
`tilelang_kernel.tilelang_sparse_fwd` which selects between
`sparse_mla_fwd_decode_partial[_fp8]` + `sparse_mla_fwd_decode_combine`.

Key properties of the FP8 partial kernel
(`tilelang_kernel.py:1085-1330`):

* **FP8-resident KV.** `kv_fp8: T.Tensor[(b, num_pages, kv_group, d_v + d_tail), fp8]`.
  No bf16 dequant in the GEMM; the dequant is folded into the per-(BI=64) scale fragment.
* **Split-K accumulator.** D=512 is split into 4 × group_size=128 tiles
  (`q_tile0..3`, `kv_tile0..3`). The four sub-GEMMs each clear/accumulate into a
  separate `acc_tile` then add into `acc_s`, breaking the MFMA accumulation
  dependency chain. d_tail=64 (RoPE) is appended as a 5th GEMM with
  `T.GemmWarpPolicy.FullCol`.
* **Softmax rescaled before the second GEMM.** After the online softmax,
  `s_fp8_shared[h, bi] = clamp(acc_s[h, bi] * fp8_max_val, ±fp8_max_val)` so the
  S × V GEMM is FP8 × FP8 → FP32 again. The inverse scale `1/fp8_max_val` is folded
  into the accumulator update. Safe because softmax outputs are in `[0, 1]`.
* **16 heads per block.** 128 heads ÷ 8 (TP=8) = 16 heads/rank/layer → exactly
  one head-block per query in this deployment, removing the outer head loop.
* **block_I=64 keys per outer iter.** On gfx95 (MI350/MI355) the kernel uses
  `block_per_cu=2, threads=256, cu=256`. `inner_iter` is auto-tuned to the batch.
* **Two-stage partial→combine.** The partial kernel writes
  `partial_o: [b, sq, n_groups, num_heads, d_v]` and `partial_lse: [b, sq, n_groups, num_heads]`;
  the combine kernel reduces across `n_groups` with online LSE merge.

### 2c. Why `tilelang` is faster than current vLLM Triton on MI350x
Conceptually:
1. FP8×FP8 GEMM at half the L2/HBM read bytes vs bf16(K).
2. 4-way split-K removes the MFMA write→read accumulator stall on CDNA3/CDNA4.
3. No per-iter dequant arithmetic on the K side; scales are loaded once per
   BI block as a small fragment.
4. Partial+combine matches the "split-KV decode" pattern that fills the
   wavefront frontier evenly across CUs even at small batch.

The current vLLM kernel `_sparse_attn_decode_ragged_kernel` does:
1. Per-iter `x_fp8.to(tl.bfloat16) * scales.to(tl.bfloat16)` (full dequant to bf16).
2. Single bf16 GEMM for nope, separate bf16 GEMM for rope.
3. No split-K; no partial+combine.
4. NaN-replace via `tl.where(k_nope == k_nope, k_nope, zero_nope)` (extra masking).

## 3. vLLM Side Mapping

### 3a. Where the kernel is called
`vllm/v1/attention/ops/rocm_aiter_mla_sparse.py::rocm_sparse_attn_decode` (l. 1684)
→ `_rocm_sparse_attn_decode_triton` (l. 1581) → `_rocm_sparse_attn_decode_ragged_triton` (l. 1471)
→ `_sparse_attn_decode_ragged_kernel` (l. 1143).

Inputs:
* `q: [sq, num_heads, head_dim]`, `head_dim = 448 + 64 = 512` (nope+rope), bf16.
* `main_cache (swa_k_cache): [num_blocks, swa_block_size, 576+]` packed
  uint8 with layout `[fp8_data: nope_dim + bf16: rope_dim, ..., fp8_scales]`.
  See the kernel at lines 1216–1247: each token is 576 bytes
  (= 448 fp8 + 64 bf16 + 8 scale bytes packed per row).
* `main_indices`: ragged 1D int32, with `main_indptr` of shape `[sq+1]` (CSR).
* Optional `extra_cache`: full KV for the compressed (topk) path when
  `compress_ratio > 1`. Same 576-byte layout.
* `attn_sink`: per-head fp32, optional.

Outputs: `[sq, num_heads, head_dim]` bf16.

### 3b. Layout differences vs SGLang TileLang

| Aspect | vLLM (current) | SGLang TileLang |
|---|---|---|
| KV dtype in GEMM | bf16 (after per-iter dequant) | fp8 (no dequant) |
| Block size | swa_block_size (e.g. 64) | BI=64 (= block_per_cu access tile) |
| K layout | packed `[fp8_nope || bf16_rope || fp8_scales]` 576-byte rows | `[fp8_nope || fp8_rope]` 512-byte rows + separate scale region |
| Two-cache structure | main (SWA) + optional extra (topk) | single `kv_fp8` |
| Scale layout | per-group fp8 e8m0 (encoded), per row, expanded inline | per-block fp32, copied once per BI |
| Heads per block | BLOCK_H=16 (Triton) | h_per_block=16 (TileLang) |
| Tail (RoPE) | bf16, packed in same row, separate GEMM | fp8, separate region, separate GEMM |
| Split-K | none | D=512 → 4 × 128 |
| Partial + Combine | none (single pass) | 2-stage |

### 3c. Why we cannot just "drop in" SGLang's path
1. The cache layout is different. DSV4 stores rope as bf16 in the same row as
   the fp8 nope, plus 4-byte fp32 scale per (block, pos). SGLang's TileLang
   kernel expects rope to be FP8 too, in a separate region, with fp32 scales
   per block. Either we change the cache layout or we adapt the kernel.
2. The two-cache structure (SWA + topk extra) is unique to DSV4 + compress_ratio=4.
   SGLang's `_forward_aiter` collapses everything into the topk page table,
   which works only with compress_ratio=1.
3. vLLM's KV cache spec for the DSV4 indexer is locked to
   `kernel_block_sizes=[256]` in
   `vllm/v1/attention/backends/mla/indexer.py::DeepseekV4IndexerBackend`,
   versus SGLang's preshuffle 64. (This is the **indexer** cache, not the
   main MLA cache, but it is part of the same data flow.)

## 4. Proposed Port Strategy

We do **not** port TileLang as a build-time dep (it would pull LLVM + a JIT runtime
and is not on the vLLM dependency surface). Instead, we re-implement the
TileLang algorithm in Triton, reusing the kernel infrastructure already in
`vllm/v1/attention/ops/rocm_aiter_mla_sparse.py`.

### 4a. Phase-1 target (this branch, end-state)

A new Triton kernel `_sparse_attn_decode_fp8_resident_kernel` that:

1. Treats the K NoPE region as FP8 throughout the QK GEMM (no bf16 dequant).
   The scale fragment is loaded once per BI-block of K rows and applied as a
   per-row multiplier on the FP32 accumulator after the FP8 GEMM, not before it.
   This matches the TileLang `s_scale_const = 1.0 / fp8_max_val` trick by folding
   the fp8 scale into the running softmax statistics instead of dequantizing K.
2. Splits the d_nope=448 GEMM into 4 × 112-element tiles (or 7 × 64; precise tile
   factor is a tuning knob). This is the analogue of TileLang's 4×128 split.
3. Keeps the bf16 RoPE region GEMM as a separate, single small GEMM (since the
   rope dim is only 64).
4. Optionally adds a partial+combine variant for very long topk windows; the
   single-pass variant covers the common case (topk=2048, single n_group).
5. Lives behind `VLLM_ROCM_DSV4_SGL_BACKEND` env flag; off by default; old
   `_sparse_attn_decode_ragged_kernel` remains the fallback.

### 4b. Phase-2 target (later, optional)

Migrate the SWA branch (the `main_cache` portion) of the same kernel to the
SGLang `_forward_aiter` strategy: when `compress_ratio == 1` for the SWA portion,
call `rocm_aiter_ops.mla_decode_fwd` with `page_size=1` over the topk indices.
This requires re-laying the SWA cache to `kv_c_and_k_pe_cache` (the layout
`mla_decode_fwd` expects), so it is a separate cache-spec patch.

### 4c. Out of scope for this port

* The lightning-indexer kernel (`rocm_fp8_paged_mqa_logits`) is *already* on the
  AITER `deepgemm_fp8_paged_mqa_logits` path. The current Preshuffle gate is
  `block_size == 64` (line 436 of `rocm_aiter_mla_sparse.py`); the DSV4 indexer
  block_size is 256 (see `DeepseekV4IndexerBackend.get_supported_kernel_block_sizes`),
  so Preshuffle is currently *off* on DSV4. Whether to push the indexer cache
  block_size down to 64 (matching SGLang) is a separate cache-spec change with
  its own correctness implications, tracked as a follow-up.

## 5. Concrete File-level Plan

### 5a. Env flag (`vllm/envs.py`)
```python
VLLM_ROCM_DSV4_SGL_BACKEND: bool = False
```
Default off. When set, `rocm_sparse_attn_decode` selects the new kernel.

### 5b. New kernel file
`vllm/v1/attention/ops/rocm_dsv4_sgl_sparse_attn.py`

Functions to author (signatures match the existing
`_rocm_sparse_attn_decode_ragged_triton` so the dispatch site is a one-liner):

```python
def _rocm_sparse_attn_decode_fp8_resident(
    q: torch.Tensor,                # [sq, H, D=512] bf16
    main_cache: torch.Tensor,       # [num_blocks, block_size, 576] uint8
    main_indices: torch.Tensor,     # ragged int32 [nnz_main]
    main_indptr: torch.Tensor,      # int32 [sq+1]
    scale: float,
    attn_sink: torch.Tensor | None, # [H] float32 or None
    nope_head_dim: int,             # 448
    rope_head_dim: int,             # 64
    extra_cache: torch.Tensor | None,
    extra_indices: torch.Tensor | None,  # ragged int32
    extra_indptr: torch.Tensor | None,
) -> torch.Tensor:   # [sq, H, D] bf16
    ...
```

Internal Triton kernel `_sparse_attn_decode_fp8_resident_kernel`:

* Tile sizes: BLOCK_H=16, BLOCK_K=64, NOPE_TILE=112 (4× over 448) or 128 (only on
  d_nope alignment that fits — DSV4 is 448 not 512, so 4×112 or 7×64 — try
  4×112 first).
* Accumulator dependency split: `acc_s` accumulates the 4 sub-GEMMs separately
  via local `acc_tile` to mirror the TileLang pattern.
* `q_fp8` cast: we cast `q[..., :nope_dim]` to fp8 once per query at kernel
  prologue (in shared mem). The dynamic scale per query is `q_scale = max(|q|) / 448`
  but for the QK score we only care up to a scalar; this is identical to what
  `act_quant` does in SGLang's `tilelang_kernel.py:142`.

### 5c. Dispatch wiring
`rocm_sparse_attn_decode` in `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py`
gets a guard:
```python
import vllm.envs as envs
if envs.VLLM_ROCM_DSV4_SGL_BACKEND and head_dim == 512:
    out = _rocm_sparse_attn_decode_fp8_resident(...)
else:
    out = _rocm_sparse_attn_decode_triton(...)
```

### 5d. Unit test
`tests/kernels/attention/test_dsv4_sgl_decode.py`

* Build a small synthetic [sq=4, H=16, D=512] case with random topk+swa indices
  and known-good reference (slow torch) values.
* Assert `_sparse_attn_decode_fp8_resident` and `_sparse_attn_decode_ragged_triton`
  produce outputs within tolerance (bf16: 5e-2 atol, 1e-2 rtol typical for
  paged MQA-style attention).
* Skip on non-ROCm.

### 5e. Validation harness (later)
Re-use `bench_results/dsv4_conc128_repro/run_bench.sh` with
`VLLM_ROCM_DSV4_SGL_BACKEND=1` set in the server env. Compare TPOT at
conc=128 vs baseline.

## 6. Risks / Known Unknowns

1. **K layout mismatch.** DSV4 packs rope as bf16 inline with fp8 nope. We must
   either re-quantize rope to fp8 on the read side, or keep rope as a bf16
   tail GEMM. The latter is simpler and matches the current Triton kernel's
   shape; that is what 4a step (3) above assumes.
2. **FP8 GEMM accumulator handling on Triton + ROCm.** Triton 3.6 on gfx950
   supports FP8 MFMA, but tile-size + num_warps combinations are not always
   tuned. Expect to spend tuning iterations on `BLOCK_K, NOPE_TILE, num_warps`.
3. **CUDA graph capture compatibility.** The new kernel must be CG-safe;
   no dynamic Python branches inside the kernel selection. The existing
   `eager_break_during_capture` decorator on `rocm_aiter_sparse_attn_indexer`
   already handles the indexer side; decode side is plain Triton.
4. **DSV4 MTP path.** DSV4 nextn (MTP) doubles `next_n`. The current
   `_rocm_sparse_attn_decode_ragged_triton` accepts arbitrary `sq`; the new
   kernel must too.

## 7. Validation Plan

* Phase-1 commit: kernel skeleton + env flag + unit test passing on small shapes.
  No production wiring at first; flag defaults off.
* Phase-2 commit: dispatch wired; run GSM8K eval (must match the
  `bench_results/gsm8k_eval_20260523_025642` exact_match=0.959 baseline)
  with flag on.
* Phase-3 commit: full conc=128 1k/1k bench at MI355X TP=8. Accept if TPOT at
  conc=128 drops below 200 ms (i.e. > 25% over baseline). Reject if it does
  not, and document the negative result.

## 8. References

* SGLang TileLang FP8 kernel:
  `/shared/amdgpu/home/fai_qle/sglang-ref/python/sglang/srt/layers/attention/dsa/tilelang_kernel.py:1085`
* SGLang DSA decode dispatcher:
  `/shared/amdgpu/home/fai_qle/sglang-ref/python/sglang/srt/layers/attention/dsa_backend.py:1539`
* SGLang aiter decode wrapper:
  `/shared/amdgpu/home/fai_qle/sglang-ref/python/sglang/srt/layers/attention/dsa_backend.py:1928`
* vLLM current sparse-attn decode:
  `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py:1684`
* vLLM existing Triton decode kernel:
  `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py:1143`
* vLLM DSV4 indexer cache spec:
  `vllm/v1/attention/backends/mla/indexer.py:158`
* InferenceX baseline CI run:
  https://github.com/SemiAnalysisAI/InferenceX/actions/runs/26207210484/attempts/1
