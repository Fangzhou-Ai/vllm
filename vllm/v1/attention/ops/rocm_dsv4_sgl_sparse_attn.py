# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP8-resident sparse-attn decode kernel for DSV4 on ROCm.

WIP scaffold for porting SGLang's `sparse_mla_fwd_decode_partial_fp8`
(`sglang/srt/layers/attention/dsa/tilelang_kernel.py:1085`) algorithm into
a vLLM-native Triton kernel. The motivation, algorithm, and full design are
documented in `bench_results/sgl_paged_mqa_port/PORT_DESIGN.md`.

Status:
    - Public entry point and dispatcher hook exist.
    - The kernel itself is intentionally NOT IMPLEMENTED.
      `_sparse_attn_decode_fp8_resident_kernel` raises NotImplementedError when
      called so that accidentally setting `VLLM_ROCM_DSV4_SGL_SPARSE_DECODE=1`
      cannot silently ship a broken result.

Continuation notes live in
`bench_results/sgl_paged_mqa_port/CONTINUATION_NOTES.md`.
"""

from __future__ import annotations

import torch


def _sparse_attn_decode_fp8_resident_unimplemented(*args, **kwargs):  # noqa: D401
    """Placeholder for the FP8-resident decode kernel.

    Raises NotImplementedError. See PORT_DESIGN.md section 4 for the algorithm
    and section 5 for the function signature this must conform to. Replace this
    function with the real implementation in a follow-up commit.
    """
    raise NotImplementedError(
        "VLLM_ROCM_DSV4_SGL_SPARSE_DECODE is enabled but the FP8-resident "
        "sparse-attn decode kernel is not yet implemented. See "
        "bench_results/sgl_paged_mqa_port/PORT_DESIGN.md."
    )


def rocm_sparse_attn_decode_fp8_resident(
    q: torch.Tensor,
    main_cache: torch.Tensor,
    main_indices: torch.Tensor,
    main_indptr: torch.Tensor,
    scale: float,
    attn_sink: torch.Tensor | None,
    nope_head_dim: int,
    rope_head_dim: int,
    extra_cache: torch.Tensor | None = None,
    extra_indices: torch.Tensor | None = None,
    extra_indptr: torch.Tensor | None = None,
) -> torch.Tensor:
    """Port-target signature, matched to `_rocm_sparse_attn_decode_ragged_triton`.

    Args:
        q: [sq, num_heads, nope_head_dim + rope_head_dim] bf16. Concatenated
            NoPE + RoPE; consumer of this kernel must guarantee this layout.
        main_cache: [num_blocks, block_size, 576] uint8 packed
            `[fp8_nope || bf16_rope || fp8_scales]`. Same layout as the existing
            DSV4 ROCm SWA k_cache.
        main_indices: ragged int32 [nnz_main]; CSR values.
        main_indptr: int32 [sq+1]; CSR row pointers for `main_indices`.
        scale: 1/sqrt(d) scaling applied before softmax.
        attn_sink: optional per-head fp32 sink logits, shape [num_heads].
        nope_head_dim: must be 448 for DSV4.
        rope_head_dim: must be 64 for DSV4.
        extra_cache: optional second cache (e.g. compressed KV for the topk
            global path) with the same uint8 packed layout.
        extra_indices, extra_indptr: ragged CSR for `extra_cache`.

    Returns:
        [sq, num_heads, nope_head_dim + rope_head_dim] bf16 output, matching the
        existing ROCm decode kernel.
    """
    return _sparse_attn_decode_fp8_resident_unimplemented(
        q,
        main_cache,
        main_indices,
        main_indptr,
        scale,
        attn_sink,
        nope_head_dim,
        rope_head_dim,
        extra_cache,
        extra_indices,
        extra_indptr,
    )
