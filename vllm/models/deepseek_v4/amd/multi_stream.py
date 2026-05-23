# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm-only multi-stream helpers for DeepSeek-V4 CSA attention overlap.

Aligned with the DeepSeek-V4 CSA design:

* One auxiliary stream alongside the default stream.
* The auxiliary stream runs the lightning indexer; the default stream runs
  ``wq_b`` + Q norm/RoPE + SWA KV insert + main KV compression.
* Input-GEMM overlap and C128A compressor overlap stay off on ROCm: at small
  decode batch sizes the per-layer event / stream-switch overhead exceeds the
  achievable savings and regresses TPOT.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

import vllm.envs as envs

if TYPE_CHECKING:
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.attention.backends.mla.sparse_swa import (
        DeepseekSparseSWAMetadata,
    )


def create_dsv4_rocm_aux_stream_list() -> list[torch.cuda.Stream] | None:
    """Create the single CSA auxiliary stream used by DeepSeek-V4 on ROCm."""
    if not envs.VLLM_DSV4_ROCM_MULTI_STREAM:
        return None
    return [torch.cuda.Stream()]


def _has_decode_tokens(
    attn_metadata: dict[str, AttentionMetadata] | list | None,
    swa_cache_prefix: str,
) -> bool:
    if not isinstance(attn_metadata, dict):
        return False
    swa_metadata = cast(
        "DeepseekSparseSWAMetadata | None",
        attn_metadata.get(swa_cache_prefix),
    )
    if swa_metadata is None:
        return False
    return swa_metadata.num_decodes > 0


def should_overlap_dsv4_rocm_indexer(
    aux_stream_list: list[torch.cuda.Stream] | None,
    attn_metadata: dict[str, AttentionMetadata] | list | None,
    swa_cache_prefix: str,
) -> bool:
    """Whether to overlap the lightning indexer with main attention prep.

    This intentionally has no decode-batch-size floor. With the opt-in env set,
    small decode graphs also take the aux-stream path so benchmark runs expose
    the observed low-concurrency regression.
    """
    if aux_stream_list is None:
        return False
    if envs.VLLM_DSV4_ROCM_MULTI_STREAM_DECODE_ONLY:
        return _has_decode_tokens(attn_metadata, swa_cache_prefix)
    return True
