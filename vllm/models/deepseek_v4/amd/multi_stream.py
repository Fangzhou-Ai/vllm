# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm-only multi-stream helpers for DeepSeek-V4 CSA attention overlap.

Aligned with the DeepSeek-V4 blog/CSA design (issue #41820):

* One auxiliary stream alongside the default stream.
* The auxiliary stream runs the lightning indexer; the default stream runs
  ``wq_b`` + Q norm/RoPE + SWA KV insert + main KV compression.
* Input-GEMM overlap and C128A compressor overlap are intentionally left off
  on ROCm: at small decode batch sizes the per-layer event / stream-switch
  overhead exceeds the achievable savings, regressing TPOT (observed on
  1k1k conc=4).
* C4A indexer overlap is also gated by active decode batch size. By default the
  gate is above the standard CUDA graph capture range, because graph replay
  fixes the captured stream topology and 1k1k conc=4 regresses when the aux
  path contends with default-stream GEMMs.
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
    """Create the single CSA auxiliary stream used by DeepSeek-V4 on ROCm.

    Opt-in via ``VLLM_DSV4_ROCM_MULTI_STREAM`` because multi-stream was
    previously disabled here due to a hang (#41820). The list shape matches
    the CUDA model entry point, but ROCm only consumes ``aux_stream_list[0]``
    for the indexer-vs-default overlap.
    """
    if not envs.VLLM_DSV4_ROCM_MULTI_STREAM:
        return None
    return [torch.cuda.Stream()]


def _active_decode_batch_size(
    attn_metadata: dict[str, AttentionMetadata] | list | None,
    swa_cache_prefix: str,
) -> int:
    if not isinstance(attn_metadata, dict):
        return 0
    swa_metadata = cast(
        "DeepseekSparseSWAMetadata | None",
        attn_metadata.get(swa_cache_prefix),
    )
    if swa_metadata is None:
        return 0
    num_decodes = int(swa_metadata.num_decodes)
    query_start_loc_cpu = getattr(swa_metadata, "query_start_loc_cpu", None)
    if query_start_loc_cpu is None or num_decodes == 0:
        return num_decodes
    query_lens = (
        query_start_loc_cpu[1 : num_decodes + 1] - query_start_loc_cpu[:num_decodes]
    )
    return int(torch.count_nonzero(query_lens).item())


def should_overlap_dsv4_rocm_indexer(
    aux_stream_list: list[torch.cuda.Stream] | None,
    attn_metadata: dict[str, AttentionMetadata] | list | None,
    swa_cache_prefix: str,
) -> bool:
    """Whether to overlap the lightning indexer with main attention prep.

    Gated on the master switch (via ``aux_stream_list`` being non-None), decode
    presence by default, and a minimum decode-batch threshold. Prefill-only and
    small captured decode graphs stay serial because the GEMMs already saturate
    the GPU and aux-stream overhead would only hurt.
    """
    if aux_stream_list is None:
        return False
    num_decodes = _active_decode_batch_size(attn_metadata, swa_cache_prefix)
    if envs.VLLM_DSV4_ROCM_MULTI_STREAM_DECODE_ONLY and num_decodes == 0:
        return False
    return num_decodes >= envs.VLLM_DSV4_ROCM_MULTI_STREAM_MIN_DECODE_BATCH
