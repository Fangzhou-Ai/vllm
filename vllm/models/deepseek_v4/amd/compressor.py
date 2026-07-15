# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from vllm.models.deepseek_v4.common.ops.fused_indexer_q import (
    _fp32x2_to_fp4x2,
)
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import np_to_pinned_tensor

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.models.deepseek_v4.compressor import (
        CompressorMetadata,
        DeepseekCompressor,
    )
    from vllm.v1.attention.backend import CommonAttentionMetadata


def infer_compress_ratio(block_size: int, sliding_window: int | None) -> int:
    geometry = (block_size, sliding_window)
    if geometry == (4, 8):
        return 4
    if geometry == (8, 128):
        return 128
    raise ValueError(f"Unsupported ROCm compressor state geometry: {geometry}")


def use_tail_only_state_writes(vllm_config: "VllmConfig") -> bool:
    kv_transfer_config = vllm_config.kv_transfer_config
    has_kv_connector = (
        kv_transfer_config is not None and kv_transfer_config.is_kv_transfer_instance
    )
    return not (vllm_config.cache_config.enable_prefix_caching or has_kv_connector)


def make_compression_plan_template(
    query_start_loc_cpu: torch.Tensor | np.ndarray,
    compress_ratio: int,
) -> np.ndarray:
    """Build compact candidate rows from query lengths only.

    Each candidate is ``[query_start, query_end, request, ordinal]``. The GPU
    finalizer resolves the first aligned boundary from the authoritative
    positions tensor and turns candidates into compressor plan rows.
    """
    starts = np.asarray(query_start_loc_cpu, dtype=np.int32)
    query_lens = np.diff(starts)
    counts = (query_lens + compress_ratio - 1) // compress_ratio
    num_candidates = int(counts.sum())
    if num_candidates == 0:
        return np.empty((0, 4), dtype=np.int32)

    request_ids = np.repeat(np.arange(query_lens.size, dtype=np.int32), counts)
    candidate_starts = np.empty(query_lens.size + 1, dtype=np.int32)
    candidate_starts[0] = 0
    np.cumsum(counts, out=candidate_starts[1:])
    ordinals = np.arange(num_candidates, dtype=np.int32)
    ordinals -= np.repeat(candidate_starts[:-1], counts)

    return np.stack(
        (
            starts[request_ids],
            starts[request_ids + 1],
            request_ids,
            ordinals,
        ),
        axis=1,
    )


def compression_plan_capacity(
    num_tokens: int,
    num_requests: int,
    compress_ratio: int,
) -> int:
    """Return a graph-descriptor-stable upper bound on boundary candidates.

    For query lengths ``q_i``, ``sum(ceil(q_i / R))`` is at most both
    ``sum(q_i)`` and ``floor(sum(q_i) / R) + B``.
    """
    return min(num_tokens, num_tokens // compress_ratio + num_requests)


@triton.jit
def _finalize_compression_plan_kernel(
    plan_ptr,
    positions_ptr,
    state_slot_mapping_ptr,
    COMPRESS_RATIO: tl.constexpr,
    STATE_WINDOW: tl.constexpr,
):
    candidate_idx = tl.program_id(0)
    plan = plan_ptr + candidate_idx * 4
    query_start = tl.load(plan)
    query_end = tl.load(plan + 1)
    request_idx = tl.load(plan + 2)
    ordinal = tl.load(plan + 3)

    first_position = tl.load(positions_ptr + query_start)
    first_offset = COMPRESS_RATIO - 1 - (first_position % COMPRESS_RATIO)
    token_idx = query_start + first_offset + ordinal * COMPRESS_RATIO
    is_valid = token_idx < query_end
    safe_token_idx = tl.where(is_valid, token_idx, query_start)
    is_valid &= tl.load(state_slot_mapping_ptr + safe_token_idx) >= 0

    position = tl.load(positions_ptr + safe_token_idx)
    token_offset = safe_token_idx - query_start
    window_len = tl.maximum(
        0,
        STATE_WINDOW - tl.minimum(token_offset + 1, STATE_WINDOW),
    )

    tl.store(plan, safe_token_idx)
    tl.store(plan + 1, request_idx)
    tl.store(plan + 2, tl.where(is_valid, position, -1))
    tl.store(plan + 3, window_len)


def build_compression_plan(
    plan_buffer: torch.Tensor,
    common_attn_metadata: "CommonAttentionMetadata",
    compress_ratio: int,
) -> torch.Tensor:
    template = make_compression_plan_template(
        common_attn_metadata.query_start_loc_cpu, compress_ratio
    )
    num_candidates = template.shape[0]
    plan_capacity = compression_plan_capacity(
        common_attn_metadata.num_actual_tokens,
        common_attn_metadata.num_reqs,
        compress_ratio,
    )
    if plan_capacity > plan_buffer.shape[0]:
        raise ValueError(
            f"ROCm compressor descriptor capacity {plan_capacity} exceeds "
            f"its plan buffer capacity {plan_buffer.shape[0]}"
        )
    if num_candidates > plan_capacity:
        raise ValueError(
            f"ROCm compressor plan needs {num_candidates} rows, but its "
            f"descriptor capacity is {plan_capacity}"
        )

    plan = plan_buffer[:plan_capacity]
    plan.fill_(-1)
    if num_candidates == 0:
        return plan
    plan[:num_candidates].copy_(np_to_pinned_tensor(template), non_blocking=True)
    positions = common_attn_metadata.positions
    if positions is None:
        raise ValueError("ROCm compressor planning requires token positions")
    state_window = compress_ratio * (2 if compress_ratio == 4 else 1)
    _finalize_compression_plan_kernel[(num_candidates,)](
        plan[:num_candidates],
        positions,
        common_attn_metadata.slot_mapping,
        COMPRESS_RATIO=compress_ratio,
        STATE_WINDOW=state_window,
    )
    return plan


@triton.jit
def _compress_current_and_paged_state_kernel(
    state_cache_ptr,
    state_cache_stride0,
    state_cache_stride1,
    kv_ptr,
    kv_stride,
    score_ptr,
    score_stride,
    ape_ptr,
    ape_stride,
    plan_ptr,
    block_table_ptr,
    block_table_stride,
    block_size,
    rms_norm_weight_ptr,
    rms_norm_eps,
    cos_sin_cache_ptr,
    cos_sin_stride,
    k_cache_ptr,
    kv_slot_mapping_ptr,
    kv_cache_block_size,
    HEAD_SIZE: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
    STATE_WIDTH: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    OVERLAP: tl.constexpr,
    ROPE_HEAD_DIM: tl.constexpr,
    FP8_MAX: tl.constexpr,
    QUANT_BLOCK: tl.constexpr,
    TOKEN_STRIDE: tl.constexpr,
    SCALE_DIM: tl.constexpr,
    KV_BLOCK_STRIDE: tl.constexpr,
    USE_FP4_CACHE: tl.constexpr,
):
    plan_idx = tl.program_id(0)
    plan = plan_ptr + plan_idx * 4
    token_idx = tl.load(plan)
    req_idx = tl.load(plan + 1)
    position = tl.load(plan + 2)
    window_len = tl.load(plan + 3)
    if position < 0:
        return

    STATE_WINDOW: tl.constexpr = (1 + OVERLAP) * COMPRESS_RATIO
    state_tokens = tl.arange(0, STATE_WINDOW)
    source_positions = position - STATE_WINDOW + 1 + state_tokens
    safe_positions = tl.maximum(source_positions, 0)
    head_offset = (
        (state_tokens >= COMPRESS_RATIO).to(tl.int32) * HEAD_SIZE if OVERLAP else 0
    )
    block = tl.arange(0, TRITON_BLOCK_SIZE)
    head_mask = block < HEAD_SIZE
    if window_len == 0:
        input_rows = token_idx - (STATE_WINDOW - 1 - state_tokens)
        input_row = input_rows * kv_stride + head_offset
        kv = tl.load(
            kv_ptr + input_row[:, None] + block[None, :],
            mask=head_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        score = tl.load(
            score_ptr
            + (input_rows * score_stride + head_offset)[:, None]
            + block[None, :],
            mask=head_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        score += tl.load(
            ape_ptr
            + ((source_positions % COMPRESS_RATIO) * ape_stride + head_offset)[:, None]
            + block[None, :],
            mask=head_mask[None, :],
            other=0.0,
        )
    else:
        state_mask = (state_tokens < window_len) & (source_positions >= 0)
        block_numbers = tl.load(
            block_table_ptr
            + req_idx * block_table_stride
            + safe_positions // block_size,
            mask=state_mask,
            other=0,
        ).to(tl.int64)
        state_row = (
            state_cache_ptr
            + block_numbers * state_cache_stride0
            + safe_positions % block_size * state_cache_stride1
            + head_offset
        )
        state_load_mask = state_mask[:, None] & head_mask[None, :]
        state_score = tl.load(
            state_row[:, None] + STATE_WIDTH + block[None, :],
            mask=state_load_mask,
            other=float("-inf"),
        )
        state_kv = tl.load(
            state_row[:, None] + block[None, :],
            mask=state_load_mask,
            other=0.0,
        )

        if window_len == STATE_WINDOW - 1:
            FINAL_HEAD_OFFSET: tl.constexpr = HEAD_SIZE if OVERLAP else 0
            decode_input_kv = tl.load(
                kv_ptr + token_idx * kv_stride + FINAL_HEAD_OFFSET + block,
                mask=head_mask,
                other=0.0,
            ).to(tl.float32)
            decode_input_score = tl.load(
                score_ptr + token_idx * score_stride + FINAL_HEAD_OFFSET + block,
                mask=head_mask,
                other=0.0,
            ).to(tl.float32)
            decode_input_score += tl.load(
                ape_ptr
                + (position % COMPRESS_RATIO) * ape_stride
                + FINAL_HEAD_OFFSET
                + block,
                mask=head_mask,
                other=0.0,
            )
            is_last = state_tokens == STATE_WINDOW - 1
            kv = tl.where(is_last[:, None], decode_input_kv[None, :], state_kv)
            score = tl.where(is_last[:, None], decode_input_score[None, :], state_score)
        else:
            input_mask = state_tokens >= window_len
            input_rows = token_idx - (STATE_WINDOW - 1 - state_tokens)
            safe_input_rows = tl.maximum(input_rows, 0)
            input_load_mask = input_mask[:, None] & head_mask[None, :]
            mixed_input_kv = tl.load(
                kv_ptr
                + (safe_input_rows * kv_stride + head_offset)[:, None]
                + block[None, :],
                mask=input_load_mask,
                other=0.0,
            ).to(tl.float32)
            mixed_input_score = tl.load(
                score_ptr
                + (safe_input_rows * score_stride + head_offset)[:, None]
                + block[None, :],
                mask=input_load_mask,
                other=0.0,
            ).to(tl.float32)
            mixed_input_score += tl.load(
                ape_ptr
                + ((safe_positions % COMPRESS_RATIO) * ape_stride + head_offset)[
                    :, None
                ]
                + block[None, :],
                mask=input_load_mask,
                other=0.0,
            )
            kv = tl.where(input_mask[:, None], mixed_input_kv, state_kv)
            score = tl.where(input_mask[:, None], mixed_input_score, state_score)

    weights = tl.softmax(score, dim=0)
    compressed_kv = tl.sum(kv * weights, axis=0)

    rms_weight = tl.load(rms_norm_weight_ptr + block, mask=head_mask, other=0.0)
    variance = tl.sum(compressed_kv * compressed_kv, axis=0) / HEAD_SIZE
    normed = compressed_kv * tl.rsqrt(variance + rms_norm_eps) * rms_weight

    kv_slot_idx = tl.load(kv_slot_mapping_ptr + token_idx)
    if kv_slot_idx < 0:
        return
    kv_block_idx = kv_slot_idx // kv_cache_block_size
    kv_pos_in_block = kv_slot_idx % kv_cache_block_size
    cache_block_ptr = k_cache_ptr + kv_block_idx.to(tl.int64) * KV_BLOCK_STRIDE
    value_ptr = cache_block_ptr + kv_pos_in_block * TOKEN_STRIDE
    scale_ptr = (
        cache_block_ptr
        + kv_cache_block_size * TOKEN_STRIDE
        + kv_pos_in_block * SCALE_DIM
    )

    NOPE_HEAD_DIM: tl.constexpr = HEAD_SIZE - ROPE_HEAD_DIM
    HALF_ROPE: tl.constexpr = ROPE_HEAD_DIM // 2
    NUM_PAIRS: tl.constexpr = TRITON_BLOCK_SIZE // 2
    NOPE_PAIRS: tl.constexpr = NOPE_HEAD_DIM // 2

    pairs = tl.reshape(normed, (NUM_PAIRS, 2))
    even, odd = tl.split(pairs)
    pair_idx = tl.arange(0, NUM_PAIRS)
    rope_pair_local = pair_idx - NOPE_PAIRS
    is_rope_pair = rope_pair_local >= 0
    cs_idx = tl.maximum(rope_pair_local, 0)
    compressed_pos = (position // COMPRESS_RATIO) * COMPRESS_RATIO
    cache_base = cos_sin_cache_ptr + compressed_pos * cos_sin_stride
    cos = tl.load(cache_base + cs_idx, mask=is_rope_pair, other=1.0)
    sin = tl.load(cache_base + HALF_ROPE + cs_idx, mask=is_rope_pair, other=0.0)
    new_even = even * cos - odd * sin
    new_odd = odd * cos + even * sin

    if HEAD_SIZE == 512:
        result = tl.interleave(new_even, new_odd)
        N_QUANT_BLOCKS: tl.constexpr = TRITON_BLOCK_SIZE // QUANT_BLOCK
        N_NOPE_BLOCKS: tl.constexpr = NOPE_HEAD_DIM // QUANT_BLOCK
        quant_input = normed.to(tl.bfloat16).to(tl.float32)
        quant_2d = tl.reshape(quant_input, (N_QUANT_BLOCKS, QUANT_BLOCK))
        block_absmax = tl.max(tl.abs(quant_2d), axis=1)
        block_absmax = tl.maximum(block_absmax, 1e-4)
        exponents = tl.ceil(tl.log2(block_absmax * (1.0 / FP8_MAX)))
        inv_scales = tl.reshape(tl.exp2(-exponents), (N_QUANT_BLOCKS, 1))
        x_scaled = tl.clamp(quant_2d * inv_scales, -FP8_MAX, FP8_MAX)
        x_uint8 = x_scaled.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
        x_uint8 = tl.reshape(x_uint8, (TRITON_BLOCK_SIZE,))
        tl.store(value_ptr + block, x_uint8, mask=block < NOPE_HEAD_DIM)

        scale_idx = tl.arange(0, N_QUANT_BLOCKS)
        encoded = tl.maximum(tl.minimum(exponents + 127.0, 255.0), 0.0)
        tl.store(
            scale_ptr + scale_idx,
            encoded.to(tl.uint8),
            mask=scale_idx < N_NOPE_BLOCKS,
        )
        tl.store(scale_ptr + N_NOPE_BLOCKS, tl.zeros((), dtype=tl.uint8))

        bf16_ptr = (value_ptr + NOPE_HEAD_DIM).to(tl.pointer_type(tl.bfloat16))
        tl.store(
            bf16_ptr + block - NOPE_HEAD_DIM,
            result.to(tl.bfloat16),
            mask=(block >= NOPE_HEAD_DIM) & head_mask,
        )
    elif USE_FP4_CACHE:
        tl.static_assert(TRITON_BLOCK_SIZE == HEAD_SIZE)
        tl.static_assert(HEAD_SIZE % QUANT_BLOCK == 0)
        N_FP4_BLOCKS: tl.constexpr = HEAD_SIZE // QUANT_BLOCK
        HALF_BLOCK: tl.constexpr = QUANT_BLOCK // 2
        new_even = new_even.to(tl.bfloat16).to(tl.float32)
        new_odd = new_odd.to(tl.bfloat16).to(tl.float32)
        even_2d = tl.reshape(new_even, (N_FP4_BLOCKS, HALF_BLOCK))
        odd_2d = tl.reshape(new_odd, (N_FP4_BLOCKS, HALF_BLOCK))
        amax = tl.maximum(
            tl.max(tl.abs(even_2d), axis=1),
            tl.max(tl.abs(odd_2d), axis=1),
        )
        amax = tl.maximum(amax, 6.0 * (2**-126))
        exponents = tl.ceil(tl.log2(amax * (1.0 / 6.0)))
        exponents = tl.minimum(tl.maximum(exponents, -127.0), 127.0)
        inv_scale = tl.reshape(tl.exp2(-exponents), (N_FP4_BLOCKS, 1))
        packed = _fp32x2_to_fp4x2(even_2d * inv_scale, odd_2d * inv_scale)
        tl.store(
            value_ptr + tl.arange(0, TOKEN_STRIDE),
            tl.reshape(packed, (TOKEN_STRIDE,)),
        )
        tl.store(
            scale_ptr + tl.arange(0, SCALE_DIM),
            (exponents + 127.0).to(tl.uint8),
        )
    else:
        tl.static_assert(TRITON_BLOCK_SIZE == QUANT_BLOCK)
        result = tl.interleave(new_even, new_odd).to(tl.bfloat16).to(tl.float32)
        absmax = tl.maximum(tl.max(tl.abs(result), axis=0), 1e-4)
        exponent = tl.ceil(tl.log2(absmax * (1.0 / FP8_MAX)))
        x_scaled = tl.clamp(result * tl.exp2(-exponent), -FP8_MAX, FP8_MAX)
        x_uint8 = x_scaled.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
        tl.store(value_ptr + block, x_uint8, mask=head_mask)
        tl.store(scale_ptr.to(tl.pointer_type(tl.float32)), tl.exp2(exponent))


def compress_current_and_paged_state(
    *,
    state_cache: torch.Tensor,
    kv: torch.Tensor,
    score: torch.Tensor,
    ape: torch.Tensor,
    plan: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    state_width: int,
    cos_sin_cache: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    head_dim: int,
    rope_head_dim: int,
    compress_ratio: int,
    overlap: bool,
    use_fp4_cache: bool,
    quant_block: int,
    token_stride: int,
    scale_dim: int,
) -> None:
    if plan.shape[0] == 0:
        return
    if plan.dtype != torch.int32 or plan.shape[1:] != (4,):
        raise ValueError(f"Invalid ROCm compressor plan: {plan.shape}, {plan.dtype}")
    if kv.stride(-1) != 1 or score.stride(-1) != 1:
        raise ValueError("ROCm compressor inputs must have contiguous columns")

    num_warps = 4 if head_dim == 512 else 1
    _compress_current_and_paged_state_kernel[(plan.shape[0],)](
        state_cache,
        state_cache.stride(0),
        state_cache.stride(1),
        kv,
        kv.stride(0),
        score,
        score.stride(0),
        ape,
        ape.stride(0),
        plan,
        block_table,
        block_table.stride(0),
        block_size,
        rms_norm_weight,
        rms_norm_eps,
        cos_sin_cache,
        cos_sin_cache.stride(0),
        kv_cache,
        kv_slot_mapping,
        kv_cache.shape[1],
        HEAD_SIZE=head_dim,
        TRITON_BLOCK_SIZE=triton.next_power_of_2(head_dim),
        STATE_WIDTH=state_width,
        COMPRESS_RATIO=compress_ratio,
        OVERLAP=overlap,
        ROPE_HEAD_DIM=rope_head_dim,
        FP8_MAX=448.0,
        QUANT_BLOCK=quant_block,
        TOKEN_STRIDE=token_stride,
        SCALE_DIM=scale_dim,
        KV_BLOCK_STRIDE=kv_cache.stride(0),
        USE_FP4_CACHE=use_fp4_cache,
        num_warps=num_warps,
    )


@triton.jit
def _save_compressor_states_kernel(
    kv_ptr,
    kv_stride,
    score_ptr,
    score_stride,
    ape_ptr,
    ape_stride,
    positions_ptr,
    seq_lens_ptr,
    token_to_req_indices_ptr,
    state_cache_ptr,
    state_cache_stride0,
    state_cache_stride1,
    slot_mapping_ptr,
    block_size,
    HEAD_SIZE: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
    STATE_WIDTH: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    STATE_WINDOW: tl.constexpr,
    TAIL_ONLY: tl.constexpr,
):
    token_idx = tl.program_id(0)
    slot_id = tl.load(slot_mapping_ptr + token_idx)
    if slot_id < 0:
        return

    position = tl.load(positions_ptr + token_idx)
    request_idx = tl.load(token_to_req_indices_ptr + token_idx)
    if TAIL_ONLY:
        seq_len = tl.load(seq_lens_ptr + request_idx)
        if position < tl.maximum(seq_len - STATE_WINDOW, 0):
            return

    block_idx = slot_id // block_size
    pos_in_block = slot_id % block_size
    state_row = (
        state_cache_ptr
        + block_idx * state_cache_stride0
        + pos_in_block * state_cache_stride1
    )
    block = tl.arange(0, TRITON_BLOCK_SIZE)
    mask = block < HEAD_SIZE
    kv = tl.load(kv_ptr + token_idx * kv_stride + block, mask=mask)
    score = tl.load(score_ptr + token_idx * score_stride + block, mask=mask)
    ape = tl.load(ape_ptr + (position % COMPRESS_RATIO) * ape_stride + block, mask=mask)
    tl.store(state_row + block, kv, mask=mask)
    tl.store(state_row + STATE_WIDTH + block, score + ape, mask=mask)


def save_compressor_states(
    *,
    kv: torch.Tensor,
    score: torch.Tensor,
    ape: torch.Tensor,
    positions: torch.Tensor,
    seq_lens: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    state_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
    state_width: int,
    compress_ratio: int,
    overlap: bool,
    tail_only: bool,
) -> None:
    num_tokens = slot_mapping.shape[0]
    if num_tokens == 0:
        return
    head_size = kv.shape[-1]
    _save_compressor_states_kernel[(num_tokens,)](
        kv,
        kv.stride(0),
        score,
        score.stride(0),
        ape,
        ape.stride(0),
        positions,
        seq_lens,
        token_to_req_indices,
        state_cache,
        state_cache.stride(0),
        state_cache.stride(1),
        slot_mapping,
        block_size,
        HEAD_SIZE=head_size,
        TRITON_BLOCK_SIZE=triton.next_power_of_2(head_size),
        STATE_WIDTH=state_width,
        COMPRESS_RATIO=compress_ratio,
        STATE_WINDOW=compress_ratio * (2 if overlap else 1),
        TAIL_ONLY=tail_only,
    )


def rocm_compressor_forward(
    compressor: "DeepseekCompressor",
    kv: torch.Tensor,
    score: torch.Tensor,
    positions: torch.Tensor,
    rotary_emb: Any,
    state_metadata: "CompressorMetadata",
    attn_metadata: dict[str, Any],
) -> None:
    plan = state_metadata.compression_plan
    if plan is None or state_metadata.seq_lens is None:
        raise RuntimeError("ROCm compressor metadata is missing its boundary plan")

    state_cache = compressor.state_cache.kv_cache
    state_width = state_cache.shape[-1] // 2
    k_cache_metadata = attn_metadata[compressor.k_cache_prefix]
    k_cache_layer = compressor._static_forward_context[compressor.k_cache_prefix]
    kv_cache = k_cache_layer.kv_cache

    compress_current_and_paged_state(
        state_cache=state_cache,
        kv=kv,
        score=score,
        ape=compressor.ape,
        plan=plan,
        block_table=state_metadata.block_table,
        block_size=state_metadata.block_size,
        state_width=state_width,
        cos_sin_cache=rotary_emb.cos_sin_cache,
        kv_cache=kv_cache,
        kv_slot_mapping=k_cache_metadata.slot_mapping,
        rms_norm_weight=compressor.norm.weight,
        rms_norm_eps=compressor.rms_norm_eps,
        head_dim=compressor.head_dim,
        rope_head_dim=compressor.rope_head_dim,
        compress_ratio=compressor.compress_ratio,
        overlap=compressor.overlap,
        use_fp4_cache=compressor.use_fp4_cache,
        quant_block=compressor._quant_block,
        token_stride=compressor._token_stride,
        scale_dim=compressor._scale_dim,
    )
    save_compressor_states(
        kv=kv,
        score=score,
        ape=compressor.ape,
        positions=positions,
        seq_lens=state_metadata.seq_lens,
        token_to_req_indices=state_metadata.token_to_req_indices,
        state_cache=state_cache,
        slot_mapping=state_metadata.slot_mapping,
        block_size=state_metadata.block_size,
        state_width=state_width,
        compress_ratio=compressor.compress_ratio,
        overlap=compressor.overlap,
        tail_only=state_metadata.tail_only,
    )
