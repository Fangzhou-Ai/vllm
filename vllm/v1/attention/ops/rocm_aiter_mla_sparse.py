# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools
import importlib
import math
from importlib.util import find_spec

import torch
import torch.nn.functional as F

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.forward_context import get_forward_context
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import LayerNameType
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadata
from vllm.v1.attention.ops.common import pack_seq_triton, unpack_seq_triton

if current_platform.is_rocm():
    from vllm.platforms.rocm import _ON_GFX942, _ON_GFX950
else:
    _ON_GFX942 = False
    _ON_GFX950 = False


@triton.jit
def _indexer_k_quant_and_cache_kernel(
    k_ptr,  # [num_tokens, head_dim]
    kv_cache_ptr,  # [n_blks, blk_size//tile_block, head_dim // 16B, tile_block, 16B]
    # [n_blocks, blk_size, head_dim]
    kv_cache_scale_ptr,  # [n_blks, blk_size]
    slot_mapping_ptr,  # [num_tokens]
    kv_cache_scale_stride,
    kv_cache_value_stride,
    block_size,
    num_tokens,
    head_dim: tl.constexpr,
    LAYOUT: tl.constexpr,
    BLOCK_TILE_SIZE: tl.constexpr,
    HEAD_TILE_SIZE: tl.constexpr,
    IS_FNUZ: tl.constexpr,
    USE_UE8M0: tl.constexpr,
):
    tid = tl.program_id(0)
    offset = tl.arange(0, head_dim)
    if LAYOUT == "SHUFFLE":
        tile_offset = (
            offset // HEAD_TILE_SIZE * BLOCK_TILE_SIZE * HEAD_TILE_SIZE
            + offset % HEAD_TILE_SIZE
        )
    else:
        tile_offset = offset
    tile_store_offset = tile_offset
    # for idx in tl.range(tid, num_tokens, n_program):
    src_ptr = k_ptr + tid * head_dim
    slot_id = tl.load(slot_mapping_ptr + tid)
    if slot_id < 0:
        return
    block_id = slot_id // block_size
    block_offset = slot_id % block_size
    tile_block_id = block_offset // BLOCK_TILE_SIZE
    tile_block_offset = block_offset % BLOCK_TILE_SIZE
    val = tl.load(src_ptr + offset)
    amax = tl.max(val.abs(), axis=-1).to(tl.float32)
    if IS_FNUZ:
        scale = tl.maximum(1e-4, amax) / 224.0
    else:
        scale = tl.maximum(1e-4, amax) / 448.0

    if USE_UE8M0:
        scale = tl.exp2(tl.ceil(tl.log2(scale)))

    fp8_val = (val.to(tl.float32) / scale).to(kv_cache_ptr.type.element_ty)
    if LAYOUT == "SHUFFLE":
        dst_ptr = (
            kv_cache_ptr
            + block_id * kv_cache_value_stride
            + tile_block_id * BLOCK_TILE_SIZE * head_dim
            + tile_block_offset * HEAD_TILE_SIZE
        )
    else:
        dst_ptr = (
            kv_cache_ptr + block_id * kv_cache_value_stride + block_offset * head_dim
        )
    tl.store(dst_ptr + tile_store_offset, fp8_val)
    dst_scale_ptr = kv_cache_scale_ptr + block_id * kv_cache_scale_stride + block_offset
    tl.store(dst_scale_ptr, scale)


def indexer_k_quant_and_cache_triton(
    k: torch.Tensor,
    kv_cache: torch.Tensor,  # [num_blocks, block_size, head_dim + 4]
    slot_mapping: torch.Tensor,
    quant_block_size,
    scale_fmt,
    block_tile_size=16,
    head_tile_size=16,
):
    num_blocks = kv_cache.shape[0]
    head_dim = k.shape[-1]
    num_tokens = slot_mapping.shape[0]
    block_size = kv_cache.shape[1]
    # In real layout, we store the first portion as kv cache value
    # and second portion as kv cache scale
    kv_cache = kv_cache.view(num_blocks, -1)
    fp8_dtype = current_platform.fp8_dtype()
    kv_cache_value = kv_cache[:, : block_size * head_dim].view(fp8_dtype)
    kv_cache_scale = kv_cache[:, block_size * head_dim :].view(torch.float32)
    head_tile_size = head_tile_size // kv_cache.element_size()
    layout = "NORMAL" if block_size == 1 else "SHUFFLE"
    grid = (num_tokens,)
    _indexer_k_quant_and_cache_kernel[grid](
        k,
        kv_cache_value,
        kv_cache_scale,
        slot_mapping,
        kv_cache_scale.stride(0),
        kv_cache_value.stride(0),
        block_size,
        num_tokens,
        head_dim,
        layout,
        block_tile_size,
        head_tile_size,
        IS_FNUZ=current_platform.fp8_dtype() == torch.float8_e4m3fnuz,
        USE_UE8M0=scale_fmt == "ue8m0",
    )


@triton.jit
def _cp_gather_indexer_quant_cache_kernel(
    kv_cache_ptr,  # [n_blks,blk_size//tile_blk,head_dim//16B,tile_blk,16B]
    # [n_blks, blk_size, head_dim]
    kv_cache_scale_ptr,  # [n_blks, blk_size]
    k_fp8_ptr,  # [num_tokens, head_dim]
    k_scale_ptr,  # [num_tokens]
    block_table_ptr,  # [batch_size, block_table_stride]
    cu_seqlen_ptr,  # [batch_size + 1]
    token_to_seq_ptr,  # [num_tokens]
    block_size,
    block_table_stride,
    kv_cache_stride,
    kv_cache_scale_stride,
    LAYOUT: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_TILE_SIZE: tl.constexpr,
    HEAD_TILE_SIZE: tl.constexpr,
    NUM_TOKENS: tl.constexpr,
    NUM_BATCHES: tl.constexpr,
    BLOCK_TABLE_WIDTH: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
):
    tid = tl.program_id(0)
    offset = tl.arange(0, HEAD_DIM)
    valid_tid = tid < NUM_TOKENS
    batch_id = tl.load(token_to_seq_ptr + tid, mask=valid_tid, other=-1)
    valid_batch = (batch_id >= 0) & (batch_id < NUM_BATCHES)
    safe_batch_id = tl.where(valid_batch, batch_id, 0)
    batch_start = tl.load(cu_seqlen_ptr + safe_batch_id, mask=valid_batch, other=0)
    batch_end = tl.load(cu_seqlen_ptr + safe_batch_id + 1, mask=valid_batch, other=0)
    batch_offset = tid - batch_start
    valid_token = valid_tid & valid_batch & (tid >= batch_start) & (tid < batch_end)
    if not valid_token:
        return
    block_table_id = batch_offset // block_size
    block_offset = batch_offset % block_size
    valid_block_table = (
        valid_token
        & (block_table_id >= 0)
        & (block_table_id < BLOCK_TABLE_WIDTH)
        & (block_offset >= 0)
        & (block_offset < block_size)
    )
    safe_block_table_id = tl.where(valid_block_table, block_table_id, 0)
    block_table_offset = safe_batch_id * block_table_stride + safe_block_table_id
    block_id = tl.load(
        block_table_ptr + block_table_offset, mask=valid_block_table, other=-1
    )
    valid_block = valid_block_table & (block_id >= 0) & (block_id < NUM_BLOCKS)
    safe_block_id = tl.where(valid_block, block_id, 0)
    safe_block_offset = tl.where(valid_block, block_offset, 0)
    tiled_block_offset = safe_block_offset % BLOCK_TILE_SIZE
    if LAYOUT == "SHUFFLE":
        src_cache_offset = (
            safe_block_id * kv_cache_stride
            + (safe_block_offset // BLOCK_TILE_SIZE) * HEAD_DIM * BLOCK_TILE_SIZE
            + tiled_block_offset * HEAD_TILE_SIZE
        )
    else:
        src_cache_offset = (
            safe_block_id * kv_cache_stride + safe_block_offset * HEAD_DIM
        )
    src_scale_offset = safe_block_id * kv_cache_scale_stride + safe_block_offset
    dst_offset = tid * HEAD_DIM
    src_scale_ptr = kv_cache_scale_ptr + src_scale_offset
    src_cache_ptr = kv_cache_ptr + src_cache_offset
    dst_k_ptr = k_fp8_ptr + dst_offset
    scale_val = tl.load(src_scale_ptr, mask=valid_block, other=0.0)
    tl.store(k_scale_ptr + tid, scale_val)
    if LAYOUT == "SHUFFLE":
        tiled_src_offset = (
            offset // HEAD_TILE_SIZE * HEAD_TILE_SIZE * BLOCK_TILE_SIZE
            + offset % HEAD_TILE_SIZE
        )
    else:
        tiled_src_offset = offset
    val = tl.load(src_cache_ptr + tiled_src_offset)
    tl.store(dst_k_ptr + offset, val, mask=valid_block)


def cp_gather_indexer_k_quant_cache_triton(
    k_cache: torch.Tensor,  # [num_blocks, block_size, head_dim + 4]
    k_fp8: torch.Tensor,
    k_fp8_scale: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlen: torch.Tensor,
    token_to_seq: torch.Tensor,
    block_tile_size: int = 16,
    head_tile_size: int = 16,
):
    num_tokens = k_fp8.size(0)
    block_size = k_cache.size(1)
    block_table_stride = block_table.stride(0)
    head_dim = k_fp8.shape[-1]
    num_blocks = k_cache.shape[0]
    # we assume the kv cache already been split to 2 portion
    k_cache = k_cache.view(num_blocks, -1)
    fp8_dtype = current_platform.fp8_dtype()
    k_cache_value = k_cache[:, : block_size * head_dim].view(fp8_dtype)
    k_cache_scale = k_cache[:, block_size * head_dim :].view(torch.float32)
    grid = (num_tokens,)
    k_fp8_scale = k_fp8_scale.view(torch.float32)
    layout = "NORMAL" if block_size == 1 else "SHUFFLE"
    _cp_gather_indexer_quant_cache_kernel[grid](
        k_cache_value,
        k_cache_scale,
        k_fp8,
        k_fp8_scale,
        block_table,
        cu_seqlen,
        token_to_seq,
        block_size,
        block_table_stride,
        k_cache_value.stride(0),
        k_cache_scale.stride(0),
        layout,
        head_dim,
        block_tile_size,
        head_tile_size,
        num_tokens,
        cu_seqlen.shape[0] - 1,
        block_table.shape[1],
        num_blocks,
    )


# Taken from https://github.com/deepseek-ai/DeepGEMM/blob/main/tests/test_attention.py#L156
def fp8_paged_mqa_logits_torch(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
):
    from vllm.utils.math_utils import cdiv

    fp8_dtype = current_platform.fp8_dtype()
    batch_size, next_n, _, dim = q.size()
    if next_n == 1:
        block_size = kv_cache.shape[1]
        logits = torch.full(
            [batch_size, max_model_len],
            float("-inf"),
            device=q.device,
            dtype=torch.float32,
        )
        if context_lens.dim() > 1:
            context_lens = context_lens.squeeze(-1)
        kv_cache_flat = kv_cache.view(-1, block_size * (dim + 4))
        for i in range(batch_size):
            q_i = q[i, 0].to(torch.float32)
            q_scale = weights[i]
            seq_len = int(context_lens[i].item())
            assert seq_len <= max_model_len
            num_pages = cdiv(seq_len, block_size)
            padded_seq_len = num_pages * block_size
            pages = block_tables[i, :num_pages]
            cache = kv_cache_flat[pages]
            scale_offset = block_size * dim
            cache_value = (
                cache[..., :scale_offset].view(dtype=fp8_dtype).to(torch.float32)
            )
            cache_scale = (
                cache[..., scale_offset:].view(dtype=torch.float32).contiguous()
            )
            cache_value = cache_value.view(padded_seq_len, dim)
            cache_scale = cache_scale.view(padded_seq_len)
            score = F.linear(cache_value, q_i)
            score = F.relu(score)
            score *= q_scale[None, :]
            score = score.sum(dim=1)
            score *= cache_scale
            logits[i, :seq_len] = score[:seq_len]
        return logits

    kv_cache, scale = kv_cache[..., :dim], kv_cache[..., dim:]
    scale = scale.contiguous().view(torch.float)
    q = q.float()
    kv_cache = kv_cache.view(fp8_dtype).float() * scale
    num_block, block_size, _, dim = kv_cache.size()
    logits = torch.full(
        [batch_size * next_n, max_model_len],
        float("-inf"),
        device=q.device,
        dtype=torch.float32,
    )
    for i in range(batch_size):
        context_len = context_lens[i]
        if context_len.ndim == 0:
            context_len_i = int(context_len.item())
            q_offsets = torch.arange(
                context_len_i - next_n, context_len_i, device=q.device
            )
            context_limit = torch.full(
                (next_n,), context_len_i, dtype=torch.int32, device=q.device
            )
        else:
            context_limit = context_len.to(device=q.device, dtype=torch.int32)
            q_offsets = context_limit - 1
        weight_slice = (
            weights[i * next_n : (i + 1) * next_n, :].transpose(0, 1).contiguous()
        )
        max_context_len = int(context_limit.max().item())
        for block_rk in range(cdiv(max_context_len, block_size)):
            block_idx = block_tables[i][block_rk]
            qx, kx = q[i], kv_cache[block_idx]
            k_offsets = torch.arange(
                block_rk * block_size, (block_rk + 1) * block_size, device=q.device
            )
            mask = (k_offsets[None, :] < context_limit[:, None]) & (
                k_offsets[None, :] <= q_offsets[:, None]
            )
            s = torch.where(
                mask[None, :, :],
                (qx.transpose(0, 1) @ kx.transpose(0, 1).transpose(1, 2)).to(
                    logits.dtype
                ),
                float("-inf"),
            )
            s = torch.relu(s) * weight_slice[..., None]
            s = s.sum(dim=0)
            logits[
                i * next_n : (i + 1) * next_n,
                block_rk * block_size : (block_rk + 1) * block_size,
            ] = torch.where(k_offsets[None, :] <= q_offsets[:, None], s, float("-inf"))
    return logits


@functools.lru_cache
def paged_mqa_logits_module():
    paged_mqa_logits_module_path = None
    if find_spec("aiter.ops.triton.pa_mqa_logits") is not None:
        paged_mqa_logits_module_path = "aiter.ops.triton.pa_mqa_logits"
    elif find_spec("aiter.ops.triton.attention.pa_mqa_logits") is not None:
        paged_mqa_logits_module_path = "aiter.ops.triton.attention.pa_mqa_logits"

    if paged_mqa_logits_module_path is not None:
        try:
            module = importlib.import_module(paged_mqa_logits_module_path)
            return module
        except ImportError:
            return None
    return None


def rocm_fp8_paged_mqa_logits(
    q_fp8: torch.Tensor,
    kv_cache_fp8: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    schedule_metadata: torch.Tensor,
    max_model_len: int,
) -> torch.Tensor:
    """Compute FP8 MQA logits using paged KV-cache.

    Args:
        q_fp8: Query tensor of shape [B, next_n, H, D]. Casted to
            `torch.float8_e4m3fn` by caller.
        kv_cache_fp8: Paged KV-cache in packed FP8+scale layout with shape
            [num_blocks, block_size, 1, D+4], dtype `torch.uint8`. The last
            4 bytes per (block,pos) store the `float` dequant scale.
        weights: Tensor of shape [B * next_n, H], dtype `torch.float32`.
        context_lens: Tensor of shape [B], dtype int32; effective context length
            for each batch element.
        block_tables: Tensor of shape [B, max_blocks], dtype int32; maps logical
            block indices to physical blocks in the paged cache.
        schedule_metadata: Returned by `get_paged_mqa_logits_metadata`;
            used to distribute work across SMs.
        max_model_len: Maximum sequence length used to size the logits output.

    Returns:
        Logits tensor of shape [B * next_n, max_model_len], dtype
        `torch.float32`.
    """
    from vllm._aiter_ops import rocm_aiter_ops

    aiter_paged_mqa_logits_module = None
    # if rocm_aiter_ops.is_enabled():
    batch_size, next_n, heads, head_dim = q_fp8.shape
    num_blocks, block_size, _, _ = kv_cache_fp8.shape

    if rocm_aiter_ops.is_enabled():
        aiter_paged_mqa_logits_module = paged_mqa_logits_module()

    if aiter_paged_mqa_logits_module is not None:
        if _ON_GFX942 or _ON_GFX950:
            deepgemm_fp8_paged_mqa_logits = (
                aiter_paged_mqa_logits_module.deepgemm_fp8_paged_mqa_logits
            )
            batch_size, next_n, heads, _ = q_fp8.shape
            out_logits = torch.full(
                [batch_size * next_n, max_model_len],
                float("-inf"),
                device="cuda",
                dtype=torch.float32,
            )
            deepgemm_fp8_paged_mqa_logits(
                q_fp8,
                kv_cache_fp8,
                weights,
                out_logits,
                context_lens,
                block_tables,
                max_model_len,
                ChunkK=256,
                Preshuffle=block_size > 1,
                KVBlockSize=block_size,
                WavePerEU=2,
            )
            return out_logits
        deepgemm_fp8_paged_mqa_logits_stage1 = (
            aiter_paged_mqa_logits_module.deepgemm_fp8_paged_mqa_logits_stage1
        )
        batch_size, next_n, heads, _ = q_fp8.shape
        out_qk = torch.full(
            (heads, batch_size * next_n, max_model_len),
            float("-inf"),
            device="cuda",
            dtype=torch.float32,
        )
        deepgemm_fp8_paged_mqa_logits_stage1(
            q_fp8,
            kv_cache_fp8,
            weights,
            out_qk,
            context_lens,
            block_tables,
            max_model_len,
            ChunkQ=heads,
        )
        return out_qk.sum(dim=0)
    else:
        return fp8_paged_mqa_logits_torch(
            q_fp8, kv_cache_fp8, weights, context_lens, block_tables, max_model_len
        )


# Take from https://github.com/deepseek-ai/DeepGEMM/blob/main/tests/test_attention.py#L84
def fp8_mqa_logits_torch(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> torch.Tensor:
    """Compute FP8 MQA logits for a single sequence without KV paging.

    Args:
        q: Query tensor of shape [M, H, D]. Casted to
            `torch.float8_e4m3fn` by caller.
        kv: Tuple `(k_fp8, k_scales)` where `k_fp8` has shape [N, D] with
            dtype `torch.float8_e4m3fn` and `k_scales` has shape [N] (or
            [N, 1]) with dtype `torch.float32`.
        weights: weights of shape [M, H], dtype `torch.float32`.
        cu_seqlen_ks: Start indices (inclusive) for valid K per query position,
            shape [M], dtype int32.
        cu_seqlen_ke: End indices (exclusive) for valid K per query position,
            shape [M], dtype int32.

    Returns:
        Logits tensor of shape [M, N], dtype `torch.float32`.
    """
    k_fp8, scale = kv
    seq_len_kv = k_fp8.shape[0]
    k = k_fp8.to(torch.bfloat16)
    q = q.to(torch.bfloat16)
    device = q.device

    mask_lo = (
        torch.arange(0, seq_len_kv, device=device)[None, :] >= cu_seqlen_ks[:, None]
    )
    mask_hi = (
        torch.arange(0, seq_len_kv, device=device)[None, :] < cu_seqlen_ke[:, None]
    )
    mask = mask_lo & mask_hi

    score = torch.einsum("mhd,nd->hmn", q, k).float() * scale
    logits = (score.relu() * weights.unsqueeze(-1).transpose(0, 1)).sum(dim=0)
    logits = logits.masked_fill(~mask, float("-inf"))

    return logits


@functools.lru_cache
def mqa_logits_module():
    mqa_logits_module_path = None
    if find_spec("aiter.ops.triton.fp8_mqa_logits") is not None:
        mqa_logits_module_path = "aiter.ops.triton.fp8_mqa_logits"
    elif find_spec("aiter.ops.triton.attention.fp8_mqa_logits") is not None:
        mqa_logits_module_path = "aiter.ops.triton.attention.fp8_mqa_logits"

    if mqa_logits_module_path is not None:
        try:
            module = importlib.import_module(mqa_logits_module_path)
            return module
        except ImportError:
            return None
    return None


def rocm_fp8_mqa_logits(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> torch.Tensor:
    """Compute FP8 MQA logits for a single sequence without KV paging.

    Args:
        q: Query tensor of shape [M, H, D]. Casted to
            `torch.float8_e4m3fn` by caller.
        kv: Tuple `(k_fp8, k_scales)` where `k_fp8` has shape [N, D] with
            dtype `torch.float8_e4m3fn` and `k_scales` has shape [N] (or
            [N, 1]) with dtype `torch.float32`.
        weights: weights of shape [M, H], dtype `torch.float32`.
        cu_seqlen_ks: Start indices (inclusive) for valid K per query position,
            shape [M], dtype int32.
        cu_seqlen_ke: End indices (exclusive) for valid K per query position,
            shape [M], dtype int32.

    Returns:
        Logits tensor of shape [M, N], dtype `torch.float32`.
    """

    # TODO(ganyi): Temporarily workaround, will remove the module check and reference
    # path after aiter merge this kernel into main
    from vllm._aiter_ops import rocm_aiter_ops

    aiter_mqa_logits_module = None
    if rocm_aiter_ops.is_enabled():
        aiter_mqa_logits_module = mqa_logits_module()

    if aiter_mqa_logits_module is not None:
        fp8_mqa_logits = aiter_mqa_logits_module.fp8_mqa_logits
        k_fp8, scale = kv
        return fp8_mqa_logits(q, k_fp8, scale, weights, cu_seqlen_ks, cu_seqlen_ke)
    else:
        return fp8_mqa_logits_torch(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke)


def _topk_indices_torch(
    logits: torch.Tensor,
    topk_tokens: int,
    row_starts: torch.Tensor | None = None,
) -> torch.Tensor:
    k = min(topk_tokens, logits.shape[-1])
    values, indices = torch.topk(logits, k=k, dim=-1)
    indices = indices.to(torch.int32)
    indices = torch.where(
        values == float("-inf"),
        torch.full_like(indices, -1, dtype=torch.int32),
        indices,
    )
    if row_starts is not None:
        # Match the CUDA top_k_per_row_prefill contract: indices are local to
        # each row's valid [row_start, row_end) range, not columns in the
        # concatenated chunk logits matrix.
        starts = row_starts.to(dtype=torch.int32).view(-1, 1)
        indices = torch.where(indices < 0, indices, indices - starts)
    if k == topk_tokens:
        return indices
    padded = torch.full(
        (logits.shape[0], topk_tokens),
        -1,
        dtype=torch.int32,
        device=logits.device,
    )
    padded[:, :k] = indices
    return padded


def rocm_aiter_sparse_attn_indexer_fake(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_fp8: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor | None,
    skip_k_cache_insert: bool = False,
) -> torch.Tensor:
    return topk_indices_buffer


@eager_break_during_capture
def rocm_aiter_sparse_attn_indexer(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_fp8: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor | None,
    skip_k_cache_insert: bool = False,
) -> torch.Tensor:
    # careful! this will be None in dummy run
    attn_metadata = get_forward_context().attn_metadata
    fp8_dtype = current_platform.fp8_dtype()
    from vllm import _custom_ops as ops
    from vllm.utils.torch_utils import _resolve_layer_name

    k_cache_prefix = _resolve_layer_name(k_cache_prefix)
    # assert isinstance(attn_metadata, dict)
    if not isinstance(attn_metadata, dict):
        return rocm_aiter_sparse_attn_indexer_fake(
            hidden_states,
            k_cache_prefix,
            kv_cache,
            q_fp8,
            k,
            weights,
            quant_block_size,
            scale_fmt,
            topk_tokens,
            head_dim,
            max_model_len,
            total_seq_lens,
            topk_indices_buffer,
            skip_k_cache_insert,
        )
    layer_attn_metadata = attn_metadata[k_cache_prefix]
    assert isinstance(layer_attn_metadata, DeepseekV32IndexerMetadata)
    assert topk_indices_buffer is not None
    assert scale_fmt is not None
    slot_mapping = layer_attn_metadata.slot_mapping
    has_decode = layer_attn_metadata.num_decodes > 0
    has_prefill = layer_attn_metadata.num_prefills > 0
    num_decode_tokens = layer_attn_metadata.num_decode_tokens
    device = hidden_states.device if k is None else k.device

    # during speculative decoding, k may be padded to the CUDA graph batch
    # size while slot_mapping only covers actual tokens.
    num_tokens = slot_mapping.shape[0]
    if k is not None:
        k = k[:num_tokens]
    elif not skip_k_cache_insert:
        raise ValueError("k must be provided when skip_k_cache_insert is False")

    if not skip_k_cache_insert:
        if _ON_GFX942:
            ops.indexer_k_quant_and_cache(
                k,
                kv_cache,
                slot_mapping,
                quant_block_size,
                scale_fmt,
            )
        else:
            indexer_k_quant_and_cache_triton(
                k,
                kv_cache,
                slot_mapping,
                quant_block_size,
                scale_fmt,
            )

    topk_indices_buffer[: hidden_states.shape[0]] = -1
    if has_prefill:
        prefill_metadata = layer_attn_metadata.prefill
        assert prefill_metadata is not None
        for chunk in prefill_metadata.chunks:
            k_fp8 = torch.empty(
                [chunk.total_seq_lens, head_dim],
                device=device,
                dtype=fp8_dtype,
            )
            k_scale = torch.empty(
                [chunk.total_seq_lens, 4],
                device=device,
                dtype=torch.uint8,
            )
            if _ON_GFX942:
                ops.cp_gather_indexer_k_quant_cache(
                    kv_cache,
                    k_fp8,
                    k_scale,
                    chunk.block_table,
                    chunk.cu_seq_lens,
                )
            else:
                cp_gather_indexer_k_quant_cache_triton(
                    kv_cache,
                    k_fp8,
                    k_scale,
                    chunk.block_table,
                    chunk.cu_seq_lens,
                    token_to_seq=chunk.token_to_seq,
                )

            logits = rocm_fp8_mqa_logits(
                q_fp8[chunk.token_start : chunk.token_end],
                (k_fp8, k_scale.view(torch.float32)),
                weights[chunk.token_start : chunk.token_end],
                chunk.cu_seqlen_ks,
                chunk.cu_seqlen_ke,
            )
            topk_indices = topk_indices_buffer[
                chunk.token_start : chunk.token_end, :topk_tokens
            ]

            num_rows = logits.shape[0]

            torch.ops._C.top_k_per_row_prefill(
                logits,
                chunk.cu_seqlen_ks,
                chunk.cu_seqlen_ke,
                topk_indices,
                num_rows,
                logits.stride(0),
                logits.stride(1),
                topk_tokens,
            )

    if has_decode:
        decode_metadata = layer_attn_metadata.decode
        assert decode_metadata is not None
        # kv_cache size requirement [num_block, block_size, n_head, head_dim],
        # we only have [num_block, block_size, head_dim],
        kv_cache = kv_cache.unsqueeze(-2)
        decode_lens = decode_metadata.decode_lens
        if decode_metadata.requires_padding:
            # pad in edge case where we have short chunked prefill length <
            # decode_threshold since we unstrictly split
            # prefill and decode by decode_threshold
            # (currently set to 1 + speculative tokens)
            padded_q_fp8_decode_tokens = pack_seq_triton(
                q_fp8[:num_decode_tokens], decode_lens
            )
        else:
            padded_q_fp8_decode_tokens = q_fp8[:num_decode_tokens].reshape(
                decode_lens.shape[0], -1, *q_fp8.shape[1:]
            )
        # TODO: move and optimize below logic with triton kernels
        batch_size = padded_q_fp8_decode_tokens.shape[0]
        next_n = padded_q_fp8_decode_tokens.shape[1]
        assert batch_size == decode_metadata.seq_lens.shape[0]
        num_padded_tokens = batch_size * next_n

        logits = rocm_fp8_paged_mqa_logits(
            padded_q_fp8_decode_tokens,
            kv_cache,
            weights[:num_padded_tokens],
            decode_metadata.seq_lens,
            decode_metadata.block_table,
            decode_metadata.schedule_metadata,
            max_model_len=max_model_len,
        )

        topk_indices = topk_indices_buffer[:num_padded_tokens, :topk_tokens]
        num_rows = logits.shape[0]

        torch.ops._C.top_k_per_row_decode(
            logits,
            next_n,
            decode_metadata.seq_lens,
            topk_indices,
            num_rows,
            logits.stride(0),
            logits.stride(1),
            topk_tokens,
        )

        if decode_metadata.requires_padding:
            # if padded, we need to unpack
            # the topk indices removing padded tokens
            topk_indices = unpack_seq_triton(
                topk_indices.reshape(batch_size, next_n, topk_indices.shape[-1]),
                decode_lens,
            )
            topk_indices_buffer[:num_decode_tokens, : topk_indices.shape[-1]] = (
                topk_indices
            )

    return topk_indices_buffer


def _decode_e8m0_scales(scale: torch.Tensor) -> torch.Tensor:
    if scale.dtype == torch.float8_e8m0fnu:
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            _upcast_e8m0_to_fp32,
        )

        return _upcast_e8m0_to_fp32(scale).contiguous()
    return scale.to(torch.float32)


def _expand_2d_block_scales(
    scale: torch.Tensor,
    rows: int,
    cols: int,
) -> torch.Tensor:
    scale = _decode_e8m0_scales(scale)
    row_blocks, col_blocks = scale.shape[-2:]
    row_block = math.ceil(rows / row_blocks)
    col_block = math.ceil(cols / col_blocks)
    scale = torch.repeat_interleave(scale, row_block, dim=-2)[..., :rows, :]
    scale = torch.repeat_interleave(scale, col_block, dim=-1)[..., :, :cols]
    return scale


def _apply_gptj_inv_rope_ref(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rope_dim: int,
) -> torch.Tensor:
    if rope_dim == 0 or x.numel() == 0:
        return x
    half_rot = rope_dim // 2
    nope_dim = x.shape[-1] - rope_dim
    dtype = x.dtype
    x = x.to(torch.float32)
    cache = cos_sin_cache.index_select(0, positions.to(torch.long))
    cos = cache[:, :half_rot].to(torch.float32)
    sin = cache[:, half_rot : 2 * half_rot].to(torch.float32)
    view_shape = (positions.shape[0],) + (1,) * (x.dim() - 2) + (half_rot,)
    cos = cos.view(view_shape)
    sin = sin.view(view_shape)
    rope = x[..., nope_dim:]
    y_even = rope[..., 0::2]
    y_odd = rope[..., 1::2]
    rope_out = torch.stack(
        (y_even * cos + y_odd * sin, y_odd * cos - y_even * sin),
        dim=-1,
    ).flatten(-2)
    x = x.clone()
    x[..., nope_dim:] = rope_out
    return x.to(dtype)


def _apply_inv_rope_ref(
    rotary_emb: torch.nn.Module,
    x: torch.Tensor,
    positions: torch.Tensor,
    rope_dim: int,
) -> torch.Tensor:
    if hasattr(rotary_emb, "forward_native"):
        try:
            query, _ = rotary_emb.forward_native(
                positions,
                x.clone(),
                None,
                inverse=True,
            )
            return query
        except TypeError:
            pass
    return _apply_gptj_inv_rope_ref(x, positions, rotary_emb.cos_sin_cache, rope_dim)


def _get_cached_rocm_wo_a_weight(
    wo_a: torch.nn.Module,
    n_local_groups: int,
    o_lora_rank: int,
    hidden_dim: int,
) -> torch.Tensor:
    expected_shape = (n_local_groups, o_lora_rank, hidden_dim)
    cached = getattr(wo_a, "_vllm_rocm_bf16_weight", None)
    if (
        isinstance(cached, torch.Tensor)
        and cached.shape == expected_shape
        and cached.dtype == torch.bfloat16
        and cached.device == wo_a.weight.device
    ):
        return cached

    if hasattr(wo_a, "weight_scale_inv"):
        wo_a_weight = wo_a.weight.view(expected_shape).to(torch.float32)
        wo_a_scale = _expand_2d_block_scales(
            wo_a.weight_scale_inv.view(
                n_local_groups, -1, wo_a.weight_scale_inv.shape[-1]
            ),
            o_lora_rank,
            hidden_dim,
        )
        wo_a_weight = (wo_a_weight * wo_a_scale).to(torch.bfloat16)
    else:
        wo_a_weight = wo_a.weight.view(expected_shape).to(torch.bfloat16)

    object.__setattr__(wo_a, "_vllm_rocm_bf16_weight", wo_a_weight)
    return wo_a_weight


@triton.jit
def _dsv4_inv_rope_grouped_kernel(
    o_ptr,
    positions_ptr,
    cos_sin_cache_ptr,
    out_ptr,
    o_stride_t,
    o_stride_h,
    o_stride_d,
    cache_stride_pos,
    out_stride_t,
    out_stride_g,
    out_stride_d,
    num_heads: tl.constexpr,
    heads_per_group: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    dim_offsets = tl.arange(0, BLOCK_D)
    dim_mask = dim_offsets < HEAD_DIM
    rope_offsets = dim_offsets - NOPE_DIM
    rope_mask = dim_mask & (dim_offsets >= NOPE_DIM)
    rope_pair = rope_offsets // 2
    pair_even = NOPE_DIM + rope_pair * 2
    pair_odd = pair_even + 1

    row_ptr = o_ptr + token_idx * o_stride_t + head_idx * o_stride_h
    vals = tl.load(
        row_ptr + dim_offsets * o_stride_d,
        mask=dim_mask,
        other=0.0,
    )
    even_vals = tl.load(
        row_ptr + pair_even * o_stride_d,
        mask=rope_mask,
        other=0.0,
    ).to(tl.float32)
    odd_vals = tl.load(
        row_ptr + pair_odd * o_stride_d,
        mask=rope_mask,
        other=0.0,
    ).to(tl.float32)

    pos = tl.load(positions_ptr + token_idx)
    cos = tl.load(
        cos_sin_cache_ptr + pos * cache_stride_pos + rope_pair,
        mask=rope_mask,
        other=1.0,
    )
    sin = tl.load(
        cos_sin_cache_ptr + pos * cache_stride_pos + ROPE_DIM // 2 + rope_pair,
        mask=rope_mask,
        other=0.0,
    )
    even_out = even_vals * cos + odd_vals * sin
    odd_out = odd_vals * cos - even_vals * sin
    rope_out = tl.where(rope_offsets % 2 == 0, even_out, odd_out)
    out_vals = tl.where(rope_mask, rope_out, vals)

    group_idx = head_idx // heads_per_group
    head_in_group = head_idx - group_idx * heads_per_group
    grouped_dim = head_in_group * HEAD_DIM + dim_offsets
    tl.store(
        out_ptr
        + token_idx * out_stride_t
        + group_idx * out_stride_g
        + grouped_dim * out_stride_d,
        out_vals,
        mask=(head_idx < num_heads) & dim_mask,
    )


def _dsv4_inv_rope_grouped(
    rotary_emb: torch.nn.Module,
    o: torch.Tensor,
    positions: torch.Tensor,
    rope_head_dim: int,
    n_local_groups: int,
) -> torch.Tensor:
    num_tokens, num_heads, head_dim = o.shape
    assert rope_head_dim == _DSV4_SPARSE_ROPE_DIM
    assert head_dim == _DSV4_SPARSE_NOPE_DIM + _DSV4_SPARSE_ROPE_DIM
    assert num_heads % n_local_groups == 0
    assert not getattr(rotary_emb, "is_neox_style", False)
    cos_sin_cache = rotary_emb.cos_sin_cache
    assert cos_sin_cache.dtype == torch.float32
    heads_per_group = num_heads // n_local_groups
    out = torch.empty(
        (num_tokens, n_local_groups, heads_per_group * head_dim),
        dtype=torch.bfloat16,
        device=o.device,
    )
    _dsv4_inv_rope_grouped_kernel[(num_tokens, num_heads)](
        o,
        positions,
        cos_sin_cache,
        out,
        o.stride(0),
        o.stride(1),
        o.stride(2),
        cos_sin_cache.stride(0),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        num_heads,
        heads_per_group,
        HEAD_DIM=head_dim,
        NOPE_DIM=_DSV4_SPARSE_NOPE_DIM,
        ROPE_DIM=rope_head_dim,
        BLOCK_D=triton.next_power_of_2(head_dim),
        num_warps=8,
    )
    return out


def rocm_inv_rope_einsum(
    rotary_emb: torch.nn.Module,
    o: torch.Tensor,
    positions: torch.Tensor,
    rope_head_dim: int,
    n_local_groups: int,
    o_lora_rank: int,
    wo_a: torch.nn.Module,
) -> torch.Tensor:
    """Reference inverse-RoPE + cached BF16 WO_A einsum path used on ROCm."""
    if (
        o.shape[-1] == _DSV4_SPARSE_NOPE_DIM + _DSV4_SPARSE_ROPE_DIM
        and rope_head_dim == _DSV4_SPARSE_ROPE_DIM
        and o.shape[1] % n_local_groups == 0
        and hasattr(rotary_emb, "cos_sin_cache")
        and rotary_emb.cos_sin_cache.dtype == torch.float32
        and not getattr(rotary_emb, "is_neox_style", False)
    ):
        o_ref = _dsv4_inv_rope_grouped(
            rotary_emb,
            o,
            positions,
            rope_head_dim,
            n_local_groups,
        )
    else:
        o_ref = _apply_inv_rope_ref(rotary_emb, o, positions, rope_head_dim).to(
            torch.bfloat16
        )
        o_ref = o_ref.view(o.shape[0], n_local_groups, -1)

    hidden_dim = o_ref.shape[-1]
    wo_a_weight = _get_cached_rocm_wo_a_weight(
        wo_a, n_local_groups, o_lora_rank, hidden_dim
    )
    return torch.bmm(o_ref.transpose(0, 1), wo_a_weight.transpose(1, 2)).transpose(0, 1)


_DSV4_SPARSE_NOPE_DIM = 448
_DSV4_SPARSE_ROPE_DIM = 64
_LOG2E = 1.4426950408889634


def _validate_dsv4_sparse_dims(
    head_dim: int,
    nope_head_dim: int,
    rope_head_dim: int,
    op_name: str,
) -> None:
    assert head_dim == nope_head_dim + rope_head_dim, (
        f"{op_name} expected head_dim={nope_head_dim + rope_head_dim}, got {head_dim}"
    )
    assert (
        nope_head_dim == _DSV4_SPARSE_NOPE_DIM
        and rope_head_dim == _DSV4_SPARSE_ROPE_DIM
    ), (
        f"{op_name} expects {_DSV4_SPARSE_NOPE_DIM} NoPE dims and "
        f"{_DSV4_SPARSE_ROPE_DIM} RoPE dims"
    )


@triton.jit
def _pack_dense_prefix_to_ragged_kernel(
    indices_ptr,
    lengths_ptr,
    indptr_ptr,
    out_ptr,
    indices_stride0,
    num_rows_limit,
    row_width,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    row_len = tl.load(lengths_ptr + row_idx)
    if block_idx * BLOCK_SIZE >= row_len:
        return

    mask = offsets < row_len
    safe_offsets = tl.where(offsets < row_width, offsets, 0)
    vals = tl.load(
        indices_ptr + row_idx * indices_stride0 + safe_offsets,
        mask=mask & (offsets < row_width),
        other=-1,
    ).to(tl.int32)
    if num_rows_limit >= 0:
        vals = tl.where((vals >= 0) & (vals < num_rows_limit), vals, -1)

    out_start = tl.load(indptr_ptr + row_idx)
    tl.store(out_ptr + out_start + offsets, vals, mask=mask)


def build_ragged_indices_from_dense(
    indices: torch.Tensor,
    lengths: torch.Tensor,
    num_rows: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    indices = indices.reshape(indices.shape[0], -1)
    lengths = lengths.to(device=indices.device, dtype=torch.int32).reshape(-1)
    assert lengths.numel() == indices.shape[0], (
        f"Expected one length per row, got {lengths.shape} for indices {indices.shape}"
    )

    max_width = indices.shape[1] if indices.ndim == 2 else 0
    lengths = lengths.clamp(min=0, max=max_width).contiguous()

    indptr = torch.zeros(indices.shape[0] + 1, dtype=torch.int32, device=indices.device)
    torch.cumsum(lengths, dim=0, out=indptr[1:])

    if indices.numel() == 0:
        flat = torch.empty(0, dtype=torch.int32, device=indices.device)
    else:
        flat = torch.empty(
            indices.shape[0] * max_width,
            dtype=torch.int32,
            device=indices.device,
        )
        if flat.numel() > 0:
            block_size = 128
            _pack_dense_prefix_to_ragged_kernel[
                (indices.shape[0], triton.cdiv(max_width, block_size))
            ](
                indices,
                lengths,
                indptr,
                flat,
                indices.stride(0),
                int(num_rows),
                max_width,
                BLOCK_SIZE=block_size,
            )

    return flat, indptr


def _as_int32_contiguous_1d(x: torch.Tensor) -> torch.Tensor:
    if x.dtype == torch.int32 and x.ndim == 1 and x.is_contiguous():
        return x
    return x.to(torch.int32).contiguous()


@triton.jit
def _sparse_attn_prefill_ragged_kernel(
    q_ptr,
    kv_ptr,
    kv_indices_ptr,
    kv_indptr_ptr,
    attn_sink_ptr,
    out_ptr,
    q_stride_t,
    q_stride_h,
    q_stride_d,
    kv_stride_n,
    kv_stride_d,
    out_stride_t,
    out_stride_h,
    out_stride_d,
    num_heads,
    head_dim,
    num_kv,
    scale,
    HAS_ATTN_SINK: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    query_idx = tl.program_id(0)
    pid_h = tl.program_id(1)

    head_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    dim_offsets = tl.arange(0, BLOCK_D)
    head_mask = head_offsets < num_heads
    dim_mask = dim_offsets < head_dim

    q = tl.load(
        q_ptr
        + query_idx * q_stride_t
        + head_offsets[:, None] * q_stride_h
        + dim_offsets[None, :] * q_stride_d,
        mask=head_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )

    neg_large = -3.4028234663852886e38
    m_i = tl.full((BLOCK_H,), neg_large, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, BLOCK_D), dtype=tl.float32)

    kv_start = tl.load(kv_indptr_ptr + query_idx)
    kv_end = tl.load(kv_indptr_ptr + query_idx + 1)
    kv_len = kv_end - kv_start

    k_offsets = tl.arange(0, BLOCK_K)
    for k_start in tl.range(0, kv_len, BLOCK_K):
        k_pos = k_start + k_offsets
        in_range = k_pos < kv_len
        slot = tl.load(kv_indices_ptr + kv_start + k_pos, mask=in_range, other=-1)
        valid = in_range & (slot >= 0) & (slot < num_kv)
        safe_slot = tl.where(valid, slot, 0)

        kv = tl.load(
            kv_ptr
            + safe_slot[:, None] * kv_stride_n
            + dim_offsets[None, :] * kv_stride_d,
            mask=valid[:, None] & dim_mask[None, :],
            other=0.0,
        )
        kv = tl.where(valid[:, None] & dim_mask[None, :], kv, 0.0)

        scores = tl.dot(q, tl.trans(kv)) * scale
        scores = tl.where(head_mask[:, None] & valid[None, :], scores, neg_large)

        m_block = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(scores - m_new[:, None])
        p = tl.where(head_mask[:, None] & valid[None, :], p, 0.0)
        l_new = l_i * alpha + tl.sum(p, axis=1)

        acc = acc * alpha[:, None] + tl.dot(p.to(kv.dtype), kv)
        m_i = m_new
        l_i = l_new

    if HAS_ATTN_SINK:
        sink = (
            tl.load(attn_sink_ptr + head_offsets, mask=head_mask, other=neg_large).to(
                tl.float32
            )
            * 1.4426950408889634
        )
        m_final = tl.maximum(m_i, sink)
        alpha = tl.exp2(m_i - m_final)
        l_final = l_i * alpha + tl.exp2(sink - m_final)
        denom = tl.maximum(l_final, 1.0e-30)
        out = tl.where(
            l_final[:, None] > 0.0,
            (acc * alpha[:, None]) / denom[:, None],
            0.0,
        )
    else:
        denom = tl.maximum(l_i, 1.0e-30)
        out = tl.where(l_i[:, None] > 0.0, acc / denom[:, None], 0.0)

    tl.store(
        out_ptr
        + query_idx * out_stride_t
        + head_offsets[:, None] * out_stride_h
        + dim_offsets[None, :] * out_stride_d,
        out,
        mask=head_mask[:, None] & dim_mask[None, :],
    )


@triton.jit
def _sparse_attn_decode_ragged_kernel(
    q_ptr,
    main_cache_ptr,
    main_indices_ptr,
    main_indptr_ptr,
    extra_cache_ptr,
    extra_indices_ptr,
    extra_indptr_ptr,
    attn_sink_ptr,
    out_ptr,
    q_stride0,
    q_stride1,
    out_stride0,
    out_stride1,
    main_cache_stride0,
    extra_cache_stride0,
    main_num_rows,
    extra_num_rows,
    main_block_size,
    extra_block_size,
    scale,
    num_heads,
    HAS_ATTN_SINK: tl.constexpr,
    HAS_EXTRA: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    NOPE_BLOCK: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    IS_FNUZ: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    query_idx = tl.program_id(0)
    pid_h = tl.program_id(1)

    head_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    head_mask = head_offsets < num_heads
    nope_offsets = tl.arange(0, NOPE_BLOCK)
    nope_mask = nope_offsets < NOPE_DIM
    rope_offsets = tl.arange(0, ROPE_DIM)

    q_row_ptr = q_ptr + query_idx * q_stride0 + head_offsets[:, None] * q_stride1
    q_nope = tl.load(
        q_row_ptr + nope_offsets[None, :],
        mask=head_mask[:, None] & nope_mask[None, :],
        other=0.0,
    )
    q_rope = tl.load(
        q_row_ptr + NOPE_DIM + rope_offsets[None, :],
        mask=head_mask[:, None],
        other=0.0,
    )

    neg_large = -3.4028234663852886e38
    m_i = tl.full((BLOCK_H,), neg_large, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc_nope = tl.zeros((BLOCK_H, NOPE_BLOCK), dtype=tl.float32)
    acc_rope = tl.zeros((BLOCK_H, ROPE_DIM), dtype=tl.float32)
    k_offsets = tl.arange(0, BLOCK_K)

    main_start = tl.load(main_indptr_ptr + query_idx)
    main_end = tl.load(main_indptr_ptr + query_idx + 1)
    main_len = main_end - main_start

    zero_nope = tl.zeros((BLOCK_K, NOPE_BLOCK), dtype=tl.bfloat16)
    zero_rope = tl.zeros((BLOCK_K, ROPE_DIM), dtype=tl.bfloat16)

    for k_start in tl.range(0, main_len, BLOCK_K):
        k_pos = k_start + k_offsets
        in_range = k_pos < main_len
        slot = tl.load(main_indices_ptr + main_start + k_pos, mask=in_range, other=-1)
        valid = in_range & (slot >= 0) & (slot < main_num_rows)
        safe_slot = tl.where(valid, slot, 0)

        block_idx = safe_slot // main_block_size
        pos_in_block = safe_slot % main_block_size
        cache_block_ptr = main_cache_ptr + block_idx.to(tl.int64) * main_cache_stride0
        token_data_ptr = cache_block_ptr + pos_in_block * 576
        token_scale_ptr = cache_block_ptr + main_block_size * 576 + pos_in_block * 8

        x_uint8 = tl.load(
            token_data_ptr[:, None] + nope_offsets[None, :],
            mask=valid[:, None] & nope_mask[None, :],
            other=0,
        )
        if IS_FNUZ:
            x_fp8 = x_uint8.to(tl.float8e4b15, bitcast=True)
        else:
            x_fp8 = x_uint8.to(tl.float8e4nv, bitcast=True)
        encoded_scales = tl.load(
            token_scale_ptr[:, None] + nope_offsets[None, :] // 64,
            mask=valid[:, None] & nope_mask[None, :],
            other=127,
        )
        scales = tl.exp2(encoded_scales.to(tl.float32) - 127.0)
        k_nope = x_fp8.to(tl.bfloat16) * scales.to(tl.bfloat16)
        k_nope = tl.where(valid[:, None] & nope_mask[None, :], k_nope, zero_nope)
        k_nope = tl.where(k_nope == k_nope, k_nope, zero_nope)

        rope_ptr = (token_data_ptr + NOPE_DIM).to(tl.pointer_type(tl.bfloat16))
        k_rope = tl.load(
            rope_ptr[:, None] + rope_offsets[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        k_rope = tl.where(valid[:, None], k_rope, zero_rope)
        k_rope = tl.where(k_rope == k_rope, k_rope, zero_rope)

        scores = tl.dot(q_nope, tl.trans(k_nope)) + tl.dot(q_rope, tl.trans(k_rope))
        scores *= scale
        scores = tl.where(head_mask[:, None] & valid[None, :], scores, neg_large)

        m_block = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(scores - m_new[:, None])
        p = tl.where(head_mask[:, None] & valid[None, :], p, 0.0)
        l_new = l_i * alpha + tl.sum(p, axis=1)

        acc_nope = acc_nope * alpha[:, None] + tl.dot(p.to(k_nope.dtype), k_nope)
        acc_rope = acc_rope * alpha[:, None] + tl.dot(p.to(k_rope.dtype), k_rope)
        m_i = m_new
        l_i = l_new

    if HAS_EXTRA:
        extra_start = tl.load(extra_indptr_ptr + query_idx)
        extra_end = tl.load(extra_indptr_ptr + query_idx + 1)
        extra_len = extra_end - extra_start

        for k_start in tl.range(0, extra_len, BLOCK_K):
            k_pos = k_start + k_offsets
            in_range = k_pos < extra_len
            slot = tl.load(
                extra_indices_ptr + extra_start + k_pos, mask=in_range, other=-1
            )
            valid = in_range & (slot >= 0) & (slot < extra_num_rows)
            safe_slot = tl.where(valid, slot, 0)

            block_idx = safe_slot // extra_block_size
            pos_in_block = safe_slot % extra_block_size
            cache_block_ptr = (
                extra_cache_ptr + block_idx.to(tl.int64) * extra_cache_stride0
            )
            token_data_ptr = cache_block_ptr + pos_in_block * 576
            token_scale_ptr = (
                cache_block_ptr + extra_block_size * 576 + pos_in_block * 8
            )

            x_uint8 = tl.load(
                token_data_ptr[:, None] + nope_offsets[None, :],
                mask=valid[:, None] & nope_mask[None, :],
                other=0,
            )
            if IS_FNUZ:
                x_fp8 = x_uint8.to(tl.float8e4b15, bitcast=True)
            else:
                x_fp8 = x_uint8.to(tl.float8e4nv, bitcast=True)
            encoded_scales = tl.load(
                token_scale_ptr[:, None] + nope_offsets[None, :] // 64,
                mask=valid[:, None] & nope_mask[None, :],
                other=127,
            )
            scales = tl.exp2(encoded_scales.to(tl.float32) - 127.0)
            k_nope = x_fp8.to(tl.bfloat16) * scales.to(tl.bfloat16)
            k_nope = tl.where(valid[:, None] & nope_mask[None, :], k_nope, zero_nope)
            k_nope = tl.where(k_nope == k_nope, k_nope, zero_nope)

            rope_ptr = (token_data_ptr + NOPE_DIM).to(tl.pointer_type(tl.bfloat16))
            k_rope = tl.load(
                rope_ptr[:, None] + rope_offsets[None, :],
                mask=valid[:, None],
                other=0.0,
            )
            k_rope = tl.where(valid[:, None], k_rope, zero_rope)
            k_rope = tl.where(k_rope == k_rope, k_rope, zero_rope)

            scores = tl.dot(q_nope, tl.trans(k_nope)) + tl.dot(
                q_rope,
                tl.trans(k_rope),
            )
            scores *= scale
            scores = tl.where(head_mask[:, None] & valid[None, :], scores, neg_large)

            m_block = tl.max(scores, axis=1)
            m_new = tl.maximum(m_i, m_block)
            alpha = tl.exp2(m_i - m_new)
            p = tl.exp2(scores - m_new[:, None])
            p = tl.where(head_mask[:, None] & valid[None, :], p, 0.0)
            l_new = l_i * alpha + tl.sum(p, axis=1)

            acc_nope = acc_nope * alpha[:, None] + tl.dot(p.to(k_nope.dtype), k_nope)
            acc_rope = acc_rope * alpha[:, None] + tl.dot(p.to(k_rope.dtype), k_rope)
            m_i = m_new
            l_i = l_new

    if HAS_ATTN_SINK:
        sink = (
            tl.load(attn_sink_ptr + head_offsets, mask=head_mask, other=neg_large).to(
                tl.float32
            )
            * 1.4426950408889634
        )
        m_final = tl.maximum(m_i, sink)
        alpha = tl.exp2(m_i - m_final)
        l_final = l_i * alpha + tl.exp2(sink - m_final)
        denom = tl.maximum(l_final, 1.0e-30)
        out_nope = tl.where(
            l_final[:, None] > 0.0,
            (acc_nope * alpha[:, None]) / denom[:, None],
            0.0,
        )
        out_rope = tl.where(
            l_final[:, None] > 0.0,
            (acc_rope * alpha[:, None]) / denom[:, None],
            0.0,
        )
    else:
        denom = tl.maximum(l_i, 1.0e-30)
        out_nope = tl.where(l_i[:, None] > 0.0, acc_nope / denom[:, None], 0.0)
        out_rope = tl.where(l_i[:, None] > 0.0, acc_rope / denom[:, None], 0.0)

    out_row_ptr = (
        out_ptr + query_idx * out_stride0 + head_offsets[:, None] * out_stride1
    )
    tl.store(
        out_row_ptr + nope_offsets[None, :],
        out_nope,
        mask=head_mask[:, None] & nope_mask[None, :],
    )
    tl.store(
        out_row_ptr + NOPE_DIM + rope_offsets[None, :],
        out_rope,
        mask=head_mask[:, None],
    )


@triton.jit
def _sparse_attn_decode_ragged_split_kernel(
    q_ptr,
    main_cache_ptr,
    main_indices_ptr,
    main_indptr_ptr,
    extra_cache_ptr,
    extra_indices_ptr,
    extra_indptr_ptr,
    m_partial_ptr,
    l_partial_ptr,
    acc_partial_ptr,
    q_stride0,
    q_stride1,
    main_cache_stride0,
    extra_cache_stride0,
    mp_stride_t,
    mp_stride_k,
    mp_stride_h,
    lp_stride_t,
    lp_stride_k,
    lp_stride_h,
    ap_stride_t,
    ap_stride_k,
    ap_stride_h,
    ap_stride_d,
    main_num_rows,
    extra_num_rows,
    main_block_size,
    extra_block_size,
    scale,
    num_heads,
    KV_SPLITS: tl.constexpr,
    HAS_EXTRA: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    NOPE_BLOCK: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    IS_FNUZ: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    query_idx = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_k = tl.program_id(2)

    head_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    head_mask = head_offsets < num_heads
    nope_offsets = tl.arange(0, NOPE_BLOCK)
    nope_mask = nope_offsets < NOPE_DIM
    rope_offsets = tl.arange(0, ROPE_DIM)

    q_row_ptr = q_ptr + query_idx * q_stride0 + head_offsets[:, None] * q_stride1
    q_nope = tl.load(
        q_row_ptr + nope_offsets[None, :],
        mask=head_mask[:, None] & nope_mask[None, :],
        other=0.0,
    )
    q_rope = tl.load(
        q_row_ptr + NOPE_DIM + rope_offsets[None, :],
        mask=head_mask[:, None],
        other=0.0,
    )

    main_start = tl.load(main_indptr_ptr + query_idx)
    main_end = tl.load(main_indptr_ptr + query_idx + 1)
    main_len = main_end - main_start
    extra_start = tl.load(extra_indptr_ptr + query_idx)
    extra_end = tl.load(extra_indptr_ptr + query_idx + 1)
    extra_len = tl.where(HAS_EXTRA, extra_end - extra_start, 0)
    total_len = main_len + extra_len

    tiles_per_segment = tl.cdiv(total_len, KV_SPLITS * BLOCK_K)
    if pid_k * tiles_per_segment * BLOCK_K >= total_len:
        return
    num_tiles = tl.cdiv(total_len, BLOCK_K)
    tile_start = pid_k * tiles_per_segment
    tile_end = tl.minimum((pid_k + 1) * tiles_per_segment, num_tiles)

    neg_large = -3.4028234663852886e38
    m_i = tl.full((BLOCK_H,), neg_large, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc_nope = tl.zeros((BLOCK_H, NOPE_BLOCK), dtype=tl.float32)
    acc_rope = tl.zeros((BLOCK_H, ROPE_DIM), dtype=tl.float32)
    k_offsets = tl.arange(0, BLOCK_K)
    zero_nope = tl.zeros((BLOCK_K, NOPE_BLOCK), dtype=tl.bfloat16)
    zero_rope = tl.zeros((BLOCK_K, ROPE_DIM), dtype=tl.bfloat16)

    for tile_idx in tl.range(tile_start, tile_end):
        k_pos = tile_idx * BLOCK_K + k_offsets
        in_range = k_pos < total_len
        use_extra = k_pos >= main_len
        main_pos = k_pos
        extra_pos = k_pos - main_len

        main_slot = tl.load(
            main_indices_ptr + main_start + main_pos,
            mask=in_range & ~use_extra,
            other=-1,
        )
        extra_slot = tl.load(
            extra_indices_ptr + extra_start + extra_pos,
            mask=in_range & use_extra,
            other=-1,
        )

        main_valid = (
            in_range & ~use_extra & (main_slot >= 0) & (main_slot < main_num_rows)
        )
        extra_valid = (
            in_range & use_extra & (extra_slot >= 0) & (extra_slot < extra_num_rows)
        )

        main_safe_slot = tl.where(main_valid, main_slot, 0)
        main_block_idx = main_safe_slot // main_block_size
        main_pos_in_block = main_safe_slot % main_block_size
        main_cache_block_ptr = (
            main_cache_ptr + main_block_idx.to(tl.int64) * main_cache_stride0
        )
        main_token_data_ptr = main_cache_block_ptr + main_pos_in_block * 576
        main_token_scale_ptr = (
            main_cache_block_ptr + main_block_size * 576 + main_pos_in_block * 8
        )

        main_x_uint8 = tl.load(
            main_token_data_ptr[:, None] + nope_offsets[None, :],
            mask=main_valid[:, None] & nope_mask[None, :],
            other=0,
        )
        if IS_FNUZ:
            main_x_fp8 = main_x_uint8.to(tl.float8e4b15, bitcast=True)
        else:
            main_x_fp8 = main_x_uint8.to(tl.float8e4nv, bitcast=True)
        main_encoded_scales = tl.load(
            main_token_scale_ptr[:, None] + nope_offsets[None, :] // 64,
            mask=main_valid[:, None] & nope_mask[None, :],
            other=127,
        )
        main_scales = tl.exp2(main_encoded_scales.to(tl.float32) - 127.0)
        main_k_nope = main_x_fp8.to(tl.bfloat16) * main_scales.to(tl.bfloat16)
        main_k_nope = tl.where(
            main_valid[:, None] & nope_mask[None, :], main_k_nope, zero_nope
        )
        main_k_nope = tl.where(main_k_nope == main_k_nope, main_k_nope, zero_nope)

        main_rope_ptr = (main_token_data_ptr + NOPE_DIM).to(
            tl.pointer_type(tl.bfloat16)
        )
        main_k_rope = tl.load(
            main_rope_ptr[:, None] + rope_offsets[None, :],
            mask=main_valid[:, None],
            other=0.0,
        )
        main_k_rope = tl.where(main_valid[:, None], main_k_rope, zero_rope)
        main_k_rope = tl.where(main_k_rope == main_k_rope, main_k_rope, zero_rope)

        extra_safe_slot = tl.where(extra_valid, extra_slot, 0)
        extra_block_idx = extra_safe_slot // extra_block_size
        extra_pos_in_block = extra_safe_slot % extra_block_size
        extra_cache_block_ptr = (
            extra_cache_ptr + extra_block_idx.to(tl.int64) * extra_cache_stride0
        )
        extra_token_data_ptr = extra_cache_block_ptr + extra_pos_in_block * 576
        extra_token_scale_ptr = (
            extra_cache_block_ptr + extra_block_size * 576 + extra_pos_in_block * 8
        )

        extra_x_uint8 = tl.load(
            extra_token_data_ptr[:, None] + nope_offsets[None, :],
            mask=extra_valid[:, None] & nope_mask[None, :],
            other=0,
        )
        if IS_FNUZ:
            extra_x_fp8 = extra_x_uint8.to(tl.float8e4b15, bitcast=True)
        else:
            extra_x_fp8 = extra_x_uint8.to(tl.float8e4nv, bitcast=True)
        extra_encoded_scales = tl.load(
            extra_token_scale_ptr[:, None] + nope_offsets[None, :] // 64,
            mask=extra_valid[:, None] & nope_mask[None, :],
            other=127,
        )
        extra_scales = tl.exp2(extra_encoded_scales.to(tl.float32) - 127.0)
        extra_k_nope = extra_x_fp8.to(tl.bfloat16) * extra_scales.to(tl.bfloat16)
        extra_k_nope = tl.where(
            extra_valid[:, None] & nope_mask[None, :], extra_k_nope, zero_nope
        )
        extra_k_nope = tl.where(extra_k_nope == extra_k_nope, extra_k_nope, zero_nope)

        extra_rope_ptr = (extra_token_data_ptr + NOPE_DIM).to(
            tl.pointer_type(tl.bfloat16)
        )
        extra_k_rope = tl.load(
            extra_rope_ptr[:, None] + rope_offsets[None, :],
            mask=extra_valid[:, None],
            other=0.0,
        )
        extra_k_rope = tl.where(extra_valid[:, None], extra_k_rope, zero_rope)
        extra_k_rope = tl.where(extra_k_rope == extra_k_rope, extra_k_rope, zero_rope)

        k_nope = main_k_nope + extra_k_nope
        k_rope = main_k_rope + extra_k_rope
        valid = main_valid | extra_valid

        scores = tl.dot(q_nope, tl.trans(k_nope)) + tl.dot(q_rope, tl.trans(k_rope))
        scores *= scale
        scores = tl.where(head_mask[:, None] & valid[None, :], scores, neg_large)

        m_block = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(scores - m_new[:, None])
        p = tl.where(head_mask[:, None] & valid[None, :], p, 0.0)
        l_new = l_i * alpha + tl.sum(p, axis=1)

        acc_nope = acc_nope * alpha[:, None] + tl.dot(p.to(k_nope.dtype), k_nope)
        acc_rope = acc_rope * alpha[:, None] + tl.dot(p.to(k_rope.dtype), k_rope)
        m_i = m_new
        l_i = l_new

    m_base = query_idx * mp_stride_t + pid_k * mp_stride_k
    tl.store(m_partial_ptr + m_base + head_offsets * mp_stride_h, m_i, mask=head_mask)
    l_base = query_idx * lp_stride_t + pid_k * lp_stride_k
    tl.store(l_partial_ptr + l_base + head_offsets * lp_stride_h, l_i, mask=head_mask)
    acc_base = query_idx * ap_stride_t + pid_k * ap_stride_k
    tl.store(
        acc_partial_ptr
        + acc_base
        + head_offsets[:, None] * ap_stride_h
        + nope_offsets[None, :] * ap_stride_d,
        acc_nope,
        mask=head_mask[:, None] & nope_mask[None, :],
    )
    tl.store(
        acc_partial_ptr
        + acc_base
        + head_offsets[:, None] * ap_stride_h
        + (NOPE_DIM + rope_offsets[None, :]) * ap_stride_d,
        acc_rope,
        mask=head_mask[:, None],
    )


@triton.jit
def _sparse_attn_decode_ragged_reduce_kernel(
    m_partial_ptr,
    l_partial_ptr,
    acc_partial_ptr,
    main_indptr_ptr,
    extra_indptr_ptr,
    attn_sink_ptr,
    out_ptr,
    mp_stride_t,
    mp_stride_k,
    mp_stride_h,
    lp_stride_t,
    lp_stride_k,
    lp_stride_h,
    ap_stride_t,
    ap_stride_k,
    ap_stride_h,
    ap_stride_d,
    out_stride0,
    out_stride1,
    KV_SPLITS: tl.constexpr,
    BLOCK_K_DECODE: tl.constexpr,
    HAS_ATTN_SINK: tl.constexpr,
    HAS_EXTRA: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    query_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    d_offsets = tl.arange(0, BLOCK_D)
    split_offsets = tl.arange(0, KV_SPLITS)
    d_mask = d_offsets < HEAD_DIM

    main_start = tl.load(main_indptr_ptr + query_idx)
    main_end = tl.load(main_indptr_ptr + query_idx + 1)
    main_len = main_end - main_start
    extra_start = tl.load(extra_indptr_ptr + query_idx)
    extra_end = tl.load(extra_indptr_ptr + query_idx + 1)
    extra_len = tl.where(HAS_EXTRA, extra_end - extra_start, 0)
    total_len = main_len + extra_len
    tiles_per_segment = tl.cdiv(total_len, KV_SPLITS * BLOCK_K_DECODE)
    act_num_segments = tl.cdiv(
        total_len, tl.maximum(tiles_per_segment, 1) * BLOCK_K_DECODE
    )
    split_mask = split_offsets < act_num_segments

    neg_large = -3.4028234663852886e38
    m_p = tl.load(
        m_partial_ptr
        + query_idx * mp_stride_t
        + split_offsets * mp_stride_k
        + head_idx * mp_stride_h,
        mask=split_mask,
        other=neg_large,
    )
    l_p = tl.load(
        l_partial_ptr
        + query_idx * lp_stride_t
        + split_offsets * lp_stride_k
        + head_idx * lp_stride_h,
        mask=split_mask,
        other=0.0,
    )
    acc_p = tl.load(
        acc_partial_ptr
        + query_idx * ap_stride_t
        + split_offsets[:, None] * ap_stride_k
        + head_idx * ap_stride_h
        + d_offsets[None, :] * ap_stride_d,
        mask=split_mask[:, None] & d_mask[None, :],
        other=0.0,
    )

    m_max = tl.max(m_p, axis=0)
    alpha_split = tl.exp2(m_p - m_max)
    l_combined = tl.sum(l_p * alpha_split, axis=0)
    acc_combined = tl.sum(acc_p * alpha_split[:, None], axis=0)

    if HAS_ATTN_SINK:
        sink = tl.load(attn_sink_ptr + head_idx).to(tl.float32) * 1.4426950408889634
        m_final = tl.maximum(m_max, sink)
        alpha_kv = tl.exp2(m_max - m_final)
        alpha_sink = tl.exp2(sink - m_final)
        l_final = l_combined * alpha_kv + alpha_sink
        acc_final = acc_combined * alpha_kv
    else:
        l_final = l_combined
        acc_final = acc_combined

    denom = tl.maximum(l_final, 1.0e-30)
    out = tl.where(l_final > 0.0, acc_final / denom, 0.0)
    tl.store(
        out_ptr + query_idx * out_stride0 + head_idx * out_stride1 + d_offsets,
        out,
        mask=d_mask,
    )


def _rocm_sparse_attn_prefill_ragged_triton(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    indptr: torch.Tensor,
    scale: float,
    attn_sink: torch.Tensor | None,
    nope_head_dim: int,
    rope_head_dim: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    assert q.ndim == 3, f"expected q=[sq,h,d], got {q.shape}"
    assert kv.ndim == 2, f"expected kv=[skv,d], got {kv.shape}"
    assert indices.ndim == 1, f"expected indices=[nnz], got {indices.shape}"
    assert indptr.ndim == 1, f"expected indptr=[sq+1], got {indptr.shape}"
    assert not q.is_cpu and not kv.is_cpu and not indices.is_cpu and not indptr.is_cpu

    indices = _as_int32_contiguous_1d(indices)
    indptr = _as_int32_contiguous_1d(indptr)
    has_attn_sink = attn_sink is not None
    if attn_sink is None:
        attn_sink = torch.empty(1, device=q.device, dtype=torch.float32)
    else:
        attn_sink = attn_sink.contiguous()

    num_queries, num_heads, head_dim = q.shape
    assert indptr.numel() == num_queries + 1, (
        f"expected indptr shape [{num_queries + 1}], got {indptr.shape}"
    )
    _validate_dsv4_sparse_dims(
        head_dim,
        nope_head_dim,
        rope_head_dim,
        "_rocm_sparse_attn_prefill_ragged_triton",
    )

    if out is None:
        out = torch.empty_like(q, dtype=torch.bfloat16)
    else:
        assert out.shape == q.shape, f"expected out shape {q.shape}, got {out.shape}"
        assert out.dtype == torch.bfloat16, f"expected bf16 out, got {out.dtype}"
        assert not out.is_cpu

    block_h = 16
    block_d = triton.next_power_of_2(head_dim)
    block_k = 32
    _sparse_attn_prefill_ragged_kernel[(num_queries, triton.cdiv(num_heads, block_h))](
        q,
        kv,
        indices,
        indptr,
        attn_sink,
        out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        kv.stride(0),
        kv.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        num_heads,
        head_dim,
        kv.shape[0],
        float(scale) * _LOG2E,
        HAS_ATTN_SINK=has_attn_sink,
        BLOCK_H=block_h,
        BLOCK_D=block_d,
        BLOCK_K=block_k,
        num_warps=8,
    )
    return out


def _rocm_sparse_attn_prefill_triton(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    scale: float,
    attn_sink: torch.Tensor | None,
    nope_head_dim: int,
    rope_head_dim: int,
    topk_length: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    ragged_indices, ragged_indptr = build_ragged_indices_from_dense(
        indices,
        topk_length
        if topk_length is not None
        else (indices >= 0).sum(dim=-1, dtype=torch.int32),
        num_rows=kv.shape[0],
    )
    return _rocm_sparse_attn_prefill_ragged_triton(
        q=q,
        kv=kv,
        indices=ragged_indices,
        indptr=ragged_indptr,
        scale=scale,
        attn_sink=attn_sink,
        nope_head_dim=nope_head_dim,
        rope_head_dim=rope_head_dim,
        out=out,
    )


def _rocm_sparse_attn_decode_ragged_triton(
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
    out: torch.Tensor | None = None,
    kv_splits: int = 1,
) -> torch.Tensor:
    assert q.ndim == 3, f"expected q=[b,h,d], got {q.shape}"
    assert main_cache.ndim == 3, (
        f"expected main_cache=[blocks,block,bytes], got {main_cache.shape}"
    )
    assert main_indices.ndim == 1, (
        f"expected main_indices=[nnz], got {main_indices.shape}"
    )
    assert main_indptr.ndim == 1, f"expected main_indptr=[b+1], got {main_indptr.shape}"
    assert (
        not q.is_cpu
        and not main_cache.is_cpu
        and not main_indices.is_cpu
        and not main_indptr.is_cpu
    )

    main_indices = _as_int32_contiguous_1d(main_indices)
    main_indptr = _as_int32_contiguous_1d(main_indptr)
    has_attn_sink = attn_sink is not None
    if attn_sink is None:
        attn_sink = torch.empty(1, device=q.device, dtype=torch.float32)
    else:
        attn_sink = attn_sink.contiguous()

    num_queries, num_heads, head_dim = q.shape
    assert main_indptr.numel() == num_queries + 1, (
        f"expected main_indptr shape [{num_queries + 1}], got {main_indptr.shape}"
    )
    _validate_dsv4_sparse_dims(
        head_dim,
        nope_head_dim,
        rope_head_dim,
        "_rocm_sparse_attn_decode_ragged_triton",
    )

    has_extra = (
        extra_cache is not None
        and extra_indices is not None
        and extra_indptr is not None
    )
    if has_extra:
        assert extra_cache is not None
        assert extra_indices is not None
        assert extra_indptr is not None
        assert extra_indices.ndim == 1, (
            f"expected extra_indices=[nnz], got {extra_indices.shape}"
        )
        assert extra_indptr.ndim == 1, (
            f"expected extra_indptr=[b+1], got {extra_indptr.shape}"
        )
        extra_indices = _as_int32_contiguous_1d(extra_indices)
        extra_indptr = _as_int32_contiguous_1d(extra_indptr)
        assert extra_indptr.numel() == num_queries + 1, (
            f"expected extra_indptr shape [{num_queries + 1}], got {extra_indptr.shape}"
        )
    else:
        extra_cache = main_cache
        extra_indices = main_indices
        extra_indptr = main_indptr

    if out is None:
        out = torch.empty_like(q, dtype=torch.bfloat16)
    else:
        assert out.shape == q.shape, f"expected out shape {q.shape}, got {out.shape}"
        assert out.dtype == torch.bfloat16, f"expected bf16 out, got {out.dtype}"
        assert not out.is_cpu

    block_h = 16
    block_k = 16 if head_dim >= 256 else 32
    if kv_splits > 1:
        head_blocks = triton.cdiv(num_heads, block_h)
        h_padded = head_blocks * block_h
        m_partial = torch.empty(
            (num_queries, kv_splits, h_padded),
            dtype=torch.float32,
            device=q.device,
        )
        l_partial = torch.empty_like(m_partial)
        acc_partial = torch.empty(
            (num_queries, kv_splits, h_padded, head_dim),
            dtype=torch.float32,
            device=q.device,
        )
        _sparse_attn_decode_ragged_split_kernel[(num_queries, head_blocks, kv_splits)](
            q,
            main_cache,
            main_indices,
            main_indptr,
            extra_cache,
            extra_indices,
            extra_indptr,
            m_partial,
            l_partial,
            acc_partial,
            q.stride(0),
            q.stride(1),
            main_cache.stride(0),
            extra_cache.stride(0),
            m_partial.stride(0),
            m_partial.stride(1),
            m_partial.stride(2),
            l_partial.stride(0),
            l_partial.stride(1),
            l_partial.stride(2),
            acc_partial.stride(0),
            acc_partial.stride(1),
            acc_partial.stride(2),
            acc_partial.stride(3),
            main_cache.shape[0] * main_cache.shape[1],
            extra_cache.shape[0] * extra_cache.shape[1],
            main_cache.shape[1],
            extra_cache.shape[1],
            scale * _LOG2E,
            num_heads,
            kv_splits,
            HAS_EXTRA=has_extra,
            NOPE_DIM=nope_head_dim,
            NOPE_BLOCK=triton.next_power_of_2(nope_head_dim),
            ROPE_DIM=rope_head_dim,
            IS_FNUZ=current_platform.is_fp8_fnuz(),
            BLOCK_H=block_h,
            BLOCK_K=block_k,
            num_warps=4,
        )
        _sparse_attn_decode_ragged_reduce_kernel[(num_queries, num_heads)](
            m_partial,
            l_partial,
            acc_partial,
            main_indptr,
            extra_indptr,
            attn_sink,
            out,
            m_partial.stride(0),
            m_partial.stride(1),
            m_partial.stride(2),
            l_partial.stride(0),
            l_partial.stride(1),
            l_partial.stride(2),
            acc_partial.stride(0),
            acc_partial.stride(1),
            acc_partial.stride(2),
            acc_partial.stride(3),
            out.stride(0),
            out.stride(1),
            kv_splits,
            block_k,
            HAS_ATTN_SINK=has_attn_sink,
            HAS_EXTRA=has_extra,
            HEAD_DIM=head_dim,
            BLOCK_D=triton.next_power_of_2(head_dim),
            num_warps=4,
        )
        return out

    _sparse_attn_decode_ragged_kernel[(num_queries, triton.cdiv(num_heads, block_h))](
        q,
        main_cache,
        main_indices,
        main_indptr,
        extra_cache,
        extra_indices,
        extra_indptr,
        attn_sink,
        out,
        q.stride(0),
        q.stride(1),
        out.stride(0),
        out.stride(1),
        main_cache.stride(0),
        extra_cache.stride(0),
        main_cache.shape[0] * main_cache.shape[1],
        extra_cache.shape[0] * extra_cache.shape[1],
        main_cache.shape[1],
        extra_cache.shape[1],
        scale * _LOG2E,
        num_heads,
        HAS_ATTN_SINK=has_attn_sink,
        HAS_EXTRA=has_extra,
        NOPE_DIM=nope_head_dim,
        NOPE_BLOCK=triton.next_power_of_2(nope_head_dim),
        ROPE_DIM=rope_head_dim,
        IS_FNUZ=current_platform.is_fp8_fnuz(),
        BLOCK_H=block_h,
        BLOCK_K=block_k,
        num_warps=8,
    )
    return out


def _rocm_sparse_attn_decode_triton(
    q: torch.Tensor,
    main_cache: torch.Tensor,
    main_indices: torch.Tensor,
    scale: float,
    attn_sink: torch.Tensor | None,
    nope_head_dim: int,
    rope_head_dim: int,
    extra_cache: torch.Tensor | None = None,
    extra_indices: torch.Tensor | None = None,
    main_lengths: torch.Tensor | None = None,
    extra_lengths: torch.Tensor | None = None,
    main_ragged_indices: torch.Tensor | None = None,
    main_ragged_indptr: torch.Tensor | None = None,
    extra_ragged_indices: torch.Tensor | None = None,
    extra_ragged_indptr: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    kv_splits: int = 1,
) -> torch.Tensor:
    if main_ragged_indices is None or main_ragged_indptr is None:
        main_ragged_indices, main_ragged_indptr = build_ragged_indices_from_dense(
            main_indices,
            main_lengths
            if main_lengths is not None
            else (main_indices >= 0).sum(dim=-1, dtype=torch.int32),
            num_rows=main_cache.shape[0] * main_cache.shape[1],
        )

    if (
        (extra_ragged_indices is None or extra_ragged_indptr is None)
        and extra_cache is not None
        and extra_indices is not None
    ):
        extra_ragged_indices, extra_ragged_indptr = build_ragged_indices_from_dense(
            extra_indices,
            extra_lengths
            if extra_lengths is not None
            else (extra_indices >= 0).sum(dim=-1, dtype=torch.int32),
            num_rows=extra_cache.shape[0] * extra_cache.shape[1],
        )

    return _rocm_sparse_attn_decode_ragged_triton(
        q=q,
        main_cache=main_cache,
        main_indices=main_ragged_indices,
        main_indptr=main_ragged_indptr,
        scale=scale,
        attn_sink=attn_sink,
        nope_head_dim=nope_head_dim,
        rope_head_dim=rope_head_dim,
        extra_cache=extra_cache,
        extra_indices=extra_ragged_indices,
        extra_indptr=extra_ragged_indptr,
        out=out,
        kv_splits=kv_splits,
    )


def rocm_sparse_attn_prefill(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    topk_length: torch.Tensor | None,
    scale: float,
    head_dim: int,
    nope_head_dim: int,
    rope_head_dim: int,
    attn_sink: torch.Tensor | None,
    output: torch.Tensor,
    ragged_indices: torch.Tensor | None = None,
    ragged_indptr: torch.Tensor | None = None,
) -> None:
    assert kv.ndim == 3 and kv.shape[1] == 1, (
        f"ROCm Triton sparse prefill expects kv=[skv,1,d], got {kv.shape}"
    )
    _validate_dsv4_sparse_dims(
        head_dim,
        nope_head_dim,
        rope_head_dim,
        "rocm_sparse_attn_prefill",
    )
    output_arg = output if output.dtype == torch.bfloat16 else None
    if ragged_indices is not None and ragged_indptr is not None:
        output_chunk = _rocm_sparse_attn_prefill_ragged_triton(
            q=q,
            kv=kv.squeeze(1),
            indices=ragged_indices,
            indptr=ragged_indptr,
            scale=scale,
            attn_sink=None if attn_sink is None else attn_sink[: q.shape[1]],
            nope_head_dim=nope_head_dim,
            rope_head_dim=rope_head_dim,
            out=output_arg,
        )
    else:
        indices_2d = indices.reshape(indices.shape[0], -1)
        output_chunk = _rocm_sparse_attn_prefill_triton(
            q=q,
            kv=kv.squeeze(1),
            indices=indices_2d,
            scale=scale,
            attn_sink=None if attn_sink is None else attn_sink[: q.shape[1]],
            nope_head_dim=nope_head_dim,
            rope_head_dim=rope_head_dim,
            topk_length=topk_length,
            out=output_arg,
        )
    if output_chunk is not output:
        output.copy_(output_chunk.to(output.dtype))


def rocm_sparse_attn_decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor | None,
    swa_k_cache: torch.Tensor,
    swa_only: bool,
    topk_indices: torch.Tensor | None,
    topk_lens: torch.Tensor | None,
    swa_indices: torch.Tensor,
    swa_lens: torch.Tensor,
    swa_ragged_indices: torch.Tensor | None,
    swa_ragged_indptr: torch.Tensor | None,
    topk_ragged_indices: torch.Tensor | None,
    topk_ragged_indptr: torch.Tensor | None,
    attn_sink: torch.Tensor | None,
    scale: float,
    head_dim: int,
    nope_head_dim: int,
    rope_head_dim: int,
    output: torch.Tensor,
    kv_splits: int = 1,
) -> None:
    assert swa_k_cache.dtype == torch.uint8, (
        "ROCm Triton sparse decode expects uint8 fp8_ds_mla SWA cache, "
        f"got {swa_k_cache.dtype}"
    )
    _validate_dsv4_sparse_dims(
        head_dim,
        nope_head_dim,
        rope_head_dim,
        "rocm_sparse_attn_decode",
    )

    main_indices = swa_indices.reshape(swa_indices.shape[0], -1)

    extra_cache = None
    extra_indices = None
    if not swa_only:
        assert kv_cache is not None
        assert topk_indices is not None or (
            topk_ragged_indices is not None and topk_ragged_indptr is not None
        )
        assert kv_cache.dtype == torch.uint8, (
            "ROCm Triton sparse decode expects uint8 fp8_ds_mla extra cache, "
            f"got {kv_cache.dtype}"
        )
        extra_cache = kv_cache
        if topk_indices is not None:
            extra_indices = topk_indices.reshape(topk_indices.shape[0], -1)

    output_arg = output if output.dtype == torch.bfloat16 else None
    attn_out = _rocm_sparse_attn_decode_triton(
        q=q,
        main_cache=swa_k_cache,
        main_indices=main_indices,
        scale=scale,
        attn_sink=None if attn_sink is None else attn_sink[: q.shape[1]],
        nope_head_dim=nope_head_dim,
        rope_head_dim=rope_head_dim,
        extra_cache=extra_cache,
        extra_indices=extra_indices,
        main_lengths=swa_lens,
        extra_lengths=topk_lens,
        main_ragged_indices=swa_ragged_indices,
        main_ragged_indptr=swa_ragged_indptr,
        extra_ragged_indices=topk_ragged_indices,
        extra_ragged_indptr=topk_ragged_indptr,
        out=output_arg,
        kv_splits=kv_splits,
    )
    if attn_out is not output:
        output.copy_(attn_out.to(output.dtype))
