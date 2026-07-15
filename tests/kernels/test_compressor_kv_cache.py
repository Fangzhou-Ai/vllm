# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Round-trip tests for compressor → FP8 quant + KV cache insert → gather + dequant.

These tests cover:
  A) DeepseekV4 Attention: head_dim=512 (448 FP8 nope + 64 bf16 rope), quant_block=64
  B) Fused dequant+gather K cache
  C) Indexer:       head_dim=128 (all FP8), quant_block=128
  D) DeepseekV4 Attention magnitude range: correctness across small/large values
  E) Indexer fused Triton kernel: compress+norm+rope+quant+insert
"""

import math
from types import SimpleNamespace

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.models.deepseek_v4.amd.compressor import (
    build_compression_plan,
    compress_current_and_paged_state,
    compression_plan_capacity,
    make_compression_plan_template,
    save_compressor_states,
    use_tail_only_state_writes,
)
from vllm.models.deepseek_v4.common.ops import (
    dequantize_and_gather_k_cache,
    quantize_and_insert_k_cache,
)
from vllm.models.deepseek_v4.common.ops.fused_compress_quant_cache import (
    _fused_kv_compress_norm_rope_insert_indexer_attn,
    _fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn,
)

from .test_fused_indexer_q_rope_quant import quantize_to_mxfp4


def test_rocm_compressor_plan_template_is_compact():
    query_start_loc = torch.tensor([0, 1, 6, 6, 140], dtype=torch.int32)

    c4_plan = make_compression_plan_template(query_start_loc, 4)
    c128_plan = make_compression_plan_template(query_start_loc, 128)

    assert c4_plan.shape == (37, 4)
    assert c128_plan.tolist() == [
        [0, 1, 0, 0],
        [1, 6, 1, 0],
        [6, 140, 3, 0],
        [6, 140, 3, 1],
    ]
    assert compression_plan_capacity(140, 4, 4) == 39
    assert c4_plan.shape[0] <= compression_plan_capacity(140, 4, 4)


@pytest.mark.parametrize(
    ("prefix_caching", "has_connector", "expected"),
    [(False, False, True), (True, False, False), (False, True, False)],
)
def test_rocm_tail_state_writes_are_safe_only_without_cache_reuse(
    prefix_caching: bool, has_connector: bool, expected: bool
):
    config = SimpleNamespace(
        cache_config=SimpleNamespace(enable_prefix_caching=prefix_caching),
        kv_transfer_config=(
            SimpleNamespace(is_kv_transfer_instance=True) if has_connector else None
        ),
    )

    assert use_tail_only_state_writes(config) is expected


def _ue8m0_reference(x: torch.Tensor, block_size: int, fp8_max: float):
    """PyTorch reference for UE8M0 FP8 quantization (per-block, power-of-2 scale).

    Returns (x_fp8, scales) where x_fp8 is float8_e4m3fn and scales are float32.
    """
    assert x.dim() == 1
    n = x.numel()
    n_blocks = math.ceil(n / block_size)
    x_fp8 = torch.zeros(n, dtype=torch.float8_e4m3fn, device=x.device)
    scales = torch.zeros(n_blocks, dtype=torch.float32, device=x.device)

    for i in range(n_blocks):
        start = i * block_size
        end = min(start + block_size, n)
        block = x[start:end].float()
        amax = block.abs().max().clamp(min=1e-4)
        raw_scale = amax / fp8_max
        exponent = math.ceil(math.log2(raw_scale.item()))
        scale = 2.0**exponent
        scales[i] = scale
        quantized = (block / scale).clamp(-fp8_max, fp8_max)
        x_fp8[start:end] = quantized.to(torch.float8_e4m3fn)

    return x_fp8, scales


# ── Test A: DeepseekV4 Attention path ──────────────────────────────────────────────


@pytest.mark.parametrize("num_tokens", [1, 4, 8, 17])
@pytest.mark.parametrize("block_size", [16, 64])
def test_deepseek_v4_attention_quant_cache_roundtrip(num_tokens: int, block_size: int):
    """compressed_kv → quantize_and_insert_k_cache → dequantize_and_gather_k_cache
    → compare against original."""

    HEAD_DIM = 512
    NOPE_DIM = 448
    HEAD_BYTES = 584  # 448 fp8 + 128 bf16 + 8 uint8 scale
    FP8_MAX = 448.0
    QUANT_BLOCK = 64

    num_blocks = (num_tokens + block_size - 1) // block_size + 1
    device = "cuda"

    # Random compressed_kv (simulates compressor output)
    compressed_kv = torch.randn(
        num_tokens, HEAD_DIM, dtype=torch.bfloat16, device=device
    )

    # ── Quant + insert ──────────────────────────────────────────────────
    k_cache = torch.zeros(
        num_blocks, block_size, HEAD_BYTES, dtype=torch.uint8, device=device
    )
    k_cache_2d = k_cache.view(num_blocks, -1)

    # Sequential slot mapping: token i → slot i
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)

    quantize_and_insert_k_cache(
        compressed_kv, k_cache_2d, slot_mapping, block_size=block_size
    )

    # ── Gather + dequant ────────────────────────────────────────────────
    num_reqs = 1
    max_blocks_per_seq = num_blocks
    out = torch.zeros(
        num_reqs, num_tokens, HEAD_DIM, dtype=torch.bfloat16, device=device
    )
    seq_lens = torch.tensor([num_tokens], dtype=torch.int32, device=device)
    # block_table: request 0 uses physical blocks 0, 1, ...
    block_table = torch.arange(
        max_blocks_per_seq, dtype=torch.int32, device=device
    ).unsqueeze(0)

    dequantize_and_gather_k_cache(
        out, k_cache, seq_lens, None, block_table, block_size, offset=0
    )

    recovered = out[0, :num_tokens]

    # ── NoPE portion (first 448): FP8 quantized, expect UE8M0 error ──
    nope_orig = compressed_kv[:, :NOPE_DIM].float()
    nope_recv = recovered[:, :NOPE_DIM].float()
    nope_diff = (nope_recv - nope_orig).abs()

    # Per-token check: FP8 e4m3 (3-bit mantissa) worst-case error is
    # half-ULP at the largest representable value.  At y ≈ 448 (max),
    # ULP = 2^(8-3) = 32, so error ≤ 16 * scale.
    for t in range(num_tokens):
        _, scales = _ue8m0_reference(
            compressed_kv[t, :NOPE_DIM].float(), QUANT_BLOCK, FP8_MAX
        )
        max_allowed = 16.0 * scales.max().item()
        token_diff = nope_diff[t].max().item()
        assert token_diff <= max_allowed, (
            f"Token {t} nope diff {token_diff} exceeds max_allowed "
            f"{max_allowed} (scale={scales.max().item()})"
        )

    # ── RoPE portion (last 64): stored as bf16, should be exact ─────
    rope_diff = (recovered[:, NOPE_DIM:] - compressed_kv[:, NOPE_DIM:]).abs()
    assert rope_diff.max().item() == 0.0, (
        f"RoPE portion should be exact but got max diff {rope_diff.max().item()}"
    )


# ── Test B: Fused dequant+gather K cache ────────────────────────────────────


def _dequantize_and_gather_k_cache_reference(
    out: torch.Tensor,
    k_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor | None,
    block_table: torch.Tensor,
    block_size: int,
    offset: int,
) -> None:
    fp8_dim = 448
    bf16_dim = 64
    scale_dim = 8
    quant_block = 64
    token_data_size = fp8_dim + bf16_dim * 2

    for req_id in range(seq_lens.shape[0]):
        seq_len = seq_lens[req_id].item()
        gather_len = gather_lens[req_id].item() if gather_lens is not None else seq_len
        start_pos = seq_len - gather_len

        for i in range(gather_len):
            pos = start_pos + i
            pos_in_block = pos % block_size
            block_idx = block_table[req_id, pos // block_size].item()
            cache_block = k_cache[block_idx].view(-1)

            token_data_start = pos_in_block * token_data_size
            fp8_bytes = cache_block[token_data_start : token_data_start + fp8_dim]
            fp8_vals = fp8_bytes.view(torch.float8_e4m3fn).float()

            scale_start = block_size * token_data_size + pos_in_block * scale_dim
            encoded_scales = cache_block[scale_start : scale_start + scale_dim]
            scales = torch.exp2(encoded_scales[:7].float() - 127.0)
            dequant = fp8_vals * scales.repeat_interleave(quant_block)

            bf16_start = token_data_start + fp8_dim
            bf16_bytes = cache_block[bf16_start : bf16_start + bf16_dim * 2]
            bf16_tail = bf16_bytes.view(torch.bfloat16)

            out[req_id, offset + i, :fp8_dim] = dequant
            out[req_id, offset + i, fp8_dim:] = bf16_tail


@pytest.mark.parametrize(
    ("seq_lens_host", "gather_lens_host", "offset"),
    [
        ([9, 23, 7], None, 0),
        ([19, 8, 257], [6, 8, 129], 5),
    ],
)
def test_dequantize_and_gather_k_cache(
    seq_lens_host: list[int],
    gather_lens_host: list[int] | None,
    offset: int,
):
    block_size = 64
    head_dim = 512
    nope_dim = 448
    scale_dim = 8
    head_bytes = nope_dim + (head_dim - nope_dim) * 2 + scale_dim
    device = "cuda"
    num_reqs = len(seq_lens_host)
    num_tokens = sum(seq_lens_host)
    max_gather_len = max(gather_lens_host or seq_lens_host)
    max_blocks_per_seq = math.ceil(max(seq_lens_host) / block_size)
    num_blocks = sum(math.ceil(seq_len / block_size) for seq_len in seq_lens_host)

    compressed_kv = torch.randn(
        num_tokens, head_dim, dtype=torch.bfloat16, device=device
    )

    # Randomize physical pages so the test covers block-table translation.
    # Keep padded block-table entries invalid to catch accidental reads.
    physical_blocks = torch.randperm(num_blocks, device=device)
    block_table = torch.full(
        (num_reqs, max_blocks_per_seq), int(-1e6), dtype=torch.int32, device=device
    )
    start = 0
    for req_id, seq_len in enumerate(seq_lens_host):
        num_req_blocks = math.ceil(seq_len / block_size)
        req_blocks = physical_blocks[start : start + num_req_blocks]
        block_table[req_id, :num_req_blocks] = req_blocks
        start += num_req_blocks

    # Build slot_mapping for quantize_and_insert_k_cache.
    slot_mapping = torch.empty(num_tokens, dtype=torch.int64, device=device)
    start = 0
    for req_id, seq_len in enumerate(seq_lens_host):
        logical_pos = torch.arange(seq_len, dtype=torch.int64, device=device)
        block_idx = block_table[req_id, logical_pos // block_size].to(torch.int64)
        token_slots = block_idx * block_size + logical_pos % block_size
        slot_mapping[start : start + seq_len] = token_slots
        start += seq_len

    # Insert compressed K into the paged cache layout used by the gather op.
    k_cache = torch.empty(
        num_blocks, block_size, head_bytes, dtype=torch.uint8, device=device
    )
    k_cache_2d = k_cache.view(num_blocks, -1)
    quantize_and_insert_k_cache(compressed_kv, k_cache_2d, slot_mapping, block_size)

    out_shape = (num_reqs, offset + max_gather_len + 3, head_dim)
    ref_out = torch.empty(out_shape, dtype=torch.bfloat16, device=device)
    actual_out = torch.empty_like(ref_out)
    seq_lens = torch.tensor(seq_lens_host, dtype=torch.int32, device=device)
    gather_lens = (
        torch.tensor(gather_lens_host, dtype=torch.int32, device=device)
        if gather_lens_host is not None
        else None
    )

    # Compare production gather against a PyTorch reference for valid output rows.
    _dequantize_and_gather_k_cache_reference(
        ref_out, k_cache, seq_lens, gather_lens, block_table, block_size, offset
    )
    dequantize_and_gather_k_cache(
        actual_out, k_cache, seq_lens, gather_lens, block_table, block_size, offset
    )
    torch.accelerator.synchronize()

    # only check non-padded content
    for req_id, seq_len in enumerate(seq_lens_host):
        gather_len = (
            gather_lens_host[req_id] if gather_lens_host is not None else seq_len
        )
        actual = actual_out[req_id, offset : offset + gather_len]
        expected = ref_out[req_id, offset : offset + gather_len]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


# ── Test C: Indexer path ────────────────────────────────────────────────────


@pytest.mark.parametrize("num_tokens", [1, 4, 8, 17])
@pytest.mark.parametrize("block_size", [16, 64])
def test_indexer_quant_cache_roundtrip(num_tokens: int, block_size: int):
    """k → indexer_k_quant_and_cache → cp_gather_indexer_k_quant_cache
    → manual dequant → compare against original."""

    HEAD_DIM = 128
    QUANT_BLOCK_SIZE = 128
    # cache_stride = head_dim + (head_dim * 4 / quant_block_size) = 128 + 4 = 132
    CACHE_STRIDE = HEAD_DIM + HEAD_DIM * 4 // QUANT_BLOCK_SIZE

    num_blocks = (num_tokens + block_size - 1) // block_size + 1
    device = "cuda"

    # Random K (simulates compressor output for indexer)
    k = torch.randn(num_tokens, HEAD_DIM, dtype=torch.bfloat16, device=device)

    # ── Quant + insert ──────────────────────────────────────────────────
    kv_cache = torch.zeros(
        num_blocks, block_size, CACHE_STRIDE, dtype=torch.uint8, device=device
    )
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)

    ops.indexer_k_quant_and_cache(k, kv_cache, slot_mapping, QUANT_BLOCK_SIZE, "ue8m0")

    # ── Gather ──────────────────────────────────────────────────────────
    max_blocks_per_seq = num_blocks
    block_table = torch.arange(
        max_blocks_per_seq, dtype=torch.int32, device=device
    ).unsqueeze(0)
    cu_seq_lens = torch.tensor([0, num_tokens], dtype=torch.int32, device=device)

    # dst_k: [total_seq_len, head_dim] as uint8 (raw FP8 bytes)
    dst_k = torch.zeros(num_tokens, HEAD_DIM, dtype=torch.uint8, device=device)
    # dst_scale: [total_seq_len, head_dim/quant_block*4] as uint8 (raw float32 bytes)
    num_scale_bytes = HEAD_DIM * 4 // QUANT_BLOCK_SIZE  # 4
    dst_scale = torch.zeros(
        num_tokens, num_scale_bytes, dtype=torch.uint8, device=device
    )

    ops.cp_gather_indexer_k_quant_cache(
        kv_cache, dst_k, dst_scale, block_table, cu_seq_lens
    )

    # ── Manual dequant ──────────────────────────────────────────────────
    k_fp8 = dst_k.view(torch.float8_e4m3fn).float()  # [num_tokens, 128]
    scale = dst_scale.view(torch.float32)  # [num_tokens, 1]
    k_recovered = k_fp8 * scale  # [num_tokens, 128]

    # ── Compare ─────────────────────────────────────────────────────────
    diff = (k_recovered - k.float()).abs()
    k_abs = k.float().abs()

    for t in range(num_tokens):
        amax = k_abs[t].max().clamp(min=1e-4).item()
        # UE8M0: scale = 2^ceil(log2(amax / 448))
        exponent = math.ceil(math.log2(amax / 448.0))
        ue8m0_scale = 2.0**exponent
        # FP8 e4m3 (3-bit mantissa): worst-case error = 16 * scale
        max_allowed = 16.0 * ue8m0_scale
        token_diff = diff[t].max().item()
        assert token_diff <= max_allowed, (
            f"Token {t} diff {token_diff} exceeds max_allowed "
            f"{max_allowed} (scale={ue8m0_scale})"
        )


def test_indexer_gather_accepts_upper_bound_output():
    """Gather only exact cu_seq_lens even when dst is over-allocated."""

    head_dim = 128
    quant_block_size = 128
    cache_stride = head_dim + head_dim * 4 // quant_block_size
    valid_tokens = 9
    upper_bound_tokens = 13
    block_size = 16
    num_blocks = 2
    sentinel = 123
    device = "cuda"

    k = torch.randn(valid_tokens, head_dim, dtype=torch.bfloat16, device=device)
    kv_cache = torch.zeros(
        num_blocks, block_size, cache_stride, dtype=torch.uint8, device=device
    )
    slot_mapping = torch.arange(valid_tokens, dtype=torch.int64, device=device)
    ops.indexer_k_quant_and_cache(k, kv_cache, slot_mapping, quant_block_size, "ue8m0")

    block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).unsqueeze(
        0
    )
    cu_seq_lens = torch.tensor([0, valid_tokens], dtype=torch.int32, device=device)
    dst_k = torch.full(
        (upper_bound_tokens, head_dim), sentinel, dtype=torch.uint8, device=device
    )
    num_scale_bytes = head_dim * 4 // quant_block_size
    dst_scale = torch.full(
        (upper_bound_tokens, num_scale_bytes),
        sentinel,
        dtype=torch.uint8,
        device=device,
    )

    ops.cp_gather_indexer_k_quant_cache(
        kv_cache, dst_k, dst_scale, block_table, cu_seq_lens
    )
    torch.accelerator.synchronize()

    k_recovered = dst_k[:valid_tokens].view(torch.float8_e4m3fn).float() * dst_scale[
        :valid_tokens
    ].view(torch.float32)
    diff = (k_recovered - k.float()).abs()
    max_allowed = (16.0 * dst_scale[:valid_tokens].view(torch.float32).max()).item()
    assert diff.max().item() <= max_allowed
    assert torch.all(dst_k[valid_tokens:] == sentinel)
    assert torch.all(dst_scale[valid_tokens:] == sentinel)


# ── Test D: DeepseekV4 attention with values at different magnitudes ───────────


def test_deepseek_v4_quant_magnitude_range():
    """Test that quantization handles a range of magnitudes correctly."""

    HEAD_DIM = 512
    NOPE_DIM = 448
    HEAD_BYTES = 584
    block_size = 16
    num_tokens = 4
    num_blocks = 2
    device = "cuda"

    # Create inputs with varying magnitudes: small, medium, large
    compressed_kv = torch.zeros(
        num_tokens, HEAD_DIM, dtype=torch.bfloat16, device=device
    )
    compressed_kv[0] = 0.001  # very small
    compressed_kv[1] = 1.0  # unit scale
    compressed_kv[2] = 100.0  # large
    compressed_kv[3] = torch.randn(HEAD_DIM, dtype=torch.bfloat16, device=device)

    k_cache = torch.zeros(
        num_blocks, block_size, HEAD_BYTES, dtype=torch.uint8, device=device
    )
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)

    quantize_and_insert_k_cache(
        compressed_kv, k_cache.view(num_blocks, -1), slot_mapping, block_size
    )

    out = torch.zeros(1, num_tokens, HEAD_DIM, dtype=torch.bfloat16, device=device)
    seq_lens = torch.tensor([num_tokens], dtype=torch.int32, device=device)
    block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).unsqueeze(
        0
    )

    dequantize_and_gather_k_cache(
        out, k_cache, seq_lens, None, block_table, block_size, offset=0
    )

    recovered = out[0, :num_tokens]

    # RoPE portion must be exact
    rope_diff = (recovered[:, NOPE_DIM:] - compressed_kv[:, NOPE_DIM:]).abs().max()
    assert rope_diff.item() == 0.0, f"RoPE diff {rope_diff.item()}"

    # NoPE: relative error should be reasonable
    for t in range(num_tokens):
        orig = compressed_kv[t, :NOPE_DIM].float()
        recv = recovered[t, :NOPE_DIM].float()
        abs_diff = (recv - orig).abs().max().item()
        magnitude = orig.abs().max().item()
        if magnitude > 0.01:
            rel_err = abs_diff / magnitude
            assert rel_err < 0.15, (
                f"Token {t}: rel_err={rel_err:.4f}, abs_diff={abs_diff:.6f}, "
                f"magnitude={magnitude:.4f}"
            )


# ── Test E: Indexer fused K-cache insert (Triton kernels) ────────────────────
#
# Both kernels share the same Triton signature; use_fp4 selects between them.
# Full pipeline: state-cache gather → softmax-weighted compress → RMSNorm →
#   GPT-J RoPE → quant (MXFP4 or FP8) → paged cache insert.


def _reference_kv_compress_norm_rope(
    state_cache: torch.Tensor,
    block_table: torch.Tensor,
    positions: torch.Tensor,
    rms_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    compress_ratio: int = 1,
    overlap: int = 0,
    use_fp4: bool = False,
    rms_eps: float = 1e-6,
    fp8_max: float = 448.0,
    return_full_cache: bool = False,
):
    """Compress → RMSNorm → GPT-J RoPE → quantize.

    Gathers (1+overlap)*compress_ratio state entries per output token, applies
    per-element softmax over the scores, and computes the weighted kv sum.
    Returns (quantized_values, scale) matching the kernel's output layout.
    """
    device = state_cache.device
    head_dim = rms_weight.shape[0]
    rope_dim = cos_sin_cache.shape[-1]
    state_block_size = state_cache.shape[1]
    state_width = state_cache.shape[-1] // 2
    nope_dim = head_dim - rope_dim
    total = (1 + overlap) * compress_ratio
    results = []
    for pos in positions.tolist():
        src = torch.arange(pos - total + 1, pos + 1, dtype=torch.int64, device=device)
        valid = src >= 0
        idx = src.clamp(min=0)
        pages = block_table[0, idx // state_block_size]
        offsets = idx % state_block_size
        raw = state_cache[pages, offsets].float()  # [total, state_dim]

        # Group 0 (tokens 0..cr-1):   kv[:H],   score[SW:SW+H]
        # Group 1 (tokens cr..2cr-1): kv[H:2H], score[SW+H:SW+2H]
        if overlap:
            sw = state_width
            g0_kv = raw[:compress_ratio, :head_dim]
            g1_kv = raw[compress_ratio:, head_dim : 2 * head_dim]
            g0_scores = raw[:compress_ratio, sw : sw + head_dim]
            g1_scores = raw[compress_ratio:, sw + head_dim : sw + 2 * head_dim]
            kv = torch.cat([g0_kv, g1_kv])
            scores = torch.cat([g0_scores, g1_scores])
        else:
            kv = raw[:, :head_dim]
            scores = raw[:, state_width : state_width + head_dim]

        scores[~valid] = float("-inf")
        kv[~valid] = 0.0
        weights = torch.softmax(scores, dim=0)
        compressed = (kv * weights).sum(dim=0)  # [H]
        var = (compressed * compressed).mean()
        normed = compressed * torch.rsqrt(var + rms_eps) * rms_weight.float()
        compressed_pos = (pos // compress_ratio) * compress_ratio
        cos, sin = cos_sin_cache[compressed_pos].float().chunk(2)
        nope, rope = normed.split([nope_dim, rope_dim])
        rope = torch.stack(
            [rope[0::2] * cos - rope[1::2] * sin, rope[1::2] * cos + rope[0::2] * sin],
            dim=-1,
        ).reshape(rope_dim)
        results.append(torch.cat([nope, rope]).to(state_cache.dtype))
    result = torch.stack(results)

    if return_full_cache:
        # Contiguous 512-wide bf16 row (nope unrotated + rope rotated), matching
        # the FlashInfer full-cache layout before any per-tensor fp8 quant. The
        # kernel rounds the fp32 result to bf16 once at the store.
        return result.to(torch.bfloat16)

    if use_fp4:
        return quantize_to_mxfp4(result)
    else:
        pairs = [
            _ue8m0_reference(result[t], head_dim, fp8_max) for t in range(len(result))
        ]
        quants, scales = zip(*pairs)
        return torch.stack(quants), torch.cat(scales)


def test_rocm_compressor_plan_resolves_aligned_boundaries():
    device = "cuda"
    positions = torch.tensor(
        [5, 6, 7, 8, 9, 12, 13, 14, 15, 16],
        dtype=torch.int64,
        device=device,
    )
    slot_mapping = torch.arange(10, dtype=torch.int64, device=device)
    metadata = SimpleNamespace(
        query_start_loc_cpu=torch.tensor([0, 5, 10], dtype=torch.int32),
        positions=positions,
        slot_mapping=slot_mapping,
        num_actual_tokens=10,
        num_reqs=2,
    )
    plan_buffer = torch.empty((10, 4), dtype=torch.int32, device=device)

    plan = build_compression_plan(plan_buffer, metadata, compress_ratio=4).cpu()

    assert plan.shape == (4, 4)
    assert plan[0].tolist() == [2, 0, 7, 5]
    assert plan[2].tolist() == [8, 1, 15, 4]
    assert plan[[1, 3], 2].tolist() == [-1, -1]


def test_rocm_compressor_plan_reuse_clears_descriptor_stable_tail():
    device = "cuda"
    plan_buffer = torch.full((8, 4), 99, dtype=torch.int32, device=device)
    slot_mapping = torch.arange(8, dtype=torch.int64, device=device)
    first_metadata = SimpleNamespace(
        query_start_loc_cpu=torch.tensor([0, 1, 8], dtype=torch.int32),
        positions=torch.tensor(
            [3, 1, 2, 3, 4, 5, 6, 7], dtype=torch.int64, device=device
        ),
        slot_mapping=slot_mapping,
        num_actual_tokens=8,
        num_reqs=2,
    )
    second_metadata = SimpleNamespace(
        query_start_loc_cpu=torch.tensor([0, 4, 8], dtype=torch.int32),
        positions=torch.arange(8, dtype=torch.int64, device=device),
        slot_mapping=slot_mapping,
        num_actual_tokens=8,
        num_reqs=2,
    )

    first_plan = build_compression_plan(plan_buffer, first_metadata, 4)
    first_shape = first_plan.shape
    assert first_plan[2, 2].item() >= 0

    second_plan = build_compression_plan(plan_buffer, second_metadata, 4)

    assert second_plan.shape == first_shape == (4, 4)
    assert second_plan[:, 2].tolist() == [3, 7, -1, -1]


def test_rocm_compressor_mixes_paged_prefix_and_current_chunk():
    device = "cuda"
    torch.manual_seed(31)
    head_dim = 128
    rope_dim = 64
    ratio = 4
    state_block_size = 4
    state_width = 2 * head_dim
    num_query_tokens = 8

    kv = torch.randn(num_query_tokens, state_width, dtype=torch.float32, device=device)
    score = torch.randn_like(kv)
    ape = torch.randn(ratio, state_width, dtype=torch.float32, device=device)
    state_cache = torch.randn(
        4,
        state_block_size,
        2 * state_width,
        dtype=torch.float32,
        device=device,
    )
    block_table = torch.arange(4, dtype=torch.int32, device=device).unsqueeze(0)
    plan = torch.tensor(
        [
            [0, 0, 7, 7],
            [2, 0, 7, 5],
            [6, 0, 11, 1],
            [7, 0, 15, 0],
        ],
        dtype=torch.int32,
        device=device,
    )
    rms_weight = torch.randn(head_dim, dtype=torch.bfloat16, device=device)
    cos_sin_cache = torch.randn(16, rope_dim, dtype=torch.float32, device=device)

    kv_block_size = 16
    token_stride = head_dim
    scale_dim = 4
    kv_cache = torch.zeros(
        1,
        kv_block_size,
        token_stride + scale_dim,
        dtype=torch.uint8,
        device=device,
    )
    kv_slot_mapping = torch.full(
        (num_query_tokens,), -1, dtype=torch.int64, device=device
    )
    kv_slot_mapping[0] = 0
    kv_slot_mapping[2] = 1
    kv_slot_mapping[6] = 2
    kv_slot_mapping[7] = 3

    compress_current_and_paged_state(
        state_cache=state_cache,
        kv=kv,
        score=score,
        ape=ape,
        plan=plan,
        block_table=block_table,
        block_size=state_block_size,
        state_width=state_width,
        cos_sin_cache=cos_sin_cache,
        kv_cache=kv_cache,
        kv_slot_mapping=kv_slot_mapping,
        rms_norm_weight=rms_weight,
        rms_norm_eps=1e-6,
        head_dim=head_dim,
        rope_head_dim=rope_dim,
        compress_ratio=ratio,
        overlap=True,
        use_fp4_cache=False,
        quant_block=head_dim,
        token_stride=token_stride,
        scale_dim=scale_dim,
    )

    expected_values = []
    expected_scales = []
    for token_idx, position, window_len in [
        (0, 7, 7),
        (2, 7, 5),
        (6, 11, 1),
        (7, 15, 0),
    ]:
        kv_rows = []
        score_rows = []
        for source_idx in range(2 * ratio):
            source_position = position - 2 * ratio + 1 + source_idx
            offset = head_dim if source_idx >= ratio else 0
            dims = slice(offset, offset + head_dim)
            if source_idx < window_len:
                page = block_table[0, source_position // state_block_size]
                row = state_cache[page, source_position % state_block_size]
                kv_rows.append(row[dims])
                score_rows.append(
                    row[state_width + offset : state_width + offset + head_dim]
                )
            else:
                input_row = token_idx - (2 * ratio - 1 - source_idx)
                kv_rows.append(kv[input_row, dims])
                score_rows.append(
                    score[input_row, dims] + ape[source_position % ratio, dims]
                )
        kv_stack = torch.stack(kv_rows)
        weights = torch.softmax(torch.stack(score_rows), dim=0)
        compressed = (kv_stack * weights).sum(dim=0)
        normed = compressed * torch.rsqrt(compressed.square().mean() + 1e-6)
        normed *= rms_weight.float()
        cos, sin = cos_sin_cache[position // ratio * ratio].chunk(2)
        rope = normed[-rope_dim:]
        rotated = torch.stack(
            [
                rope[0::2] * cos - rope[1::2] * sin,
                rope[1::2] * cos + rope[0::2] * sin,
            ],
            dim=-1,
        ).flatten()
        result = torch.cat((normed[:-rope_dim], rotated)).to(torch.bfloat16)
        quant, scales = _ue8m0_reference(result.float(), head_dim, 448.0)
        expected_values.append(quant.view(torch.uint8))
        expected_scales.append(scales)

    flat_cache = kv_cache.view(-1)
    values = []
    scales = []
    for slot in range(4):
        values.append(flat_cache[slot * token_stride : (slot + 1) * token_stride])
        scale_start = kv_block_size * token_stride + slot * scale_dim
        scales.append(
            flat_cache[scale_start : scale_start + scale_dim].view(torch.float32)
        )
    assert torch.equal(torch.stack(values), torch.stack(expected_values))
    assert torch.equal(torch.cat(scales), torch.cat(expected_scales))


def test_rocm_compressor_tail_write_preserves_only_future_state():
    device = "cuda"
    head_size = 8
    positions = torch.tensor(
        [*range(10), *range(20, 25)], dtype=torch.int64, device=device
    )
    num_tokens = positions.shape[0]
    kv = torch.arange(num_tokens * head_size, dtype=torch.float32, device=device).view(
        num_tokens, head_size
    )
    score = -kv
    ape = torch.arange(4 * head_size, dtype=torch.float32, device=device).view(
        4, head_size
    )
    state_cache = torch.full(
        (4, 4, 2 * head_size), -999.0, dtype=torch.float32, device=device
    )
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)
    token_to_req = torch.tensor([0] * 10 + [1] * 5, dtype=torch.int32, device=device)

    save_compressor_states(
        kv=kv,
        score=score,
        ape=ape,
        positions=positions,
        seq_lens=torch.tensor([10, 25], dtype=torch.int32, device=device),
        token_to_req_indices=token_to_req,
        state_cache=state_cache,
        slot_mapping=slot_mapping,
        block_size=4,
        state_width=head_size,
        compress_ratio=4,
        overlap=True,
        tail_only=True,
    )

    state_rows = state_cache.view(-1, 2 * head_size)
    assert torch.all(state_rows[:2] == -999.0)
    assert torch.equal(state_rows[2:num_tokens, :head_size], kv[2:])
    assert torch.equal(
        state_rows[2:num_tokens, head_size:],
        score[2:] + ape[positions[2:] % 4],
    )
    assert torch.all(state_rows[num_tokens:] == -999.0)


@pytest.mark.parametrize("num_tokens", [1, 7, 32])
@pytest.mark.parametrize("kv_block_size", [16, 32])
@pytest.mark.parametrize("use_fp4", [False, True])
def test_fused_kv_insert_indexer(num_tokens: int, kv_block_size: int, use_fp4: bool):
    """Fused K compress+norm+rope+quant+insert for the indexer KV cache."""
    HEAD_DIM = 128
    ROPE_DIM = 64
    BLOCK_SIZE = 16
    RMS_EPS = 1e-6
    FP8_MAX = 448.0

    device = "cuda"
    torch.manual_seed(42)
    compress_ratio = 4

    if use_fp4:
        TOKEN_STRIDE = HEAD_DIM // 2  # packed nibbles: 64 bytes
        SCALE_DIM = HEAD_DIM // 32  # ue8m0 bytes: 4
        QUANT_BLOCK = 32
        kernel = _fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn
    else:
        TOKEN_STRIDE = HEAD_DIM  # FP8 bytes: 128
        SCALE_DIM = 4  # 1 float32: 4 bytes
        QUANT_BLOCK = HEAD_DIM
        kernel = _fused_kv_compress_norm_rope_insert_indexer_attn

    # overlap=1 whenever compress_ratio==4, matching DeepseekCompressor logic.
    overlap = 1 if compress_ratio == 4 else 0
    coff = 1 + overlap  # multiplier for state_dim per entry

    num_pages = (compress_ratio * num_tokens - 1) // BLOCK_SIZE + 2
    state_cache = torch.randn(
        num_pages,
        BLOCK_SIZE,
        2 * coff * HEAD_DIM,  # kv_state + score_state, each coff*HEAD_DIM wide
        dtype=torch.bfloat16,
        device=device,
    )
    block_table = torch.arange(num_pages, dtype=torch.int32, device=device).unsqueeze(0)
    token_to_req = torch.zeros(num_tokens, dtype=torch.int32, device=device)
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)
    positions = torch.arange(
        compress_ratio - 1,
        compress_ratio * num_tokens,
        compress_ratio,
        dtype=torch.int64,
        device=device,
    )
    rms_weight = torch.randn(HEAD_DIM, dtype=torch.bfloat16, device=device)
    cos_sin_cache = torch.randn(compress_ratio * num_tokens, ROPE_DIM, device=device)

    kv_n_blocks = (num_tokens + kv_block_size - 1) // kv_block_size + 1
    kv_cache = torch.zeros(
        kv_n_blocks,
        kv_block_size * (TOKEN_STRIDE + SCALE_DIM),
        dtype=torch.uint8,
        device=device,
    )

    kernel[(num_tokens,)](
        state_cache,
        state_cache.stride(0),
        state_cache.stride(1),
        token_to_req,
        positions,
        slot_mapping,
        block_table,
        block_table.stride(0),
        BLOCK_SIZE,
        rms_weight,
        RMS_EPS,
        cos_sin_cache,
        cos_sin_cache.stride(0),
        kv_cache,
        slot_mapping,
        kv_block_size,
        HEAD_SIZE=HEAD_DIM,
        TRITON_BLOCK_SIZE=HEAD_DIM,
        STATE_WIDTH=coff * HEAD_DIM,
        COMPRESS_RATIO=compress_ratio,
        OVERLAP=overlap,
        ROPE_HEAD_DIM=ROPE_DIM,
        FP8_MAX=FP8_MAX,
        QUANT_BLOCK=QUANT_BLOCK,
        TOKEN_STRIDE=TOKEN_STRIDE,
        SCALE_DIM=SCALE_DIM,
        KV_BLOCK_STRIDE=kv_cache.stride(0),
        num_warps=1,
    )

    k_quant, scale = _reference_kv_compress_norm_rope(
        state_cache,
        block_table,
        positions,
        rms_weight,
        cos_sin_cache,
        compress_ratio,
        overlap,
        use_fp4,
        rms_eps=RMS_EPS,
        fp8_max=FP8_MAX,
    )

    if use_fp4:
        for i in range(num_tokens):
            blk, pos = i // kv_block_size, i % kv_block_size
            val_off = pos * TOKEN_STRIDE
            fp4_actual = kv_cache[blk, val_off : val_off + TOKEN_STRIDE]
            assert torch.equal(k_quant[i], fp4_actual), (
                f"token {i}: packed nibbles differ, "
                f"{(k_quant[i] != fp4_actual).sum()} "
                f"/ {TOKEN_STRIDE}"
            )

            scale_off = kv_block_size * TOKEN_STRIDE + pos * SCALE_DIM
            scale_actual = kv_cache[blk, scale_off : scale_off + SCALE_DIM]
            assert torch.equal(scale_actual, scale[i]), (
                f"token {i}: ue8m0 {scale_actual.tolist()} != {scale[i].tolist()}"
            )

    else:
        k_quant = k_quant.view(torch.uint8)
        for i in range(num_tokens):
            blk, pos = i // kv_block_size, i % kv_block_size
            val_off = pos * TOKEN_STRIDE
            assert torch.equal(
                k_quant[i], kv_cache[blk, val_off : val_off + TOKEN_STRIDE]
            ), f"token {i}: FP8 bytes differ"

            scale_off = kv_block_size * TOKEN_STRIDE + pos * SCALE_DIM
            actual_scale = kv_cache[blk, scale_off : scale_off + SCALE_DIM].view(
                torch.float32
            )
            assert torch.equal(actual_scale, scale[i : i + 1]), (
                f"token {i}: scale {actual_scale.item()} != {scale[i].item()}"
            )


@pytest.mark.parametrize("compress_ratio", [4, 128])
@pytest.mark.parametrize("store_fp8", [False, True])
def test_cutedsl_full_cache_store(compress_ratio: int, store_fp8: bool):
    """CuTeDSL compressor full-cache (FlashInfer) store parity for head=512.

    Exercises the contiguous bf16 / per-tensor fp8 store branch of both the C4
    fused kernel and the C128 split kernel against the PyTorch reference.
    """
    cutedsl = pytest.importorskip("cutlass")  # noqa: F841
    from vllm.models.deepseek_v4.nvidia.ops.sparse_attn_compress_cutedsl import (
        fused_kv_compress_norm_rope_insert_sparse_attn_cutedsl,
        split_kv_compress_norm_rope_insert_sparse_attn_cutedsl,
    )

    HEAD_DIM = 512
    ROPE_DIM = 64
    RMS_EPS = 1e-6
    FP8_MAX = 448.0
    # C128 compress (Block8 kernel) requires state-cache block_size=8; C4 uses 16.
    BLOCK_SIZE = 8 if compress_ratio == 128 else 16
    KV_BLOCK_SIZE = 64
    device = "cuda"
    torch.manual_seed(7)

    overlap = 1 if compress_ratio == 4 else 0
    coff = 1 + overlap
    num_tokens = 8

    num_pages = (compress_ratio * num_tokens - 1) // BLOCK_SIZE + 2
    # The production CompressorStateCache is fp32.
    state_cache = torch.randn(
        num_pages, BLOCK_SIZE, 2 * coff * HEAD_DIM, dtype=torch.float32, device=device
    )
    block_table = torch.arange(num_pages, dtype=torch.int32, device=device).unsqueeze(0)
    token_to_req = torch.zeros(num_tokens, dtype=torch.int32, device=device)
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)
    positions = torch.arange(
        compress_ratio - 1,
        compress_ratio * num_tokens,
        compress_ratio,
        dtype=torch.int64,
        device=device,
    )
    rms_weight = torch.randn(HEAD_DIM, dtype=torch.bfloat16, device=device)
    cos_sin_cache = torch.randn(
        compress_ratio * num_tokens, ROPE_DIM, dtype=torch.float32, device=device
    )

    dtype = torch.float8_e4m3fn if store_fp8 else torch.bfloat16
    kv_n_blocks = (num_tokens + KV_BLOCK_SIZE - 1) // KV_BLOCK_SIZE + 1
    k_cache = torch.zeros(
        kv_n_blocks, KV_BLOCK_SIZE, HEAD_DIM, dtype=dtype, device=device
    )
    fp8_scale = torch.tensor(
        [0.5 if store_fp8 else 1.0], dtype=torch.float32, device=device
    )

    if compress_ratio == 4:
        fused_kv_compress_norm_rope_insert_sparse_attn_cutedsl(
            state_cache,
            token_to_req,
            positions,
            slot_mapping,
            block_table,
            BLOCK_SIZE,
            rms_weight,
            RMS_EPS,
            cos_sin_cache,
            k_cache,
            slot_mapping,
            KV_BLOCK_SIZE,
            k_cache.stride(0),
            head_size=HEAD_DIM,
            state_width=coff * HEAD_DIM,
            rope_head_dim=ROPE_DIM,
            fp8_max=FP8_MAX,
            quant_block=64,
            token_stride=576,
            scale_dim=8,
            compress_ratio=compress_ratio,
            overlap=True,
            store_full_kv=True,
            store_full_fp8=store_fp8,
            fp8_scale=fp8_scale,
        )
    else:
        compressed_kv = torch.empty(
            (num_tokens, HEAD_DIM), dtype=torch.float32, device=device
        )
        split_kv_compress_norm_rope_insert_sparse_attn_cutedsl(
            state_cache,
            token_to_req,
            positions,
            slot_mapping,
            block_table,
            BLOCK_SIZE,
            compressed_kv,
            rms_weight,
            RMS_EPS,
            cos_sin_cache,
            k_cache,
            slot_mapping,
            KV_BLOCK_SIZE,
            k_cache.stride(0),
            head_size=HEAD_DIM,
            state_width=coff * HEAD_DIM,
            rope_head_dim=ROPE_DIM,
            fp8_max=FP8_MAX,
            quant_block=64,
            token_stride=576,
            scale_dim=8,
            compress_ratio=compress_ratio,
            overlap=bool(overlap),
            store_full_kv=True,
            store_full_fp8=store_fp8,
            fp8_scale=fp8_scale,
        )

    ref = _reference_kv_compress_norm_rope(
        state_cache,
        block_table,
        positions,
        rms_weight,
        cos_sin_cache,
        compress_ratio,
        overlap,
        rms_eps=RMS_EPS,
        return_full_cache=True,
    )  # [num_tokens, HEAD_DIM] bf16

    actual = torch.stack(
        [k_cache[i // KV_BLOCK_SIZE, i % KV_BLOCK_SIZE] for i in range(num_tokens)]
    )
    if store_fp8:
        ref_fp8 = torch.clamp(ref.float() / fp8_scale, -FP8_MAX, FP8_MAX).to(
            torch.float8_e4m3fn
        )
        torch.testing.assert_close(actual.float(), ref_fp8.float(), rtol=0.0, atol=0.3)
    else:
        torch.testing.assert_close(actual.float(), ref.float(), rtol=3e-2, atol=3e-2)
