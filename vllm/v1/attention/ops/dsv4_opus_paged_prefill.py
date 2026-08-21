"""Direct paged-cache adapter for the DeepSeek-V4 opus fp8 sparse prefill.

The regular ROCm prefill path dequantises and gathers ``fp8_ds_mla`` pages into
a temporary BF16 slab before attention.  The opus paged AITER kernel addresses
those pages by slot index instead, so this module builds zero-copy NoPE/RoPE
views and dispatches attention before the gather ever runs.

This is a sibling of ``dsv4_h40_prefill``, and the difference that matters is
what it does *not* do.  The h40 path has to rewrite the cache in place --
collapsing seven per-64 exponents per token down to one -- and carry a global
``kv_max_e`` accumulator for its fixed softmax frame.  This kernel consumes the
pool's per-64 UE8M0 scales natively, because a per-64 exponent applies exactly
to both of the per-32 halves the MFMA wants.  So there is no collapse pass, no
mutation of the KV cache, and no process-lifetime accumulator that a long
request can ratchet.  Q likewise arrives as plain bf16 and is packed inside the
kernel, so neither operand needs a pre-pass.

The path is opt-in.  Returning ``False`` means nothing was modified and the
caller must run its normal fallback; since this adapter never mutates the
cache, declining is always safe, at any point.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass

import torch

_D_HEAD = 512
_D_NOPE = 448
_D_ROPE = 64
_ROW_BYTES = _D_NOPE + _D_ROPE * 2
_SCALE_BYTES = 8
# The kernel compiles only the T_M=8 pipeline; H <= 32 is a different pipeline
# with no paged variant.  The op exports its own threshold and we take the
# stricter of the two.
_MIN_HEADS = 33

_ENABLED: bool | None = None
_SERVED = 0
_DECLINED: dict[str, int] = {}
_SCRATCH: dict[tuple[str, torch.device], torch.Tensor] = {}


@dataclass(frozen=True)
class _PagedCache:
    nope: torch.Tensor
    rope: torch.Tensor
    page_size: int
    rows_per_page: int
    page_shift: int
    scale_offset: int


def enabled() -> bool:
    """Whether the process opted into the opus paged sparse prefill."""
    global _ENABLED
    if _ENABLED is None:
        _ENABLED = os.getenv("VLLM_DSV4_OPUS_PAGED_PREFILL", "0") == "1"
    return _ENABLED


def _scratch(
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    key = (name, device)
    tensor = _SCRATCH.get(key)
    if tensor is None or tensor.shape != shape or tensor.dtype != dtype:
        tensor = torch.empty(shape, dtype=dtype, device=device)
        _SCRATCH[key] = tensor
    return tensor


def _decline(reason: str) -> bool:
    first = reason not in _DECLINED
    _DECLINED[reason] = _DECLINED.get(reason, 0) + 1
    if first:
        print(f"[opus-paged] declining sparse prefill: {reason}", flush=True)
    return False


def _is_gfx950(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    arch = getattr(torch.cuda.get_device_properties(device), "gcnArchName", "")
    return str(arch).split(":", 1)[0] == "gfx950"


def _load_aiter_op() -> tuple[Callable[..., torch.Tensor], int]:
    from aiter.ops.dsv4_mla_prefill_opus import (
        DSV4_MLA_PREFILL_OPUS_MIN_H,
        dsv4_mla_prefill_opus,
    )

    return dsv4_mla_prefill_opus, int(DSV4_MLA_PREFILL_OPUS_MIN_H)


def _paged_cache_views(cache: torch.Tensor, page_size: int) -> _PagedCache:
    """Describe a vLLM 584-byte-row cache without copying its allocation."""
    if cache.dtype != torch.uint8:
        raise ValueError(f"cache dtype must be uint8, got {cache.dtype}")
    if cache.dim() < 2 or cache.shape[0] == 0:
        raise ValueError(f"cache must contain pages, got shape={tuple(cache.shape)}")
    if cache.stride(-1) != 1:
        raise ValueError(f"cache innermost stride must be 1, got {cache.stride()}")
    if page_size <= 0 or page_size & (page_size - 1):
        raise ValueError(f"page_size must be a power of two, got {page_size}")
    # The kernel has no flat layout: page_shift 0 would make it read RoPE bytes
    # as exponents rather than fall back to anything.
    if page_size < 2:
        raise ValueError(f"page_size must be at least 2, got {page_size}")

    num_pages = cache.shape[0]
    bytes_per_page = cache.stride(0)
    scale_offset = page_size * _ROW_BYTES
    required_bytes = scale_offset + page_size * _SCALE_BYTES
    if bytes_per_page < required_bytes or bytes_per_page % _ROW_BYTES:
        raise ValueError(
            "invalid fp8_ds_mla page layout: "
            f"stride={bytes_per_page}, required={required_bytes}, "
            f"row_bytes={_ROW_BYTES}"
        )
    rows_per_page = bytes_per_page // _ROW_BYTES

    # Packed KV groups give each layer a non-zero storage offset and use the
    # whole packed block as stride(0).  Span only the bytes physically available
    # after this layer's base pointer.
    storage_bytes = cache.untyped_storage().nbytes()
    byte_offset = cache.storage_offset() * cache.element_size()
    available_bytes = storage_bytes - byte_offset
    view_rows = min(available_bytes // _ROW_BYTES, num_pages * rows_per_page)
    if view_rows < (num_pages - 1) * rows_per_page + page_size:
        raise ValueError(
            "fp8_ds_mla cache storage is too short for its page stride: "
            f"available={available_bytes}, pages={num_pages}, "
            f"rows_per_page={rows_per_page}, page_size={page_size}"
        )
    # The op wants a whole number of pages; in the normal case the line above
    # already gives exactly num_pages * rows_per_page.
    view_rows -= view_rows % rows_per_page

    byte_stream = cache.as_strided(
        (view_rows * _ROW_BYTES,),
        (1,),
        storage_offset=cache.storage_offset(),
    )
    flat_fp8 = byte_stream.view(torch.float8_e4m3fn)
    nope = flat_fp8.as_strided(
        (view_rows, _D_HEAD),
        (_ROW_BYTES, 1),
        storage_offset=flat_fp8.storage_offset(),
    )
    flat_bf16 = byte_stream.view(torch.bfloat16)
    rope = flat_bf16.as_strided(
        (view_rows, _D_ROPE),
        (_ROW_BYTES // 2, 1),
        storage_offset=flat_bf16.storage_offset() + _D_NOPE // 2,
    )
    return _PagedCache(
        nope=nope,
        rope=rope,
        page_size=page_size,
        rows_per_page=rows_per_page,
        page_shift=page_size.bit_length() - 1,
        scale_offset=scale_offset,
    )


def _dense_indices(
    indices: torch.Tensor,
    lens: torch.Tensor,
    tokens: int,
    name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if indices.dtype != torch.int32:
        raise ValueError(f"{name} indices must be int32, got {indices.dtype}")
    if lens.dtype != torch.int32:
        raise ValueError(f"{name} lengths must be int32, got {lens.dtype}")
    if indices.shape[0] != tokens or lens.shape != (tokens,):
        raise ValueError(
            f"{name} metadata shape mismatch: indices={tuple(indices.shape)}, "
            f"lens={tuple(lens.shape)}, tokens={tokens}"
        )
    dense = indices.reshape(tokens, -1)
    if dense.stride(1) != 1:
        raise ValueError(f"{name} indices must be dense by row, got {dense.stride()}")
    return dense, lens


def try_opus_paged(
    *,
    q: torch.Tensor,
    output: torch.Tensor,
    compressed_k_cache: torch.Tensor | None,
    swa_k_cache: torch.Tensor,
    compressed_indices: torch.Tensor | None,
    compressed_lens: torch.Tensor | None,
    swa_indices: torch.Tensor,
    swa_lens: torch.Tensor,
    compressed_page_size: int | None,
    swa_page_size: int,
    kv_cache_dtype: str,
    attn_sink: torch.Tensor | None,
    softmax_scale: float,
    head_dim: int,
    nope_head_dim: int,
    rope_head_dim: int,
) -> bool:
    """Try the paged opus prefill; return ``False`` if it did not run.

    Nothing is mutated on any path, so declining is safe at any point -- unlike
    the h40 adapter, which becomes committed once its cache collapse starts.
    """
    global _SERVED

    if not enabled():
        return False
    if kv_cache_dtype != "fp8_ds_mla":
        return _decline(f"kv_cache_dtype={kv_cache_dtype}")
    if (head_dim, nope_head_dim, rope_head_dim) != (_D_HEAD, _D_NOPE, _D_ROPE):
        return _decline(f"head dims {head_dim}/{nope_head_dim}/{rope_head_dim}")
    if q.dim() != 3 or q.dtype != torch.bfloat16 or q.shape[-1] != _D_HEAD:
        return _decline(f"q shape/dtype {tuple(q.shape)} {q.dtype}")

    tokens, heads, _ = q.shape
    if tokens == 0:
        return _decline("empty query")
    if not _is_gfx950(q.device):
        return _decline(f"device {q.device} is not gfx950")
    if q.stride(2) != 1 or q.stride(0) != heads * q.stride(1):
        return _decline(f"q layout {q.stride()}")
    if (
        output.shape != q.shape
        or output.dtype != torch.bfloat16
        or output.stride(2) != 1
        or output.stride(0) != heads * output.stride(1)
    ):
        return _decline(
            f"output shape/dtype/layout {tuple(output.shape)} "
            f"{output.dtype} {output.stride()}"
        )
    if attn_sink is None:
        return _decline("attn_sink is None")
    if (
        attn_sink.dtype != torch.float32
        or attn_sink.numel() < heads
        or not attn_sink.is_contiguous()
    ):
        return _decline(
            f"attn_sink shape/dtype/layout {tuple(attn_sink.shape)} "
            f"{attn_sink.dtype} contiguous={attn_sink.is_contiguous()}"
        )

    try:
        prefill, aiter_min_heads = _load_aiter_op()
    except (ImportError, AttributeError) as exc:
        return _decline(f"AITER opus paged symbols unavailable: {exc}")
    if heads < max(_MIN_HEADS, aiter_min_heads):
        return _decline(f"H={heads}, minimum={max(_MIN_HEADS, aiter_min_heads)}")

    try:
        swa = _paged_cache_views(swa_k_cache, swa_page_size)
        swa_dense, swa_lens = _dense_indices(swa_indices, swa_lens, tokens, "SWA")
        if compressed_k_cache is None:
            if compressed_indices is not None or compressed_lens is not None:
                return _decline("compressed metadata without compressed cache")
            # SWA-only: point the prefix segment at the same pool with an empty
            # index range.  The kernel's per-segment page descriptors then agree
            # with the views, which is what the op requires.
            prefix = swa
            prefix_dense = _scratch(
                "empty_prefix_indices", (tokens, 1), torch.int32, q.device
            ).fill_(0)
            prefix_lens = _scratch(
                "empty_prefix_lens", (tokens,), torch.int32, q.device
            ).zero_()
        else:
            if (
                compressed_indices is None
                or compressed_lens is None
                or compressed_page_size is None
            ):
                return _decline("incomplete compressed cache metadata")
            prefix = _paged_cache_views(compressed_k_cache, compressed_page_size)
            prefix_dense, prefix_lens = _dense_indices(
                compressed_indices, compressed_lens, tokens, "compressed"
            )
    except ValueError as exc:
        return _decline(str(exc))

    # The dense index form: no CSR is built, so kv_indptr goes unused.
    empty_indptr = _scratch("empty_indptr", (0,), torch.int32, q.device)
    prefill(
        q_nope=q[..., :_D_NOPE],
        # q_nope may stay a view -- the kernel takes its row stride as an
        # argument -- but q_rope may not: that layout has 64 baked in at compile
        # time, so a [..., 448:] slice of a [T, H, 512] Q addresses wrongly.
        # The copy is ~50 us per layer at T=8192 H=128 against ~1.8 ms of
        # attention.  Teaching the RoPE layout to take a runtime stride, the way
        # q_nope already does, would remove it.
        q_rope=q[..., _D_NOPE:].contiguous(),
        unified_kv_nope=prefix.nope,
        unified_kv_rope=prefix.rope,
        kv_indices_prefix=prefix_dense.reshape(-1),
        kv_indptr_prefix=empty_indptr,
        kv_nope=swa.nope,
        kv_rope=swa.rope,
        kv_indices_extend=swa_dense.reshape(-1),
        kv_indptr_extend=empty_indptr,
        attn_sink=attn_sink[:heads],
        softmax_scale=float(softmax_scale),
        page_shift_prefix=prefix.page_shift,
        rows_per_page_prefix=prefix.rows_per_page,
        scale_off_prefix=prefix.scale_offset,
        page_shift_extend=swa.page_shift,
        rows_per_page_extend=swa.rows_per_page,
        scale_off_extend=swa.scale_offset,
        out=output,
        kv_lens_prefix=prefix_lens,
        kv_lens_extend=swa_lens,
        kv_stride_q_prefix=prefix_dense.shape[1],
        kv_stride_q_extend=swa_dense.shape[1],
    )

    _SERVED += 1
    if _SERVED == 1 or _SERVED % 2000 == 0:
        declines = ", ".join(
            f"{reason}={count}" for reason, count in sorted(_DECLINED.items())
        )
        print(
            f"[opus-paged] serving sparse prefill: T={tokens} H={heads} "
            f"prefix={prefix_dense.shape[1]} swa={swa_dense.shape[1]} "
            f"served={_SERVED} declined[{declines or 'none'}]",
            flush=True,
        )
    return True


def _reset_for_tests() -> None:
    global _ENABLED, _SERVED
    _ENABLED = None
    _SERVED = 0
    _DECLINED.clear()
    _SCRATCH.clear()


__all__ = ["enabled", "try_opus_paged"]
