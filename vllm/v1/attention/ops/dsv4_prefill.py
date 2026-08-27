"""Direct paged-cache adapter for DeepSeek-V4 sparse prefill.

The regular ROCm prefill path gathers ``fp8_ds_mla`` pages into a temporary
BF16 slab before attention.  The AITER MLA prefill kernel addresses those pages
directly, so this module builds zero-copy NoPE/RoPE views and dispatches
attention before the BF16 gather.  The cache is read exactly as it was written:
the kernel requantises each staged tile in LDS, so nothing here rewrites the
seven per-64 exponents a page stores per token.

The path is deliberately opt-in.  Returning ``False`` means that no output
tensor was modified and the caller must execute its normal fallback.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_D_HEAD = 512
_D_NOPE = 448
_D_ROPE = 64
_ROW_BYTES = _D_NOPE + _D_ROPE * 2
_SCALE_BYTES = 8
_MIN_HEADS = 16
_MAX_BUFFER_BYTES = 1 << 32

_ENABLED: bool | None = None
_SERVED = 0
_DECLINED: dict[str, int] = {}
_SCRATCH: dict[tuple[str, torch.device], torch.Tensor] = {}
# The kernel ignores kv_max_e -- it derives the softmax frame per KV tile -- but
# the op still takes it.  One zeroed scalar per device, made once.
_IGNORED_MAX_E: dict[torch.device, torch.Tensor] = {}


@dataclass(frozen=True)
class _PagedCache:
    buffer: torch.Tensor
    nope: torch.Tensor
    rope: torch.Tensor
    page_size: int
    bytes_per_page: int
    rows_per_page: int
    page_shift: int
    scale_offset: int


def enabled() -> bool:
    """Whether this process takes the direct AITER prefill path.

    On by default.  Set ``VLLM_ROCM_DSV4_MLA_PREFILL=0`` to fall back to the Triton
    sparse prefill.  Declining is always safe -- every check below runs before
    anything is written -- so a tree without the prefill code object simply
    takes the old path, silently.
    """
    global _ENABLED
    if _ENABLED is None:
        _ENABLED = os.getenv("VLLM_ROCM_DSV4_MLA_PREFILL", "1") == "1"
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
    _DECLINED[reason] = _DECLINED.get(reason, 0) + 1
    logger.warning_once("dsv4 sparse prefill declined: %s", reason)
    return False


def _is_gfx950(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    props = torch.cuda.get_device_properties(device)
    arch = getattr(props, "gcnArchName", "")
    return str(arch).split(":", 1)[0] == "gfx950"


def _load_aiter_ops() -> tuple[
    Callable[..., torch.Tensor],
    int,
    bool,
]:
    # The kernels come from AITER as a prebuilt code object, so the schedule of
    # a kernel that sits at 504 of 512 ArchVGPRs is fixed at ship time rather
    # than left to whichever clang the image carries. The JIT-compiled build of
    # the same kernels lives in aiter.ops.pa_sparse_prefill_opus and exports the
    # attention op under its older name.
    from aiter.ops.dsv4_mla_prefill import (
        PA_FP8_GLOBAL64,
        PA_FP8_MIN_H,
        dsv4_mla_prefill,
    )

    return (
        dsv4_mla_prefill,
        int(PA_FP8_MIN_H),
        bool(PA_FP8_GLOBAL64),
    )


def _paged_cache_views(
    cache: torch.Tensor,
    page_size: int,
    *,
    allow_global64: bool = False,
) -> _PagedCache:
    """Describe a vLLM 584-byte-row cache without copying its allocation."""
    if cache.dtype != torch.uint8:
        raise ValueError(f"cache dtype must be uint8, got {cache.dtype}")
    if cache.dim() < 2 or cache.shape[0] == 0:
        raise ValueError(f"cache must contain pages, got shape={tuple(cache.shape)}")
    if cache.stride(-1) != 1:
        raise ValueError(f"cache innermost stride must be 1, got {cache.stride()}")
    if page_size <= 0 or page_size & (page_size - 1):
        raise ValueError(f"page_size must be a power of two, got {page_size}")

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
    address_span = (num_pages - 1) * bytes_per_page + required_bytes
    if address_span > _MAX_BUFFER_BYTES and not allow_global64:
        raise ValueError(
            "fp8_ds_mla cache exceeds the kernel's 32-bit buffer addressing: "
            f"span={address_span}, limit={_MAX_BUFFER_BYTES}, "
            f"pages={num_pages}, page_stride={bytes_per_page}"
        )

    rows_per_page = bytes_per_page // _ROW_BYTES
    page_view = cache.as_strided(
        (num_pages, required_bytes),
        (cache.stride(0), 1),
        storage_offset=cache.storage_offset(),
    )

    # Packed KV groups give each layer a non-zero storage offset and use the
    # whole packed block as stride(0).  A synthetic [pages, stride(0)] view
    # would overrun storage for every layer except the first.  Span only the
    # bytes physically available after this layer's base pointer; the kernel's page
    # descriptor still jumps between pages with rows_per_page.
    storage_bytes = cache.untyped_storage().nbytes()
    byte_offset = cache.storage_offset() * cache.element_size()
    available_bytes = storage_bytes - byte_offset
    view_rows = min(
        available_bytes // _ROW_BYTES,
        num_pages * rows_per_page,
    )
    if view_rows < (num_pages - 1) * rows_per_page + page_size:
        raise ValueError(
            "fp8_ds_mla cache storage is too short for its page stride: "
            f"available={available_bytes}, pages={num_pages}, "
            f"rows_per_page={rows_per_page}, page_size={page_size}"
        )
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
        buffer=page_view,
        nope=nope,
        rope=rope,
        page_size=page_size,
        bytes_per_page=bytes_per_page,
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


def _ignored_max_e(device: torch.device) -> torch.Tensor:
    t = _IGNORED_MAX_E.get(device)
    if t is None:
        t = torch.zeros(1, dtype=torch.int32, device=device)
        _IGNORED_MAX_E[device] = t
    return t


def try_dsv4_prefill(
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
    """Try the direct prefill path; return ``False`` before touching output."""
    global _SERVED

    if not enabled():
        return False
    if kv_cache_dtype != "fp8_ds_mla":
        return _decline(f"kv_cache_dtype={kv_cache_dtype}")
    if (head_dim, nope_head_dim, rope_head_dim) != (
        _D_HEAD,
        _D_NOPE,
        _D_ROPE,
    ):
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
        mla_prefill, aiter_min_heads, global64 = _load_aiter_ops()
    except (ImportError, AttributeError) as exc:
        return _decline(f"AITER paged prefill symbols unavailable: {exc}")
    if heads < max(_MIN_HEADS, aiter_min_heads):
        return _decline(f"H={heads}, minimum={max(_MIN_HEADS, aiter_min_heads)}")

    try:
        swa = _paged_cache_views(
            swa_k_cache,
            swa_page_size,
            allow_global64=global64,
        )
        swa_dense, swa_lens = _dense_indices(swa_indices, swa_lens, tokens, "SWA")
        if compressed_k_cache is None:
            if compressed_indices is not None or compressed_lens is not None:
                return _decline("compressed metadata without compressed cache")
            prefix = swa
            prefix_dense = _scratch(
                "empty_prefix_indices",
                (tokens, 1),
                torch.int32,
                q.device,
            ).fill_(-1)
            prefix_lens = _scratch(
                "empty_prefix_lens",
                (tokens,),
                torch.int32,
                q.device,
            ).zero_()
        else:
            if (
                compressed_indices is None
                or compressed_lens is None
                or compressed_page_size is None
            ):
                return _decline("incomplete compressed cache metadata")
            prefix = _paged_cache_views(
                compressed_k_cache,
                compressed_page_size,
                allow_global64=global64,
            )
            prefix_dense, prefix_lens = _dense_indices(
                compressed_indices,
                compressed_lens,
                tokens,
                "compressed",
            )
    except ValueError as exc:
        return _decline(str(exc))

    kv_max_e = _ignored_max_e(q.device)

    empty_indptr = _scratch("empty_indptr", (0,), torch.int32, q.device)
    mla_prefill(
        q_nope=q[..., :_D_NOPE],
        q_rope=q[..., _D_NOPE:],
        unified_kv_nope=prefix.nope,
        unified_kv_rope=prefix.rope,
        kv_indices_prefix=prefix_dense.reshape(-1),
        kv_indptr_prefix=empty_indptr,
        kv_nope=swa.nope,
        kv_rope=swa.rope,
        kv_indices_extend=swa_dense.reshape(-1),
        kv_indptr_extend=empty_indptr,
        attn_sink=attn_sink[:heads],
        kv_max_e=kv_max_e,
        softmax_scale=float(softmax_scale),
        out=output,
        kv_lens_prefix=prefix_lens,
        kv_lens_extend=swa_lens,
        kv_stride_q_prefix=prefix_dense.shape[1],
        kv_stride_q_extend=swa_dense.shape[1],
        page_shift_prefix=prefix.page_shift,
        rows_per_page_prefix=prefix.rows_per_page,
        scale_off_prefix=prefix.scale_offset,
        page_shift_extend=swa.page_shift,
        rows_per_page_extend=swa.rows_per_page,
        scale_off_extend=swa.scale_offset,
    )

    _SERVED += 1
    if _SERVED == 1 or _SERVED % 2000 == 0:
        declines = ", ".join(
            f"{reason}={count}" for reason, count in sorted(_DECLINED.items())
        )
        logger.info(
            "dsv4 sparse prefill: T=%d H=%d prefix=%d swa=%d served=%d declined[%s]",
            tokens,
            heads,
            prefix_dense.shape[1],
            swa_dense.shape[1],
            _SERVED,
            declines or "none",
        )
    return True


__all__ = ["enabled", "try_dsv4_prefill"]
