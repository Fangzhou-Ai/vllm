"""Direct paged-cache adapter for the DeepSeek-V4 h40 fp8 sparse *decode*.

Sibling of :mod:`dsv4_prefill`, and deliberately the same kernel: decode's
sparse attention is the same operation as prefill's -- N query rows, each
gathering its own list of compressed rows plus its own sliding window out of
the same ``fp8_ds_mla`` pool -- only with a much smaller N.  So this module
does not introduce a second kernel; it feeds ``dsv4_mla_sparse`` the tensors
the decode path already has, through the same two-segment contract the prefill
adapter uses for prefix and extend.

All 64 attention layers of a decode step go through it: the 61 target layers
map their compressed top-k rows onto the first segment and their sliding window
onto the second, and the three DSpark draft layers -- whose ``compress_ratio``
is 0, so they have no compressed cache at all -- describe the first segment as
empty and use only the second.

Two things differ from the prefill adapter and both are decode-specific:

* **CSR, not the dense form.**  The ratio-4 layers only ever materialise a
  ragged global index list (``compute_global_topk_ragged_indices_and_indptr``),
  and the ROCm SWA builder already keeps its ragged decode buffers in
  persistent graph-capture storage.  Taking the CSR form therefore costs
  nothing to build and makes padded rows unambiguous: a padded row is one whose
  ``indptr`` interval is empty, rather than one whose length happens to be
  zeroed.

* **It runs under a captured CUDA graph.**  Decode is
  ``FULL_DECODE_ONLY``-captured, so every buffer this adapter hands the kernel
  must have an address that survives replay.  Everything here is either a view
  of a caller tensor or a per-shape entry in a cache that is never resized in
  place -- see ``_zero_indptr`` and ``_max_e``.  The op is loaded in
  :func:`enabled` rather than at first call for the same reason: loading
  registers a code object, which is illegal while a stream is capturing, and
  under ``FULL_DECODE_ONLY`` the first call can land inside a capture.

Nothing is mutated on any path: the cache is read only and Q is packed inside
the kernel.  Returning ``False`` is safe at any point and the caller runs its
normal Triton path.

The kernel pins its accumulator to the first KV tile's log2 frame instead of
rescaling an online maximum, which is what makes it 1.22-1.34x per call at
decode shapes.  The documented bound is that a row whose later logits beat its
first tile's by ~127 octaves overflows to NaN.  Measured drift at this shape is
under one octave, but the first tile at decode is wherever the indexer pointed
rather than the start of the sequence, so the path stays opt-in.
"""

from __future__ import annotations

import os
from collections.abc import Callable

import torch

from vllm.logger import init_logger

# Same pool, same 584-byte-row geometry, same validation -- importing it keeps
# one description of the layout rather than two that can drift apart.  It also
# carries the 32-bit buffer-addressing guard, which this kernel needs: it read
# rows past a 4 GiB span as zero until the loads were widened, and
# ``PA_FP8_GLOBAL64`` is how the shipped object reports that it was.
from vllm.v1.attention.ops.dsv4_prefill import (
    _D_HEAD,
    _D_NOPE,
    _D_ROPE,
    _paged_cache_views,
)

logger = init_logger(__name__)

# Floor of this adapter's own; the shipped object reports its real minimum as
# ``PA_FP8_MIN_H`` and the larger of the two governs.
_MIN_HEADS = 16

_ENABLED: bool | None = None
_OP: Callable[..., torch.Tensor] | None = None
_AITER_MIN_HEADS = 0
_GLOBAL64 = False
_HAS_ROW_MAP = False
_SERVED = 0
_DECLINED: dict[str, int] = {}
# Keyed by (device, length) and never resized: a captured graph holds the
# address of whatever it was given, so an entry that a later, differently
# shaped call could reallocate would leave the earlier graph pointing at freed
# memory.  vLLM captures one graph per batch size, so this holds one small
# tensor per captured size.
# Grid-position -> query-row permutation, keyed by (device, tokens) and never
# resized, for the same reason _ZERO_INDPTR is: a captured graph holds the
# address.  It depends only on the token count, so there is one entry per
# captured batch size and no per-step work.
_ROW_MAP: dict[tuple[torch.device, int, int], torch.Tensor] = {}
# XCC count per device, read once from aiter.  1 disables the placement, which
# is both the right answer for a single-XCD partition and the safe fallback for
# an aiter that cannot be asked.
_XCC: dict[torch.device, int] = {}

_ZERO_INDPTR: dict[tuple[torch.device, int], torch.Tensor] = {}
# The op keeps ``kv_max_e`` for call compatibility and ignores it, but it still
# dereferences the pointer, so the scalar has to outlive every captured graph.
_MAX_E: dict[torch.device, torch.Tensor] = {}
# Stand-in index list for the empty compressed segment of an SWA-only layer.
_EMPTY_INDICES: dict[torch.device, torch.Tensor] = {}


def enabled() -> bool:
    """Whether the process opted into the direct AITER sparse decode."""
    global _ENABLED, _OP, _AITER_MIN_HEADS, _GLOBAL64, _HAS_ROW_MAP
    if _ENABLED is None:
        _ENABLED = os.getenv("VLLM_ROCM_DSV4_MLA_DECODE", "0") == "1"
        if _ENABLED:
            try:
                _OP, _AITER_MIN_HEADS, _GLOBAL64, _HAS_ROW_MAP = _load_aiter_ops()
            except (ImportError, AttributeError) as exc:
                logger.warning(
                    "dsv4 sparse decode: AITER symbols unavailable (%s); "
                    "falling back to Triton",
                    exc,
                )
                _ENABLED = False
    return _ENABLED


def _decline(reason: str) -> bool:
    _DECLINED[reason] = _DECLINED.get(reason, 0) + 1
    logger.warning_once("dsv4 sparse decode declined: %s", reason)
    return False


def _is_gfx950(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    arch = getattr(torch.cuda.get_device_properties(device), "gcnArchName", "")
    return str(arch).split(":", 1)[0] == "gfx950"


def _load_aiter_ops() -> tuple[Callable[..., torch.Tensor], int, bool, bool]:
    # Same import the prefill adapter uses, so a build carrying one carries the
    # other; the object is prebuilt, so its schedule does not depend on
    # whichever clang the image happens to ship.
    from aiter.ops.dsv4_mla_sparse import (
        PA_FP8_GLOBAL64,
        PA_FP8_MIN_H,
        dsv4_mla_sparse,
    )

    # getattr on the module, not a from-import: a missing name raises
    # ImportError, and enabled() handles that by disabling the whole fast path.
    # Losing the kernel over an absent tuning knob would be a bad trade.
    from aiter.ops import dsv4_mla_sparse as _module

    has_row_map = bool(getattr(_module, "PA_FP8_HAS_ROW_MAP", False))
    return dsv4_mla_sparse, int(PA_FP8_MIN_H), bool(PA_FP8_GLOBAL64), has_row_map


def _csr(
    indices: torch.Tensor | None,
    indptr: torch.Tensor | None,
    tokens: int,
    name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if indices is None or indptr is None:
        raise ValueError(f"{name} ragged metadata is missing")
    if indices.dtype != torch.int32 or indptr.dtype != torch.int32:
        raise ValueError(
            f"{name} ragged metadata must be int32, got {indices.dtype}/{indptr.dtype}"
        )
    if indices.dim() != 1 or indptr.dim() != 1:
        raise ValueError(
            f"{name} ragged metadata must be 1-D, got "
            f"{tuple(indices.shape)}/{tuple(indptr.shape)}"
        )
    if indptr.numel() != tokens + 1:
        raise ValueError(
            f"{name} indptr must hold {tokens + 1} entries, got {indptr.numel()}"
        )
    if not indices.is_contiguous() or not indptr.is_contiguous():
        raise ValueError(f"{name} ragged metadata must be contiguous")
    return indices, indptr


def _empty_indices(device: torch.device) -> torch.Tensor:
    """A zero-length int32 index list for a segment with no rows."""
    tensor = _EMPTY_INDICES.get(device)
    if tensor is None:
        tensor = torch.zeros(0, dtype=torch.int32, device=device)
        _EMPTY_INDICES[device] = tensor
    return tensor


def _xcc(device: torch.device) -> int:
    """XCCs in the device's current compute partition, or 1 if unknown.

    Only aiter can answer this: torch does not surface it, and calling
    libamdhip64 from here would mean hardcoding
    ``hipDeviceAttributeNumberOfXccs``' numeric value, which moves with the ROCm
    version -- a worse constant than the one it replaces.  An aiter without the
    entry point returns 1 and the placement simply does not apply, which is the
    same behaviour as before this optimisation existed.
    """
    cached = _XCC.get(device)
    if cached is None:
        try:
            from aiter.ops.dsv4_mla_sparse import xcc_count

            cached = int(xcc_count(device.index or 0))
        except Exception:
            cached = 1
        _XCC[device] = cached
    return cached


def _row_map(
    tokens: int, groups: int, device: torch.device
) -> torch.Tensor | None:
    """Grid position -> query row, so one request's rows share an XCD.

    Workgroups go round-robin over the XCDs (``XCC_ID = blockIdx.x % n``, n =
    the device's XCC count) and each XCD has its own L2.  Decode rows are
    request-major, so a request's rows land on that many *different* XCDs, and
    every one of those L2s pulls the same KV separately -- the rows of one
    request read the *same* compressed cache (byte-identical index lists on the
    ratio-128 layers, ~0.98 tile overlap on ratio-4).

    Cutting the row order into ``n`` contiguous runs and giving run ``c`` to the
    grid positions with ``b % n == c`` fixes that, because a request's rows are
    adjacent.  But contiguous runs also hand each XCD a contiguous *slice of the
    batch*, and when per-row work correlates with position in the batch that is
    a load imbalance: measured on a synthetic batch whose context length rises
    with request index, one XCD drew 4x another's tiles and the placement cost
    2.7% instead of paying.  So the requests are dealt out ``n``-ways first --
    run ``c`` takes requests c, c+n, c+2n, ... -- which samples the whole batch
    into every run and leaves the co-location intact.  Measured: -2.67% -> +1.28%
    on that batch, and +2.74/+3.94% -> +2.60/+3.55% on the real ratio-4 and
    ratio-128 shapes, a difference well inside the intervals.

    Dealing requests out needs to know how many rows one contributes, which the
    plain form did not.  When that is unavailable or does not divide the row
    count, this falls back to the contiguous form: it is what the production
    batch measures best on anyway, since there every row carries identical work
    (captured ratio-4 and ratio-128 batches both have zero spread) and the
    imbalance this guards against cannot arise.

    ``n`` is read from the device rather than written here, because it is a
    property of the *current compute partition* -- amd-smi can set CPX and make
    it 1 -- not of the chip.  A literal would silently stop placing anything
    under a non-default partition: the permutation would still be valid, so
    nothing would fail, it would just quietly do nothing.

    Returns None when the count is not a multiple of ``n``, or when the device
    has a single XCD and there is nothing to place, which leaves the kernel on
    its default order (reverse, so the heaviest rows dispatch first).
    """
    n = _xcc(device)
    if n < 2 or tokens < n or tokens % n:
        return None
    key = (device, tokens, groups)
    row_map = _ROW_MAP.get(key)
    if row_map is None:
        rows = torch.arange(tokens, dtype=torch.int32, device=device)
        m = tokens // groups if groups > 0 else 0
        if m > 1 and groups * m == tokens and groups >= n:
            # Requests dealt n-ways: run c takes c, c+n, c+2n, ...  Wrapping
            # rather than reshaping keeps this valid for any request count --
            # the production batch is 30 requests, which n does not divide.
            order = torch.cat(
                [torch.arange(c, groups, n, device=device) for c in range(n)]
            )
            rows = (
                order.unsqueeze(1) * m
                + torch.arange(m, device=device)
            ).reshape(-1).to(torch.int32)
        row_map = rows[
            torch.arange(tokens, device=device).view(n, tokens // n).t().reshape(-1)
        ]
        _ROW_MAP[key] = row_map
        # Which form was taken is not visible in the results -- the placement is
        # bit-identical either way -- so say it once per shape.  A build that
        # cannot read the speculative config degrades to the contiguous form
        # silently otherwise.
        logger.info(
            "dsv4 sparse decode: XCD row_map for %d rows, %d XCDs, %s",
            tokens, n,
            f"{groups} requests dealt {n}-ways, {m} rows each"
            if m > 1 and groups * m == tokens and groups >= n
            else f"contiguous runs (requests={groups}, rows={tokens})",
        )
    return row_map


def _max_e(device: torch.device) -> torch.Tensor:
    tensor = _MAX_E.get(device)
    if tensor is None:
        tensor = torch.zeros(1, dtype=torch.int32, device=device)
        _MAX_E[device] = tensor
    return tensor


def _zero_indptr(tokens: int, device: torch.device) -> torch.Tensor:
    """An all-zero ``[tokens + 1]`` CSR pointer: every row empty."""
    key = (device, tokens + 1)
    tensor = _ZERO_INDPTR.get(key)
    if tensor is None:
        tensor = torch.zeros(tokens + 1, dtype=torch.int32, device=device)
        _ZERO_INDPTR[key] = tensor
    return tensor


def try_dsv4_decode(
    *,
    q: torch.Tensor,
    output: torch.Tensor,
    compressed_k_cache: torch.Tensor | None,
    swa_k_cache: torch.Tensor,
    compressed_ragged_indices: torch.Tensor | None,
    compressed_ragged_indptr: torch.Tensor | None,
    swa_ragged_indices: torch.Tensor | None,
    swa_ragged_indptr: torch.Tensor | None,
    compressed_page_size: int | None,
    swa_page_size: int,
    kv_cache_dtype: str,
    attn_sink: torch.Tensor | None,
    softmax_scale: float,
    head_dim: int,
    nope_head_dim: int,
    rope_head_dim: int,
    num_decodes: int = 0,
) -> bool:
    """Try the direct AITER sparse decode; return ``False`` if it did not run."""
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

    assert _OP is not None  # enabled() loads it, or turns itself off
    floor = max(_MIN_HEADS, _AITER_MIN_HEADS)
    if heads < floor:
        return _decline(f"H={heads}, minimum={floor}")

    try:
        swa = _paged_cache_views(swa_k_cache, swa_page_size, allow_global64=_GLOBAL64)
        swa_indices, swa_indptr = _csr(
            swa_ragged_indices, swa_ragged_indptr, tokens, "SWA"
        )
        if compressed_k_cache is None:
            # SWA-only layers -- the three DSpark draft layers, whose
            # compress_ratio is 0.  The op has no single-segment form, so the
            # compressed segment is described as empty: the SWA pool stands in
            # for its base pointer (the op dereferences a segment's base even
            # when the indptr says no row is read) and every row's interval is
            # empty.  Both stand-ins are per-shape cache entries, so a captured
            # graph keeps holding a live address.
            prefix = swa
            prefix_indices = _empty_indices(q.device)
            prefix_indptr = _zero_indptr(tokens, q.device)
        else:
            if compressed_page_size is None:
                return _decline("compressed cache without a page size")
            prefix = _paged_cache_views(
                compressed_k_cache, compressed_page_size, allow_global64=_GLOBAL64
            )
            prefix_indices, prefix_indptr = _csr(
                compressed_ragged_indices,
                compressed_ragged_indptr,
                tokens,
                "compressed",
            )
    except ValueError as exc:
        return _decline(str(exc))

    # Empty rows are expressed by the indptr interval, so a graph-padded row
    # needs no special case here -- but the op still has to be handed a pointer
    # for a segment with no rows at all.
    if prefix_indices.numel() == 0:
        prefix_indptr = _zero_indptr(tokens, q.device)
    if swa_indices.numel() == 0:
        swa_indptr = _zero_indptr(tokens, q.device)

    row_map_kwarg: dict[str, torch.Tensor] = {}
    if _HAS_ROW_MAP:
        row_map = _row_map(tokens, num_decodes, q.device)
        if row_map is not None:
            row_map_kwarg["row_map"] = row_map

    _OP(
        q_nope=q[..., :_D_NOPE],
        # Both Q views use the explicit per-head stride accepted by the kernel.
        q_rope=q[..., _D_NOPE:],
        unified_kv_nope=prefix.nope,
        unified_kv_rope=prefix.rope,
        kv_indices_prefix=prefix_indices,
        kv_indptr_prefix=prefix_indptr,
        kv_nope=swa.nope,
        kv_rope=swa.rope,
        kv_indices_extend=swa_indices,
        kv_indptr_extend=swa_indptr,
        attn_sink=attn_sink[:heads],
        softmax_scale=float(softmax_scale),
        page_shift_prefix=prefix.page_shift,
        rows_per_page_prefix=prefix.rows_per_page,
        scale_off_prefix=prefix.scale_offset,
        page_shift_extend=swa.page_shift,
        rows_per_page_extend=swa.rows_per_page,
        scale_off_extend=swa.scale_offset,
        out=output,
        # CSR form: kv_lens must stay None or the op reads the indices as a
        # dense [N, stride] array instead.
        kv_lens_prefix=None,
        kv_lens_extend=None,
        # Kept for call compatibility and ignored, but still dereferenced.
        kv_max_e=_max_e(q.device),
        # Placement only: which CU computes which row.  The result is
        # bit-identical -- see _row_map.  Omitted entirely on an AITER that
        # predates the argument, since an unknown keyword fails at call time.
        **row_map_kwarg,
    )

    _SERVED += 1
    if _SERVED == 1 or _SERVED % 20000 == 0:
        declines = ", ".join(
            f"{reason}={count}" for reason, count in sorted(_DECLINED.items())
        )
        logger.info(
            "dsv4 sparse decode: N=%d H=%d prefix_nnz=%d swa_nnz=%d "
            "served=%d declined[%s]",
            tokens,
            heads,
            prefix_indices.numel(),
            swa_indices.numel(),
            _SERVED,
            declines or "none",
        )
    return True


def _reset_for_tests() -> None:
    global _ENABLED, _SERVED, _OP, _AITER_MIN_HEADS, _GLOBAL64, _HAS_ROW_MAP
    _ENABLED = None
    _OP = None
    _AITER_MIN_HEADS = 0
    _GLOBAL64 = False
    _HAS_ROW_MAP = False
    _SERVED = 0
    _DECLINED.clear()
    _ROW_MAP.clear()
    _ZERO_INDPTR.clear()
    _MAX_E.clear()
    _EMPTY_INDICES.clear()


__all__ = ["enabled", "try_dsv4_decode"]
