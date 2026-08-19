# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""gfx950 MLA decode for the Kimi-K3 MTP target-verify group.

The verify block is causal, but every one of a request's ``query_len``
positions still attends to the same committed KV prefix. The assembly path
tiles by head, so each tile re-reads the whole KV span; this kernel keeps the
block folded into a single workgroup tile and reads the span once.

It also takes ``q`` at its **real** head width. The assembly kernel implements
16/32/64/128 heads only, so a 24-head rank is padded to 32 before it is called
-- at ``query_len`` 6 that is 192 rows against 144 real ones, a third of the
fold computed and thrown away. This kernel derives its work from the row count
it is given, so the caller hands it the unpadded tensors. Padding is not
removed globally: it is skipped exactly on the calls this function serves, and
every path that falls back still gets the padded ``q``.

The kernel itself lives in aiter (``kimi_k3_verify_mla_decode``), which also owns
the split count, the row tile and the partial-accumulator workspace. Keeping
one copy means the two cannot drift, and drift is not benign -- the workspace
has to be sized from the same block count the kernel derives, or the launch
writes past it.

Everything below is therefore admission control: decide whether aiter can serve
this shape and return False if not, so the caller falls back to the assembly
kernel. The checks have to be a superset of aiter's, which raises rather than
declines.
"""

import os

import torch

from vllm.platforms import current_platform
from vllm.platforms.rocm import on_gfx950

D_QK = 576  # 512 latent + 64 rope
D_V = 512
KV_TILE = 128

# Opt-in. Declining is silent -- control falls through to the padded assembly
# path and nothing is logged -- so a deployment that has not asked for this
# should not be quietly running it, and one that has can be told from one that
# has not. Read once: a server does not change it mid-run, and this sits on the
# decode path.
_ENABLED = os.environ.get("VLLM_AITER_USE_K3_VERIFY_MLA", "0") == "1"


def enabled() -> bool:
    """Whether the deployment asked for this path at all.

    Exposed so the metadata builder can fold it into the one decision it makes
    at startup, instead of every call re-deriving something that cannot change.
    """
    return _ENABLED


def _supported() -> bool:
    # v_mfma_scale_f32_32x32x64_f8f6f4 and ds_read_b64_tr_b8 are gfx950-only.
    return _ENABLED and current_platform.is_rocm() and on_gfx950()


_SCALES: dict[int, tuple[torch.Tensor, float]] = {}


def _fscale(t) -> float | None:
    """Scalar value of a scale that may arrive as a device tensor.

    ``.item()`` synchronizes, which is illegal once a graph is capturing, so
    the value is cached the first time it is seen eagerly and only the cached
    float is read afterwards. Returns None when it would have to synchronize
    mid-capture, so the caller declines instead of aborting the capture.
    """
    if t is None:
        return 1.0
    if not isinstance(t, torch.Tensor):
        return float(t)
    hit = _SCALES.get(t.data_ptr())
    if hit is not None and hit[0] is t:
        return hit[1]
    if torch.cuda.is_current_stream_capturing():
        return None
    value = float(t.reshape(()).item())
    _SCALES[t.data_ptr()] = (t, value)
    return value


def can_serve(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    query_len: int,
    q_scale: float,
    kv_scale: float,
) -> bool:
    """Whether this call's shapes are serveable, decided without touching ``out``.

    Separate from the decode so a caller can ask before allocating the output
    buffer: the answer is no often enough -- a qlen that came back to 1, a scale
    first seen mid-capture -- that allocating first and discarding on a decline
    wastes a buffer per call. It reads nothing and writes nothing except the
    scale cache, which is idempotent.
    """
    if not _supported():
        return False
    if q.dtype is not torch.float8_e4m3fn or kv_cache.dtype is not torch.float8_e4m3fn:
        return False
    if q.dim() != 3 or q.size(2) != D_QK:
        return False
    if not q.is_contiguous():
        return False

    num_reqs = seq_lens.numel()
    # query_len == 1 is plain decode, which the assembly kernel already serves
    # without a fold; this path exists for the multi-position verify block.
    if query_len < 2 or q.size(0) != num_reqs * query_len:
        return False
    if block_table.size(0) < num_reqs:
        return False
    # Never copy the block table: under graph capture a .contiguous() here would
    # bake a fresh pointer into the graph, so every replay would read the page
    # map as it stood at capture time. The kernel takes a row stride, so a
    # row-sliced or over-wide table is passed through untouched.
    if block_table.dim() != 2 or block_table.stride(1) != 1:
        return False
    if block_table.dtype is not torch.int32 or seq_lens.dtype is not torch.int32:
        return False
    if kv_cache.size(1) % KV_TILE:
        # A KV tile would straddle a page; the kernel resolves one page per tile.
        return False

    if _fscale(q_scale) is None or _fscale(kv_scale) is None:
        return False

    # One name, whichever way aiter built the kernel. An aiter without it falls
    # back to the assembly path rather than raising.
    try:
        from aiter import kimi_k3_verify_mla_decode  # noqa: F401
    except ImportError:
        return False
    return True


def k3_verify_mla_decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    out: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    query_len: int,
    sm_scale: float,
    q_scale: float,
    kv_scale: float,
) -> bool:
    """Fill ``out`` in place. Returns False if the shape is not supported.

    Re-runs :func:`can_serve` even when the caller has already asked. That is a
    dozen attribute comparisons against a wrong answer for any caller that has
    not: this stays safe to call directly.

    ``q`` and ``out`` carry the rank's real head count, not the padded one.
    """
    if not can_serve(q, kv_cache, block_table, seq_lens, query_len, q_scale, kv_scale):
        return False
    if out.dtype is not torch.bfloat16 or out.size(-1) != D_V:
        return False
    if not out.is_contiguous():
        return False

    from aiter import kimi_k3_verify_mla_decode as _decode

    _decode(
        q,
        kv_cache,
        out,
        block_table,
        seq_lens,
        query_len,
        int(kv_cache.size(0)) * int(kv_cache.size(1)),
        sm_scale,
        _fscale(q_scale),
        _fscale(kv_scale),
    )
    return True
