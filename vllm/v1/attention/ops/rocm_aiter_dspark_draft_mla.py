# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""gfx950 MLA decode for the Kimi-K3 DSpark draft group.

The draft block is non-causal: every one of a request's ``query_len`` positions
attends to the same committed KV prefix. The Triton path expresses that by
flattening the block to one decode row per query token, which makes each row
re-read the whole KV span -- ``query_len`` times the traffic. This kernel keeps
the block folded into a single workgroup tile and reads the span once.

The kernel itself lives in aiter (``dspark_draft_mla_decode``), which also owns
the split count, the row tile and the partial-accumulator workspace. Those were
duplicated here while the kernel was vendored into vLLM; keeping one copy means
the two cannot drift, and drift is not benign -- the workspace has to be sized
from the same block count the kernel derives, or the launch writes past it.

Everything below is therefore admission control: decide whether aiter can serve
this shape and return False if not, so the caller falls back to Triton. The
checks have to be a superset of aiter's, which raises rather than declines.
"""

import torch

from vllm.platforms import current_platform
from vllm.platforms.rocm import on_gfx950

D_QK = 576  # 512 latent + 64 rope
D_V = 512
KV_TILE = 128


_SCALE_CACHE: dict[int, tuple[torch.Tensor, float]] = {}


def _scalar(t) -> float:
    """float(t) without a per-call device->host sync.

    The entry holds the tensor, not just the value: keying on the address alone
    is a silent-wrong-answer bug, because freeing a scale tensor lets the
    allocator hand its address to the next one, which would inherit the cached
    value. Keeping a reference makes that impossible.
    """
    if t is None:
        return 1.0
    if not isinstance(t, torch.Tensor):
        return float(t)
    hit = _SCALE_CACHE.get(t.data_ptr())
    if hit is not None and hit[0] is t:
        return hit[1]
    value = float(t.reshape(()).item())
    _SCALE_CACHE[t.data_ptr()] = (t, value)
    return value


def _supported() -> bool:
    # v_mfma_scale_f32_16x16x128_f8f6f4 and ds_read_b64_tr_b8 are gfx950-only.
    return current_platform.is_rocm() and on_gfx950()


def dspark_draft_mla_decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    out: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    query_len: int,
    sm_scale: float,
    q_scale: torch.Tensor | float,
    kv_scale: torch.Tensor | float,
) -> bool:
    """Fill ``out`` in place. Returns False if the shape is not supported."""
    if not _supported():
        return False
    if q.dtype is not torch.float8_e4m3fn or kv_cache.dtype is not torch.float8_e4m3fn:
        return False
    if out.dtype is not torch.bfloat16:
        return False
    if q.dim() != 3 or q.size(2) != D_QK or out.size(-1) != D_V:
        return False

    num_reqs = seq_lens.numel()
    if query_len < 2 or q.size(0) != num_reqs * query_len:
        return False
    # Never copy the block table: under graph capture a .contiguous() here would
    # bake a fresh pointer into the graph, so every replay would read the page
    # map as it stood at capture time. The kernel takes a row stride, so a
    # row-sliced or over-wide table is passed through untouched.
    if block_table.dim() != 2 or block_table.stride(1) != 1:
        return False
    if kv_cache.size(1) % KV_TILE:
        # A KV tile would straddle a page; the kernel resolves one page per tile.
        return False

    # aiter ships this kernel either JIT-compiled or as a prebuilt code
    # object, and the two spell the entry point differently. Naming only one
    # makes the other aiter fall back to Triton with no error.
    try:
        from aiter import dspark_draft_mla_decode as _decode
    except ImportError:
        return False

    _decode(
        q,
        kv_cache,
        out,
        block_table,
        seq_lens,
        query_len,
        int(kv_cache.size(0)) * int(kv_cache.size(1)),
        sm_scale,
        _scalar(q_scale),
        _scalar(kv_scale),
    )
    return True
