# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Kimi-K3 MTP target-verify MLA path: admission control, and the answer it gives.

The kernel is aiter's; what is tested here is the vLLM side of the contract --
that unsupported shapes decline instead of raising or writing garbage, and that
a supported one matches a dense reference computed the way the model defines
the verify block: causal over query positions, one shared KV prefix per
request, all heads folded into one tile.
"""

import math

import pytest
import torch

import vllm.v1.attention.ops.rocm_aiter_k3_verify_mla as k3_verify_mla
from vllm.v1.attention.ops.rocm_aiter_k3_verify_mla import k3_verify_mla_decode


@pytest.fixture(autouse=True)
def _opt_in(monkeypatch):
    """Answer the kernel's question, not the deployment's.

    The op is opt-in through VLLM_AITER_USE_K3_VERIFY_MLA, read once at import,
    so an unset environment would make every test below assert against a
    decline it did not mean to test.
    """
    monkeypatch.setattr(k3_verify_mla, "_ENABLED", True)

from vllm.platforms import current_platform
from vllm.platforms.rocm import on_gfx950

D_QK, D_V = 576, 512
PAGE = 128

pytestmark = pytest.mark.skipif(
    not (current_platform.is_rocm() and on_gfx950()),
    reason="K3 verify MLA is gfx950-only",
)


def _case(T=2, qlen=6, nhead=24, npage=4, device="cuda", seed=0):
    torch.manual_seed(seed)
    q = (torch.randn(T * qlen, nhead, D_QK, device=device) * 0.3).to(
        torch.float8_e4m3fn
    )
    kv = (torch.randn(npage, PAGE, 1, D_QK, device=device) * 0.3).to(
        torch.float8_e4m3fn
    )
    block_table = (
        torch.arange(T * npage, device=device, dtype=torch.int32).view(T, npage) % npage
    )
    seq_lens = torch.full((T,), PAGE * npage, device=device, dtype=torch.int32)
    out = torch.zeros(T * qlen, nhead, D_V, device=device, dtype=torch.bfloat16)
    return q, kv, out, block_table, seq_lens, T, qlen


def _reference(q, kv, block_table, seq_lens, T, qlen, sm_scale):
    """Dense causal reference over the folded verify block."""
    nhead = q.size(1)
    ref = torch.empty(T * qlen, nhead, D_V, dtype=torch.float32, device=q.device)
    for t in range(T):
        pages = block_table[t].tolist()
        k = torch.cat([kv[p, :, 0, :] for p in pages], dim=0).float()
        kv_len = int(seq_lens[t])
        k = k[:kv_len]
        for p in range(qlen):
            row = q[t * qlen + p].float()  # [nhead, D_QK]
            logits = row @ k.t() * sm_scale
            # The verify block is causal in query position: position p may not
            # see the KV written by the positions after it.
            visible = kv_len - (qlen - 1 - p)
            logits[:, visible:] = float("-inf")
            probs = torch.softmax(logits, dim=-1)
            ref[t * qlen + p] = probs @ k[:, :D_V]
    return ref


def test_declines_unsupported_shapes():
    q, kv, out, bt, sl, T, qlen = _case()
    sm = 1.0 / math.sqrt(D_QK)

    # query_len 1 is plain decode: the assembly kernel serves it, no fold.
    assert not k3_verify_mla_decode(
        q[:T], kv, out[:T], bt, sl, 1, sm, 1.0, 1.0
    )
    # bf16 q is not this kernel's input.
    assert not k3_verify_mla_decode(
        q.to(torch.bfloat16), kv, out, bt, sl, qlen, sm, 1.0, 1.0
    )
    # A block table whose innermost stride is not 1 would need a copy, and a
    # copy under graph capture bakes a stale pointer into the graph.
    assert not k3_verify_mla_decode(
        q, kv, out, bt.t().contiguous().t(), sl, qlen, sm, 1.0, 1.0
    ) or bt.stride(1) == 1


def test_matches_dense_reference():
    q, kv, out, bt, sl, T, qlen = _case()
    sm = 1.0 / math.sqrt(D_QK)

    assert k3_verify_mla_decode(q, kv, out, bt, sl, qlen, sm, 1.0, 1.0)

    ref = _reference(q, kv, bt, sl, T, qlen, sm)
    got = out.float()
    rel = ((got - ref).norm() / ref.norm().clamp_min(1e-30)).item()
    # fp8 e4m3 carries ~2 decimal digits; the whole point of the kernel is that
    # it keeps the KV in fp8, so the tolerance is the format's, not the math's.
    assert rel < 5e-2, f"rel_l2 {rel:.3e}"
    assert torch.isfinite(got).all()


def test_output_is_flat_across_query_rows():
    """A mask or page-mapping error puts a subset of rows far off, not all."""
    q, kv, out, bt, sl, T, qlen = _case()
    sm = 1.0 / math.sqrt(D_QK)
    assert k3_verify_mla_decode(q, kv, out, bt, sl, qlen, sm, 1.0, 1.0)
    ref = _reference(q, kv, bt, sl, T, qlen, sm)

    per_row = []
    for p in range(qlen):
        idx = [t * qlen + p for t in range(T)]
        a, b = out.float()[idx], ref[idx]
        per_row.append(((a - b).norm() / b.norm().clamp_min(1e-30)).item())
    assert max(per_row) < 5e-2, per_row
    assert max(per_row) - min(per_row) < 2e-2, per_row
