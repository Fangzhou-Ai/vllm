"""Microbenchmark for the ROCm sparse-attn decode kernel.

Times one call of:
  * baseline `_rocm_sparse_attn_decode_ragged_triton` (BLOCK_K=16, warps=8)
  * v2      `rocm_sparse_attn_decode_fp8_resident`   (env-configurable)

at the conc=128 DSV4 / TP=8 operating point. Use this to iterate kernel
configs in seconds rather than chewing 13 minutes per full server bench.
"""

from __future__ import annotations

import os
import sys

# The venv has `include-system-site-packages=true`, so a stale system-wide
# vllm at /usr/local/lib/python3.12/dist-packages/vllm shadows our editable
# checkout for sub-packages like `vllm.v1.attention.ops`. Prepending the
# repo root to sys.path makes our local source win. Top-level `vllm` is
# already loaded from the editable; this fixes namespace resolution for
# subpackages.
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402

from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (  # noqa: E402
    _rocm_sparse_attn_decode_ragged_triton,
)
from vllm.v1.attention.ops.rocm_dsv4_sgl_sparse_attn import (  # noqa: E402
    rocm_sparse_attn_decode_fp8_resident,
)

NOPE_HEAD_DIM = 448
ROPE_HEAD_DIM = 64
HEAD_DIM = NOPE_HEAD_DIM + ROPE_HEAD_DIM
NUM_HEADS = 16  # 128 model heads / TP=8
BATCH = 128
TOPK = 2048
SWA_WINDOW = 1024
BLOCK_SIZE_MAIN = 16
BLOCK_SIZE_EXTRA = 16

# Need at least 2 K main rows + 2 K extra rows to cover the indices we
# sample at this batch size. Use a smaller cache so the pack runs in a few
# seconds. Index *aliasing* (the same slot getting hit many times) is fine
# for timing purposes — the kernel pays the same memory cost regardless.
NUM_MAIN_BLOCKS = 256  # 4 K rows
NUM_EXTRA_BLOCKS = 256


def _pack_fp8_cache(kv: torch.Tensor, block_size: int) -> torch.Tensor:
    """Pack into vLLM's FP8 DS-MLA paged layout (matches the test helper).

    Storage per token = 576 B (FP8 NoPE + bf16 RoPE). Per-token scale = 8 B
    placed after `block_size * 576` bytes of token data within each block.
    Scale bytes filled with 127 (i.e. 2**0 = 1.0) for simplicity, which
    matches the e2e test setup; absolute kernel timing is unaffected.
    """
    from vllm.platforms import current_platform

    num_tokens = kv.shape[0]
    num_blocks = (num_tokens + block_size - 1) // block_size
    cache = torch.zeros(
        (num_blocks, block_size, 584), dtype=torch.uint8, device=kv.device
    )
    cache_flat = cache.view(torch.uint8).flatten()
    kv_nope_fp8 = (
        kv[:, :NOPE_HEAD_DIM].to(current_platform.fp8_dtype()).view(torch.uint8)
    )
    kv_rope_u8 = kv[:, NOPE_HEAD_DIM:].contiguous().view(torch.uint8)

    for slot in range(num_tokens):
        block_idx = slot // block_size
        pos = slot % block_size
        block_base = block_idx * cache.stride(0)
        token_base = block_base + pos * 576
        scale_base = block_base + block_size * 576 + pos * 8
        cache_flat[token_base : token_base + NOPE_HEAD_DIM].copy_(kv_nope_fp8[slot])
        cache_flat[
            token_base + NOPE_HEAD_DIM : token_base + NOPE_HEAD_DIM + ROPE_HEAD_DIM * 2
        ].copy_(kv_rope_u8[slot])
        cache_flat[scale_base : scale_base + 7].fill_(127)
    return cache


def _build_inputs(device: torch.device, seed: int = 0):
    torch.manual_seed(seed)
    q = torch.randn(BATCH, NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device) * 0.125

    main_rows = NUM_MAIN_BLOCKS * BLOCK_SIZE_MAIN
    extra_rows = NUM_EXTRA_BLOCKS * BLOCK_SIZE_EXTRA
    main_kv = torch.randn(main_rows, HEAD_DIM, dtype=torch.bfloat16, device=device) * 0.125
    extra_kv = torch.randn(extra_rows, HEAD_DIM, dtype=torch.bfloat16, device=device) * 0.125

    main_cache = _pack_fp8_cache(main_kv, BLOCK_SIZE_MAIN)
    extra_cache = _pack_fp8_cache(extra_kv, BLOCK_SIZE_EXTRA)

    # Realistic main (SWA) length per query: pre-clamped to swa_window.
    main_len = SWA_WINDOW
    extra_len = TOPK
    main_idx = torch.randint(0, main_rows, (BATCH * main_len,), dtype=torch.int32, device=device)
    extra_idx = torch.randint(
        0, extra_rows, (BATCH * extra_len,), dtype=torch.int32, device=device
    )
    main_indptr = torch.arange(0, BATCH * main_len + 1, main_len, dtype=torch.int32, device=device)
    extra_indptr = torch.arange(
        0, BATCH * extra_len + 1, extra_len, dtype=torch.int32, device=device
    )

    return q, main_cache, main_idx, main_indptr, extra_cache, extra_idx, extra_indptr


@torch.inference_mode()
def bench_once(fn, args, warmup: int = 5, iters: int = 50) -> float:
    for _ in range(warmup):
        out = fn(*args)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        out = fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms


def main() -> None:
    device = torch.device("cuda")
    scale = HEAD_DIM**-0.5

    print("[*] Building inputs (batch=128, swa=1024, topk=2048)...")
    q, main_cache, main_idx, main_indptr, extra_cache, extra_idx, extra_indptr = (
        _build_inputs(device)
    )
    print(f"    main_cache={tuple(main_cache.shape)}, dtype={main_cache.dtype}")
    print(f"    extra_cache={tuple(extra_cache.shape)}")

    common_kwargs = dict(
        q=q,
        main_cache=main_cache,
        main_indices=main_idx,
        main_indptr=main_indptr,
        scale=scale,
        attn_sink=None,
        nope_head_dim=NOPE_HEAD_DIM,
        rope_head_dim=ROPE_HEAD_DIM,
        extra_cache=extra_cache,
        extra_indices=extra_idx,
        extra_indptr=extra_indptr,
    )

    def call_baseline():
        return _rocm_sparse_attn_decode_ragged_triton(**common_kwargs)

    def call_v2():
        return rocm_sparse_attn_decode_fp8_resident(**common_kwargs)

    print("[*] Warming up baseline...")
    base_ms = bench_once(call_baseline, ())
    print(f"baseline: {base_ms*1000:.1f} us/call")

    # Sweep: (BLOCK_H, BLOCK_K, num_warps, num_stages, loop_stages).
    # Optimal so far on MI355 + DSV4 is BLOCK_H=16 BLOCK_K=32 W=4 S=2 L=1.
    configs = [
        # Best known
        ("BH=16 BK=32 W=4 S=2 L=1", "16", "32", "4", "2", "1"),
        # Try smaller BLOCK_H to expose more parallelism (batch * cdiv(16, BH) CTAs)
        ("BH=8  BK=32 W=4 S=2 L=1", "8", "32", "4", "2", "1"),
        ("BH=8  BK=64 W=4 S=2 L=1", "8", "64", "4", "2", "1"),
        ("BH=4  BK=32 W=4 S=2 L=1", "4", "32", "4", "2", "1"),
        # Try BLOCK_H=32 if num_heads gets padded
        ("BH=32 BK=32 W=4 S=2 L=1", "32", "32", "4", "2", "1"),
    ]

    for label, bh, bk, nw, ns, ls in configs:
        os.environ["VLLM_ROCM_DSV4_SGL_BLOCK_H"] = bh
        os.environ["VLLM_ROCM_DSV4_SGL_BLOCK_K"] = bk
        os.environ["VLLM_ROCM_DSV4_SGL_NUM_WARPS"] = nw
        os.environ["VLLM_ROCM_DSV4_SGL_NUM_STAGES"] = ns
        os.environ["VLLM_ROCM_DSV4_SGL_LOOP_STAGES"] = ls
        try:
            ms = bench_once(call_v2, ())
        except Exception as e:
            print(f"{label:28s} FAIL: {type(e).__name__}: {str(e)[:80]}")
            continue
        speedup = base_ms / ms
        delta = (ms - base_ms) / base_ms * 100
        print(
            f"{label:28s} {ms*1000:8.1f} us/call "
            f"({speedup:.3f}x vs baseline, {delta:+.1f}%)"
        )


if __name__ == "__main__":
    main()
