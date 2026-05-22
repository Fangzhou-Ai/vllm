# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Minimal probe for DSV4 c4a indexer multi-stream overlap on ROCm.

Rebuilds the per-layer pattern that ``DeepseekV4MLAAttention.forward`` uses
when ``self.indexer is not None``::

    q, _ = maybe_execute_in_parallel(
        wq_b_kv_insert_and_compress,  # default stream
        indexer(...),  # aux stream
        ln_events[0],
        ln_events[1],
        aux_stream,
    )

but at synthetic c4a shapes (DSV4-Pro decode, B=4, n_indexer_heads=64,
index_head_dim=128, q_lora_rank=1536). Calls the real AITER
``deepgemm_fp8_paged_mqa_logits`` kernel on the aux side via
``rocm_fp8_paged_mqa_logits``, so the GPU work matches production.

Goals:
  1. Measure per-iter wall time and per-stream GPU time for three modes:
       - serial: fn0(); fn1(); on default stream.
       - mstream: maybe_execute_in_parallel(fn0, fn1, e0, e1, aux).
       - aux_only: only fn1(), to bound aux's standalone GPU cost.
  2. Capture a chrome trace via torch.profiler so the user can open it and
     verify (a) whether default and aux kernels actually run concurrently
     and (b) how long HIP event ops take on the CPU.
  3. Probe HIP event-op cost in isolation (record/wait, no kernels).

Run::

    VLLM_ROCM_USE_AITER=1 python benchmarks/profiling/dsv4_indexer_multi_stream_probe.py

Outputs::

    /tmp/dsv4_probe_trace.json   # chrome trace, open in chrome://tracing
    stdout summary
"""

from __future__ import annotations

import argparse
import statistics
import time
from contextlib import nullcontext

import torch

from vllm.utils.multi_stream_utils import maybe_execute_in_parallel
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
    _get_decode_logits_buffer,
    rocm_fp8_paged_mqa_logits,
)

DEVICE = torch.device("cuda")
FP8 = torch.float8_e4m3fn


# ---------------------------------------------------------------------------
# Synthetic c4a shapes (DSV4-Pro decode at TP=1; absolute numbers don't matter
# for the overlap question — what matters is that the indexer-side GPU work is
# substantial enough to be worth overlapping against the smaller default-side
# work, which is the same regime the real model is in).
# ---------------------------------------------------------------------------
BATCH = 4
NEXT_N = 1
N_HEAD = 64  # indexer heads
HEAD_DIM = 128  # indexer head dim
Q_LORA_RANK = 1536
HIDDEN_SIZE = 7168

# Default-stream prep proxies wq_b + qnorm/rope/kv_insert + compressor:
# we approximate with one small GEMM + one elementwise op.
N_HEADS_DEFAULT = 16  # n_local_heads at TP=8 for DSV4-Pro MLA
HEAD_DIM_DEFAULT = 192  # MLA head_dim


def build_kv_cache(
    seq_len: int, block_size: int = 64
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Build a paged FP8 KV cache for the AITER indexer kernel.

    Returns (kv_cache_fp8, block_tables, context_lens, max_model_len).
    """
    # Layout per `rocm_fp8_paged_mqa_logits` docstring:
    #   [num_blocks, block_size, 1, D + 4 bytes scale]  uint8
    max_model_len = ((seq_len + block_size - 1) // block_size) * block_size
    num_blocks_per_seq = max_model_len // block_size
    num_blocks = num_blocks_per_seq * BATCH + 8  # a few spares
    kv_cache_fp8 = torch.randint(
        0,
        255,
        (num_blocks, block_size, 1, HEAD_DIM + 4),
        device=DEVICE,
        dtype=torch.uint8,
    )
    block_tables = torch.zeros(
        (BATCH, num_blocks_per_seq), device=DEVICE, dtype=torch.int32
    )
    for b in range(BATCH):
        for j in range(num_blocks_per_seq):
            block_tables[b, j] = b * num_blocks_per_seq + j
    context_lens = torch.full((BATCH,), seq_len, device=DEVICE, dtype=torch.int32)
    return kv_cache_fp8, block_tables, context_lens, max_model_len


def build_inputs(seq_len: int):
    qr = torch.randn(BATCH * NEXT_N, Q_LORA_RANK, device=DEVICE, dtype=torch.bfloat16)

    # Indexer wq_b: (qr) -> (B, n_head * head_dim)
    w_indexer_wq_b = torch.randn(
        Q_LORA_RANK, N_HEAD * HEAD_DIM, device=DEVICE, dtype=torch.bfloat16
    )
    # Default wq_b proxy: (qr) -> (B, n_heads_default * head_dim_default)
    w_default_wq_b = torch.randn(
        Q_LORA_RANK,
        N_HEADS_DEFAULT * HEAD_DIM_DEFAULT,
        device=DEVICE,
        dtype=torch.bfloat16,
    )

    weights = torch.randn(BATCH * NEXT_N, N_HEAD, device=DEVICE, dtype=torch.float32)

    kv_cache_fp8, block_tables, context_lens, max_model_len = build_kv_cache(seq_len)

    return {
        "qr": qr,
        "w_indexer_wq_b": w_indexer_wq_b,
        "w_default_wq_b": w_default_wq_b,
        "weights": weights,
        "kv_cache_fp8": kv_cache_fp8,
        "block_tables": block_tables,
        "context_lens": context_lens,
        "max_model_len": max_model_len,
    }


def make_fn0(inp):
    """Default-stream callable: small GEMM + elementwise touch.

    Mimics ``wq_b(qr) + _fused_qnorm_rope_kv_insert + compressor``. We don't
    need to be exact; we just need the GPU time profile to roughly match.
    """
    qr = inp["qr"]
    w = inp["w_default_wq_b"]

    def fn0():
        q_default = qr @ w  # tiny GEMM, output (B, 16*192) bf16
        q_default = torch.nn.functional.silu(q_default)  # elementwise touch
        return q_default

    return fn0


def make_fn1(inp, *, use_persistent_buf: bool):
    """Aux-stream callable: indexer wq_b + AITER paged_mqa_logits.

    Optionally uses the persistent ``_get_decode_logits_buffer`` so the
    AITER kernel writes into a pre-allocated buffer instead of a per-call
    ``torch.full`` allocation.
    """
    qr = inp["qr"]
    w = inp["w_indexer_wq_b"]
    weights = inp["weights"]
    kv_cache_fp8 = inp["kv_cache_fp8"]
    block_tables = inp["block_tables"]
    context_lens = inp["context_lens"]
    max_model_len = inp["max_model_len"]

    def fn1():
        q_idx = qr @ w  # (B, 64*128) bf16
        q_idx = q_idx.view(BATCH, NEXT_N, N_HEAD, HEAD_DIM)
        q_fp8 = q_idx.to(FP8)
        out_logits = None
        if use_persistent_buf:
            out_logits = _get_decode_logits_buffer(
                BATCH * NEXT_N, max_model_len, DEVICE
            )
        logits = rocm_fp8_paged_mqa_logits(
            q_fp8,
            kv_cache_fp8,
            weights,
            context_lens,
            block_tables,
            None,  # schedule_metadata: unused on gfx950 branch
            max_model_len=max_model_len,
            out_logits=out_logits,
        )
        return logits

    return fn1


def time_iters(
    fn,
    n_iters: int,
    warmup: int = 20,
    profile_path: str | None = None,
) -> dict:
    """Run fn() n_iters times, return wall-time / GPU-time stats.

    If profile_path is set, the run is wrapped with torch.profiler and the
    chrome trace is dumped to that path.
    """
    for _ in range(warmup):
        fn()
    torch.accelerator.synchronize()

    profiler_ctx: object = nullcontext()
    prof = None
    if profile_path is not None:
        prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=False,
            with_stack=False,
        )
        profiler_ctx = prof

    iter_us: list[float] = []
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    with profiler_ctx:
        # Whole-loop timing for wall-clock measurement.
        wall_t0 = time.perf_counter()
        for _ in range(n_iters):
            start_evt.record()
            fn()
            end_evt.record()
            end_evt.synchronize()
            iter_us.append(start_evt.elapsed_time(end_evt) * 1e3)
        wall_t1 = time.perf_counter()

    if prof is not None and profile_path is not None:
        prof.export_chrome_trace(profile_path)

    iter_us.sort()
    return {
        "wall_us_per_iter": (wall_t1 - wall_t0) * 1e6 / n_iters,
        "gpu_us_median": statistics.median(iter_us),
        "gpu_us_mean": statistics.mean(iter_us),
        "gpu_us_p05": iter_us[int(0.05 * len(iter_us))],
        "gpu_us_p95": iter_us[int(0.95 * len(iter_us))],
        "n_iters": n_iters,
    }


def probe_event_overhead(n_iters: int = 1000) -> dict:
    """Measure HIP event record/wait CPU cost in isolation (no kernels)."""
    aux = torch.cuda.Stream(device=DEVICE)
    e0 = torch.cuda.Event()
    e1 = torch.cuda.Event()

    # Warmup
    for _ in range(50):
        e0.record()
        with torch.cuda.stream(aux):
            e0.wait()
            e1.record()
        e1.wait()
    torch.accelerator.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iters):
        e0.record()
        with torch.cuda.stream(aux):
            e0.wait()
            e1.record()
        e1.wait()
    torch.accelerator.synchronize()
    t1 = time.perf_counter()

    return {
        "per_call_us": (t1 - t0) * 1e6 / n_iters,
        "n_iters": n_iters,
    }


def run(seq_len: int, n_iters: int, trace_dir: str) -> None:
    print(
        f"\n=== probe @ B={BATCH}, next_n={NEXT_N}, seq_len={seq_len}, "
        f"n_heads={N_HEAD}, head_dim={HEAD_DIM} ==="
    )

    inp = build_inputs(seq_len)
    aux_stream = torch.cuda.Stream(device=DEVICE)
    e0 = torch.cuda.Event()
    e1 = torch.cuda.Event()

    fn0 = make_fn0(inp)
    # Two variants of fn1: with and without persistent out_logits buffer.
    fn1_alloc = make_fn1(inp, use_persistent_buf=False)
    fn1_persist = make_fn1(inp, use_persistent_buf=True)

    def serial_alloc():
        fn0()
        fn1_alloc()

    def serial_persist():
        fn0()
        fn1_persist()

    def mstream_alloc():
        maybe_execute_in_parallel(fn0, fn1_alloc, e0, e1, aux_stream)

    def mstream_persist():
        maybe_execute_in_parallel(fn0, fn1_persist, e0, e1, aux_stream)

    def aux_only_persist():
        fn1_persist()

    cases = [
        ("aux-only       (fn1 alone, default stream)", aux_only_persist, None),
        ("serial         (fn0+fn1, default, torch.full)", serial_alloc, None),
        ("serial-persist (fn0+fn1, default, persistent buf)", serial_persist, None),
        (
            "mstream        (fn0||fn1, multi-stream, torch.full)",
            mstream_alloc,
            f"{trace_dir}/mstream_alloc_seq{seq_len}.json",
        ),
        (
            "mstream-persist(fn0||fn1, multi-stream, persistent buf)",
            mstream_persist,
            f"{trace_dir}/mstream_persist_seq{seq_len}.json",
        ),
    ]

    print(
        f"{'mode':<55} {'wall_us':>10} {'gpu_med':>10} {'gpu_p05':>10} {'gpu_p95':>10}"
    )
    for name, fn, trace_path in cases:
        stats = time_iters(fn, n_iters=n_iters, warmup=30, profile_path=trace_path)
        print(
            f"{name:<55} "
            f"{stats['wall_us_per_iter']:>10.1f} "
            f"{stats['gpu_us_median']:>10.1f} "
            f"{stats['gpu_us_p05']:>10.1f} "
            f"{stats['gpu_us_p95']:>10.1f}"
        )

    evt = probe_event_overhead()
    print(
        f"\nHIP event round-trip (2 records + 2 waits, no kernels): "
        f"{evt['per_call_us']:.2f} us / call (n={evt['n_iters']})"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[1024, 4096, 9472])
    parser.add_argument("--n-iters", type=int, default=200)
    parser.add_argument("--trace-dir", type=str, default="/tmp/dsv4_probe")
    args = parser.parse_args()

    import os

    os.makedirs(args.trace_dir, exist_ok=True)

    torch.manual_seed(0)
    print(
        f"torch={torch.__version__} hip={torch.version.hip} "
        f"device={torch.cuda.get_device_name(0)}"
    )

    for seq_len in args.seq_lens:
        run(seq_len, args.n_iters, args.trace_dir)

    print(f"\nChrome traces dumped under {args.trace_dir}/")
    print("Open in chrome://tracing to inspect per-stream kernel timeline.")


if __name__ == "__main__":
    main()
