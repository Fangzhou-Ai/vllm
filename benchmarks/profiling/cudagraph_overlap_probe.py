# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Probe: does multi-stream overlap work when the pattern is captured in a
HIP/CUDA graph?

The eager probes show no overlap because PyTorch's per-step launch + event op
CPU overhead (~25-50 us per iter) exceeds the savings (~17 us) at decode B=4
kernel sizes. CUDA graph / HIP graph capture removes that CPU overhead — the
whole sequence is captured once and replayed as a single dispatch — so the
question becomes: on GPU replay, do the captured streams actually run
concurrently?

This is the regime DSV4 production uses when ``compilation_config.cudagraph_mode
== FULL``. If GPU-side overlap shows up here, the multi-stream design is viable
under FULL cudagraph; if not, GPU command-processor serialization is the
binding constraint and the design cannot win on this stack period.
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch


def build_workload(M: int, K: int, N: int, device: torch.device):
    a0 = torch.randn(M, K, device=device, dtype=torch.bfloat16)
    b0 = torch.randn(K, N, device=device, dtype=torch.bfloat16)
    a1 = torch.randn(M, K, device=device, dtype=torch.bfloat16)
    b1 = torch.randn(K, N, device=device, dtype=torch.bfloat16)
    c0 = torch.empty(M, N, device=device, dtype=torch.bfloat16)
    c1 = torch.empty(M, N, device=device, dtype=torch.bfloat16)
    return a0, b0, c0, a1, b1, c1


def capture_serial(M, K, N, n_repeats, device):
    a0, b0, c0, a1, b1, c1 = build_workload(M, K, N, device)
    # warmup
    for _ in range(3):
        torch.matmul(a0, b0, out=c0)
        torch.matmul(a1, b1, out=c1)
    torch.accelerator.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n_repeats):
            torch.matmul(a0, b0, out=c0)
            torch.matmul(a1, b1, out=c1)
    return g


def capture_mstream(M, K, N, n_repeats, device):
    a0, b0, c0, a1, b1, c1 = build_workload(M, K, N, device)
    aux = torch.cuda.Stream(device=device)
    e_start = torch.cuda.Event()
    e_done = torch.cuda.Event()
    # warmup
    for _ in range(3):
        e_start.record()
        torch.matmul(a0, b0, out=c0)
        with torch.cuda.stream(aux):
            e_start.wait()
            torch.matmul(a1, b1, out=c1)
            e_done.record()
        e_done.wait()
    torch.accelerator.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n_repeats):
            e_start.record()
            torch.matmul(a0, b0, out=c0)
            with torch.cuda.stream(aux):
                e_start.wait()
                torch.matmul(a1, b1, out=c1)
                e_done.record()
            e_done.wait()
    return g


def time_replay(g, n_iters: int, warmup: int = 30) -> dict:
    for _ in range(warmup):
        g.replay()
    torch.accelerator.synchronize()

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    iter_us: list[float] = []
    t0 = time.perf_counter()
    for _ in range(n_iters):
        start_evt.record()
        g.replay()
        end_evt.record()
        end_evt.synchronize()
        iter_us.append(start_evt.elapsed_time(end_evt) * 1e3)
    t1 = time.perf_counter()
    iter_us.sort()
    return {
        "wall_us": (t1 - t0) * 1e6 / n_iters,
        "gpu_med": statistics.median(iter_us),
        "gpu_p05": iter_us[int(0.05 * len(iter_us))],
        "gpu_p95": iter_us[int(0.95 * len(iter_us))],
    }


def run(M: int, N: int, K: int, n_repeats: int, n_iters: int) -> None:
    device = torch.device("cuda")
    print(f"\n[M={M}, N={N}, K={K}, n_repeats={n_repeats} per replay]")

    # Eager baselines
    a0, b0, c0, a1, b1, c1 = build_workload(M, K, N, device)
    aux = torch.cuda.Stream(device=device)
    e_start = torch.cuda.Event()
    e_done = torch.cuda.Event()

    def eager_serial():
        for _ in range(n_repeats):
            torch.matmul(a0, b0, out=c0)
            torch.matmul(a1, b1, out=c1)

    def eager_mstream():
        for _ in range(n_repeats):
            e_start.record()
            torch.matmul(a0, b0, out=c0)
            with torch.cuda.stream(aux):
                e_start.wait()
                torch.matmul(a1, b1, out=c1)
                e_done.record()
            e_done.wait()

    def measure_eager(fn, label):
        for _ in range(20):
            fn()
        torch.accelerator.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        iter_us: list[float] = []
        for _ in range(n_iters):
            s.record()
            fn()
            e.record()
            e.synchronize()
            iter_us.append(s.elapsed_time(e) * 1e3)
        iter_us.sort()
        print(
            f"  {label:<28} "
            f"gpu_med={statistics.median(iter_us):>8.1f}us "
            f"gpu_p05={iter_us[int(0.05 * len(iter_us))]:>8.1f}us "
            f"gpu_p95={iter_us[int(0.95 * len(iter_us))]:>8.1f}us"
        )

    measure_eager(eager_serial, "eager-serial")
    measure_eager(eager_mstream, "eager-mstream")

    # Captured
    g_serial = capture_serial(M, K, N, n_repeats, device)
    stats_serial = time_replay(g_serial, n_iters=n_iters)
    print(
        f"  {'graph-serial':<28} "
        f"gpu_med={stats_serial['gpu_med']:>8.1f}us "
        f"gpu_p05={stats_serial['gpu_p05']:>8.1f}us "
        f"gpu_p95={stats_serial['gpu_p95']:>8.1f}us"
    )

    g_mstream = capture_mstream(M, K, N, n_repeats, device)
    stats_mstream = time_replay(g_mstream, n_iters=n_iters)
    print(
        f"  {'graph-mstream':<28} "
        f"gpu_med={stats_mstream['gpu_med']:>8.1f}us "
        f"gpu_p05={stats_mstream['gpu_p05']:>8.1f}us "
        f"gpu_p95={stats_mstream['gpu_p95']:>8.1f}us"
    )

    speedup = stats_serial["gpu_med"] / stats_mstream["gpu_med"]
    print(
        f"  -> graph-mstream / graph-serial = {1/speedup:.2f}x  "
        f"(speedup: {speedup:.2f}x)"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-iters", type=int, default=200)
    args = parser.parse_args()

    print(f"torch={torch.__version__} hip={torch.version.hip}")

    # n_repeats > 1 amortizes any per-graph overhead and makes the per-iter
    # overlap math easier to read.
    for size in [(256, 256, 256), (1024, 1024, 1024),
                 (4096, 4096, 4096), (8192, 8192, 8192)]:
        run(*size, n_repeats=8, n_iters=args.n_iters)


if __name__ == "__main__":
    main()
