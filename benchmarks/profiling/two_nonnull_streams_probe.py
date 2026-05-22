# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Re-do the stream overlap probe but with TWO non-NULL streams.

The previous probe used the per-thread default stream (cuda_stream=0x0 on this
build) as one of the two. On CUDA, the legacy NULL stream serializes against
every other stream by design; HIP inherits the same semantic. If that's the
real cause of the "no overlap" observation, two explicit streams should
overlap fine.
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch


def run(M: int, N: int, K: int, n_iters: int, warmup: int = 30) -> None:
    device = torch.device("cuda")
    a0 = torch.randn(M, K, device=device, dtype=torch.bfloat16)
    b0 = torch.randn(K, N, device=device, dtype=torch.bfloat16)
    a1 = torch.randn(M, K, device=device, dtype=torch.bfloat16)
    b1 = torch.randn(K, N, device=device, dtype=torch.bfloat16)

    s0 = torch.cuda.Stream(device=device)
    s1 = torch.cuda.Stream(device=device)
    e0 = torch.cuda.Event()
    e1 = torch.cuda.Event()
    e_start = torch.cuda.Event()

    def fn(a, b):
        return a @ b

    def serial_on_s0():
        with torch.cuda.stream(s0):
            fn(a0, b0)
            fn(a1, b1)

    def mstream_two_nonnull():
        # Record a start event on the current (NULL/default) stream so both
        # explicit streams have a single fan-out point.
        e_start.record()
        with torch.cuda.stream(s0):
            e_start.wait()
            fn(a0, b0)
            e0.record()
        with torch.cuda.stream(s1):
            e_start.wait()
            fn(a1, b1)
            e1.record()
        e0.wait()
        e1.wait()

    def mstream_default_plus_aux():
        # The pattern actually used in DeepSeek-V4: one half on the current
        # (default) stream, the other half on an aux stream.
        e_start.record()
        fn(a0, b0)  # default
        with torch.cuda.stream(s1):
            e_start.wait()
            fn(a1, b1)
            e1.record()
        e1.wait()

    def measure(fn, label):
        for _ in range(warmup):
            fn()
        torch.accelerator.synchronize()
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        iter_us: list[float] = []
        t0 = time.perf_counter()
        for _ in range(n_iters):
            start_evt.record()
            fn()
            end_evt.record()
            end_evt.synchronize()
            iter_us.append(start_evt.elapsed_time(end_evt) * 1e3)
        t1 = time.perf_counter()
        iter_us.sort()
        print(
            f"  {label:<35} wall={((t1 - t0) * 1e6 / n_iters):>7.1f}us  "
            f"gpu_med={statistics.median(iter_us):>7.1f}us  "
            f"gpu_p05={iter_us[int(0.05 * len(iter_us))]:>7.1f}us  "
            f"gpu_p95={iter_us[int(0.95 * len(iter_us))]:>7.1f}us"
        )

    print(f"\n[M={M}, N={N}, K={K}, dtype=bf16]")
    measure(serial_on_s0, "serial(x2 on s0)")
    measure(mstream_two_nonnull, "mstream(x2 on s0,s1 — both nonnull)")
    measure(mstream_default_plus_aux, "mstream(default + s1 aux)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-iters", type=int, default=200)
    args = parser.parse_args()

    print(f"torch={torch.__version__} hip={torch.version.hip}")
    print(f"current_stream: {torch.cuda.current_stream()}")
    s = torch.cuda.Stream()
    print(f"new stream: {s} (nonnull: {s.cuda_stream != 0})")

    for size in [
        (256, 256, 256),
        (1024, 1024, 1024),
        (4096, 4096, 4096),
        (8192, 8192, 8192),
    ]:
        run(*size, n_iters=args.n_iters)


if __name__ == "__main__":
    main()
