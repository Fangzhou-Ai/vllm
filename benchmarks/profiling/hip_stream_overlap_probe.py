# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Minimal probe: do two independent CUDA streams overlap on this ROCm setup?

If yes, multi-stream is fundamentally workable and the c4a regression must come
from event/scheduling overhead specifically. If no, multi-stream cannot help on
this hardware/driver combo and the whole CSA-overlap strategy is moot here.

We launch a same-size compute-heavy GEMM on two streams. We expect:
  - serial:   ~2 * gemm_time
  - mstream:  ~1 * gemm_time   (true overlap)
  - mstream:  ~2 * gemm_time   (no overlap)

Both branches are timed with CUDA events and via wall-clock.
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch


def run(M: int, N: int, K: int, n_iters: int, warmup: int = 20) -> None:
    device = torch.device("cuda")
    a0 = torch.randn(M, K, device=device, dtype=torch.bfloat16)
    b0 = torch.randn(K, N, device=device, dtype=torch.bfloat16)
    a1 = torch.randn(M, K, device=device, dtype=torch.bfloat16)
    b1 = torch.randn(K, N, device=device, dtype=torch.bfloat16)

    aux = torch.cuda.Stream(device=device)
    e0 = torch.cuda.Event()
    e1 = torch.cuda.Event()

    def fn_default():
        return a0 @ b0

    def fn_aux():
        return a1 @ b1

    def serial():
        fn_default()
        fn_aux()

    def mstream():
        e0.record()
        fn_default()
        with torch.cuda.stream(aux):
            e0.wait()
            fn_aux()
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
            f"  {label:<10} wall={((t1 - t0) * 1e6 / n_iters):>7.1f}us  "
            f"gpu_med={statistics.median(iter_us):>7.1f}us  "
            f"gpu_p05={iter_us[int(0.05 * len(iter_us))]:>7.1f}us  "
            f"gpu_p95={iter_us[int(0.95 * len(iter_us))]:>7.1f}us"
        )

    print(f"\n[M={M}, N={N}, K={K}, dtype=bf16]")
    measure(fn_default, "single")
    measure(serial, "serial(x2)")
    measure(mstream, "mstream(x2)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-iters", type=int, default=200)
    args = parser.parse_args()

    print(
        f"torch={torch.__version__} hip={torch.version.hip} "
        f"device={torch.cuda.get_device_name(0)}"
    )

    # Small (launch-bound) → larger (compute-bound) sweep.
    for size in [
        (32, 32, 64),
        (256, 256, 256),
        (1024, 1024, 1024),
        (4096, 4096, 4096),
        (8192, 8192, 8192),
    ]:
        run(*size, n_iters=args.n_iters)


if __name__ == "__main__":
    main()
