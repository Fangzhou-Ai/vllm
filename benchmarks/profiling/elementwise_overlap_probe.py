# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Probe: do elementwise (non-BLAS) kernels overlap across streams on ROCm?

If GEMMs don't overlap because hipBLAS holds a device-wide lock, then plain
elementwise kernels (which go straight through HIP without BLAS) might. If
elementwise ALSO doesn't overlap, then the serialization is at a deeper level
(GPU command processor / driver) and not BLAS-specific.
"""

from __future__ import annotations

import argparse
import statistics

import torch


def run(n_elems: int, n_iters: int, warmup: int = 30) -> None:
    device = torch.device("cuda")
    a0 = torch.randn(n_elems, device=device, dtype=torch.float32)
    a1 = torch.randn(n_elems, device=device, dtype=torch.float32)
    b0 = torch.empty_like(a0)
    b1 = torch.empty_like(a1)

    s0 = torch.cuda.Stream(device=device)
    s1 = torch.cuda.Stream(device=device)
    e_start = torch.cuda.Event()
    e0 = torch.cuda.Event()
    e1 = torch.cuda.Event()

    def kernel0():
        torch.sin(a0, out=b0)

    def kernel1():
        torch.sin(a1, out=b1)

    def single():
        kernel0()

    def serial():
        kernel0()
        kernel1()

    def mstream():
        e_start.record()
        with torch.cuda.stream(s0):
            e_start.wait()
            kernel0()
            e0.record()
        with torch.cuda.stream(s1):
            e_start.wait()
            kernel1()
            e1.record()
        e0.wait()
        e1.wait()

    def measure(fn, label):
        for _ in range(warmup):
            fn()
        torch.accelerator.synchronize()
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        iter_us: list[float] = []
        for _ in range(n_iters):
            start_evt.record()
            fn()
            end_evt.record()
            end_evt.synchronize()
            iter_us.append(start_evt.elapsed_time(end_evt) * 1e3)
        iter_us.sort()
        print(
            f"  {label:<12} gpu_med={statistics.median(iter_us):>7.1f}us  "
            f"gpu_p05={iter_us[int(0.05 * len(iter_us))]:>7.1f}us  "
            f"gpu_p95={iter_us[int(0.95 * len(iter_us))]:>7.1f}us"
        )

    print(f"\n[n_elems={n_elems}]")
    measure(single, "single")
    measure(serial, "serial(x2)")
    measure(mstream, "mstream(x2)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-iters", type=int, default=300)
    args = parser.parse_args()

    print(f"torch={torch.__version__} hip={torch.version.hip}")
    # Range from tiny (launch-bound) to huge (compute-bound).
    for n in [1024, 65536, 1 << 20, 1 << 24, 1 << 26]:
        run(n, n_iters=args.n_iters)


if __name__ == "__main__":
    main()
