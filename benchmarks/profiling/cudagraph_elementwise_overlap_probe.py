# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Probe: graph-captured stream overlap for compute-light elementwise kernels.

Companion to ``cudagraph_overlap_probe.py``. That probe used GEMMs which might
be serialized by hipBLAS internal locks. This one uses elementwise sin/cos to
rule that out — if the GPU command processor can run two streams concurrently,
two independent elementwise kernels of equal duration should achieve close to
2x throughput vs serial.
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch


def build(n_elems: int, device: torch.device):
    a0 = torch.randn(n_elems, device=device, dtype=torch.float32)
    a1 = torch.randn(n_elems, device=device, dtype=torch.float32)
    c0 = torch.empty_like(a0)
    c1 = torch.empty_like(a1)
    return a0, c0, a1, c1


def capture_serial(n_elems, n_repeats, device):
    a0, c0, a1, c1 = build(n_elems, device)
    for _ in range(3):
        torch.sin(a0, out=c0)
        torch.sin(a1, out=c1)
    torch.accelerator.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n_repeats):
            torch.sin(a0, out=c0)
            torch.sin(a1, out=c1)
    return g


def capture_mstream(n_elems, n_repeats, device):
    a0, c0, a1, c1 = build(n_elems, device)
    aux = torch.cuda.Stream(device=device)
    e_start = torch.cuda.Event()
    e_done = torch.cuda.Event()
    for _ in range(3):
        e_start.record()
        torch.sin(a0, out=c0)
        with torch.cuda.stream(aux):
            e_start.wait()
            torch.sin(a1, out=c1)
            e_done.record()
        e_done.wait()
    torch.accelerator.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n_repeats):
            e_start.record()
            torch.sin(a0, out=c0)
            with torch.cuda.stream(aux):
                e_start.wait()
                torch.sin(a1, out=c1)
                e_done.record()
            e_done.wait()
    return g


def time_replay(g, n_iters: int, warmup: int = 30) -> dict:
    for _ in range(warmup):
        g.replay()
    torch.accelerator.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    iter_us: list[float] = []
    t0 = time.perf_counter()
    for _ in range(n_iters):
        s.record()
        g.replay()
        e.record()
        e.synchronize()
        iter_us.append(s.elapsed_time(e) * 1e3)
    t1 = time.perf_counter()
    iter_us.sort()
    return {
        "wall_us": (t1 - t0) * 1e6 / n_iters,
        "gpu_med": statistics.median(iter_us),
    }


def run(n_elems: int, n_repeats: int, n_iters: int) -> None:
    device = torch.device("cuda")
    g_ser = capture_serial(n_elems, n_repeats, device)
    g_ms = capture_mstream(n_elems, n_repeats, device)
    s_ser = time_replay(g_ser, n_iters)
    s_ms = time_replay(g_ms, n_iters)
    per_pair_ser = s_ser["gpu_med"] / n_repeats
    per_pair_ms = s_ms["gpu_med"] / n_repeats
    ratio = s_ms["gpu_med"] / s_ser["gpu_med"]
    speedup = 1.0 / ratio
    print(
        f"n_elems={n_elems:>10}  repeats={n_repeats:>2}  "
        f"graph-serial={s_ser['gpu_med']:>9.1f}us "
        f"(per-pair {per_pair_ser:>7.2f}us)   "
        f"graph-mstream={s_ms['gpu_med']:>9.1f}us "
        f"(per-pair {per_pair_ms:>7.2f}us)   "
        f"speedup={speedup:>4.2f}x"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-iters", type=int, default=200)
    args = parser.parse_args()
    print(f"torch={torch.__version__} hip={torch.version.hip}")
    for n_elems in [1 << 12, 1 << 16, 1 << 20, 1 << 22, 1 << 24, 1 << 26]:
        run(n_elems=n_elems, n_repeats=8, n_iters=args.n_iters)


if __name__ == "__main__":
    main()
