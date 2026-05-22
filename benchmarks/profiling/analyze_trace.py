# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize a torch.profiler chrome trace: per-stream kernel timelines.

Answers: do default-stream and aux-stream GPU kernels actually run in parallel,
or is one waiting for the other?

Reports for each stream:
  - total GPU busy time
  - first/last kernel start
  - inferred occupancy (busy_time / wall_span)
And for the pair:
  - overlap_us = sum over time of min(default_active, aux_active)
  - serial_us  = sum_per_stream busy_time
  - overlap_pct = overlap_us / min(default_busy, aux_busy)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict


def load_kernels(path: str) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    events = data.get("traceEvents", [])
    kernels = []
    for ev in events:
        if ev.get("ph") != "X":  # complete events only
            continue
        cat = ev.get("cat", "").lower()
        # ROCm puts GPU kernels under "kernel" / "Kernel" or with the device
        # name in args.
        args = ev.get("args", {})
        if "kernel" not in cat and "Stream" not in args and "gpu" not in cat:
            continue
        # Identify stream id; torch.profiler stores it in args.
        stream = args.get("stream") or args.get("Stream") or ev.get("tid")
        start = ev.get("ts")
        dur = ev.get("dur", 0)
        if start is None or dur <= 0:
            continue
        kernels.append(
            {
                "name": ev.get("name", ""),
                "start": start,
                "end": start + dur,
                "dur": dur,
                "stream": stream,
            }
        )
    return kernels


def per_stream_stats(kernels: list[dict]):
    by_stream = defaultdict(list)
    for k in kernels:
        by_stream[k["stream"]].append(k)
    out = {}
    for s, ks in by_stream.items():
        ks.sort(key=lambda x: x["start"])
        busy = sum(k["dur"] for k in ks)
        span = ks[-1]["end"] - ks[0]["start"] if ks else 0
        out[s] = {
            "count": len(ks),
            "busy_us": busy,
            "span_us": span,
            "occupancy": busy / span if span else 0.0,
            "first": ks[0]["start"] if ks else 0,
            "last": ks[-1]["end"] if ks else 0,
        }
    return out


def pairwise_overlap(kernels: list[dict], stream_a, stream_b) -> int:
    """Compute total time during which both streams have a kernel running."""
    a = [(k["start"], k["end"]) for k in kernels if k["stream"] == stream_a]
    b = [(k["start"], k["end"]) for k in kernels if k["stream"] == stream_b]
    a.sort()
    b.sort()
    events = []
    for s, e in a:
        events.append((s, +1, "a"))
        events.append((e, -1, "a"))
    for s, e in b:
        events.append((s, +1, "b"))
        events.append((e, -1, "b"))
    events.sort()
    overlap = 0
    in_a = 0
    in_b = 0
    last_t = None
    for t, delta, who in events:
        if in_a > 0 and in_b > 0 and last_t is not None:
            overlap += t - last_t
        if who == "a":
            in_a += delta
        else:
            in_b += delta
        last_t = t
    return overlap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=str)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    kernels = load_kernels(args.trace)
    if not kernels:
        print(f"No GPU kernels found in {args.trace}", file=sys.stderr)
        sys.exit(1)

    print(f"Trace: {args.trace}")
    print(f"Total GPU kernel events: {len(kernels)}")

    stats = per_stream_stats(kernels)
    streams = sorted(stats.keys(), key=lambda s: -stats[s]["busy_us"])
    print("\nPer-stream stats (sorted by busy time):")
    hdr = (
        f"  {'stream':<20} {'count':>8} {'busy_us':>10} "
        f"{'span_us':>10} {'occ':>8} {'first':>12} {'last':>12}"
    )
    print(hdr)
    for s in streams[:8]:
        st = stats[s]
        print(
            f"  {str(s):<20} {st['count']:>8} {st['busy_us']:>10} "
            f"{st['span_us']:>10} {st['occupancy']:>8.2%} "
            f"{st['first']:>12} {st['last']:>12}"
        )

    if len(streams) >= 2:
        a, b = streams[0], streams[1]
        overlap = pairwise_overlap(kernels, a, b)
        busy_a = stats[a]["busy_us"]
        busy_b = stats[b]["busy_us"]
        print(f"\nTop-2 stream overlap ({a} vs {b}):")
        print(f"  overlap_us         = {overlap}")
        print(f"  busy_a / busy_b    = {busy_a} / {busy_b}")
        print(f"  overlap / min(a,b) = {overlap / max(1, min(busy_a, busy_b)):.2%}")
        if overlap == 0:
            print(
                "  -> The two streams DO NOT overlap at any point. "
                "Multi-stream has zero parallelism benefit on this workload."
            )
        elif overlap < 0.1 * min(busy_a, busy_b):
            print(
                "  -> Negligible overlap (<10%). Multi-stream is effectively "
                "serial here."
            )

    # Top kernel-name breakdown
    by_name = defaultdict(lambda: [0, 0])
    for k in kernels:
        by_name[k["name"]][0] += 1
        by_name[k["name"]][1] += k["dur"]
    rows = sorted(by_name.items(), key=lambda x: -x[1][1])
    print(f"\nTop {args.top_k} kernels by total time:")
    print(f"  {'count':>8} {'total_us':>10}  name")
    for name, (cnt, total) in rows[: args.top_k]:
        print(f"  {cnt:>8} {total:>10}  {name}")


if __name__ == "__main__":
    main()
