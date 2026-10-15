# A shared kernel substrate for persistent ("mono") decode kernels

Status: draft proposal. Everything below comes from static reading of the PR heads listed in
[Method and caveats](#method-and-caveats). Nothing was compiled, run or measured: no GPU was
available, so every claim about behavior is "by reading", and every migration step needs the
verification listed in [Verification](#verification) before it can land.

Related: RFC [#60904](https://github.com/vllm-project/vllm/issues/60904) (a standard for
MonoKernels), [#61311](https://github.com/vllm-project/vllm/pull/61311) (runtime-layer draft),
[#60397](https://github.com/vllm-project/vllm/pull/60397) (DeepSeek-V4.1-Flash, merged),
[#61482](https://github.com/vllm-project/vllm/pull/61482) (DeepSeek-V4-Pro on a shared stage package).

## Summary

- Every mono PR carries its own copy of the same plumbing: a tagged-pair mailbox with an epoch
  tag, a symmetric peer-memory buffer, a build/JIT key, CTA task placement, wave/MX/MFMA device
  helpers and an in-kernel rank-ordered all-reduce. The five codebases we compared hold 0.5-1.3k
  lines of it each.
- The wire format is identical everywhere (8-byte `(value, tag)` pairs, polls batched by 12,
  system scope for peers). What diverges is the *contract around it*: the epoch/tag scheme (four, counting
  ATOM's), the wait policy (unbounded, 10 s timeout, retry-bounded), the peer allocator,
  the `Mailbox` constructor, and the build key. The copies already behave differently where a
  rank fails, where a tag wraps and where a pair is double-buffered.
- The RFC's runtime shell (refusals, routing, fail-stop) does not touch any of this, and the stage
  code (seam, attention, MoE) is genuinely model-specific. The part worth sharing is the layer in
  between: the **kernel substrate**.
- Proposal: one substrate package with a compile-time *policy* for each place the copies diverge,
  so that a model that does not opt into a policy gets the machine code it has today. Adopt ATOM's
  `atom/mono` device/plan/runtime layering as the API baseline (it is the stated ancestor of the
  DeepSeek and MiniMax copies and is already model-independent) rather than inventing a fourth
  shape.
- Migrate in phases, each a pure refactor for the models already merged, gated by a compile-only
  machine-code diff. The first phase is inside #61482 itself, which currently has two `Mailbox`
  classes.

## Evidence: the substrate is copied, not shared

### Codebases compared

| Codebase | PR | Substrate files (lines) | Lineage claimed in the file headers |
| --- | --- | --- | --- |
| DeepSeek-V4.1 / V4-Pro | #60397, #61482 | `common/{ops 301, sync 177, plan 142, peer_memory 104, mx 78}` and `attention/device.py 504` | ATOM `97359d6` |
| Kimi-K3 (Qwen3.8 imports it) | #60683, #61435, #60833, #61325 | `kimi_k3/.../common/{ops 435, sync 251, plan 237, peer_memory 133, debug 105, mx 84, ranks}`, plus `common/mono/peer.py 126` | ATOM `a526f0d` |
| GLM-5.2 / GLM-5 | #60889, #60831 | `kernel/{ops 195, ipc 156, runtime 137, layout, packing}`; the mailbox is a set of closures inside `glm/kernel.py` | FlyDSL#1204, ATOM#2435 |
| GLM-5.3-Flash | #61662 | `ops 194`, `ffn/layout 275`, `common/mono/peer.py 147` | aiter#6173 |
| MiniMax-M3 | #59705, #61273 | `kernels/common.py 437`, `peer_buffer.py 49`, `layout.py 230` | FlyDSL GLM-5 layer kernel |

Reference points outside vLLM: ROCm/ATOM `atom/mono` (about 2k lines, deliberately
model-independent, three layers `device/`, `plan/`, `runtime/`, MIT, used by ATOM's DeepSeek-V4.1
and MiniMax-M3); ROCm/FlyDSL `kernels/monokernel` (the GLM-5 / Kimi-K3 line, Apache-2.0, not
installed in the wheel and outside FlyDSL's stable-API policy); the aiter FlyDSL directory, which
vLLM already imports `buffer_ops`, `dpp_utils`, `kernel_signature` and `UncachedIpcHeap` from.

ATOM's own contract is `Mailbox(layer, diag=None, region_ids=None)` with tag `layer + 1`; its
runner zeroes every mailbox at the start of each step in a dedicated launch
(`runtime/step_begin.py`, which also fences the ranks through step-epoch slots at the buffer
tail), and its bounded wait (`_spin_bounded`) exists only in debug builds. The Kimi copy is the
closest to it (50 changed lines in `ops.py`, 139 in `sync.py` against ATOM); the DeepSeek copy is
a redesign (264 changed lines in `ops.py`): no per-step zeroing launch, a device epoch word.

The ATOM commits cited by the vLLM files (`97359d6`, `a526f0d`) are snapshots of ATOM `main`
that do not touch `atom/mono`; the framework itself landed in ATOM `0873517` (2026-10-06). The
copies are therefore forks of a moving target with no recorded sync point.

### Capability matrix

`=` same as the DeepSeek copy, `~` near-identical, `!=` different. The summary cell says how.

| Capability | DeepSeek (#61482) | Kimi, Qwen | GLM-5.x | MiniMax-M3 |
| --- | --- | --- | --- | --- |
| Pair wire format, batch-12 poll | base | = | = | = |
| `Mailbox` constructor / addressing | `Mailbox(tag)`, typed `Addr(region, space)` | `!=` `Mailbox(layer, health, alarm, region_ids)`, typed `Addr` | `!=` closures in `kernel.py`, scope passed as a string per poll | `!=` `Mailbox(scratch, step_ptr, layer)`, cache modifier per call |
| Spin / wait policy | `s_sleep 1`, unbounded | `!=` `s_sleep 1` plus 10 s timeout, health word, alarm word | `!=` `s_nop 0`, optional retry-bounded poll with per-stage error words and an abort word | `~` unbounded |
| `publish` (`s_waitcnt vmcnt(0)`, barrier, flag) | base | = | not found | `!=` fence plus barrier only (see Open questions) |
| Epoch / tag | `!=` device epoch word advanced inside the kernel; tag `2e+1` / `2e+2` | `!=` host `epoch.add_(1)` in the graph; tag `(e % 2^22)*256 + slot + 1` | `!=` host `step.add_(1)`; tag `step*launches + layer + 1` | `!=` host `step.add_(1)`; tag `step*128 + layer + 1` |
| Peer double-buffering | `e & 1` halves | `epoch % 2`, K2 only | `(step*launches + layer) & 1` | none |
| Peer memory | `UncachedIpcHeap`, collective `tp_agree` | = (plus a 256-byte alarm head) | `!=` cached `torch.zeros` plus raw HIP IPC (GLM-5.3-Flash: a smaller rewrite) | `~` same heap, no failure agreement |
| Scratch layout | `pair_layout` (align 8) then 256-byte rounding | `~` plus `KernelAbi` order check | `~` `layout()`, 256-byte regions | `~` `_layout`, 256-byte regions |
| Build / JIT key | `key_tuple` + `source_digest` + `kernel_symbol` | `~` plus `symbol_params` | `!=` none (builder closures) | `!=` none (symbol only) |
| CTA placement | `first_task(bid, base)` | = | `~` same formula, inline | `~` `stage_bases` plus a cost model |
| Device helpers | `ops` 301 + `device.py` | `~` strict superset (+177 lines) | `~` same bodies, extra fp8/fp4 conversions | `~` plus all-reduce norm helpers |
| In-kernel all-reduce | one-hop push, optional two-hop; fp32 rank-order sum, bf16 round before the residual | one-hop, rank-order fp32 | one-hop, rank-order fp32; residual added in fp32, rounded once (A/B) | all-to-all below 12 tokens, reduce-scatter + all-gather from 12 |
| Instrumentation | two stamp variants (global store; LDS) | LDS stamps, diagnostic records, build-time fault hook | timeline buffer, env-selected fault hook | LDS stamps |

Between the three GLM variants, `ops.py` is AST-identical in #60831 and #61662 and differs only
in formatting in #60889; the mailbox helpers (`put`, `put2`, `put_bf`, `poll`, `get*`) are the same
nested functions in all three; the stage bodies all differ. Qwen vendors nothing: it imports the
Kimi copy and folds it into its own source digest.

### Duplication inside #61482 itself

`attention/device.py` (504 lines) is a second copy of `common/{ops,plan,mx,sync}.py`, and the K1
front / K2 back import it while the seam and MoE import `common/`:

- two `Mailbox` classes (`attention/device.py:381`, `common/sync.py:70`), constructed with the same
  tag in the same launch (`layer.py:324` and `front.py:600` / `back.py:857`); the first is
  device-scope only on raw bases, the second adds a peer space;
- about twenty behavior-identical helpers under the same name (`traced`, `rsrc`, `first_task`,
  `butterfly`, `bf16_pair`, `fp8_pack4`, `row_sum`, `await_flags`, ...) and a handful of same-named
  functions with *different* behavior (`pair_layout`: align 8 vs 256 with another argument order;
  `stamp`: LDS vs global store; `publish`: `(put, a, i, v, who)` vs `(c, region, i)`);
- constants defined three to four times (`BLOCKS`, `THREADS`, `PAIR`, `ROWS`, `TILE`, `LOG2E`).

This is the cheapest place to start, because it needs no agreement from anyone else.

### Divergences that matter

1. **Four epoch protocols.** ATOM zeroes all mailboxes at each step's start in a separate launch
   and fences the ranks with step-epoch slots; the vLLM DeepSeek design exists to avoid that
   launch. DeepSeek's kernel advances its own epoch after every CTA has written a
   mark (graph-safe, no host op). Kimi, Qwen, GLM and MiniMax advance a device counter with a host
   `add_` captured in the graph. Tags are `2e+1/2e+2`, `(e mod 2^22)*256+slot+1`,
   `step*lps+layer+1` and `step*128+layer+1`. Wrap behavior differs: DeepSeek's tag is a 32-bit
   epoch; Kimi's is 22 bits times 256 slots; GLM's `int32` tag wraps after about `2^31 /
   launches_per_step` steps (not traced).
2. **Peer-buffer hazards rest on different arguments.** DeepSeek double-buffers every peer region
   by epoch parity. Kimi double-buffers K2 only and relies on cross-layer data dependencies for the
   M1 regions. MiniMax has no parity and relies on a "half a layer apart" argument. None of these
   arguments is wrong by inspection, but they are three different arguments for one mechanism.
3. **Failure behavior.** A hung peer spins forever in DeepSeek and MiniMax. Kimi gives up after
   10 s (an `s_memrealtime` read, a system-scope health-word load per iteration, a record and an
   alarm store), and GLM after a retry count with an abort word. These are the only features of
   the copies that DeepSeek lacks and that a production deployment wants; they must not cost
   DeepSeek anything when off.
4. **Peer memory model.** DeepSeek, Kimi and MiniMax allocate from aiter's `UncachedIpcHeap`. GLM-5.2/5
   allocate cached `torch.zeros` memory and open it over raw HIP IPC, so its correctness depends
   on every peer access carrying system-scope cache bits. Only DeepSeek and Kimi agree a failed
   handshake across ranks (`tp_agree`).
5. **Numerics contracts differ at the all-reduce.** All sum in fp32 in rank order, but the rounding
   point before the residual add differs (DeepSeek: bf16 round of the sum, then add; GLM-5.2/5:
   add in fp32, round once). Outputs are not bitwise-compatible across families; a shared
   primitive must expose the rounding point rather than pick one.
6. **Build keys.** GLM and MiniMax have no source digest, so editing a helper does not change the
   JIT key (MiniMax's own comments record a shared-cache-across-ranks bug for one constant).
   GLM-5.3-Flash's builder takes an unused `rank` argument that would add a per-rank entry if it
   became part of a key.
7. **Smaller inconsistencies.** `MonoUnsupported` is defined twice; `kernel_symbol` uses three
   namespaces (`aiter`, `atom`, `flydsl`); Kimi's slot scheme assumes `layer_idx < 128` with no
   assertion; the output rank is hard-wired to TP rank 0 in Kimi; the `Mailbox` docstring in
   Kimi's `sync.py` describes a class that does not exist.

## What is and is not the substrate

```text
 vLLM runtime shell    refusals, routing, cache re-binding, fail-stop surface   RFC #60904, #61311
 ------------------------------------------------------------------------------------------
 Kernel substrate      mailbox, epoch, peer memory, layout, build key,           this proposal
                       placement, device helpers, all-reduce primitives
 ------------------------------------------------------------------------------------------
 Stage packages        mHC seam, sparse-MLA front/back, MoE, KDA, GDN, ...       per model family
                       (#61482 is the DeepSeek-V4 family's)
```

Out of scope: the stages (their structure differs per model; Kimi's block-softmax AttnRes seam
and KDA recurrence share nothing with DeepSeek's mHC seam), spec-driven stage parameterization,
the runtime shell, and any change of scheduling model (for example generating task tables from a
dependency graph); each is a separate discussion.

## Proposed design (v0)

Layering follows ATOM's `atom/mono` (`device/` traced in kernels, `plan/` host-side pure Python,
`runtime/` host-side torch) so a later move into FlyDSL or aiter is a copy, not a redesign.
Module names below are working names.

| Module | Contents | Taken from |
| --- | --- | --- |
| `device/sync` | `Addr`, `Space`, `sreg`/`preg`, `Mailbox`, `publish`, `WaitPolicy` | DeepSeek/Kimi `sync.py` |
| `device/epoch` | `Epoch` strategies producing `(tag, parity)` | DeepSeek `layer.py`/`back.py`, Kimi `kda.step_tag` |
| `device/ops`, `device/mx` | the superset of the wave, math, packing, MFMA, buffer helpers | Kimi `ops.py` (a strict superset of DeepSeek's), `mx.py` |
| `device/ranks` | `peer_bases`, `sum_partials` | Kimi `ranks.py` |
| `device/trace` | one timeline-stamp API and the compile-time fault hook | Kimi `debug.py`, DeepSeek `stamp` |
| `plan/*` | `pair_layout`, `key_tuple`, `source_digest`, `kernel_symbol`/`symbol_params`, `KernelAbi`, `first_task`, `BLOCKS`/`THREADS` | DeepSeek/Kimi `plan.py` |
| `runtime/peer` | `PeerBuffer`, `tp_agree`, `MonoUnsupported` (one definition) | DeepSeek `peer_memory.py`, RFC `common/mono/peer.py` |

### Decisions

**D1. Wait policy is a compile-time parameter.** `WaitPolicy.unbounded()` is exactly today's
DeepSeek spin (`s_sleep 1`, reload, no extra load or timer). `WaitPolicy.bounded(...)` adds
Kimi's health/alarm record or GLM's retry limit and abort word. It is read with `const_expr`, so
a kernel built with the default emits no timeout code. This is the only way to give Kimi/GLM's
fail-stop to everyone without charging DeepSeek the extra per-iteration load. The error-word
location (in scratch vs an external tensor) becomes a field of the policy, not a fork.

**D2. Epoch strategies behind one interface; do not unify the protocols in v0.** The substrate
exposes `tag(epoch, slot)` and `parity(...)` for both the in-kernel epoch (DeepSeek) and the
host-advanced counter (everyone else); ATOM's zero-at-step-start protocol stays ATOM's, because
it needs a host launch per step that the vLLM copies avoid, with the tag layout a documented parameter. Merging them
needs a hazard review of the peer-buffer arguments (Open questions) and is a v1 decision.

**D3. Typed addresses, not scope strings.** Adopt `Addr(region, space)` with the one-space-per-poll
assertion (DeepSeek, Kimi). GLM's per-call scope string and MiniMax's per-call cache modifier become
callers of `sreg`/`preg`. The `Mailbox` constructor takes a tag and a wait policy; per-model
extras (Kimi's region ids for diagnostics) are policy fields.

**D4. One peer allocator.** `PeerBuffer` on `UncachedIpcHeap` with collective failure agreement.
GLM-5.2/5's cached-allocation IPC path is a different memory model; it is kept as a documented
non-goal for v0 (those PRs migrate to the heap, or stay behind a separate backend if their owners
need it). The host half merges with the RFC shell's `common/mono/peer.py` (126 lines in #61435), which
today duplicates the kernel-side `peer_memory.py` copies.

**D5. One build-key recipe.** `key_tuple` over frozen dataclasses, `source_digest(paths)`,
`kernel_symbol(namespace, ...)`. A model's digest includes the substrate's path (Qwen already
does this for Kimi's `common/`, so the mechanism exists). This fixes GLM/MiniMax's missing digest.

**D6. Layout and ABI checks.** One `pair_layout(items, align, start)` (the DeepSeek package has two
with different alignment and argument order) and Kimi's `KernelAbi` launcher/kernel order check.

**D7. All-reduce: primitives in the substrate, policy in the model.** The substrate provides
`push` and `sum_partials` (fp32, rank order) with the *rounding point as a parameter*; one-hop vs
two-hop vs reduce-scatter+all-gather selection (`spec.two_hop_rows`, MiniMax's 12-token switch)
stays with the model because it is a per-shape tuning decision.

**D8. One device-helper set.** Kimi's `ops.py` is a strict superset of DeepSeek's (it adds wave and
block reductions, scans, `readlane`, `global_load`, MFMA wrappers), so it is the base; DeepSeek's
duplicate `device.py` helpers disappear. Tracing keeps both existing stamp flavors under one API.

### Invariants the substrate must preserve for merged kernels

1. The mailbox instruction sequence for the default policy: the same `buffer_store` / `buffer_load`
   of `(value, tag)` pairs, `CM_DEV` for scratch and `CM_SYS` for peers, batch of 12, `s_sleep 1`
   spin, `(ep<<1)+1` / `(ep<<1)+2` tags and `ep & 1` parity for DeepSeek.
2. Offsets and order: the 256-byte region rounding, `first_task` placement, `butterfly` offset
   order, rank-order fp32 sums, `row_sum` wave order.
3. JIT identity: kernel symbol names and `key_tuple` contents, including the by-name import
   structure FlyDSL's cache key depends on (see the comment at `stages/moe.py:99`).

## Where it should live

| Home | For | Against |
| --- | --- | --- |
| vLLM, `vllm/models/common/amd/mono/common` | no new dependency; one copy for every vLLM model; fastest | still a fork of ATOM/FlyDSL; becomes the next thing that drifts from them |
| ROCm/FlyDSL `kernels/monokernel` | owns the compiler; has releases; already hosts the GLM-5/Kimi-K3 line | `kernels/` is not installed or covered by the stable-API policy; Apache-2.0 (ATOM is MIT); not a compiler concern |
| aiter (`aiter/ops/flydsl/...`) | vLLM already depends on it and imports four symbols from it; MIT | its FlyDSL directory has no API promise; aiter would have to absorb ATOM's mono code |
| ROCm/ATOM `atom/mono` | the only deliberately model-independent copy, with unit tests | vLLM cannot take a runtime dependency on another serving framework |

Recommendation: first consolidate **inside vLLM**, since every vLLM mono PR then shares one copy
and the migration phases below are owned by people who are already in these files; mirror ATOM's
layering and names so that moving the package to aiter or FlyDSL later is mechanical; keep a
`Adapted from ... at <sha>` header per file and one pinned ATOM commit. Opening that upstream
conversation with the aiter/FlyDSL owners is a follow-up, not a prerequisite.

## Migration plan

Each phase is a pure refactor for models already merged. Nothing below has been attempted.

| Phase | Change | Needs from its owner |
| --- | --- | --- |
| P0 | Inside #61482: `attention/device.py` imports `common/*`; one `Mailbox`, one `pair_layout`, one `publish`; constants defined once | DeepSeek machine code identical (compile-only diff of every K1/K2/FFN variant) |
| P1 | Extract the package as the substrate with the superset API; DeepSeek builds with the default policies | same diff |
| P2 | Kimi (and Qwen, which imports it) move onto it; add `WaitPolicy.bounded` and `KernelAbi` | Kimi's numerics tests and a hot-path cost check of the bounded policy |
| P3 | GLM-5.x move onto it: host-add epoch strategy, `Addr` for scopes, `UncachedIpcHeap` | GLM owners decide on the peer-memory model (D4) |
| P4 | MiniMax-M3: typed addresses, `publish` with the `vmcnt(0)` drain, a source digest | MiniMax owners; an ISA check of the flag-publication ordering |

P0-P1 are a diff to our own tree and can be reviewed without involving other teams; P2-P4 are PRs
by each model's owners against the extracted package.

## Verification

Nothing was run. What each phase must show before it lands:

- compile-only machine-code diff (`FLYDSL_DUMP_IR=1` with the compile-only switch, per build key)
  for every merged model's kernels: byte-identical for P0/P1; for P2-P4, identical except where
  an owner opts into a new policy, with the difference listed;
- the existing per-model suites (numerics, layer, graph-replay);
- unit tests for the host-side pieces (peer handshake and failure agreement with mocks, key
  flattening, layout disjointness);
- for `WaitPolicy.bounded`, an A/B of K2-sized kernels showing the cost of the added per-iteration
  load and timer read (Kimi's PR compares 88.9 us against 89.7 us but both arms contain the code).

## Open questions

- **Peer-buffer hazards.** Kimi flips K2's parity per step while tags change per layer, and its M1
  regions have none; MiniMax has none. A hazard review (can a rank overwrite a pair a peer has not
  yet read?) is needed before D2 can be unified; I could not establish it by reading.
- **MiniMax flag publication.** Its kernels publish flags after a workgroup-scope release fence and
  a barrier, with no `s_waitcnt vmcnt(0)`; DeepSeek and Kimi drain explicitly because a
  workgroup-scope fence does not wait for outstanding stores. The compiler may add the wait; the
  ISA was not inspected. This is a question for the MiniMax owners, not a finding.
- **Tag wrap and range.** DeepSeek 32-bit epoch, Kimi 22 bits times 256 slots (`layer_idx < 128`
  unasserted), GLM `int32` tag, MiniMax `step * 128 + layer + 1`; the substrate should assert the
  range it supports.
- **Does FlyDSL's cache key follow helpers across module paths?** Several comments assume so
  by-name; the substrate move changes the paths.
- **Naming.** #60683, #61662, #61311 and #61482 each claim a `common/mono`-like package. The
  substrate and the runtime shell are different layers and should say so in their names.
- **License mix** if the package ever moves to FlyDSL (Apache-2.0) from ATOM-derived MIT files.

## Method and caveats

Static reading only. PR heads at the time of writing:

- #61482 `154c48551d`, #60683 `df3693c8f8`, #61435 `122af47ce6`
- #60889 `2303eafe36`, #60831 `6deee07c52`, #61662 `43cfb5c8aa`
- #59705 `c1d75b6356`, #60833 `1301a7cfcf`, #61325 `b8c3d47b01`, #61311 `1bd3fd8294`

Reference checkouts: ROCm/ATOM `81d4462`;
ROCm/FlyDSL `1941889`. Line counts are from the checkouts; "identical" means the Python bodies
were compared (diff or AST), not the machine code. Not verified: any runtime behavior or
numerics; machine-code identity of the two DeepSeek `Mailbox` classes; the FlyDSL cache-key
behavior; upstream merge status of the aiter and FlyDSL pieces; TP8 behavior of MiniMax's
TP-generalized kernels (#61273); FlyDSL commit `1e28fd8a` cited by ATOM (not in the local clone).
Several claims about other authors' PRs are from reading alone and should be confirmed with
their owners.
