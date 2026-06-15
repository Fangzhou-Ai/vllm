# ATOM -> vLLM port plan (branch afz/port_atom, base 9945ac6e644)

Goal: vendor ATOM's DeepSeek-V4 (model + compute + KV runtime) into vllm/ as
self-contained native files + a thin vLLM adapter = 'vLLM-flavored ATOM',
reproducing ATOM's ~19ms TPOT. The ATOM vLLM PLUGIN already solves the hard
parts (proxy-layer KV, forward-context bridge, aiter all-reduce, cudagraph seam)
-> this is vendor-and-thin, NOT greenfield.

I have everything needed. The deepseek_v4 model imports broadly from `atom.model_ops.*`, `atom.config`, `atom.model_loader`, `atom.utils.forward_context`, and `atom.distributed` (via aiter). The plugin already solves the integration. Let me write the plan.

# Plan: Copy ATOM's DeepSeek-V4 into the vLLM tree as "vLLM-flavored ATOM"

## Key finding that reshapes the whole plan

ATOM **already ships** a working "vLLM-flavored ATOM" as an out-of-tree plugin under `/shared/amdgpu/home/fai_qle/ATOM/atom/plugin/vllm/`. It wraps ATOM's *native* `DeepseekV4ForCausalLM` (which reads ATOM's own forward context, owns its native unified_kv pool, runs aiter all-reduce) and exposes it through a thin vLLM `VllmModelForTextGeneration` wrapper. The crux mechanisms are all there:

- **KV ownership via a "proxy layer"** — `deepseek_v4_bridge.py:307 AtomDeepseekV4ProxyAttention.get_kv_cache_spec` returns ONE `FullAttentionSpec` whose `head_size` is back-computed from a giant `_proxy_page_bytes(vllm_config)` (`deepseek_v4_bridge.py:48`). vLLM's allocator/block-manager thus hands ATOM a raw `uint8` slab; `slice_deepseek_v4_proxy_cache_views` (`deepseek_v4_bridge.py:66`) re-slices that slab into ATOM's native per-layer unified_kv + compressor-state tensors. **vLLM allocates raw memory; ATOM lays out its native pool.** This is exactly the design the user wants and it is the single most important decision.
- **Forward-context bridge** — `model_wrapper.py:548 forward` binds the proxy views then enters `atom_deepseek_v4_forward_context(...)` and calls ATOM's native `self.model(input_ids, positions)` (`model_wrapper.py:609`). vLLM drives the loop; ATOM owns compute/KV inside.
- **All-reduce reuse** — `tp_group_reuse.py:117 set_custom_all_reduce(True)` rebinds aiter's `_TP/_PP/_DP` to vLLM's process groups (no duplicate ProcessGroups).
- **Cudagraph seam** — `AtomDeepseekV4ProxyBackend.forward_includes_kv_cache_update = True` (`deepseek_v4_bridge.py:262`) + metadata built in `_build_and_attach_atom_v4_md` into "persistent fixed-address buffers" (`deepseek_v4_bridge.py:210-220`) so captured kernels replay against stable addresses. vLLM owns capture; ATOM's metadata build runs outside it.

**Therefore the task is not greenfield: it is "vendor the plugin + its model/compute/KV deps into `vllm/` and thin the adapter."** The plan below copies ATOM's real code (per the user's explicit instruction) rather than re-deriving it, and uses the existing plugin as the reference adapter.

---

## A. INTEGRATION ARCHITECTURE — the cleanest thin seam

**Seam placement: the vLLM model class is a thin wrapper; ATOM owns the model-runner *step's compute and KV*, vLLM owns serving/scheduling/loop/capture.** Do NOT let ATOM own the model-runner step (that would mean importing ATOM's `model_engine/`, `scheduler.py`, `engine_core.py`, `block_manager.py` — that re-implements vLLM's serving layer and defeats the purpose). The boundary is exactly vLLM's `_model_forward` → `self.model(...)` call (`vllm/v1/worker/gpu_model_runner.py:3737`).

**The seam is a single vLLM model class** `DeepseekV4ForCausalLM` (registered at `registry.py:101`, today pointing at `vllm.models.deepseek_v4`) whose `forward(input_ids, positions, intermediate_tensors, inputs_embeds)` (matching the signature at `amd/model.py:809`) does three thin things, copied from `model_wrapper.py:548`:
1. bind proxy cache views (`bind_deepseek_v4_proxy_cache_views`),
2. enter `atom_deepseek_v4_forward_context(...)`,
3. call vendored ATOM `self.model(input_ids, positions)` and return `hidden_states` (NOT logits — runner calls `compute_logits` at `gpu_model_runner.py:3244`).

**KV decision — ATOM manages its own native pool; vLLM allocates raw memory (proxy-layer pattern). This is the choice most likely to reproduce 19ms.** Rationale grounded in the maps:
- The *current* in-tree vLLM ROCm path keeps vLLM's two `fp8_ds_mla` caches and **reconstructs ATOM's unified pool every decode step** via `gather_dequant_unified_kv_fixed_stride` (`amd/rocm.py:1039`, bridge described in `deepseek_v4_rocm_atom_bridge.py`). That per-step gather+dequant is precisely the overhead between "vLLM's DSv4 path" and ATOM's 19ms. Reusing vLLM's native KV layout forces this bridge forever.
- The proxy-layer design eliminates it: ATOM's `allocate_per_req_cache` (`deepseek_v4_attn.py:476`) native layout (per-layer `[swa_pages + compress_pages, head_dim]` BF16, plus FP8 compressor-state rings) lives directly inside the vLLM-allocated slab. `paged_decode` reads one base ptr (`paged_decode.py:21-46`) — **no gather, no dequant, native data path intact.** That is the thing that makes ATOM fast (native KV pool + the overlap + aiter all-reduce + fused FlyDSL kernels).

**What vLLM still owns** (the thin contract): the scheduler/`SchedulerOutput`, the block-manager *block-id allocation* (which feeds ATOM's `state_slot_mapping` via the per-request slot the proxy spec produces), the `InputBatch`/positions tensors, cudagraph capture dispatch, sampling/`compute_logits`. ATOM owns: the model forward, the KV *layout* inside the slab, SWA ring writes (`state_writes.py:60`), compressor state updates (`state_writes.py:237`), sparse paged decode/prefill, the compressor/Q-KV overlap, the MoE hash routing, aiter all-reduce.

---

## B. WHAT TO COPY (ATOM source → target under `vllm/`)

Copy ATOM's real code as a self-contained subtree, namespaced so it cannot collide with vLLM internals. Proposed root: **`vllm/models/deepseek_v4/atom/`** (a new sibling of `amd/`, `nvidia/`, `xpu/`), plus a vendored runtime under **`vllm/_atom_runtime/`** for the non-model deps.

**B1. Model + V4 attention (the heart):**
- `ATOM/atom/models/deepseek_v4.py` (3087 L) → `vllm/models/deepseek_v4/atom/model.py`
- `ATOM/atom/models/deepseek_v4_mtp.py` → `vllm/models/deepseek_v4/atom/mtp.py`
- `ATOM/atom/models/deepseek_v2.py` (base classes it subclasses) → `vllm/models/deepseek_v4/atom/deepseek_v2_base.py`
- `ATOM/atom/model_ops/attentions/deepseek_v4_attn.py` (2437 L — `AttentionMetaData_DSV4`, `allocate_per_req_cache:476`, `DeepseekV4AttentionMetadataBuilder`) → `vllm/models/deepseek_v4/atom/v4_attn.py`
- `ATOM/atom/model_ops/sparse_attn_v4.py` → same subtree

**B2. Compute deps (`atom.model_ops.*` — the imports at `deepseek_v4.py:68-100`):** copy the whole `ATOM/atom/model_ops/` directory (28 top-level .py + `v4_kernels/` 12 files / 4403 L + `attentions/` + `fused_moe/`) → `vllm/_atom_runtime/model_ops/`. Critical members named in the maps:
- `v4_kernels/`: `fused_compress.py`, `state_writes.py` (swa_write:60, update_compressor_states:237), `paged_decode.py`, `paged_prefill.py`, `paged_decode_indices.py`, `qk_norm_rope_maybe_quant.py`, `inverse_rope.py`, `compress_plan.py`, `csa_translate_pack.py`
- `moe.py` (FusedMoE), `topK.py` (indexer_score_topk), `linear.py`, `embed_head.py`, `layernorm.py`, `quant_v4.py`, `module_dispatch_ops.py`, `triton_rmsnorm_nw.py`

**B3. Runtime/config/loader/forward-context deps (non-`model_engine`):**
- `ATOM/atom/utils/forward_context.py` (`AttentionMetaData`, `Context`, `set_forward_context`, `get_forward_context`, `AttnState`) → `vllm/_atom_runtime/forward_context.py`
- `ATOM/atom/utils/` selected modules it pulls (`decorators.py` for `support_torch_compile`, `envs.py`, `mark_spliting_op`) → `vllm/_atom_runtime/utils/`
- `ATOM/atom/config/` (`Config`, `CompilationConfig`, `QuantizationConfig`, `ParallelConfig`) → `vllm/_atom_runtime/config/`
- `ATOM/atom/model_loader/loader.py` (`WeightsMapper`, `load_model_in_plugin_mode`) + `quant_spec.py` / `quant_v4` config → `vllm/_atom_runtime/model_loader/`
- aiter all-reduce stays an external `aiter` dep (already a runtime dep on ROCm); do NOT copy aiter. Copy only the *wiring*: `ATOM/atom/plugin/vllm/tp_group_reuse.py` → `vllm/models/deepseek_v4/atom/tp_group_reuse.py`.

**B4. The adapter glue (copy from the existing plugin, then thin it — see C):**
- `ATOM/atom/plugin/vllm/deepseek_v4_bridge.py` (1369 L — proxy layer/spec, cache-view slicing, metadata builder, forward-context cm) → `vllm/models/deepseek_v4/atom/bridge.py`
- `ATOM/atom/plugin/vllm/model_wrapper.py` (710 L — keep ONLY the V4 path) → distilled into `vllm/models/deepseek_v4/atom/wrapper.py`
- `ATOM/atom/plugin/vllm/graph_capture_patch.py` + `atom/plugin/graph_capture_patch.py` → `vllm/models/deepseek_v4/atom/graph_capture_patch.py`
- `ATOM/atom/plugin/config.py` (`_generate_atom_config_from_vllm_config`) → `vllm/models/deepseek_v4/atom/config_from_vllm.py`

**Do NOT copy:** `atom/model_engine/` (llm_engine, engine_core, scheduler, model_runner, block_manager), `atom/entrypoints/`, `atom/mesh/`, `atom/rollout/`, `atom/sampling/` — vLLM owns serving/scheduling/sampling. That's the line that keeps the interface thin.

---

## C. THE THIN ADAPTER — exact vLLM-side glue (keep it THIN)

The adapter is ~4 small touch-points; everything heavy is vendored copied code, not new logic.

1. **Registration / platform routing** — no registry change needed; reuse the existing route. `registry.py:101` already maps `DeepseekV4ForCausalLM → vllm.models.deepseek_v4`. Edit `vllm/models/deepseek_v4/__init__.py:17` to add a branch: when `current_platform.is_rocm()` **and** `envs.VLLM_DSV4_USE_ATOM`, import `from .atom.wrapper import DeepseekV4ForCausalLM` instead of `.amd.model`. (Add the env flag in `vllm/envs.py`.) This makes ATOM opt-in and keeps the current optimized AMD path as fallback.

2. **The forward adapter** — `vllm/models/deepseek_v4/atom/wrapper.py`, class `DeepseekV4ForCausalLM(nn.Module, VllmModelForTextGeneration, SupportsPP, SupportsQuant)`. `__init__(self, *, vllm_config, prefix)`: build ATOM `Config` via `config_from_vllm.py`, instantiate vendored ATOM model, call `register_deepseek_v4_proxy_layer(vllm_config)` (`bridge.py`). `forward(...)`: the exact 3-step body from `model_wrapper.py:548-611` — bind proxy views, enter `atom_deepseek_v4_forward_context`, call ATOM `self.model(input_ids, positions)`, return `hidden_states`. `compute_logits(hidden_states)`: call ATOM `compute_logits` (the hc_head reduction at `deepseek_v4.py:2870`). This is the only "new" file and it is <150 lines.

3. **KV/metadata bridge** — entirely the vendored `bridge.py`. No change to `gpu_model_runner.py` is required because the proxy layer plugs into vLLM's existing KV machinery: `register_deepseek_v4_proxy_layer` writes into `vllm_config.compilation_config.static_forward_context[ATOM_DEEPSEEK_V4_PROXY_LAYER_NAME]` (`deepseek_v4_bridge.py:330`); vLLM's `get_kv_cache_spec` walk picks up the proxy's one `FullAttentionSpec` (`deepseek_v4_bridge.py:317`) and allocates the slab into the runner's `kv_caches` list (`gpu_model_runner.py:525`); `AtomDeepseekV4ProxyMetadataBuilder.build` (`deepseek_v4_bridge.py:194`) is invoked by the runner's per-group builder loop (`gpu_model_runner.py:2414`) and writes ATOM metadata into persistent buffers. The bind step in `forward` re-slices `kv_caches[proxy_group]` into ATOM's native tensors.

4. **Weight loading** — `wrapper.load_weights(weights)` delegates to vendored `load_model_in_plugin_mode` (the V4 branch of `model_wrapper.py:625+`), which applies ATOM's `WeightsMapper` (`deepseek_v4.py:2777`), FusedMoE expert dispatch (`deepseek_v4.py:2964`), and the FP8→BF16 `wo_a` dequant (`deepseek_v4.py:2940`). Keep vLLM's `AutoWeightsLoader` out of this path (ATOM's loader owns the V4 quant formats).

5. **All-reduce wiring** — call `tp_group_reuse.reuse_vllm_tp_group(...)` once at `__init__` (or at platform init), which does `set_custom_all_reduce(True)` and rebinds aiter groups (`tp_group_reuse.py:108-117`). Plus apply `graph_capture_patch` so vLLM's `GroupCoordinator.graph_capture` nests aiter's `ca_comm.capture()`.

---

## D. PHASED PLAN (each phase gsm8k-checkable; converges on 19ms)

Per memory note, **validate accuracy via gsm8k, never curl.** Each phase ends with a gsm8k run + a TPOT measurement.

- **Phase 0 — Vendoring skeleton (no behavior change).** Copy B2/B3 (`model_ops`, `forward_context`, `config`, `model_loader`) into `vllm/_atom_runtime/` and rewrite intra-ATOM imports (`from atom.X` → `from vllm._atom_runtime.X`). Verify: the subtree imports cleanly under `.venv/bin/python -c "import vllm._atom_runtime..."`. No model wired yet. *Verifiable: import + lint only.*

- **Phase 1 — Copy kernels + standalone kernel parity.** Ensure `v4_kernels` (state_writes, fused_compress, paged_decode/prefill, qk_norm_rope) run on ROCm in isolation. Verify: a microbench comparing each kernel's output to the existing in-tree `deepseek_v4_rocm_atom_*` ops (numeric match), since the in-tree ROCm ops are ports of these same kernels.

- **Phase 2 — Copy the model; run under ATOM's own context (no vLLM yet).** Instantiate vendored `atom/model.py` with a hand-built `atom_deepseek_v4_forward_context` and ATOM-allocated `allocate_per_req_cache`. Verify: gsm8k offline through a tiny driver (bypasses vLLM scheduler). This isolates "is the copied model correct" from "is the adapter correct."

- **Phase 3 — Thin adapter + proxy KV (the hardest reconciliation).** Wire `wrapper.py` + `bridge.py`; route `__init__.py` under `VLLM_DSV4_USE_ATOM`. **Hardest reconciliation: vLLM scheduler/block-manager vs ATOM's native pool.** vLLM's block-manager allocates *block ids* per request in fixed-size 128-token blocks; ATOM's pool is indexed by *per-request state slots* (`state_slot_mapping`) over `num_slots` ring buffers, not by vLLM block ids. The proxy spec makes vLLM allocate the *bytes*, but the per-request *slot identity* must be derived from vLLM's block table inside `AtomDeepseekV4ProxyMetadataBuilder.build` (`deepseek_v4_bridge.py:194`, see `_atom_v4_slot_allocator` stash at `model_wrapper.py:592`). Verify: gsm8k online via vLLM serving at eager mode (cudagraph off). Expect correctness first, speed later.

- **Phase 4 — All-reduce + overlap.** Engage `tp_group_reuse` + aiter ca_comm; enable the compressor/Q-KV multi-stream overlap (`deepseek_v4_attn.py:1655 maybe_compressors_async`). Verify: gsm8k unchanged at TP>1; TPOT drops toward target. (Per memory: multi-stream on ROCm has been flaky — gate it, but it's needed for 19ms.)

- **Phase 5 — Cudagraph capture seam.** Enable vLLM cudagraph with `forward_includes_kv_cache_update=True` (`deepseek_v4_bridge.py:262`) + `graph_capture_patch`. **The capture seam:** ATOM's metadata build (`_build_and_attach_atom_v4_md`) must run *outside* capture into fixed-address buffers; only `_forward_decode` is captured — matches the memory note "build() runs outside capture; _forward_decode inside; topk_indices_buffer aliased under MTP." Verify: gsm8k with graphs on == gsm8k eager; measure TPOT == ~19ms. This is the phase that actually realizes the target.

- **Phase 6 — MTP / spec-decode (optional for 19ms baseline).** Wire `atom/mtp.py` + EagleProposer through vLLM spec-decode. Verify: gsm8k + accept-rate; TPOT with speculation.

---

## E. RISKS + EFFORT

**Biggest risks (ranked):**
1. **KV-ownership / slot-identity reconciliation (HIGH).** vLLM block-manager block-ids ≠ ATOM per-request ring slots. The proxy spec gets the *bytes* right, but mapping vLLM's per-request block table → ATOM's `state_slot_mapping` / SWA ring slot every step, CG-safely, is the single trickiest piece. Mitigated by the existing `_atom_v4_slot_allocator` in the plugin — but it is the part most likely to leak correctness bugs (silent wrong-token KV reads). gsm8k is the guard.
2. **Cudagraph capture (HIGH).** Anything in ATOM's metadata build that allocates or has a data-dependent grid breaks capture. The plugin's "persistent fixed-address buffers" approach (`deepseek_v4_bridge.py:210-220`) is mandatory and must be preserved verbatim. Memory's "phantom garbage from bare-prompt curl" warning applies — validate via gsm8k only.
3. **Weight/quant format (MEDIUM).** V4 mixes FP8 projections, FP4 experts, BF16 `wo_a`/Compressor (`deepseek_v4.py:2940`, `_probe_v4_routed_expert_dtype` at `model_wrapper.py:53`). vLLM's loader cannot parse these; ATOM's loader must own the path. Risk is misrouted experts/scales — caught by gsm8k.
4. **TP all-reduce (MEDIUM).** `set_custom_all_reduce(True)` rebinds aiter's groups to vLLM's (`tp_group_reuse.py`). Risk: double-init of ca_comm IPC, or graph-capture deadlock if `graph_capture_patch` isn't applied. Low novelty (plugin already does it), but ROCm stream-capture has historically hung (per memory `VLLM_DSV4_ROCM_MULTI_STREAM` kept off).
5. **Vendoring drift / import surface (LOW-MEDIUM).** ~10k+ lines copied (`model.py` 3087 + `v4_attn.py` 2437 + `model_ops`/`v4_kernels` ~5k + bridge 1369). Rewriting `from atom.*` imports across the subtree is mechanical but broad; an `atom`-vs-vLLM config/forward-context name clash is the main footgun.

**Effort estimate:** The design and most code already exist (the plugin), so this is a port-and-thin, not a build-from-scratch. Phases 0-2 (vendor + kernels + standalone model): ~2-3 days, mostly mechanical import rewrites + parity microbenches. Phase 3 (adapter + proxy KV, the hard reconciliation): ~3-5 days — this is where most engineering judgment goes. Phases 4-5 (all-reduce + cudagraph to hit 19ms): ~3-4 days, gated by ROCm stream-capture stability. Phase 6 (MTP): ~2-3 days, optional. **Total ~2-3 engineer-weeks to a gsm8k-clean, ~19ms ROCm path**, dominated by the KV-slot reconciliation and cudagraph seam — both of which are de-risked by the fact that ATOM's out-of-tree plugin already implements them.

**Reference files to mine during the port (absolute paths):**
- ATOM plugin (the de-facto adapter to copy): `/shared/amdgpu/home/fai_qle/ATOM/atom/plugin/vllm/deepseek_v4_bridge.py`, `/shared/amdgpu/home/fai_qle/ATOM/atom/plugin/vllm/model_wrapper.py`, `/shared/amdgpu/home/fai_qle/ATOM/atom/plugin/vllm/tp_group_reuse.py`, `/shared/amdgpu/home/fai_qle/ATOM/atom/plugin/vllm/graph_capture_patch.py`
- ATOM model + KV: `/shared/amdgpu/home/fai_qle/ATOM/atom/models/deepseek_v4.py`, `/shared/amdgpu/home/fai_qle/ATOM/atom/model_ops/attentions/deepseek_v4_attn.py` (`allocate_per_req_cache:476`), `/shared/amdgpu/home/fai_qle/ATOM/atom/model_ops/v4_kernels/`
- vLLM seam: `/shared/amdgpu/home/fai_qle/vllm/vllm/v1/worker/gpu_model_runner.py` (`_model_forward:3713`, `kv_caches:525`, builder loop:2414), `/shared/amdgpu/home/fai_qle/vllm/vllm/models/deepseek_v4/__init__.py:17` (platform routing), `/shared/amdgpu/home/fai_qle/vllm/vllm/model_executor/models/registry.py:101`, `/shared/amdgpu/home/fai_qle/vllm/vllm/v1/kv_cache_interface.py:160` (`AttentionSpec`/`FullAttentionSpec`)
- Overhead to delete (the per-step gather the new design avoids): `/shared/amdgpu/home/fai_qle/vllm/vllm/v1/attention/ops/deepseek_v4_rocm_atom_bridge.py`, used at `/shared/amdgpu/home/fai_qle/vllm/vllm/models/deepseek_v4/amd/rocm.py:1039`
