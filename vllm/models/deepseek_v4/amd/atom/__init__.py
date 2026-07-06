# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""ATOM DeepSeek V4 MoE module, ported into the vLLM source tree.

This package is a faithful port of AMD ATOM's DeepSeek V4 MoE (mixture-of-experts
FFN) subsystem — kept as ATOM-native code (imports repointed to this vendored
root), NOT rewritten to vLLM conventions. vLLM's original DeepSeek V4 attention /
embeddings / MHC / head are reused unchanged; only the per-layer FFN is swapped
for ATOM's MoE. It bundles:

* **MoE compute** — ``models.moe_core`` (``MoE`` + standalone ``Expert``) and
  ``model_ops.moe`` (ATOM's ``FusedMoE`` stack): FP4 routed experts
  (``Mxfp4MoEMethod``, per-1x32 UE8M0) + FP8 shared expert, sqrtsoftplus routing
  with ``e_score_correction_bias``, hash routing (``model_ops.triton_hash_topk``)
  for the first ``num_hash_layers`` layers, and the aiter ``fused_moe`` /
  ``topk_gating`` kernels via ``model_ops.topK``.
* **Supporting ops / shims** — ATOM linears (``model_ops.linear``), the quant
  spec/parsers (``quant_spec``, ``quantization.quark``), the single-node config
  singleton (``config``), the forward-context registry (``utils.forward_context``,
  used to pass hash-routing ``input_ids`` into the Dynamo-opaque MoE op) and the
  V4 model-args / quant-config (``models.v4_config``).

Scope / single-node: distributed features (data/expert parallel) are stubbed to
single-rank so the ATOM ``FusedMoE`` collapses to plain tensor-parallel sharding.
Dual-stream / shared-expert-fusion overlap is intentionally not wired here.
"""
