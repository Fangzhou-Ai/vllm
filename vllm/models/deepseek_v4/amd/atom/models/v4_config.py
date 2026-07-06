# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Standalone V4 model-args + quant-config for the MoE-only port (extracted from the ported attention's attention_core.py)."""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Tuple

import torch
from aiter import dtypes

from vllm.models.deepseek_v4.amd.atom.config import (
    LayerQuantConfig,
    QuantizationConfig,
    QuantType,
)

logger = logging.getLogger(__name__)


@dataclass
class DeepseekV4Args:
    """Mirrors `inference/model.py:ModelArgs`. Constructed from `hf_config`.

    Field names match the V4 HuggingFace `config.json` keys where possible;
    aliases are documented inline.
    """

    # Core
    vocab_size: int = 129280
    dim: int = 7168  # hidden_size
    n_layers: int = 61  # num_hidden_layers
    n_mtp_layers: int = 1  # num_nextn_predict_layers
    n_hash_layers: int = 3  # num_hash_layers
    norm_eps: float = 1e-6  # rms_norm_eps
    max_seq_len: int = 1048576  # max_position_embeddings
    max_batch_size: int = 4  # default placeholder; production driven by ATOM scheduler

    # Attention (MQA, single shared KV head)
    n_heads: int = 128  # num_attention_heads
    head_dim: int = 512
    rope_head_dim: int = 64  # qk_rope_head_dim
    q_lora_rank: int = 1536
    o_lora_rank: int = 1024
    o_groups: int = 16
    window_size: int = 128  # sliding_window

    # Per-layer attention type: 0=Dense, 4=CSA, 128 (or other large m')=HCA
    compress_ratios: Tuple[int, ...] = field(default_factory=tuple)

    # Indexer (CSA layers only)
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 1024
    use_index_cache: bool = False
    index_topk_freq: int = 1
    index_topk_pattern: Optional[Any] = None

    # MoE
    moe_inter_dim: int = 3072  # moe_intermediate_size
    n_routed_experts: int = 384
    n_shared_experts: int = 1
    n_activated_experts: int = 6  # num_experts_per_tok
    score_func: Literal["softmax", "sigmoid", "sqrtsoftplus"] = "sqrtsoftplus"
    route_scale: float = 2.5  # routed_scaling_factor
    swiglu_limit: float = 10.0

    # Hyper-Connections (mHC)
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6

    # YaRN RoPE
    rope_theta: float = 10000.0
    compress_rope_theta: float = 160000.0
    rope_factor: float = 16.0  # rope_scaling.factor
    original_seq_len: int = 65536  # rope_scaling.original_max_position_embeddings
    beta_fast: int = 32
    beta_slow: int = 1

    # Quantization (PR1 ignores; PR2+ uses)
    dtype: Literal["bf16", "fp8"] = "bf16"
    expert_dtype: Optional[Literal["fp4", "fp8"]] = None
    scale_fmt: Optional[Literal["ue8m0"]] = None

    # V4QuantizationConfig — Linear layers auto-build the right (FP8 / FP4
    # / BF16) weight + scale params. Set by DeepseekV4ForCausalLM at init.
    quant_config: Optional[Any] = None

    @classmethod
    def from_hf_config(cls, hf_config: Any) -> "DeepseekV4Args":
        # Use getattr with sensible defaults so we work whether the HF config is
        # a real V4 PretrainedConfig (all fields present) or a V3 PretrainedConfig
        # populated with extra V4 attrs (some fields may live only in the raw
        # config_dict, not on the config object — `transformers` strips unknown
        # kwargs unless they're in the schema).
        def g(k, default=None):
            return getattr(hf_config, k, default)

        rope_scaling = g("rope_scaling", {}) or {}
        return cls(
            vocab_size=g("vocab_size"),
            dim=g("hidden_size"),
            n_layers=g("num_hidden_layers"),
            n_mtp_layers=g("num_nextn_predict_layers", 1),
            n_hash_layers=g("num_hash_layers", 0),
            norm_eps=g("rms_norm_eps", 1e-6),
            max_seq_len=g("max_position_embeddings", 2048),
            n_heads=g("num_attention_heads"),
            head_dim=g("head_dim", 512),
            rope_head_dim=g("qk_rope_head_dim", 64),
            q_lora_rank=g("q_lora_rank", 1536),
            o_lora_rank=g("o_lora_rank", 256),
            o_groups=g("o_groups", 16),
            window_size=g("sliding_window", 128),
            compress_ratios=tuple(g("compress_ratios", (0,))),
            index_n_heads=g("index_n_heads", 64),
            index_head_dim=g("index_head_dim", 128),
            index_topk=g("index_topk", 1024),
            use_index_cache=bool(g("use_index_cache", False)),
            index_topk_freq=int(g("index_topk_freq", 1)),
            index_topk_pattern=g("index_topk_pattern", None),
            moe_inter_dim=g("moe_intermediate_size", 2048),
            n_routed_experts=g("n_routed_experts", 256),
            n_shared_experts=g("n_shared_experts", 1),
            n_activated_experts=g("num_experts_per_tok", 6),
            score_func=g("scoring_func", "sqrtsoftplus"),
            route_scale=g("routed_scaling_factor", 1.5),
            swiglu_limit=g("swiglu_limit", 10.0),
            hc_mult=g("hc_mult", 4),
            hc_sinkhorn_iters=g("hc_sinkhorn_iters", 20),
            hc_eps=g("hc_eps", 1e-6),
            rope_theta=g("rope_theta", 10000.0),
            compress_rope_theta=g("compress_rope_theta", 160000.0),
            rope_factor=rope_scaling.get("factor", 1.0),
            original_seq_len=rope_scaling.get("original_max_position_embeddings", 0),
            beta_fast=rope_scaling.get("beta_fast", 32),
            beta_slow=rope_scaling.get("beta_slow", 1),
            # Default to "ue8m0" matching reference ModelArgs (inference/model.py:40);
            # HF config.json does not carry this field, only inference/config.json does.
            scale_fmt=g("scale_fmt", "ue8m0"),
        )


def _wo_a_is_bf16_on_disk(model_path):
    """Return True iff this ckpt stores ``layers.0.attn.wo_a.weight`` as BF16
    (already pre-dequantized) with NO companion ``wo_a.scale`` on disk.

    V4-Flash-FP8 ships ``wo_a`` as BF16 directly; V4-Flash-Base / V4-Pro ship
    it as FP8 + UE8M0 block-scale and rely on
    ``DeepseekV4Attention.process_weights_after_loading`` to dequant at load
    time. The ATOM Linear allocator decides FP8 vs BF16 from the quant spec
    at module-init time, so we have to probe the ckpt here BEFORE building
    the model — otherwise the FP8 + scale param shapes mismatch the BF16
    tensor on disk and produce garbage attention output.
    """
    if not model_path or not os.path.isdir(model_path):
        return False
    idx_path = os.path.join(model_path, "model.safetensors.index.json")
    if not os.path.isfile(idx_path):
        return False
    try:
        with open(idx_path) as f:
            idx = json.load(f)
        wmap = idx.get("weight_map", {})
    except Exception:
        return False
    probe = "layers.0.attn.wo_a.weight"
    if probe not in wmap:
        return False
    scale_present_in_idx = "layers.0.attn.wo_a.scale" in wmap
    # Even when listed in the index, the shard may not actually contain the
    # scale (V4-Flash-FP8 had a stale index entry). Open the shard and verify.
    try:
        from safetensors import safe_open

        with safe_open(os.path.join(model_path, wmap[probe]), framework="pt") as h:
            w = h.get_slice(probe)
            w_dtype = (
                w.get_dtype() if hasattr(w, "get_dtype") else getattr(w, "dtype", None)
            )
            if w_dtype in (torch.bfloat16, "BF16"):
                return True  # BF16 weight; no scale needed regardless of index
            if not scale_present_in_idx:
                return False
            if "layers.0.attn.wo_a.scale" not in h.keys():
                # Index lies. wo_a still FP8 but no scale → loader will fail
                # anyway; safer to fall back to no_spec, although this case is
                # unexpected.
                return True
    except Exception:
        return False
    return False


def make_v4_quant_config(hf_config, model_path=None, online_quant_config=None):
    """Build a QuantizationConfig that knows V4's per-layer quant scheme.

    Two V4 SKUs supported:
      - **V4-Pro** (gfx950 / MI355X): routed experts FP4 e2m1 packed +
        per-1x32 UE8M0 scale (DeepGEMM `gemm_a4w4_quant` path).
      - **V4-Flash-Base** (gfx942 / MI308 + others): routed experts FP8 e4m3
        per-block 128x128 + UE8M0 scale (aiter `gemm_a8w8_blockscale` /
        Triton MoE per_1x128 path).

    The routed-expert spec is auto-detected from the ckpt's quantization
    layout via :func:`_detect_v4_routed_quant_spec`; SKU-agnostic projections
    (wq_a/b, wkv, wo_b, indexer.wq_b) all stay FP8 per-block 128x128.

    V4 checkpoint layout (common):
      - Most projections (wq_a/b, wkv, wo_b, indexer.wq_b, etc.): FP8 e4m3 +
        128x128 ue8m0 block scale. Picked up by ATOM's standard parser.
      - Routed expert weights (`ffn.experts.{N}.w{1,2,3}`): FP4 (V4-Pro) OR
        FP8 per-block (V4-Flash-Base) — auto-detected.
      - `wo_a`: FP8 on disk but loaded as BF16 (convert.py:137-141 dequantizes
        because the grouped-LoRA einsum needs BF16; aiter has no FP8 einsum).
      - `Compressor.wkv` / `Compressor.wgate` / `indexer.weights_proj`: BF16
        (or fp32 internally; reference declares dtype= explicitly). Loaded raw.
      - All RMSNorm weights, attn_sink, hc_*: BF16/fp32 raw, no quant.

    The optional ``online_quant_config`` is forwarded to the base
    QuantizationConfig so V4 models can also be re-quantized at load time
    (e.g. ``ptpc_fp8`` / ``mxfp4``). V4's hardcoded per-layer overrides
    (FP4 routed experts, BF16 compressor / indexer.weights_proj) are
    preserved on BOTH the source lookup AND the online lookup — returning
    the same spec on the online path triggers the FusedMoE/Linear
    ``source == online_target`` early-return so those layers stay untouched.
    """

    base = QuantizationConfig(hf_config, online_quant_config=online_quant_config)

    fp4_spec = LayerQuantConfig(quant_type=QuantType.per_1x32, quant_dtype=dtypes.fp4x2)
    # FP8 per-block 128x128 — V4-Flash-Base routed path.
    # ``dtypes.fp8`` from aiter resolves to ``float8_e4m3fnuz`` on gfx942/gfx94x
    # (MI308) and ``float8_e4m3fn`` on gfx950 / NV — picked at import time.
    fp8_block_spec = LayerQuantConfig(
        quant_type=QuantType.per_1x128,
        quant_dtype=dtypes.fp8,
    )
    no_spec = LayerQuantConfig(quant_type=QuantType.No, quant_dtype=torch.bfloat16)

    # Detect which routed-expert quant scheme this ckpt uses (FP4 or FP8-block).
    # ``base`` is consulted first — if the user's quant_method parser already
    # produced a per_1x128 fp8 spec for ``ffn.experts``, we honor it; only
    # when the parser yields no information do we fall back to V4-Pro's FP4.
    routed_spec = _detect_v4_routed_quant_spec(
        hf_config, base, fp4_spec, fp8_block_spec
    )

    # V4-Flash-FP8 ships ``wo_a`` already dequanted to BF16 on disk (no
    # ``.scale`` companion). Probe the ckpt; when wo_a is BF16, allocate it
    # as BF16 directly. Other SKUs (V4-Pro / V4-Flash-Base) keep wo_a as
    # FP8 + UE8M0 scale and rely on the load-time dequant in
    # ``DeepseekV4Attention.process_weights_after_loading``.
    wo_a_is_bf16 = _wo_a_is_bf16_on_disk(model_path)
    if wo_a_is_bf16:
        logger.info(
            "ckpt stores wo_a as BF16 on disk; allocating BF16 "
            "wo_a params (skipping FP8 + scale load-time dequant)."
        )

    orig_lookup = base.get_layer_quant_config

    def overridden(layer_name, use_online_quant=False, *, check_children=False):
        # Routed experts → SKU-detected (FP4 for V4-Pro, FP8-block for V4-Flash).
        # Match both per-expert prefix `layers.N.ffn.experts.M.w{1,2,3}` (used
        # by individual Linear lookups, with trailing `.M.w1`) AND the bare
        # `layers.N.ffn.experts` prefix (used by FusedMoE.__init__ when
        # constructing fused expert params — has NO trailing dot).
        #
        # V4 hardcoded specs apply on BOTH source AND online lookups. When
        # online_quant is enabled, returning the source spec here means
        # FusedMoE/Linear see `source == online_target` and skip the
        # dequant→requant round-trip for these layers (which would either
        # crash on the moe assert or further damage already-quantized weights).
        if ".ffn.experts" in layer_name:
            return routed_spec
        # BF16 / fp32 raw paths
        if (
            ".compressor.wkv" in layer_name
            or ".compressor.wgate" in layer_name
            or ".indexer.weights_proj" in layer_name
        ):
            return no_spec
        # V4-Flash-FP8 layout: wo_a is BF16 on disk — allocate as BF16 directly
        # so the loader receives matching dtype. Other SKUs let wo_a allocate
        # as FP8 + scale and DeepseekV4Attention dequants at load time.
        # When online_quant is enabled, also keep wo_a BF16 so
        # the dequant→requant round-trip is skipped for this layer.
        if ".wo_a" in layer_name and (wo_a_is_bf16 or use_online_quant):
            return no_spec
        return orig_lookup(
            layer_name,
            use_online_quant=use_online_quant,
            check_children=check_children,
        )

    base.get_layer_quant_config = overridden
    return base


def _detect_v4_routed_quant_spec(hf_config, base, fp4_spec, fp8_block_spec):
    """Detect V4 routed-expert quant scheme from HF config + parser output.

    Resolution order:
      1. **HF config ``expert_dtype``** — if the model's config.json declares
         ``expert_dtype`` (e.g. ``"fp8"`` or ``"fp4"``), use it directly.
      2. **Parser-derived spec for ``ffn.experts``** — if the model's
         quant_method parser (quark / generic / fp8 / ...) already produced a
         layer pattern that matches ``ffn.experts.*.w*``, honor it. This is
         the canonical path: the ckpt's own quantization_config dict declares
         ``per_1x128`` (fp8 block) or ``per_1x32`` (fp4 microscaling), and
         the parser turns it into the correct spec.
      3. **Heuristic from ``quant_method``** — when the parser doesn't carry
         per-layer detail (some compressed-tensors ckpts only set a global
         spec), look at ``hf_config.quantization_config.quant_method``:
         strings containing "fp4"/"mxfp4" → FP4; "fp8" → FP8 block.
      4. **V4-Pro default fallback** — historical V4 default (FP4 e2m1).

    Returns the chosen ``LayerQuantConfig`` (always either ``fp4_spec`` or
    ``fp8_block_spec`` — never None).
    """

    # ── 1. HF config expert_dtype hint ──
    expert_dtype = getattr(hf_config, "expert_dtype", None) or ""
    if isinstance(expert_dtype, str):
        ed = expert_dtype.lower()
        if "fp4" in ed:
            return fp4_spec
        if "fp8" in ed:
            return fp8_block_spec

    # ── 2. Parser-derived spec ──
    # Probe a representative routed-expert layer name. The parser's pattern
    # match (fnmatch) returns whatever was declared in the ckpt's
    # quantization_config -> layer_quant_config dict.
    sample = base.get_layer_quant_config("layers.0.ffn.experts.0.w1")
    if sample.is_quantized:
        # FP4: ATOM uses per_1x32 + dtypes.fp4x2 (microscaling FP4)
        if sample.quant_type == QuantType.per_1x32:
            return fp4_spec
        # FP8 per-block: per_1x128 + fp8 dtype
        if sample.quant_type == QuantType.per_1x128:
            return fp8_block_spec
        logger.warning(
            "Routed-expert layer quantized with unsupported quant_type=%s "
            "(expected per_1x32 or per_1x128). Falling through to heuristic.",
            sample.quant_type,
        )

    # ── 3. quant_method heuristic ──
    qc = getattr(hf_config, "quantization_config", None) or {}
    method = (qc.get("quant_method") or "").lower() if isinstance(qc, dict) else ""
    fmt = (qc.get("fmt") or "").lower() if isinstance(qc, dict) else ""
    method_lower = method + " " + fmt
    if "fp4" in method_lower or "mxfp4" in method_lower:
        return fp4_spec
    if "fp8" in method_lower or "deepseek_fp8" in method_lower:
        return fp8_block_spec

    # ── 4. V4-Pro default fallback ──
    logger.info(
        "routed-expert quant not auto-detected; falling back to FP4 (V4-Pro). "
        "Set expert_dtype in config.json to override."
    )
    return fp4_spec
