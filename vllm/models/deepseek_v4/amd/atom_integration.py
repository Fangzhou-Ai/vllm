# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Integration glue making the ported ATOM DeepSeek-V4 MoE the FFN compute
inside vLLM's DeepSeek-V4 model on ROCm.

This branch ports ONLY the MoE. vLLM's original DeepSeek-V4 model (embeddings /
attention / MHC / head / weight loading) is reused as-is; only the per-layer FFN
is swapped for ATOM's ``MoE`` (ported under ``amd/atom/models/moe_core.py``),
which owns ATOM's FP4 routed experts + FP8 shared expert, sqrtsoftplus / hash
routing and the aiter ``fused_moe`` / ``topk_gating`` kernels.

Selection is env-gated (``VLLM_DSV4_USE_ATOM_MOE``, default on). Hash-routed MoE
layers read their token ids from a minimal ATOM forward context
(:func:`atom_moe_forward_context`) entered around the model forward; no ATOM
attention infrastructure (proxy KV bridge / chunk-aware metadata) is involved.
"""

import os

from vllm.models.deepseek_v4.amd.atom.config import (
    Config as _AtomConfig,
    get_current_atom_config,
    set_current_atom_config,
)
from vllm.models.deepseek_v4.amd.atom.models.v4_config import (
    DeepseekV4Args,
    make_v4_quant_config,
)


def dsv4_use_atom_moe() -> bool:
    """Whether the ported ATOM MoE replaces vLLM's FusedMoE MoE path.

    Default ON (opt-out via ``VLLM_DSV4_USE_ATOM_MOE=0``). Runs on top of vLLM's
    original attention; hash-routed layers get ``input_ids`` from a minimal ATOM
    forward context (:func:`atom_moe_forward_context`).
    """
    return os.environ.get("VLLM_DSV4_USE_ATOM_MOE", "1") == "1"


def atom_moe_forward_context(input_ids, positions):
    """Minimal ATOM forward context for the MoE (vLLM-attention path).

    ATOM's hash-routed MoE layers read the flat token ids from
    ``get_forward_context().context.input_ids`` inside the Dynamo-opaque
    ``moe_forward`` op (its signature can't carry input_ids). Set one here around
    the model forward. ``input_ids`` is vLLM's model-input tensor (a persistent
    fixed-address buffer under cudagraph, updated in place each step), so the
    captured hash-routing kernels re-read fresh ids on replay.
    """
    from contextlib import contextmanager

    import vllm.models.deepseek_v4.amd.atom.utils.forward_context as _fc

    @contextmanager
    def _ctx():
        prev = _fc._forward_context
        _fc.set_forward_context(
            attn_metadata={},
            atom_config=get_current_atom_config(),
            context=_fc.Context(positions=positions, input_ids=input_ids),
        )
        try:
            yield
        finally:
            _fc._forward_context = prev

    return _ctx()


_ATOM_ARGS_CACHE = {}


def setup_atom_config_and_args(vllm_config) -> DeepseekV4Args:
    """Build (and cache) the ATOM ``DeepseekV4Args`` and initialise the ATOM
    engine-config singleton (single-node stub) for the MoE.
    """
    hf = vllm_config.model_config.hf_config
    key = id(hf)
    args = _ATOM_ARGS_CACHE.get(key)
    if args is None:
        atom_config = _AtomConfig()
        # The ported ATOM FusedMoE sizes its shared topK-metadata buffers and
        # FusedMoEConfig to atom_config.max_num_batched_tokens, and reads
        # tensor_parallel_size on the weight-prep path. Bind them to vLLM's real
        # values so the topK buffers cover the runtime token budget.
        from vllm.distributed import get_tensor_model_parallel_world_size

        atom_config.max_num_batched_tokens = (
            vllm_config.scheduler_config.max_num_batched_tokens
        )
        try:
            atom_config.tensor_parallel_size = get_tensor_model_parallel_world_size()
        except Exception:
            atom_config.tensor_parallel_size = 1
        set_current_atom_config(atom_config)
        args = DeepseekV4Args.from_hf_config(hf)
        args.quant_config = make_v4_quant_config(hf)
        _ATOM_ARGS_CACHE[key] = args
    return args


_ATOM_V4_MOE_CLS = None


def _atom_v4_moe_cls():
    """Lazily build (once) the AtomV4MoE class.

    Deferring the ``moe_core`` import until first construction keeps the ATOM
    MoE's ``aiter`` custom-op registrations out of the process until the MoE is
    actually selected.
    """
    global _ATOM_V4_MOE_CLS
    if _ATOM_V4_MOE_CLS is not None:
        return _ATOM_V4_MOE_CLS

    from vllm.models.deepseek_v4.amd.atom.models.moe_core import MoE as _AtomMoE

    class AtomV4MoE(_AtomMoE):
        """ATOM DeepSeek-V4 MoE with vLLM's decoder-layer call signature.

        vLLM's decoder calls ``ffn(hidden_states, input_ids)``; ATOM's native
        ``MoE.forward`` takes only ``x`` (hash-routed layers read ``input_ids``
        from the ATOM forward context). Force ``alt_stream=None`` (dual-stream
        overlap out of scope, so ``_use_dual_stream`` stays False).
        """

        def __init__(self, vllm_config, prefix: str, args: DeepseekV4Args):
            layer_id = int(prefix.split("layers.")[1].split(".")[0])
            super().__init__(
                layer_id=layer_id, args=args, prefix=prefix, alt_stream=None
            )

        def forward(self, hidden_states, input_ids=None):
            return super().forward(hidden_states)

    _ATOM_V4_MOE_CLS = AtomV4MoE
    return _ATOM_V4_MOE_CLS


def AtomV4MoE(vllm_config, prefix: str, args: DeepseekV4Args):
    """Construct an AtomV4MoE (builds the class lazily on first use)."""
    return _atom_v4_moe_cls()(vllm_config, prefix, args)
