# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Native (non-plugin) activation of ATOM's DeepSeek-V4 path on ROCm.

This re-homes what the now-removed ``register_model`` general plugin did, MINUS
the model-registry override: the model class itself is selected natively by
``vllm/models/deepseek_v4/__init__.py`` (the ROCm + ``VLLM_DSV4_USE_ATOM``
branch). Here we only set ATOM's vLLM backbone and install the three vLLM
monkey-patches ATOM still relies on (act_dtype default, spec-decode metadata
allow-list, aiter graph-capture nesting).

Called once per process at DeepSeek-V4 model-resolve time (and idempotent), so
it fires in every worker that loads the model — unlike a registry mutation,
which would only run engine-side.
"""

import functools
import inspect
import logging

import torch

logger = logging.getLogger("atom")

_ACTIVATED = False


def _patch_attention_act_dtype(attention) -> None:
    """Give ``process_weights_after_loading`` a default ``act_dtype`` so ATOM's
    weight handling does not need the explicit arg (relocated from the old
    ``register.py``)."""
    orig = attention.process_weights_after_loading
    if getattr(orig, "_atom_default_act_dtype_patched", False):
        return
    try:
        sig = inspect.signature(orig)
        act_dtype_param = sig.parameters.get("act_dtype")
        if act_dtype_param is not None and act_dtype_param.default is not inspect._empty:
            return
    except Exception:
        pass

    @functools.wraps(orig)
    def wrapped(self, act_dtype: "torch.dtype" = torch.bfloat16):
        return orig(self, act_dtype)

    setattr(wrapped, "_atom_default_act_dtype_patched", True)
    attention.process_weights_after_loading = wrapped


def activate_atom_dsv4_native() -> None:
    """Set ATOM's vLLM backbone and install the vLLM patches ATOM needs.

    Idempotent: a module-level guard plus the underlying patches' own
    ``_atom_*_patched`` flags make repeat calls (across processes / re-imports)
    no-ops.
    """
    global _ACTIVATED
    if _ACTIVATED:
        return

    from vllm._atom.plugin.prepare import _set_framework_backbone

    _set_framework_backbone("vllm")

    try:
        from vllm.attention.layer import Attention, MLAAttention
    except ImportError:
        from vllm.model_executor.layers.attention import Attention, MLAAttention

    _patch_attention_act_dtype(Attention)
    _patch_attention_act_dtype(MLAAttention)

    # vLLM's speculative decoder keeps an allow-list of attention metadata
    # classes; ATOM uses its own after attention isolation.
    from vllm._atom.plugin.vllm.spec_decode_patch import apply_vllm_spec_decode_patch

    apply_vllm_spec_decode_patch()

    # Nest aiter's ca_comm.capture() inside vLLM graph capture so the fused
    # all-reduce+rmsnorm avoids a per-replay hipMemcpyAsync.
    from vllm._atom.plugin.vllm.graph_capture_patch import apply_graph_capture_patch

    apply_graph_capture_patch()

    _ACTIVATED = True
    logger.info("ATOM DeepSeek-V4 native ROCm path activated")
