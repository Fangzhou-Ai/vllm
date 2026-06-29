# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4 ROCm full-attention activation.

On ROCm the DeepSeek-V4 model is unconditionally
served through the full-attention stack in
``vllm.models.deepseek_v4.amd`` (ported from ROCm/ATOM): the
``DeepseekV4RocmForCausalLM`` wrapper drives the native DeepSeek-V4 attention
(unified-KV proxy bridge + ``model_ops/v4_kernels``), bypassing vLLM's native
sparse-MLA path (``DeepseekV4ROCMAiterMLAAttention`` / ``rocm_sparse_attn_*`` /
``DeepseekSparseSWAMetadataBuilder``).

The behaviours this path needs are integrated natively elsewhere: the
attention ``act_dtype`` default is handled at the loader call-site
(``deepseek_v4_rocm/model_loader/loader.py``), and the aiter graph-capture
nesting lives in ``vllm.distributed.parallel_state`` (a no-op when aiter is
absent). This module only flags the framework backbone and re-exports the
wrapper class so ``vllm/models/deepseek_v4/__init__.py`` can resolve
``DeepseekV4ForCausalLM`` to it. ``activate_dsv4_rocm_full_attention`` is
idempotent and safe to call in every worker process.
"""

import logging

logger = logging.getLogger(__name__)

_ACTIVATED = False


def activate_dsv4_rocm_full_attention() -> None:
    """Flag the framework backbone for the DSv4 ROCm full-attention path.

    Idempotent. The wrapper ``__init__`` sets this too; doing it here covers
    import-time branches as well.
    """
    global _ACTIVATED
    if _ACTIVATED:
        return

    from vllm.models.deepseek_v4.amd.plugin.prepare import (
        _set_framework_backbone,
    )

    _set_framework_backbone("vllm")

    _ACTIVATED = True
    logger.info(
        "DeepSeek-V4 ROCm: full-attention path activated "
        "(default ROCm DeepSeek-V4 path)"
    )


def get_dsv4_rocm_causal_lm():
    """Return the DSv4 ROCm full-attention causal-LM wrapper class."""
    from vllm.models.deepseek_v4.amd.plugin.vllm.model_wrapper import (
        DeepseekV4RocmForCausalLM,
    )

    return DeepseekV4RocmForCausalLM
