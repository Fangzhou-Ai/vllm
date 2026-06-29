# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 model — hardware-isolated entry point.

The actual implementation lives under ``nvidia/`` and ``amd/``; this module
picks the right one for the current platform and re-exports the public
classes used by the model registry and quantization config lookup.
"""

from vllm.platforms import current_platform

from .quant_config import DeepseekV4FP8Config

# Pick the per-platform implementation. The NVIDIA branch is the static
# default that mypy sees; the ROCm/XPU branches override at runtime and are
# kept type-compatible via ``# type: ignore[assignment]``.
if current_platform.is_rocm():
    # DeepSeek-V4 on ROCm is served by the full-attention path in
    # ``vllm.models.deepseek_v4.amd`` (ported from ROCm/ATOM):
    # the unified-KV proxy bridge + v4_kernels, which replaces the native
    # sparse-MLA ROCm path. MTP stays native (unused without speculative
    # decoding). The native ``.amd.model`` / ``.amd.rocm`` modules are kept
    # because their components are shared (``.amd.mtp``, XPU, and the vLLM
    # attention-backend registry / spec-decode), but are no longer the served
    # ROCm DeepSeek-V4 model.
    from .amd.rocm_full_attention import (
        activate_dsv4_rocm_full_attention,
        get_dsv4_rocm_causal_lm,
    )

    activate_dsv4_rocm_full_attention()
    DeepseekV4ForCausalLM = get_dsv4_rocm_causal_lm()  # type: ignore[assignment,misc]
    from .amd.mtp import DeepSeekV4MTP
elif current_platform.is_xpu():
    from .xpu.model import DeepseekV4ForCausalLM  # type: ignore[assignment]
    from .xpu.mtp import DeepSeekV4MTP  # type: ignore[assignment]
else:
    from .nvidia.model import DeepseekV4ForCausalLM  # type: ignore[assignment]
    from .nvidia.mtp import DeepSeekV4MTP  # type: ignore[assignment]

__all__ = [
    "DeepSeekV4MTP",
    "DeepseekV4FP8Config",
    "DeepseekV4ForCausalLM",
]
