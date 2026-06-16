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
    from vllm import envs as _vllm_envs

    if _vllm_envs.VLLM_DSV4_USE_ATOM:
        # Native AMD/ROCm path backed by the vendored ATOM model
        # (vllm/_atom). Replaces the former ATOM out-of-tree plugin: set the
        # backbone + install ATOM's vLLM patches here (per-process, at
        # model-resolve time), then resolve the arch to ATOM's wrapper.
        from vllm._atom.plugin.vllm.native_activation import (
            activate_atom_dsv4_native,
        )

        activate_atom_dsv4_native()
        from vllm._atom.plugin.vllm.model_wrapper import (
            ATOMMoEForCausalLM as DeepseekV4ForCausalLM,
        )
        from vllm._atom.plugin.vllm.model_wrapper import (
            ATOMMoEForCausalLM as DeepSeekV4MTP,
        )
    else:
        from .amd.model import DeepseekV4ForCausalLM
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
