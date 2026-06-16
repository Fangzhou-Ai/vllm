# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Vendored ATOM (DeepSeek-V4 path) for vLLM.

Pruned to the DeepSeek-V4 inference path under vLLM: ATOM's own engine runtime
(model_engine: scheduler/llm_engine/model_runner) and KV disaggregation
(kv_transfer) were removed -- vLLM owns scheduling/engine/runner on the native
ROCm path. Only ``SamplingParams`` is re-exported.
"""

from vllm._atom.sampling_params import SamplingParams

__all__ = [
    "SamplingParams",
]
