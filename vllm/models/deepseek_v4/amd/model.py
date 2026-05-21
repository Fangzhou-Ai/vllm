# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm entry point for DeepSeek-V4.

Reuses the NVIDIA implementation but swaps aux-stream creation for the
opt-in ROCm multi-stream path before exporting public symbols.
"""

from vllm.models.deepseek_v4.amd.multi_stream import create_dsv4_rocm_aux_stream_list
from vllm.models.deepseek_v4.nvidia import model as _nv_model

_nv_model._dsv4_aux_stream_list = create_dsv4_rocm_aux_stream_list

from vllm.models.deepseek_v4.nvidia.model import *  # noqa: F403
