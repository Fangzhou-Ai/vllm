# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""vLLM integration for ATOM.

Stage 2: ATOM's DeepSeek-V4 is activated as a native AMD/ROCm path (see
``vllm/models/deepseek_v4/__init__.py`` + ``vllm/platforms/rocm.py`` +
``native_activation.py``), not via an out-of-tree platform/general plugin.
"""

__all__: list[str] = []
