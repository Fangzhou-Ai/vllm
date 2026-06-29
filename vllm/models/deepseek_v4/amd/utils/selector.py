# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from functools import cache
from typing import Type

from vllm.models.deepseek_v4.amd.model_ops.attentions.backends import AttentionBackend
from vllm.models.deepseek_v4.amd.utils import resolve_obj_by_qualname
from vllm.models.deepseek_v4.amd.plugin.prepare import is_sglang, is_vllm


def get_attn_backend(
    block_size: int,
    use_mla: bool = False,
    use_gdn: bool = False,
    use_v4: bool = False,
) -> Type[AttentionBackend]:
    """Selects which attention backend to use and lazily imports it."""
    return _cached_get_attn_backend(
        block_size=block_size,
        use_mla=use_mla,
        use_gdn=use_gdn,
        use_v4=use_v4,
        use_sglang=False,
        use_vllm=True,
    )


@cache
def _cached_get_attn_backend(
    block_size: int,
    use_mla: bool = False,
    use_gdn: bool = False,
    use_v4: bool = False,
    use_sglang: bool = False,
    use_vllm: bool = False,
) -> Type[AttentionBackend]:

    # get device-specific attn_backend
    attention_cls = get_attn_backend_cls(
        block_size, use_mla, use_gdn, use_v4, use_sglang, use_vllm
    )
    if not attention_cls:
        raise ValueError(f"Invalid attention backend for {attention_cls}")
    return resolve_obj_by_qualname(attention_cls)


def get_attn_backend_cls(
    block_size, use_mla, use_gdn, use_v4, use_sglang, use_vllm
) -> str:
    if use_v4:
        return "vllm.models.deepseek_v4.amd.model_ops.attentions.deepseek_v4_attn.DeepseekV4Backend"
    if use_mla:
        return "vllm.models.deepseek_v4.amd.model_ops.attentions.aiter_mla.AiterMLABackend"
    return "vllm.models.deepseek_v4.amd.model_ops.attentions.aiter_attention.AiterBackend"  # noqa: E501
