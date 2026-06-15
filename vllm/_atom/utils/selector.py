# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from functools import cache
from typing import Type

from vllm._atom.model_ops.attentions.backends import AttentionBackend
from vllm._atom.utils import resolve_obj_by_qualname
from vllm._atom.plugin.prepare import is_sglang, is_vllm
from vllm._atom.utils import envs


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
        use_sglang=is_sglang(),
        use_vllm=is_vllm(),
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
        return "vllm._atom.model_ops.attentions.deepseek_v4_attn.DeepseekV4Backend"
    if use_mla:
        if envs.ATOM_USE_TRITON_MLA:
            return "vllm._atom.model_ops.attentions.triton_mla.TritonMLABackend"
        return "vllm._atom.model_ops.attentions.aiter_mla.AiterMLABackend"
    if use_gdn:
        if use_vllm:
            return "vllm._atom.plugin.vllm.attention.backend.GDNAttentionBackend"
        if use_sglang:
            return (
                "vllm._atom.plugin.sglang.attention_backend.attention_gdn.GDNAttentionBackend"
            )
        return "vllm._atom.model_ops.attentions.gdn_attn.GDNAttentionBackend"
    if envs.ATOM_USE_UNIFIED_ATTN:
        return "vllm._atom.model_ops.attentions.triton_mha.TritonMHABackend"
    return "vllm._atom.model_ops.attentions.aiter_attention.AiterBackend"  # noqa: E501
