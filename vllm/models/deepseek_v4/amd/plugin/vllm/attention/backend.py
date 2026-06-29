from typing import Type

import torch
from vllm.v1.attention.backends.mla.prefill.base import MLAPrefillBackend


class AiterMlaBackendForVllm:
    """vLLM-facing dense MLA backend surface for ATOM attention layers."""

    accept_output_buffer: bool = True
    supported_dtypes: list = [torch.float16, torch.bfloat16]
    forward_includes_kv_cache_update: bool = True

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_supported_kernel_block_sizes():
        return [1]

    @classmethod
    def get_preferred_block_size(cls, default_block_size: int) -> int:
        return 1

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (num_blocks, block_size, head_size)

    @classmethod
    def is_mla(cls) -> bool:
        return True

    @classmethod
    def is_ssm(cls) -> bool:
        return False

    @staticmethod
    def get_required_kv_cache_layout():
        return None

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [576]

    @classmethod
    def supports_alibi_sqrt(cls) -> bool:
        return False

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        return (1, 0, 2, 3) if include_num_layers_dimension else (0, 1, 2)

    @staticmethod
    def get_builder_cls() -> Type:
        from vllm.models.deepseek_v4.amd.plugin.vllm.attention.metadata import AiterMlaMetadataBuilderForVllm

        return AiterMlaMetadataBuilderForVllm

    @staticmethod
    def get_impl_cls():
        from vllm.models.deepseek_v4.amd.plugin.vllm.attention.layer import AttentionForVllmMLA

        return AttentionForVllmMLA

    @classmethod
    def full_cls_name(cls) -> tuple[str, str]:
        return (cls.__module__, cls.__qualname__)


class AiterMLAPrefillBackend(MLAPrefillBackend):
    """vLLM 0.22 MLA prefill interface backed by ATOM's aiter path."""

    @staticmethod
    def get_name() -> str:
        return "DSV4_ROCM_AITER_MLA_PREFILL"

    def __init__(
        self,
        layer,
        num_heads: int,
        scale: float,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        vllm_config,
    ) -> None:
        super().__init__(
            num_heads=num_heads,
            scale=scale,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            vllm_config=vllm_config,
        )
        self._layer = layer

    def run_prefill_new_tokens(self, q, k, v, return_softmax_lse):
        return self._layer._run_prefill_new_tokens(
            self._prefill_metadata,
            q,
            k,
            v,
            return_softmax_lse,
        )

    def run_prefill_context_chunk(self, chunk_idx: int, q, k, v):
        return self._layer._run_prefill_context_chunk(
            self._prefill_metadata,
            chunk_idx,
            q,
            k,
            v,
        )


def build_vllm_mla_prefill_backend(layer, vllm_config):
    """Create the vLLM 0.22 MLA prefill backend for an ATOM MLA layer."""
    return AiterMLAPrefillBackend(
        layer=layer,
        num_heads=layer.num_heads,
        scale=layer.scale,
        kv_lora_rank=layer.kv_lora_rank,
        qk_nope_head_dim=layer.qk_nope_head_dim,
        qk_rope_head_dim=layer.qk_rope_head_dim,
        v_head_dim=layer.v_head_dim,
        vllm_config=vllm_config,
    )


class AiterSparseMlaBackendForVllm(AiterMlaBackendForVllm):
    """vLLM-facing sparse MLA backend surface for ATOM attention layers."""

    @staticmethod
    def get_supported_kernel_block_sizes():
        return [1, 64]

    @classmethod
    def get_preferred_block_size(cls, default_block_size: int) -> int:
        # Prefer block_size == 64 so the indexer's preshuffled path is taken.
        return 64

    @staticmethod
    def get_builder_cls() -> Type:
        from vllm.models.deepseek_v4.amd.plugin.vllm.attention.metadata import AiterMlaSparseMetadataBuilder

        return AiterMlaSparseMetadataBuilder

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @staticmethod
    def get_impl_cls():
        from vllm.models.deepseek_v4.amd.plugin.vllm.attention.layer import AttentionForVllmMLA

        return AttentionForVllmMLA

    @classmethod
    def full_cls_name(cls) -> tuple[str, str]:
        return (cls.__module__, cls.__qualname__)


class AiterSparseMlaIndexerBackendForVllm(AiterMlaBackendForVllm):
    """vLLM-facing sparse MLA indexer backend surface."""

    @staticmethod
    def get_supported_kernel_block_sizes():
        return [1, 64]

    @classmethod
    def get_preferred_block_size(cls, default_block_size: int) -> int:
        # Prefer block_size == 64 so the indexer's preshuffled path is taken.
        return 64

    @staticmethod
    def get_builder_cls() -> Type:
        from vllm.models.deepseek_v4.amd.plugin.vllm.attention.metadata import (
            AiterMlaSparseIndexerMetadataBuilder,
        )

        return AiterMlaSparseIndexerMetadataBuilder

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @staticmethod
    def get_impl_cls():
        from vllm.models.deepseek_v4.amd.plugin.vllm.attention.layer import AttentionForVllmMLA

        return AttentionForVllmMLA

    @classmethod
    def full_cls_name(cls) -> tuple[str, str]:
        return (cls.__module__, cls.__qualname__)

