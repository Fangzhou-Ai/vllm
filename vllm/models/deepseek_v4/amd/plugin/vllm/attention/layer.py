from typing import Optional

from vllm.models.deepseek_v4.amd.config import get_current_dsv4_config
from vllm.models.deepseek_v4.amd.model_ops.attention_mla import MLAModules
from vllm.models.deepseek_v4.amd.plugin.vllm.attention.layer_mla import (
    AttentionForVllmMLA,
    AttentionForVllmSparseMLA,
)
from vllm.models.deepseek_v4.amd.plugin.vllm.attention import ops as _dsv4_vllm_attention_ops  # noqa: F401


class AttentionForVllm:
    """Factory for ATOM-owned attention layers running under vLLM."""

    def __new__(
        cls,
        *args,
        use_mla: bool = False,
        mla_modules: Optional[MLAModules] = None,
        **kwargs,
    ):
        dsv4_config = get_current_dsv4_config()
        if dsv4_config is None:
            raise RuntimeError("dsv4_config is required for vLLM plugin attention")

        if use_mla:
            if mla_modules is not None and mla_modules.indexer is not None:
                return AttentionForVllmSparseMLA(
                    *args, mla_modules=mla_modules, **kwargs
                )
            return AttentionForVllmMLA(*args, mla_modules=mla_modules, **kwargs)
        # DeepSeek-V4 ROCm is MLA-only; non-MLA (MHA) attention was removed.
        raise NotImplementedError(
            "non-MLA attention is not supported in the DeepSeek-V4 ROCm build"
        )
