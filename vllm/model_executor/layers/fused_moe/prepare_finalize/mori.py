# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import mori
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.utils import aiter_mx_quantize_input
from vllm.platforms import current_platform

logger = init_logger(__name__)


class MoriPrepareAndFinalize(mk.FusedMoEPrepareAndFinalizeModular):
    """
    Prepare/Finalize using MoRI kernels.
    """

    def __init__(
        self,
        mori_op: mori.ops.EpDispatchCombineOp,
        max_tokens_per_rank: int,
        num_dispatchers: int,
        use_fp8_dispatch: bool = False,
        mxfp_dispatch_dtype: torch.dtype | None = None,
        compact_recv_layout: bool = False,
    ):
        super().__init__()
        self.mori_op = mori_op
        self.num_dispatchers_ = num_dispatchers
        self.max_tokens_per_rank = max_tokens_per_rank
        self.use_fp8_dispatch = use_fp8_dispatch
        self.mxfp_dispatch_dtype = mxfp_dispatch_dtype
        self.compact_recv_layout = compact_recv_layout

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def output_is_reduced(self) -> bool:
        return True

    @property
    def supports_mx_prequantized_inputs(self) -> bool:
        return self.mxfp_dispatch_dtype is not None

    def num_dispatchers(self):
        return self.num_dispatchers_

    def max_num_tokens_per_rank(self) -> int | None:
        return self.max_tokens_per_rank

    def topk_indices_dtype(self) -> torch.dtype | None:
        return torch.int32

    def supports_async(self) -> bool:
        return False

    def _recv_row_bound(self, num_tokens: int, arena_rows: int) -> int | None:
        """Return a safe static bound for compact receive buffers.

        Mori's intranode dispatch deduplicates each source token per
        destination rank and assigns receive rows through one destination-side
        atomic counter. Therefore, when every dispatcher carries ``num_tokens``
        rows, all valid receives are in the prefix
        ``[0, num_tokens * num_dispatchers)``.

        The bound becomes part of a captured graph, so fail closed unless the
        current topology and DP metadata prove that invariant. Inter-node Mori
        layouts are not assumed to be compact here.
        """
        if not self.compact_recv_layout or not is_forward_context_available():
            return None

        dp_metadata = get_forward_context().dp_metadata
        if dp_metadata is None:
            return None

        counts = dp_metadata.num_tokens_across_dp_cpu
        if counts.numel() != self.num_dispatchers_:
            return None
        if not bool((counts == num_tokens).all()):
            return None

        bound = num_tokens * self.num_dispatchers_
        return bound if bound < arena_rows else None

    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> mk.PrepareResultType:
        """
        Returns a tuple of:
        - quantized + dispatched a.
        - Optional quantized + dispatched a1_scales.
        - Optional ExpertTokensMetadata containing gpu/cpu tensors
          as big as the number of local experts with the information about the
          number of tokens assigned to each local expert.
        - Optional dispatched expert topk IDs
        - Optional dispatched expert topk weight
        """
        assert not apply_router_weight_on_input, (
            "mori does not support apply_router_weight_on_input=True now."
        )
        num_tokens = a1.shape[0]
        scale = None
        if self.mxfp_dispatch_dtype is not None and defer_input_quant:
            raise ValueError(
                "MXFP4/MXFP8 dispatch requires the prepare step to quantize activations"
            )
        # When defer_input_quant is True, the expert kernel handles
        # quantization internally, so skip prepare-side dispatch quantization.
        if (self.use_fp8_dispatch or self.mxfp_dispatch_dtype is not None) and not (
            defer_input_quant
        ):
            if self.mxfp_dispatch_dtype is not None:
                a1, scale = aiter_mx_quantize_input(a1, self.mxfp_dispatch_dtype)
            elif quant_config.is_block_quantized:
                from aiter import QuantType, get_hip_quant

                quant_func = get_hip_quant(QuantType.per_1x128)
                a1, scale = quant_func(a1, quant_dtype=current_platform.fp8_dtype())
            elif quant_config.is_per_act_token:
                from aiter import QuantType, get_hip_quant

                quant_func = get_hip_quant(QuantType.per_Token)
                a1, scale = quant_func(a1, quant_dtype=current_platform.fp8_dtype())

        (
            dispatch_a1,
            dispatch_weights,
            dispatch_scale,
            dispatch_ids,
            dispatch_recv_token_num,
        ) = self.mori_op.dispatch(a1, topk_weights, scale, topk_ids)

        bound = self._recv_row_bound(num_tokens, dispatch_a1.shape[0])
        if bound is not None:
            dispatch_a1 = dispatch_a1[:bound]
            dispatch_weights = dispatch_weights[:bound]
            dispatch_ids = dispatch_ids[:bound]
            if dispatch_scale is not None:
                dispatch_scale = dispatch_scale[:bound]

        expert_tokens_meta = mk.ExpertTokensMetadata(
            expert_num_tokens=dispatch_recv_token_num, expert_num_tokens_cpu=None
        )

        return (
            dispatch_a1,
            dispatch_scale,
            expert_tokens_meta,
            dispatch_ids,
            dispatch_weights,
        )

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None:
        num_token = output.shape[0]
        result = self.mori_op.combine(
            fused_expert_output,
            None,
            topk_ids,
        )[0]
        output.copy_(result[:num_token])
