# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.utils.torch_utils import direct_register_custom_op


def mhc_pre_aiter(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Forward pass for mHC pre block.

    Args:
        residual: shape (..., hc_mult, hidden_size), dtype torch.bfloat16
        fn: shape (hc_mult3, hc_mult * hidden_size), dtype torch.float32
        hc_scale: shape (3,), dtype torch.float32
        hc_base: shape (hc_mult3,), dtype torch.float32
        rms_eps: RMS normalization epsilon
        hc_pre_eps: pre-mix epsilon
        hc_sinkhorn_eps: sinkhorn epsilon
        hc_post_mult_value: post-mix multiplier value
        sinkhorn_repeat: number of sinkhorn iterations
        n_splits: split-k factor;

    Returns:
        post_mix: shape (..., hc_mult), dtype torch.float32
        comb_mix: shape (..., hc_mult, hc_mult), dtype torch.float32
        layer_input: shape (..., hidden_size), dtype torch.bfloat16
    """

    hidden_size = residual.shape[-1]
    assert hidden_size % 256 == 0
    from vllm._aiter_ops import rocm_aiter_ops

    return rocm_aiter_ops.mhc_pre(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
    )


def _mhc_pre_aiter_fake(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    outer_shape = residual.shape[:-2]

    # Create empty tensors with correct shapes for meta device / shape inference
    post_mix = torch.empty(
        *outer_shape,
        hc_mult,
        1,
        dtype=torch.float32,
        device=residual.device,
    )
    comb_mix = torch.empty(
        *outer_shape,
        hc_mult,
        hc_mult,
        dtype=torch.float32,
        device=residual.device,
    )
    layer_input = torch.empty(
        *outer_shape,
        hidden_size,
        dtype=torch.bfloat16,
        device=residual.device,
    )

    return post_mix, comb_mix, layer_input


def mhc_post_aiter(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    hidden_size = residual.shape[-1]

    assert hidden_size % 256 == 0
    from vllm._aiter_ops import rocm_aiter_ops

    return rocm_aiter_ops.mhc_post(
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
    )


def _mhc_post_aiter_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(residual)


def mhc_fused_post_pre_aiter(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """MHC post followed by the next pre, in one aiter call.

    aiter decides per call whether the fused kernel or the two-launch path is
    faster for this m, so this is not unconditionally a single kernel.
    """
    from aiter.ops.mhc import mhc_fused_post_pre

    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    residual_flat = residual.view(-1, hc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]

    post_mix, comb_mix, layer_input, next_residual = mhc_fused_post_pre(
        x.view(num_tokens, hidden_size),
        residual_flat,
        post_layer_mix.view(num_tokens, hc_mult, 1),
        comb_res_mix.view(num_tokens, hc_mult, hc_mult),
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
    )
    return next_residual, post_mix, comb_mix, layer_input


def _mhc_fused_post_pre_aiter_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    num_tokens = residual.view(-1, hc_mult, hidden_size).shape[0]
    return (
        torch.empty_like(residual),
        torch.empty(
            (num_tokens, hc_mult, 1), dtype=torch.float32, device=residual.device
        ),
        torch.empty(
            (num_tokens, hc_mult, hc_mult),
            dtype=torch.float32,
            device=residual.device,
        ),
        torch.empty(
            (num_tokens, hidden_size), dtype=residual.dtype, device=residual.device
        ),
    )


direct_register_custom_op(
    op_name="mhc_fused_post_pre_aiter",
    op_func=mhc_fused_post_pre_aiter,
    mutates_args=[],
    fake_impl=_mhc_fused_post_pre_aiter_fake,
)
direct_register_custom_op(
    op_name="mhc_pre_aiter",
    op_func=mhc_pre_aiter,
    mutates_args=[],
    fake_impl=_mhc_pre_aiter_fake,
)
direct_register_custom_op(
    op_name="mhc_post_aiter",
    op_func=mhc_post_aiter,
    mutates_args=[],
    fake_impl=_mhc_post_aiter_fake,
)
