# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _fused_q_kv_rmsnorm_kernel(
    q_ptr,
    q_out_ptr,
    q_weight_ptr,
    q_in_stride,
    q_out_stride,
    kv_ptr,
    kv_out_ptr,
    kv_weight_ptr,
    kv_in_stride,
    kv_out_stride,
    eps,
    Q_SIZE: tl.constexpr,
    KV_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # num_tokens goes on grid-x (max 2**31 - 1); task goes on grid-y.
    # CUDA's grid-y/z are capped at 65535, so putting num_tokens there crashes
    # the launch at max-num-batched-tokens >= 65536 with "invalid argument".
    # int64: q_in_stride can be ~24K (128 heads × 192) and overflows int32
    # past num_tokens ~87K under large chunked prefill.
    token_idx = tl.program_id(0).to(tl.int64)
    pid_task = tl.program_id(1)

    # RMSNorm in fp32 throughout — matches csrc/layernorm_kernels.cu's
    # `(scalar_t)(x * s_variance * w)` and DeepseekV4's compressor kernel, which
    # keep x, rrms, and w all in fp32 and perform a single cast at store.
    block = tl.arange(0, BLOCK_SIZE)
    is_q = pid_task == 0
    q_mask = (block < Q_SIZE) & is_q
    kv_mask = (block < KV_SIZE) & (pid_task != 0)

    # Keep task selection branchless. ROCm Triton can fail pointer
    # canonicalization for an scf.if that returns different pointer bases.
    q_row_in = q_ptr + token_idx * q_in_stride
    kv_row_in = kv_ptr + token_idx * kv_in_stride
    q_x = tl.load(q_row_in + block, mask=q_mask, other=0.0).to(tl.float32)
    kv_x = tl.load(kv_row_in + block, mask=kv_mask, other=0.0).to(tl.float32)
    x = q_x + kv_x

    size = tl.where(is_q, Q_SIZE, KV_SIZE)
    variance = tl.sum(x * x, axis=0) / size
    rrms = tl.rsqrt(variance + eps)
    q_w = tl.load(q_weight_ptr + block, mask=q_mask, other=0.0).to(tl.float32)
    kv_w = tl.load(kv_weight_ptr + block, mask=kv_mask, other=0.0).to(tl.float32)
    w = q_w + kv_w
    y = x * rrms * w
    q_row_out = q_out_ptr + token_idx * q_out_stride
    kv_row_out = kv_out_ptr + token_idx * kv_out_stride
    tl.store(q_row_out + block, y.to(q_out_ptr.dtype.element_ty), mask=q_mask)
    tl.store(kv_row_out + block, y.to(kv_out_ptr.dtype.element_ty), mask=kv_mask)


def fused_q_kv_rmsnorm(
    qr: torch.Tensor,
    kv: torch.Tensor,
    q_weight: torch.Tensor,
    kv_weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert qr.ndim == 2 and kv.ndim == 2
    assert qr.shape[0] == kv.shape[0], (
        f"token dim mismatch: qr={qr.shape}, kv={kv.shape}"
    )
    assert qr.stride(-1) == 1 and kv.stride(-1) == 1
    assert q_weight.is_contiguous() and kv_weight.is_contiguous()

    q_size = qr.shape[1]
    kv_size = kv.shape[1]
    num_tokens = qr.shape[0]
    qr_out = torch.empty_like(qr)
    kv_out = torch.empty_like(kv)
    if num_tokens == 0:
        return qr_out, kv_out

    block_size = triton.next_power_of_2(max(q_size, kv_size))
    _fused_q_kv_rmsnorm_kernel[(num_tokens, 2)](
        qr,
        qr_out,
        q_weight,
        qr.stride(0),
        qr_out.stride(0),
        kv,
        kv_out,
        kv_weight,
        kv.stride(0),
        kv_out.stride(0),
        eps,
        Q_SIZE=q_size,
        KV_SIZE=kv_size,
        BLOCK_SIZE=block_size,
    )
    return qr_out, kv_out
