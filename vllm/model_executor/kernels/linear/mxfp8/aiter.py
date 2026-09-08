# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP8 linear GEMM through aiter ``hipb_mm`` (hipBLASLt VEC32_UE8M0).

Consumes the same FP8 E4M3 weights + E8M0 block scales as the Triton
``RocmDotScaledMxfp8LinearKernel``, but hands the whole GEMM to hipBLASLt's
native MX scale mode instead of tiling it in Triton. Opt-in via
``VLLM_ROCM_USE_AITER_MXFP8_HIPBMM=1``; see that env var in ``vllm/envs.py``
for why it is off by default.
"""

import torch
from torch.nn.parameter import Parameter

from vllm._aiter_ops import rocm_aiter_ops
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
    MXFP8_SCALE_DTYPE,
    mxfp8_e4m3_quantize,
)
from vllm.platforms import current_platform

from .Mxfp8LinearKernel import Mxfp8LinearKernel, Mxfp8LinearLayerConfig

# hipBLASLt's MX solutions come from rocRoller, whose workgroup-tile predicates
# reject an M that does not match the tile. Token counts are arbitrary, so the
# activation is zero-padded up to the required multiple and the output sliced
# back. The multiple depends on N: a heuristic sweep of M = 16..208 on gfx950 /
# ROCm 7.2.4 returns solutions for every multiple of 16 when N is a multiple of
# 64 (e.g. N=16896), but only for multiples of 64 when it is not (N=12448, the
# Kimi-K3 KDA in_proj_qkvgfab, has solutions at 64/128/192 and nowhere else).
MXFP8_M_ALIGNMENT = 16
MXFP8_M_ALIGNMENT_UNALIGNED_N = 64
MXFP8_N_ALIGNMENT = 64


def _m_alignment(n: int) -> int:
    return MXFP8_M_ALIGNMENT if n % MXFP8_N_ALIGNMENT == 0 else (
        MXFP8_M_ALIGNMENT_UNALIGNED_N
    )


class AiterHipbMMMxfp8LinearKernel(Mxfp8LinearKernel):
    """CDNA4 MXFP8 linear via aiter ``hipb_mm``."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_rocm():
            return False, "not ROCm"
        if not rocm_aiter_ops.is_mxfp8_hipbmm_enabled():
            return (
                False,
                "requires setting `VLLM_ROCM_USE_AITER=1`, "
                "`VLLM_ROCM_USE_AITER_LINEAR=1`, "
                "and `VLLM_ROCM_USE_AITER_MXFP8_HIPBMM=1` on gfx950.",
            )
        try:
            import aiter  # noqa: F401
        except Exception:
            return False, "requires aiter library to be installed."
        if not hasattr(aiter, "hipb_mm"):
            return False, "requires aiter hipb_mm support."
        return True, None

    @classmethod
    def can_implement(cls, c: Mxfp8LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = layer.weight.data  # [N, K] fp8
        N, K = weight.shape
        scale_k = K // MXFP8_BLOCK_SIZE
        weight_scale = layer.weight_scale.data[:N, :scale_k].contiguous()
        layer.weight = Parameter(weight.contiguous(), requires_grad=False)
        layer.weight_scale = Parameter(weight_scale, requires_grad=False)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if layer.weight_scale.dtype != MXFP8_SCALE_DTYPE:
            raise ValueError(
                f"Expected {MXFP8_SCALE_DTYPE} weight_scale, got "
                f"{layer.weight_scale.dtype}."
            )
        weight = layer.weight
        out_shape = (*x.shape[:-1], weight.shape[0])
        x2d = x.reshape(-1, x.shape[-1])

        if x2d.shape[-1] % MXFP8_BLOCK_SIZE != 0:
            raise ValueError(
                f"MXFP8 requires K divisible by {MXFP8_BLOCK_SIZE}, got "
                f"{x2d.shape[-1]}."
            )

        x_q, x_scale = mxfp8_e4m3_quantize(x2d)

        num_tokens = x_q.shape[0]
        pad = -num_tokens % _m_alignment(weight.shape[0])
        if pad:
            # A zero E8M0 byte is the smallest finite exponent (2^-127), not a
            # NaN, so the padded rows contribute exactly zero.
            x_q = torch.nn.functional.pad(x_q, (0, 0, 0, pad))
            x_scale = torch.nn.functional.pad(x_scale, (0, 0, 0, pad))

        # hipb_mm's `bias` drives hipBLASLt's BIAS epilogue, which the MX
        # solutions do not carry; add it afterwards instead.
        out = rocm_aiter_ops.hipb_mm_mxfp8(
            x_q,
            weight.t(),
            x_scale,
            layer.weight_scale,
            None,
            torch.bfloat16,
        )
        if pad:
            out = out[:num_tokens]
        out = out.to(x.dtype).reshape(out_shape)
        if bias is not None:
            out = out + bias
        return out
