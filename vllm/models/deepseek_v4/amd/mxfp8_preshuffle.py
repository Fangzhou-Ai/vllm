# SPDX-License-Identifier: Apache-2.0
"""FlyDSL mxfp8 GEMM path for the DeepSeek-V4 a8w8 blockscale sites (ROCm / gfx950).

Gated by VLLM_DSV4_USE_FLYDSL_MXFP8_GEMM. The switch is off by default, and when
it is off nothing in this module runs: the CK bpreshuffle path in rocm.py /
model.py is left exactly as it was.

The weight layout is shared with that path -- FlyDSL's preshuffle is
byte-identical to shuffle_weight(w, layout=(16, 16)) -- so a site preshuffled
here can still be served by the CK bpreshuffle GEMM. That is what happens when a
call's M is not 32-row aligned, which the scaled-MFMA kernel requires (decode
steps, ragged tails).
"""

import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

# Scale is 128-granular along K and stored in 256-K chunks; the MFMA kernel reads
# B in 128-row groups.
_BLOCK_K = 128
_SCALE_CHUNK_K = 256
_BLOCK_N = 128
# Row group of the compact scale layout, which bounds the alignment of M.
_ROW_GROUP = 32


def enabled() -> bool:
    return os.getenv("VLLM_DSV4_USE_FLYDSL_MXFP8_GEMM", "0") == "1"


def make_gemm(linear) -> "Mxfp8Gemm | None":
    """Load time: preshuffle the weight in place and return the dispatcher.

    Returns None when the site is not an fp8 blockscale linear, or its shape is
    outside what the kernel takes, in which case the caller keeps its own path.
    """
    from vllm._aiter_ops import rocm_aiter_ops

    if not rocm_aiter_ops.is_enabled():
        return None
    w = getattr(linear, "weight", None)
    if w is None or w.dim() != 2:
        return None
    n, k = w.shape
    # N % 128 and K % 256 are kernel requirements; K % 128 also covers the
    # group-128 quantization and N % 16 the shuffle.
    if n % _BLOCK_N != 0 or k % _SCALE_CHUNK_K != 0:
        return None
    ws = getattr(linear, "weight_scale_inv", None)  # per-block scale
    if ws is None:
        return None
    if ws.dtype == torch.float8_e8m0fnu:
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            _upcast_e8m0_to_fp32,
        )

        ws = _upcast_e8m0_to_fp32(ws).contiguous()
    from vllm.model_executor.utils import replace_parameter

    # Shuffle in place, so no unshuffled copy is kept.
    replace_parameter(
        linear, "weight", rocm_aiter_ops.shuffle_weight(w.data, layout=(16, 16))
    )
    return Mxfp8Gemm(linear, ws)


class Mxfp8Gemm:
    """One preshuffled blockscale weight, run through the FlyDSL mxfp8 kernel."""

    def __init__(self, linear, scale: torch.Tensor):
        self.linear = linear
        self.scale = scale  # fp32 per-block scale, as the CK GEMM takes it
        self._b_scale: torch.Tensor | None = None  # compact e8m0 weight scale

    def __call__(self, x: torch.Tensor, reduce_tp: bool = False) -> torch.Tensor:
        w = self.linear.weight
        x2 = x.reshape(-1, x.shape[-1])
        m = x2.shape[0]
        n = w.shape[0]
        if m % _ROW_GROUP == 0:
            out = self._mxfp8(x2, w, m, n, x2.shape[1])
        else:
            out = self._ck(x2, w)
        if reduce_tp:
            from vllm.distributed import (
                get_tensor_model_parallel_world_size,
                tensor_model_parallel_all_reduce,
            )

            if get_tensor_model_parallel_world_size() > 1:
                out = tensor_model_parallel_all_reduce(out)
        return out if x.dim() == 2 else out.view(*x.shape[:-1], n)

    def _ck(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Fallback for an M the scaled-MFMA kernel cannot take. The preshuffled
        layout is the same, so the CK bpreshuffle GEMM reads the weight as is."""
        from vllm._aiter_ops import rocm_aiter_ops

        x_fp8, x_scale = rocm_aiter_ops.group_fp8_quant(x, transpose_scale=True)
        return rocm_aiter_ops.gemm_a8w8_blockscale_bpreshuffle(
            x_fp8, w, x_scale, self.scale, output_dtype=x.dtype
        )

    def _mxfp8(
        self, x: torch.Tensor, w: torch.Tensor, m: int, n: int, k: int
    ) -> torch.Tensor:
        from aiter import QuantType, dtypes, get_hip_quant
        from aiter.ops.flydsl.mxfp8_128_bpreshuffle_gemm_gfx950 import (
            compact_scale_w4,
            run_gemm_a8w8_mxfp8_128_bpreshuffle_gfx950,
        )

        if self._b_scale is None:
            # The weight scale's compact layout does not depend on M, so pack it
            # once. The checkpoint's scale_fmt is ue8m0, i.e. the fp32 mantissa is
            # all zeros and taking the exponent with >>23 is exact.
            ws = self.scale.reshape(n // _BLOCK_N, k // _BLOCK_K).contiguous()
            # A non-zero mantissa would be silently truncated to a smaller scale,
            # and the resulting precision loss is near impossible to trace back,
            # so assert rather than tolerate it.
            frac = ws.view(torch.int32) & 0x7FFFFF
            assert not frac.any(), (
                f"weight_scale_inv has {int((frac != 0).sum())} elements that are "
                f"not powers of two; FlyDSL's >>23 exponent extraction would lose "
                f"precision (N={n} K={k})"
            )
            logger.debug("mxfp8 GEMM active for N=%d K=%d (first M=%d)", n, k, m)
            self._b_scale = compact_scale_w4(
                (ws.view(torch.int32) >> 23).to(torch.uint8), _BLOCK_N, k
            )
        # Ask for e8m0 scales directly: scaled-MFMA only takes powers of two, so
        # this avoids a round trip through fp32's <<23 / >>23.
        from vllm._aiter_ops import FP8_DTYPE

        xq, xs = get_hip_quant(QuantType.per_1x128)(
            x.contiguous(),
            quant_dtype=FP8_DTYPE,
            transpose_scale=False,
            scale_type=dtypes.fp8_e8m0,
        )
        k_blocks = k // _BLOCK_K
        xs = xs.view(torch.uint8)
        xs = (
            xs.reshape(m, k_blocks)
            if xs.shape[0] == m
            else xs.reshape(k_blocks, m).t().contiguous()
        )
        a_scale = compact_scale_w4(xs, 1, k)
        wq = w.view(torch.float8_e4m3fn) if w.dtype == torch.uint8 else w
        out = x.new_empty(m, n)
        return run_gemm_a8w8_mxfp8_128_bpreshuffle_gfx950(
            xq, wq, a_scale, self._b_scale, out
        )
