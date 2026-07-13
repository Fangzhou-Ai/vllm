# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# This is a test for the AITER group_fp8_quant op.
# It tests if the AITER op is
# 1. correctly defined the relationship between
#    implementation and fake function
# 2. can be used with torch.compile
# 3. can be used with CUDA graphs
# This file will be skipped if AITER is not installed
# and the platform is not ROCm.

import importlib.util

import pytest
import torch
from packaging.version import Version

# this import statement is needed to ensure the ops are registered
from vllm._aiter_ops import rocm_aiter_ops
from vllm.platforms import current_platform
from vllm.platforms.rocm import on_gfx950

# Check if aiter package is installed
aiter_available = importlib.util.find_spec("aiter") is not None

pytestmark = pytest.mark.skipif(
    not (current_platform.is_rocm() and aiter_available),
    reason="AITER ops are only available on ROCm with aiter package installed",
)


@pytest.mark.parametrize("transpose_scale", [False, True])
def test_rocm_aiter_group_fp8_quant_fake_implementation(transpose_scale):
    """Test that the fake implementation is correctly
    defined for torch.ops.vllm.rocm_aiter_group_fp8_quant."""
    # Create test tensors
    M = 4
    N = 256
    group_size = 128

    input_tensor = torch.randn((M, N), dtype=torch.bfloat16, device="cuda")

    # Verify the op's fake implementation using torch.library.opcheck
    # This checks that the fake function returns tensors with correct shapes and dtypes
    torch.library.opcheck(
        torch.ops.vllm.rocm_aiter_group_fp8_quant,
        (input_tensor, group_size, transpose_scale),
        test_utils=("test_faketensor",),
    )


def test_rocm_aiter_group_fp8_quant_transposed_scale_layout():
    """Transposed scales preserve shape but store group-major bytes."""
    input_tensor = torch.randn((4, 256), dtype=torch.bfloat16, device="cuda")

    output, scales = rocm_aiter_ops.group_fp8_quant(
        input_tensor, 128, transpose_scale=False
    )
    transposed_output, transposed_scales = rocm_aiter_ops.group_fp8_quant(
        input_tensor, 128, transpose_scale=True
    )
    expected_scales = scales.t().contiguous().view_as(scales)

    assert torch.equal(output.view(torch.uint8), transposed_output.view(torch.uint8))
    assert transposed_scales.shape == scales.shape
    assert transposed_scales.is_contiguous()
    assert torch.equal(
        transposed_scales.view(torch.uint8), expected_scales.view(torch.uint8)
    )


@pytest.mark.parametrize("transpose_scale", [False, True])
def test_rocm_aiter_group_fp8_quant_torch_compile_with_cudagraph(
    transpose_scale,
):
    """Test that rocm_aiter_ops.group_fp8_quant
    can be used with torch.compile in cudagraph mode."""
    # Create test tensors
    M = 4
    N = 256
    group_size = 128

    input_tensor = torch.randn((M, N), dtype=torch.bfloat16, device="cuda")

    # Define a function that uses the op
    def group_fp8_quant_fn(x):
        return rocm_aiter_ops.group_fp8_quant(
            x, group_size, transpose_scale=transpose_scale
        )

    # Compile with cudagraph mode
    compiled_fn = torch.compile(
        group_fp8_quant_fn,
        fullgraph=True,
        backend="inductor",
        mode="reduce-overhead",
        dynamic=False,
    )

    # Run eager mode
    x_fp8_eager, scales_eager = group_fp8_quant_fn(input_tensor)

    # Run compiled version (first run will trigger compilation)
    x_fp8_compiled, scales_compiled = compiled_fn(input_tensor)

    # Verify shapes match
    assert x_fp8_compiled.shape == x_fp8_eager.shape
    assert scales_compiled.shape == scales_eager.shape

    # Verify expected shapes
    assert x_fp8_compiled.shape == (M, N)
    expected_scale_cols = (N + group_size - 1) // group_size
    assert scales_compiled.shape == (M, expected_scale_cols)

    # Verify results match
    assert torch.allclose(
        x_fp8_compiled.to(torch.float32),
        x_fp8_eager.to(torch.float32),
        rtol=1e-2,
        atol=1e-2,
    )
    assert torch.allclose(scales_compiled, scales_eager, rtol=1e-3, atol=1e-3)

    # Test with different input (reusing compiled graph)
    input_tensor_2 = torch.randn((M, N), dtype=torch.bfloat16, device="cuda")
    x_fp8_eager_2, scales_eager_2 = group_fp8_quant_fn(input_tensor_2)
    x_fp8_compiled_2, scales_compiled_2 = compiled_fn(input_tensor_2)

    # Verify second run also produces correct results
    assert torch.allclose(
        x_fp8_compiled_2.to(torch.float32),
        x_fp8_eager_2.to(torch.float32),
        rtol=1e-2,
        atol=1e-2,
    )
    assert torch.allclose(scales_compiled_2, scales_eager_2, rtol=1e-3, atol=1e-3)


def test_rocm_aiter_group_fp8_quant_different_shapes():
    """Test rocm_aiter_ops.group_fp8_quant with different input shapes."""
    group_size = 128

    test_shapes = [
        (64, 2048),
        (256, 8192),
        (32, 1024),
        (512, 4096),
    ]

    for M, N in test_shapes:
        input_tensor = torch.randn((M, N), dtype=torch.bfloat16, device="cuda")

        x_fp8, scales = rocm_aiter_ops.group_fp8_quant(input_tensor, group_size)

        # Verify shapes
        assert x_fp8.shape == (M, N)
        expected_scale_cols = (N + group_size - 1) // group_size
        assert scales.shape == (M, expected_scale_cols)

        # Verify dtypes
        from aiter import dtypes

        assert x_fp8.dtype == dtypes.fp8
        assert scales.dtype == torch.float32


@pytest.mark.skipif(
    not on_gfx950() or Version(torch.version.hip or "0") < Version("7.2"),
    reason="AITER blockscale B-preshuffle requires gfx950 and ROCm 7.2 or newer",
)
def test_rocm_aiter_blockscale_bpreshuffle_gemm_matches_unshuffled():
    """Preshuffled DSV4 WQA/WKV GEMM matches the ordinary AITER path."""
    m, n, k = 4, 2048, 7168
    block_size = [128, 128]
    fp8_dtype = current_platform.fp8_dtype()
    torch.manual_seed(0)

    x = (torch.rand((m, k), dtype=torch.float16, device="cuda") / 10).to(fp8_dtype)
    weight = (torch.rand((n, k), dtype=torch.float16, device="cuda") / 10).to(fp8_dtype)
    x_scale = torch.rand((m, k // 128), dtype=torch.float32, device="cuda")
    weight_scale = torch.rand((n // 128, k // 128), dtype=torch.float32, device="cuda")

    expected = rocm_aiter_ops.gemm_a8w8_blockscale(
        x,
        weight,
        x_scale,
        weight_scale,
        block_size,
        output_dtype=torch.bfloat16,
    )
    shuffled_weight = rocm_aiter_ops.shuffle_weight(weight, (16, 16))
    transposed_x_scale = x_scale.t().contiguous().view_as(x_scale)
    actual = rocm_aiter_ops.gemm_a8w8_blockscale_bpreshuffle(
        x,
        shuffled_weight,
        transposed_x_scale,
        weight_scale,
        block_size,
        output_dtype=torch.bfloat16,
    )

    torch.testing.assert_close(actual.float(), expected.float(), rtol=5e-2, atol=5e-2)
