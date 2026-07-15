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
from types import SimpleNamespace

import pytest
import torch
from packaging.version import Version

# this import statement is needed to ensure the ops are registered
from vllm._aiter_ops import rocm_aiter_ops
from vllm.model_executor.kernels.linear.scaled_mm.aiter import (
    AiterFp8BlockScaledMMKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import (
    FP8ScaledMMLinearLayerConfig,
)
from vllm.model_executor.layers.fusion.quant_activation import (
    QuantizedActivation,
    expose_input_quant_key,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8Dynamic128Sym,
    kFp8Static128BlockSym,
)
from vllm.models.deepseek_v4.amd.rocm import DeepseekV4ROCMAiterMLAAttention
from vllm.platforms import current_platform
from vllm.platforms.rocm import on_gfx950

# Check if aiter package is installed
aiter_available = importlib.util.find_spec("aiter") is not None

pytestmark = pytest.mark.skipif(
    not (current_platform.is_rocm() and aiter_available),
    reason="AITER ops are only available on ROCm with aiter package installed",
)


def _dsv4_attention_stub(main_wq_b, indexer_wq_b):
    return SimpleNamespace(
        wq_b=main_wq_b,
        indexer=None if indexer_wq_b is None else SimpleNamespace(wq_b=indexer_wq_b),
    )


def test_dsv4_shared_qr_quantization_fails_closed():
    def unexpected_quant(*args, **kwargs):
        raise AssertionError("invalid qr must not be externally quantized")

    quantizer_id = object()
    main_wq_b = SimpleNamespace(
        input_quant_key=kFp8Dynamic128Sym,
        input_quantizer_id=quantizer_id,
        quantize_input=unexpected_quant,
    )
    indexer_wq_b = SimpleNamespace(
        input_quant_key=kFp8Dynamic128Sym,
        input_quantizer_id=quantizer_id,
        quantize_input=unexpected_quant,
    )
    prepare = DeepseekV4ROCMAiterMLAAttention._prepare_qr_for_wq_b
    valid = torch.empty(4, 256, dtype=torch.bfloat16)
    noncontiguous = torch.empty(4, 512, dtype=torch.bfloat16)[:, ::2]
    invalid_inputs = [
        valid.float(),
        valid.view(2, 2, 256),
        valid[:0],
        noncontiguous,
        torch.empty(4, 130, dtype=torch.bfloat16),
    ]

    for qr in invalid_inputs:
        assert prepare(_dsv4_attention_stub(main_wq_b, indexer_wq_b), qr) is qr

    invalid_consumers = [
        (
            SimpleNamespace(
                input_quant_key=None,
                input_quantizer_id=quantizer_id,
                quantize_input=unexpected_quant,
            ),
            indexer_wq_b,
        ),
        (
            main_wq_b,
            SimpleNamespace(
                input_quant_key=None,
                input_quantizer_id=quantizer_id,
            ),
        ),
        (
            main_wq_b,
            SimpleNamespace(
                input_quant_key=kFp8Dynamic128Sym,
                input_quantizer_id=object(),
            ),
        ),
        (
            SimpleNamespace(
                input_quant_key=kFp8Dynamic128Sym,
                input_quantizer_id=quantizer_id,
            ),
            indexer_wq_b,
        ),
        (main_wq_b, None),
    ]
    for main, indexer in invalid_consumers:
        assert prepare(_dsv4_attention_stub(main, indexer), valid) is valid


def _dsv4_block_kernel(weight_shape):
    config = FP8ScaledMMLinearLayerConfig(
        weight_quant_key=kFp8Static128BlockSym,
        activation_quant_key=kFp8Dynamic128Sym,
        weight_shape=weight_shape,
        input_dtype=torch.bfloat16,
        out_dtype=torch.bfloat16,
    )
    kernel = AiterFp8BlockScaledMMKernel(config)
    kernel.quant_fp8.use_aiter = True
    kernel.quant_fp8._forward_method = kernel.quant_fp8.forward_hip
    return kernel


def _dsv4_linear_stub(kernel):
    n, k = kernel.config.weight_shape
    fp8_dtype = current_platform.fp8_dtype()
    layer = torch.nn.Module()
    layer.weight = torch.randn(n, k, dtype=torch.bfloat16, device="cuda").to(fp8_dtype)
    layer.weight_scale_inv = torch.rand(
        n // 128, k // 128, dtype=torch.float32, device="cuda"
    )
    layer.weight_scale = None
    layer.input_scale = None
    layer.input_scale_ub = None
    expose_input_quant_key(layer, kernel)
    return layer


def _record_quantizer_paths(kernels):
    paths = []
    for kernel in kernels:
        forward = kernel.quant_fp8._forward_method

        def counted_quant(*args, _forward=forward, **kwargs):
            paths.append(kwargs["use_triton"])
            return _forward(*args, **kwargs)

        kernel.quant_fp8._forward_method = counted_quant
    return paths


@pytest.mark.skipif(not on_gfx950(), reason="Production policy is for gfx950")
@pytest.mark.parametrize(
    ("main_shape", "indexer_shape", "expected_paths", "should_share"),
    [
        pytest.param(
            (8192, 1536),
            (8192, 1536),
            [False, False],
            True,
            id="pro-tp8-aiter-aiter",
        ),
        pytest.param(
            (8192, 1024),
            (8192, 1024),
            [True, True],
            True,
            id="flash-tp4-triton-triton",
        ),
        pytest.param(
            (4096, 1024),
            (8192, 1024),
            [False, True],
            False,
            id="flash-tp8-aiter-triton",
        ),
    ],
)
def test_dsv4_shared_qr_quantization_preserves_wq_b_outputs(
    main_shape,
    indexer_shape,
    expected_paths,
    should_share,
    default_vllm_config,
):
    torch.manual_seed(7)
    kernels = [
        _dsv4_block_kernel(main_shape),
        _dsv4_block_kernel(indexer_shape),
    ]
    assert [kernel.use_triton for kernel in kernels] == expected_paths
    layers = [_dsv4_linear_stub(kernel) for kernel in kernels]
    qr = torch.randn(4, main_shape[1], dtype=torch.bfloat16, device="cuda")
    quantizer_paths = _record_quantizer_paths(kernels)

    expected = [
        kernel.apply_weights(layer, qr) for kernel, layer in zip(kernels, layers)
    ]
    assert quantizer_paths == expected_paths

    quantizer_paths.clear()
    shared_qr = DeepseekV4ROCMAiterMLAAttention._prepare_qr_for_wq_b(
        _dsv4_attention_stub(*layers), qr
    )
    assert isinstance(shared_qr, QuantizedActivation) is should_share
    if not should_share:
        assert shared_qr is qr
    actual = [
        kernel.apply_weights(layer, shared_qr) for kernel, layer in zip(kernels, layers)
    ]
    assert quantizer_paths == ([expected_paths[0]] if should_share else expected_paths)

    for shared_output, expected_output in zip(actual, expected):
        assert torch.equal(shared_output, expected_output)


@pytest.mark.skipif(not on_gfx950(), reason="Production policy is for gfx950")
@pytest.mark.parametrize(
    ("weight_shape", "use_triton"),
    [
        ((8192, 1536), False),
        ((8192, 1024), True),
    ],
)
def test_dsv4_shared_qr_quantization_cudagraph_freshness(
    weight_shape, use_triton, default_vllm_config
):
    kernel = _dsv4_block_kernel(weight_shape)
    assert kernel.use_triton is use_triton
    main_wq_b = torch.nn.Module()
    indexer_wq_b = torch.nn.Module()
    expose_input_quant_key(main_wq_b, kernel)
    expose_input_quant_key(indexer_wq_b, _dsv4_block_kernel(weight_shape))
    attention = _dsv4_attention_stub(main_wq_b, indexer_wq_b)

    m, k = 4, weight_shape[1]
    static_input = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    caller_data = torch.empty(m, k, dtype=current_platform.fp8_dtype(), device="cuda")
    caller_scale = torch.empty(m, k // 128, dtype=torch.float32, device="cuda")

    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        for _ in range(2):
            kernel.quantize_input(static_input)
    torch.cuda.current_stream().wait_stream(side_stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        shared_qr = DeepseekV4ROCMAiterMLAAttention._prepare_qr_for_wq_b(
            attention, static_input
        )
        assert isinstance(shared_qr, QuantizedActivation)
        caller_data.copy_(shared_qr.data)
        caller_scale.copy_(shared_qr.scale)
    del shared_qr

    data_ptr = caller_data.data_ptr()
    scale_ptr = caller_scale.data_ptr()
    graph.replay()
    torch.accelerator.synchronize()
    expected = kernel.quantize_input(static_input)
    assert torch.equal(caller_data.view(torch.uint8), expected.data.view(torch.uint8))
    assert torch.equal(caller_scale, expected.scale)
    first_scale = caller_scale.clone()

    static_input.copy_(torch.randn_like(static_input) * 3)
    graph.replay()
    torch.accelerator.synchronize()
    expected = kernel.quantize_input(static_input)
    assert caller_data.data_ptr() == data_ptr
    assert caller_scale.data_ptr() == scale_ptr
    assert torch.equal(caller_data.view(torch.uint8), expected.data.view(torch.uint8))
    assert torch.equal(caller_scale, expected.scale)
    assert not torch.equal(caller_scale, first_scale)


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
