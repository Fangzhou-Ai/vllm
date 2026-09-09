# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("ROCm-specific tests", allow_module_level=True)

pytest.importorskip("mori", reason="Mori is required")
aiter = pytest.importorskip("aiter", reason="AITER is required")
from aiter import dtypes  # noqa: E402

import vllm.model_executor.layers.fused_moe.all2all_utils as all2all_utils  # noqa: E402
import vllm.model_executor.layers.fused_moe.oracle.mxfp4 as mxfp4_oracle  # noqa: E402
from vllm._aiter_ops import rocm_aiter_ops  # noqa: E402
from vllm.model_executor.layers.fused_moe.activation import (  # noqa: E402
    MoEActivation,
)
from vllm.model_executor.layers.fused_moe.config import (  # noqa: E402
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
    mxfp4_moe_quant_config,
    mxfp4_w4a8_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import (  # noqa: E402
    AiterExperts,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import (  # noqa: E402
    FusedMoEKernelModularImpl,
)
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (  # noqa: E402
    Mxfp4MoeBackend,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize.mori import (  # noqa: E402
    MoriPrepareAndFinalize,
)


class FakeMoriOp:
    def __init__(self):
        self.dispatch_args = None

    def dispatch(self, a1, topk_weights, scale, topk_ids):
        self.dispatch_args = (a1, topk_weights, scale, topk_ids)
        expert_counts = torch.tensor([a1.shape[0]], dtype=torch.int32, device=a1.device)
        return a1, topk_weights, scale, topk_ids, expert_counts


def _quant_config(activation_dtype: torch.dtype, *, for_dispatch: bool = False):
    scale = torch.empty(1, dtype=torch.uint8)
    if activation_dtype == dtypes.fp8:
        quant_config = mxfp4_w4a8_moe_quant_config(scale, scale)
    else:
        assert activation_dtype == dtypes.fp4x2
        quant_config = mxfp4_moe_quant_config(scale, scale)
    if for_dispatch:
        quant_config.dispatch_quant_dtype = activation_dtype
    return quant_config


def _moe_config(
    backend: str,
    *,
    hidden_dim: int = 64,
    activation: MoEActivation = MoEActivation.SILU,
    all2all_backend: str = "mori_low_latency",
) -> FusedMoEConfig:
    parallel_config = FusedMoEParallelConfig(
        tp_size=1,
        pcp_size=1,
        dp_size=2,
        ep_size=2,
        tp_rank=0,
        pcp_rank=0,
        dp_rank=0,
        ep_rank=0,
        sp_size=1,
        use_ep=True,
        all2all_backend=all2all_backend,
        enable_eplb=False,
    )
    return FusedMoEConfig(
        num_experts=8,
        experts_per_token=2,
        hidden_dim=hidden_dim,
        intermediate_size=64,
        num_local_experts=4,
        num_logical_experts=8,
        moe_parallel_config=parallel_config,
        activation=activation,
        in_dtype=torch.bfloat16,
        device="cuda",
        routing_method=RoutingMethodType.DeepseekV4,
        moe_backend=backend,
        max_num_tokens=16,
    )


def _require_native_fp4(activation_dtype: torch.dtype) -> None:
    if (
        activation_dtype == dtypes.fp4x2
        and getattr(torch, "float4_e2m1fn_x2", None) != dtypes.fp4x2
    ):
        pytest.skip("native torch FP4 dtype is required")


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, False), ("0", False), ("false", False), ("1", True), ("true", True)],
)
def test_all2all_prequant_env(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("VLLM_ROCM_ALL2ALL_PREQUANT", raising=False)
    else:
        monkeypatch.setenv("VLLM_ROCM_ALL2ALL_PREQUANT", value)

    getter = mxfp4_oracle.envs.environment_variables["VLLM_ROCM_ALL2ALL_PREQUANT"]
    assert getter() is expected


@pytest.mark.parametrize(
    "all2all_backend", ["mori_low_latency", "allgather_reducescatter"]
)
def test_deepseek_v4_w4a8_backend_selects_modular_aiter(monkeypatch, all2all_backend):
    monkeypatch.setattr(mxfp4_oracle.envs, "VLLM_ROCM_ALL2ALL_PREQUANT", True)
    monkeypatch.setattr(rocm_aiter_ops, "is_enabled", lambda: True)
    monkeypatch.setattr(rocm_aiter_ops, "is_fused_moe_enabled", lambda: True)
    monkeypatch.setattr(
        rocm_aiter_ops, "is_fusion_moe_shared_experts_enabled", lambda: False
    )
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx950", lambda: True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1250", lambda: False)
    fake_op = FakeMoriOp()
    manager = SimpleNamespace(
        rank=0,
        world_size=2,
        internode=False,
        get_handle=lambda _: fake_op,
    )
    monkeypatch.setattr(all2all_utils, "get_ep_all2all_manager", lambda _: manager)

    moe_config = _moe_config(
        "aiter_mxfp4_fp8",
        hidden_dim=256,
        activation=MoEActivation.SILU,
        all2all_backend=all2all_backend,
    )
    backend, experts_cls = mxfp4_oracle.select_deepseek_v4_mxfp4_moe_backend(moe_config)

    assert backend == Mxfp4MoeBackend.AITER_MXFP4_FP8
    assert experts_cls is AiterExperts
    quant_config = _quant_config(dtypes.fp8)
    kernel = mxfp4_oracle.make_mxfp4_moe_kernel(
        quant_config,
        moe_config,
        experts_cls,
        backend,
    )
    assert isinstance(kernel.fused_experts, AiterExperts)
    assert kernel.fused_experts.expects_unquantized_inputs
    assert not kernel.impl.defer_input_quant
    assert kernel.prepare_finalize.supports_mx_prequantized_inputs
    assert quant_config.dispatch_quant_dtype == dtypes.fp8
    if isinstance(kernel.prepare_finalize, MoriPrepareAndFinalize):
        assert kernel.prepare_finalize.mxfp_dispatch_dtype == dtypes.fp8


@pytest.mark.parametrize(
    "all2all_backend", ["mori_low_latency", "allgather_reducescatter"]
)
@pytest.mark.parametrize(
    ("backend_name", "expected_backend", "activation_dtype"),
    [
        (
            "aiter_mxfp4_fp8",
            Mxfp4MoeBackend.AITER_MXFP4_FP8,
            dtypes.fp8,
        ),
        (
            "aiter_mxfp4_mxfp4",
            Mxfp4MoeBackend.AITER_MXFP4_MXFP4,
            dtypes.fp4x2,
        ),
    ],
)
def test_all2all_prequant_can_be_disabled(
    monkeypatch,
    all2all_backend,
    backend_name,
    expected_backend,
    activation_dtype,
):
    monkeypatch.setattr(mxfp4_oracle.envs, "VLLM_ROCM_ALL2ALL_PREQUANT", False)
    if activation_dtype == dtypes.fp4x2:
        monkeypatch.setattr(torch, "float4_e2m1fn_x2", None)
    monkeypatch.setattr(rocm_aiter_ops, "is_enabled", lambda: True)
    monkeypatch.setattr(rocm_aiter_ops, "is_fused_moe_enabled", lambda: True)
    monkeypatch.setattr(
        rocm_aiter_ops, "is_fusion_moe_shared_experts_enabled", lambda: False
    )
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx950", lambda: True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1250", lambda: False)
    handle_args = {}

    def get_handle(args):
        handle_args.update(args)
        return FakeMoriOp()

    manager = SimpleNamespace(
        rank=0,
        world_size=2,
        internode=False,
        get_handle=get_handle,
    )
    monkeypatch.setattr(all2all_utils, "get_ep_all2all_manager", lambda _: manager)

    moe_config = _moe_config(
        backend_name,
        hidden_dim=256,
        activation=MoEActivation.SILU,
        all2all_backend=all2all_backend,
    )
    backend, experts_cls = mxfp4_oracle.select_deepseek_v4_mxfp4_moe_backend(moe_config)
    assert backend == expected_backend
    assert experts_cls is AiterExperts
    quant_config = _quant_config(activation_dtype)
    kernel = mxfp4_oracle.make_mxfp4_moe_kernel(
        quant_config,
        moe_config,
        experts_cls,
        backend,
    )

    assert quant_config.dispatch_quant_dtype is None
    assert kernel.impl.defer_input_quant
    if isinstance(kernel.prepare_finalize, MoriPrepareAndFinalize):
        assert not kernel.prepare_finalize.supports_mx_prequantized_inputs
        assert handle_args["quant_dtype"] == torch.bfloat16
        assert handle_args["scale_dim"] == 0


def test_deepseek_v4_w4a8_selection_is_transport_independent(monkeypatch):
    monkeypatch.setattr(rocm_aiter_ops, "is_enabled", lambda: True)
    monkeypatch.setattr(rocm_aiter_ops, "is_fused_moe_enabled", lambda: True)
    monkeypatch.setattr(
        rocm_aiter_ops, "is_fusion_moe_shared_experts_enabled", lambda: False
    )
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx950", lambda: True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1250", lambda: False)

    moe_config = _moe_config(
        "aiter_mxfp4_fp8",
        hidden_dim=256,
        activation=MoEActivation.SILU,
        all2all_backend="deepep_high_throughput",
    )
    backend, experts_cls = mxfp4_oracle.select_deepseek_v4_mxfp4_moe_backend(moe_config)

    assert backend == Mxfp4MoeBackend.AITER_MXFP4_FP8
    assert experts_cls is AiterExperts
    quant_config = _quant_config(dtypes.fp8, for_dispatch=True)
    experts = experts_cls(moe_config, quant_config)
    assert experts.expects_unquantized_inputs
    kernel_impl = object.__new__(FusedMoEKernelModularImpl)
    kernel_impl.fused_experts = experts
    kernel_impl.prepare_finalize = SimpleNamespace(
        supports_mx_prequantized_inputs=False
    )
    assert kernel_impl.defer_input_quant


@pytest.mark.parametrize(
    "all2all_backend",
    [
        "mori_low_latency",
        "allgather_reducescatter",
        "deepep_high_throughput",
    ],
)
def test_deepseek_v4_w4a8_uses_aiter_weight_layout(monkeypatch, all2all_backend):
    _require_native_fp4(dtypes.fp4x2)
    monkeypatch.setattr(rocm_aiter_ops, "is_fused_moe_enabled", lambda: True)
    shuffle_calls = []

    def shuffle_weight(weight, *, is_guinterleave, gate_up):
        shuffle_calls.append((is_guinterleave, gate_up))
        return weight.clone()

    def shuffle_scale(scale, num_experts, use_gu_interleave, gate_up):
        assert num_experts == 2
        shuffle_calls.append((use_gu_interleave, gate_up))
        return scale.clone()

    monkeypatch.setattr("aiter.ops.shuffle.shuffle_weight", shuffle_weight)
    monkeypatch.setattr("aiter.ops.shuffle.shuffle_scale", shuffle_scale)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1250", lambda: False)
    monkeypatch.setenv("AITER_BF16_FP8_MOE_BOUND", "17")

    layer = SimpleNamespace(
        moe_config=_moe_config(
            "aiter_mxfp4_fp8",
            activation=MoEActivation.SILU,
            all2all_backend=all2all_backend,
        )
    )
    w13 = torch.nn.Parameter(torch.zeros(2, 128, 32, dtype=torch.uint8), False)
    w2 = torch.nn.Parameter(torch.zeros(2, 64, 32, dtype=torch.uint8), False)
    w13_scale = torch.nn.Parameter(torch.zeros(2, 128, 2, dtype=torch.uint8), False)
    w2_scale = torch.nn.Parameter(torch.zeros(2, 64, 2, dtype=torch.uint8), False)

    converted = mxfp4_oracle.convert_weight_to_mxfp4_moe_kernel_format(
        Mxfp4MoeBackend.AITER_MXFP4_FP8,
        layer,
        w13,
        w2,
        w13_scale,
        w2_scale,
        activation=MoEActivation.SILU,
    )

    assert os.environ["AITER_BF16_FP8_MOE_BOUND"] == "0"
    assert shuffle_calls == [
        (True, True),
        (True, True),
        (True, False),
        (True, False),
    ]
    assert isinstance(converted[0], torch.nn.Parameter)
    assert isinstance(converted[1], torch.nn.Parameter)
    assert converted[0].is_shuffled
    assert converted[1].is_shuffled
    assert isinstance(converted[2], torch.Tensor)
    assert isinstance(converted[3], torch.Tensor)


@pytest.mark.parametrize(
    ("backend", "activation_dtype"),
    [
        ("aiter_mxfp4_fp8", dtypes.fp8),
        ("aiter_mxfp4_mxfp4", dtypes.fp4x2),
    ],
)
@pytest.mark.parametrize("num_tokens", [0, 3])
def test_mori_factory_quantizes_and_dispatches_mx(
    monkeypatch, backend, activation_dtype, num_tokens
):
    from vllm.platforms.rocm import on_gfx950

    if not on_gfx950():
        pytest.skip("gfx950 is required")
    _require_native_fp4(activation_dtype)
    monkeypatch.setattr(rocm_aiter_ops, "is_fused_moe_enabled", lambda: True)
    fake_op = FakeMoriOp()
    manager = SimpleNamespace(
        rank=1,
        world_size=2,
        internode=False,
        get_handle=lambda args: fake_op,
    )
    handle_args = {}

    def get_handle(args):
        handle_args.update(args)
        return fake_op

    manager.get_handle = get_handle
    monkeypatch.setattr(all2all_utils, "get_ep_all2all_manager", lambda _: manager)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx950", lambda: True)

    quant_config = _quant_config(activation_dtype, for_dispatch=True)
    prepare_finalize = all2all_utils.maybe_make_prepare_finalize(
        _moe_config(backend),
        quant_config,
    )

    assert isinstance(prepare_finalize, MoriPrepareAndFinalize)
    assert prepare_finalize.mori_op is fake_op
    assert prepare_finalize.mxfp_dispatch_dtype == activation_dtype
    assert prepare_finalize.compact_recv_layout
    assert handle_args == {
        "rank": 1,
        "num_ep_ranks": 2,
        "quant_dtype": activation_dtype,
        "token_hidden_size": 64,
        "scale_dim": 2,
        "scale_type_size": 1,
        "max_num_tokens_per_dp_rank": 16,
        "input_dtype": torch.bfloat16,
        "num_local_experts": 4,
        "num_experts_per_token": 2,
    }

    hidden_dim = 64
    topk_weights = torch.rand(num_tokens, 2, device="cuda")
    topk_ids = torch.zeros(num_tokens, 2, dtype=torch.int32, device="cuda")
    result = prepare_finalize.prepare(
        torch.randn(num_tokens, hidden_dim, dtype=torch.bfloat16, device="cuda"),
        topk_weights,
        topk_ids,
        num_experts=8,
        expert_map=None,
        apply_router_weight_on_input=False,
        quant_config=quant_config,
    )

    assert fake_op.dispatch_args == (result[0], topk_weights, result[1], topk_ids)
    assert result[0].shape == (
        num_tokens,
        hidden_dim // 2 if activation_dtype == dtypes.fp4x2 else hidden_dim,
    )
    assert result[0].dtype == activation_dtype
    assert result[0].is_contiguous()
    assert result[1].shape == (num_tokens, hidden_dim // 32)
    assert result[1].dtype == dtypes.fp8_e8m0
    assert result[1].is_contiguous()
    assert result[2].expert_num_tokens.tolist() == [num_tokens]
    assert result[3] is topk_ids
    assert result[4] is topk_weights


def test_aiter_mxfp4_dispatch_requires_native_torch_dtype(monkeypatch):
    monkeypatch.setattr(mxfp4_oracle.envs, "VLLM_ROCM_ALL2ALL_PREQUANT", True)
    monkeypatch.setattr(rocm_aiter_ops, "is_fused_moe_enabled", lambda: True)
    monkeypatch.setattr(torch, "float4_e2m1fn_x2", None)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx950", lambda: True)

    with pytest.raises(NotImplementedError, match="native torch FP4 dtype"):
        mxfp4_oracle.make_mxfp4_moe_kernel(
            _quant_config(dtypes.fp4x2),
            _moe_config("aiter_mxfp4_mxfp4"),
            AiterExperts,
            Mxfp4MoeBackend.AITER_MXFP4_MXFP4,
        )


@pytest.mark.parametrize(
    ("invalid_part", "error"),
    [
        ("payload_dtype", "invalid payload layout"),
        ("payload_width", "invalid payload layout"),
        ("payload_contiguity", "must be contiguous"),
        ("scale_dtype", "invalid scale layout"),
        ("scale_shape", "invalid scale layout"),
        ("scale_contiguity", "must be contiguous"),
    ],
)
def test_mori_prepare_rejects_invalid_mx_layout(monkeypatch, invalid_part, error):
    num_tokens = 3
    hidden_dim = 64
    quantized = torch.empty(num_tokens, hidden_dim, dtype=dtypes.fp8)
    scale = torch.empty(num_tokens, hidden_dim // 32, dtype=dtypes.fp8_e8m0)

    if invalid_part == "payload_dtype":
        quantized = torch.empty(num_tokens, hidden_dim, dtype=torch.bfloat16)
    elif invalid_part == "payload_width":
        quantized = torch.empty(num_tokens, hidden_dim - 1, dtype=dtypes.fp8)
    elif invalid_part == "payload_contiguity":
        quantized = torch.empty(hidden_dim, num_tokens, dtype=dtypes.fp8).T
    elif invalid_part == "scale_dtype":
        scale = torch.empty(num_tokens, hidden_dim // 32, dtype=torch.float32)
    elif invalid_part == "scale_shape":
        scale = torch.empty(num_tokens, 1, dtype=dtypes.fp8_e8m0)
    else:
        scale = torch.empty(hidden_dim // 32, num_tokens, dtype=dtypes.fp8_e8m0).T

    monkeypatch.setattr(
        aiter,
        "get_hip_quant",
        lambda _: lambda *args, **kwargs: (quantized, scale),
    )
    prepare_finalize = MoriPrepareAndFinalize(
        FakeMoriOp(),
        max_tokens_per_rank=16,
        num_dispatchers=2,
        mxfp_dispatch_dtype=dtypes.fp8,
    )
    quant_config = _quant_config(dtypes.fp8)

    with pytest.raises(ValueError, match=error):
        prepare_finalize.prepare(
            torch.randn(num_tokens, hidden_dim, dtype=torch.bfloat16),
            torch.rand(num_tokens, 2),
            torch.zeros(num_tokens, 2, dtype=torch.int32),
            num_experts=8,
            expert_map=None,
            apply_router_weight_on_input=False,
            quant_config=quant_config,
        )


@pytest.mark.parametrize("activation_dtype", [dtypes.fp8, dtypes.fp4x2])
def test_mori_mx_dispatch_rejects_deferred_quantization(activation_dtype):
    _require_native_fp4(activation_dtype)
    quant_config = _quant_config(activation_dtype)
    prepare_finalize = MoriPrepareAndFinalize(
        FakeMoriOp(),
        max_tokens_per_rank=16,
        num_dispatchers=2,
        mxfp_dispatch_dtype=activation_dtype,
    )

    with pytest.raises(ValueError, match="requires the prepare step"):
        prepare_finalize.prepare(
            torch.empty(1, 64, dtype=torch.bfloat16),
            torch.ones(1, 1),
            torch.zeros(1, 1, dtype=torch.int32),
            num_experts=8,
            expert_map=None,
            apply_router_weight_on_input=False,
            quant_config=quant_config,
            defer_input_quant=True,
        )
