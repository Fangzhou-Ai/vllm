# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kernel tests for ROCm AITER fused MoE.

This file owns the ROCm-specific fused-MoE custom-op path:
- custom-op registration and fake-tensor support
- enablement gating for ``VLLM_ROCM_USE_AITER_MOE`` and shared experts
- BF16 accuracy for the fused MoE kernel on representative shapes
- gfx950-only AITER MXFP4 W4A16 MoE support, accuracy, and determinism
- MoE-facing FP8 group-quant activation quality
- deterministic routing and representative gfx942 / gfx950 coverage

Generic fused-MoE backend selection and non-ROCm kernel coverage live in the
generic MoE test files under ``tests/kernels/moe``.

Raw ROCm AITER helper-op coverage such as ``group_fp8_quant`` and
``per_tensor_quant`` lives in ``tests/kernels/core/test_rocm_aiter_ops.py``.
This file only keeps the MoE-shaped integration angle for those helpers.
"""

import importlib
import math
import types
import warnings
from typing import Any, NamedTuple

import pytest
import torch
import torch.nn.functional as F

from tests.kernels.utils import _assert_deterministic
from vllm.platforms import current_platform
from vllm.platforms.rocm import on_gfx942, on_gfx950
from vllm.utils.torch_utils import set_random_seed

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCm-specific tests"
)


# Helpers -----------------------------------------------------------------


def _reload_envs():
    import vllm.envs as envs

    return importlib.reload(envs)


@pytest.fixture(autouse=True)
def _restore_rocm_env_state():
    """Restore global env + AITER flag state after every test."""
    yield
    _reload_envs()
    from vllm._aiter_ops import rocm_aiter_ops

    rocm_aiter_ops.refresh_env_variables()


def _assert_aiter_supported() -> None:
    from vllm._aiter_ops import is_aiter_found_and_supported

    assert is_aiter_found_and_supported(), (
        "aiter is required on supported ROCm hardware for this test"
    )


def _format_observed_rate(count: int, total: int) -> str:
    return f"{count / total:.4%} ({count}/{total})"


def _format_allowed_rate(rate: float, total: int) -> str:
    allowed_count = int(rate * total)
    return f"{rate:.4%} (<= {allowed_count}/{total})"


def _quantile(values: torch.Tensor, q: float) -> float:
    if values.numel() == 0:
        return 0.0
    return torch.quantile(values, q).item()


def _assert_close_budget(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    label: str,
    atol: float,
    rtol: float = 0.0,
    pass_rate: float = 0.99999,
    max_violation_factor: float = 3.0,
) -> None:
    actual_f = actual.detach().float().flatten()
    expected_f = expected.detach().float().flatten()
    abs_diff = (actual_f - expected_f).abs()
    allowed = atol + rtol * expected_f.abs()

    total = abs_diff.numel()
    within = abs_diff <= allowed
    passed = int(within.sum().item())
    failed = total - passed
    allowed_fail_rate = 1.0 - pass_rate

    max_abs = abs_diff.max().item()
    mean_abs = abs_diff.mean().item()
    p99_abs = _quantile(abs_diff, 0.99)
    p999_abs = _quantile(abs_diff, 0.999)
    worst_ratio = (abs_diff / allowed.clamp_min(1e-12)).max().item()
    max_atol = max_violation_factor * atol
    above_max_count = int((abs_diff > max_atol).sum().item())

    msg = (
        "[rocm_aiter_moe] "
        f"{label}: "
        f"pass={passed / total:.4%} ({passed}/{total}) "
        f"fail={_format_observed_rate(failed, total)} "
        f"allowed_fail={_format_allowed_rate(allowed_fail_rate, total)} "
        f"atol={atol:g} "
        f"rtol={rtol:g} "
        f"abs>{max_atol:g}={_format_observed_rate(above_max_count, total)} "
        f"allowed_above_max={_format_allowed_rate(0.0, total)} "
        f"max_abs={max_abs:.6g} "
        f"mean_abs={mean_abs:.6g} "
        f"p99_abs={p99_abs:.6g} "
        f"p999_abs={p999_abs:.6g} "
        f"worst_ratio={worst_ratio:.6g}"
    )
    print(msg)
    if failed > 0:
        warnings.warn(msg, stacklevel=2)

    assert passed / total >= pass_rate, msg
    assert max_abs <= max_atol, msg
    assert mean_abs <= atol * 0.25, msg


def _assert_group_quant_quality(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    label: str,
    preferred_rel: float = 0.06,
    pass_rate: float = 0.9999,
    max_rel: float = 0.55,
    max_fail_rate: float = 0.000005,
    mean_limit: float = 0.03,
) -> None:
    rel = (
        (actual.float() - expected.float()).abs()
        / expected.float().abs().clamp_min(1e-5)
    ).flatten()
    total = rel.numel()
    within_preferred_count = int((rel <= preferred_rel).sum().item())
    fail_count = total - within_preferred_count
    allowed_fail_rate = 1.0 - pass_rate
    above_max_count = int((rel > max_rel).sum().item())
    mean_rel = rel.mean().item()
    max_rel_err = rel.max().item()
    p99 = _quantile(rel, 0.99)
    p999 = _quantile(rel, 0.999)

    msg = (
        "[rocm_aiter_moe] "
        f"{label}: "
        f"rel<={preferred_rel:g} pass={within_preferred_count / total:.4%} "
        f"({within_preferred_count}/{total}) "
        f"fail={_format_observed_rate(fail_count, total)} "
        f"allowed_fail={_format_allowed_rate(allowed_fail_rate, total)} "
        f"rel>{max_rel:g}={_format_observed_rate(above_max_count, total)} "
        f"allowed_above_max={_format_allowed_rate(max_fail_rate, total)} "
        f"mean_rel={mean_rel:.6g} "
        f"max_rel={max_rel_err:.6g} "
        f"p99={p99:.6g} "
        f"p999={p999:.6g}"
    )
    print(msg)
    if fail_count > 0:
        warnings.warn(msg, stacklevel=2)

    assert within_preferred_count / total >= pass_rate, msg
    assert above_max_count / total <= max_fail_rate, msg
    assert mean_rel < mean_limit, msg


def ref_moe_forward(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
) -> torch.Tensor:
    """Float32 mask-based MoE reference for the ROCm fused-MoE kernel."""
    num_tokens, hidden_dim = hidden_states.shape
    num_experts = w1.shape[0]
    intermediate_dim = w1.shape[1] // 2
    topk = topk_ids.shape[1]
    device = hidden_states.device

    hidden_states_f = hidden_states.float()
    w1_f = w1.float()
    w2_f = w2.float()

    expanded = hidden_states_f.view(num_tokens, 1, hidden_dim).expand(
        num_tokens, topk, hidden_dim
    )
    expanded = expanded.reshape(num_tokens * topk, hidden_dim)
    output = torch.zeros(
        num_tokens * topk, hidden_dim, dtype=torch.float32, device=device
    )
    flat_topk_ids = topk_ids.view(-1).long()

    for expert_idx in range(num_experts):
        expert_mask = flat_topk_ids == expert_idx
        if expert_mask.sum() == 0:
            continue
        gate_up = expanded[expert_mask] @ w1_f[expert_idx].T
        gate = gate_up[:, :intermediate_dim]
        up = gate_up[:, intermediate_dim:]
        if activation == "silu":
            act = F.silu(gate) * up
        elif activation == "gelu":
            act = F.gelu(gate) * up
        else:
            raise ValueError(f"Unknown activation: {activation}")
        output[expert_mask] = act @ w2_f[expert_idx].T

    output = output.view(num_tokens, topk, hidden_dim)
    weights = topk_weights.float().view(num_tokens, topk, 1)
    return (output * weights).sum(dim=1)


class AiterMxfp4MoeCase(NamedTuple):
    hidden_states: torch.Tensor
    w1_kernel: torch.Tensor
    w2_kernel: torch.Tensor
    w1_ref: torch.Tensor
    w2_ref: torch.Tensor
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    moe_config: Any
    quant_config: Any


def _make_topk_ids(
    num_tokens: int,
    num_experts: int,
    topk: int,
    *,
    device: str = "cuda",
) -> torch.Tensor:
    """Generate distinct expert IDs per token with the same top-k shape as
    production routing."""
    router_logits = torch.randn(num_tokens, num_experts, device=device)
    _, topk_ids = torch.topk(torch.softmax(router_logits, dim=-1), k=topk, dim=-1)
    return topk_ids.to(torch.int32)


def _shuffle_moe_weights(
    w1: torch.Tensor,
    w2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shuffle MoE weights into the AITER CK layout used in production."""
    from vllm._aiter_ops import rocm_aiter_ops

    w1_shuffled, w2_shuffled = rocm_aiter_ops.shuffle_weights(w1, w2)
    w1_shuffled.is_shuffled = True
    w2_shuffled.is_shuffled = True
    return w1_shuffled, w2_shuffled


def _make_moe_case(
    *,
    num_tokens: int,
    hidden_dim: int,
    intermediate_dim: int,
    num_experts: int,
    topk: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    torch.set_default_device("cuda")
    set_random_seed(seed)

    hidden_states = torch.randn(num_tokens, hidden_dim, dtype=torch.bfloat16)
    w1 = torch.randn(
        num_experts,
        intermediate_dim * 2,
        hidden_dim,
        dtype=torch.bfloat16,
    ) / math.sqrt(hidden_dim)
    w2 = torch.randn(
        num_experts,
        hidden_dim,
        intermediate_dim,
        dtype=torch.bfloat16,
    ) / math.sqrt(intermediate_dim)
    topk_weights = torch.rand(num_tokens, topk, dtype=torch.float32)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_ids = _make_topk_ids(num_tokens, num_experts, topk)
    return {
        "hidden_states": hidden_states,
        "w1": w1,
        "w2": w2,
        "topk_weights": topk_weights,
        "topk_ids": topk_ids,
    }


def _interleave_gate_up_rows(t: torch.Tensor) -> torch.Tensor:
    """Reorder contiguous ``[gate; up]`` rows into gpt-oss interleaved order."""
    e, two_i = t.shape[0], t.shape[1]
    i, rest = two_i // 2, t.shape[2:]
    perm = (0, 2, 1, *range(3, 3 + len(rest)))
    return t.view(e, 2, i, *rest).permute(*perm).contiguous().view(e, two_i, *rest)


def _make_aiter_mxfp4_moe_case(
    *,
    num_tokens: int,
    hidden_dim: int,
    intermediate_dim: int,
    num_experts: int,
    topk: int,
    seed: int,
) -> AiterMxfp4MoeCase:
    from triton_kernels.numerics_details.mxfp import (
        downcast_to_mxfp,
        upcast_from_mxfp,
    )

    from tests.kernels.moe.utils import make_dummy_moe_config
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
        Mxfp4MoeBackend,
        convert_gpt_oss_weight_to_mxfp4_moe_kernel_format,
        make_mxfp4_moe_quant_config,
    )

    torch.set_default_device("cuda")
    set_random_seed(seed)

    hidden_states = torch.randn(num_tokens, hidden_dim, dtype=torch.bfloat16)
    w1 = torch.randn(
        num_experts,
        intermediate_dim * 2,
        hidden_dim,
        dtype=torch.bfloat16,
    ) / math.sqrt(hidden_dim)
    w2 = torch.randn(
        num_experts,
        hidden_dim,
        intermediate_dim,
        dtype=torch.bfloat16,
    ) / math.sqrt(intermediate_dim)
    topk_weights = torch.rand(num_tokens, topk, dtype=torch.float32)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_ids = _make_topk_ids(num_tokens, num_experts, topk)

    w1_q, w1_scale = downcast_to_mxfp(w1, torch.uint8, axis=-1)
    w2_q, w2_scale = downcast_to_mxfp(w2, torch.uint8, axis=-1)

    w1_ref = upcast_from_mxfp(w1_q, w1_scale, torch.bfloat16, axis=-1)
    w2_ref = upcast_from_mxfp(w2_q, w2_scale, torch.bfloat16, axis=-1)

    (
        w1_kernel,
        w2_kernel,
        w1_scale_kernel,
        w2_scale_kernel,
        _,
        _,
    ) = convert_gpt_oss_weight_to_mxfp4_moe_kernel_format(
        mxfp4_backend=Mxfp4MoeBackend.AITER,
        layer=torch.nn.Module(),
        # w13 rows must be gpt-oss-interleaved; the converter de-interleaves them
        # back to the contiguous gate/up blocks that ``w1_ref`` / ``ref_moe_forward``
        # use. w2 has no gate/up split, so it stays as-is.
        w13_weight=_interleave_gate_up_rows(w1_q),
        w2_weight=w2_q.clone(),
        w13_weight_scale=_interleave_gate_up_rows(w1_scale),
        w2_weight_scale=w2_scale.clone(),
    )

    moe_config = make_dummy_moe_config(
        num_experts=num_experts,
        experts_per_token=topk,
        hidden_dim=hidden_dim,
        intermediate_size=intermediate_dim,
        in_dtype=torch.bfloat16,
    )
    quant_config = make_mxfp4_moe_quant_config(
        Mxfp4MoeBackend.AITER,
        w1_scale=w1_scale_kernel,
        w2_scale=w2_scale_kernel,
    )

    return AiterMxfp4MoeCase(
        hidden_states=hidden_states,
        w1_kernel=w1_kernel,
        w2_kernel=w2_kernel,
        w1_ref=w1_ref,
        w2_ref=w2_ref,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        moe_config=moe_config,
        quant_config=quant_config,
    )


def _run_fused_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    activation_method: int,
    quant_method: int,
) -> torch.Tensor:
    w1_shuffled, w2_shuffled = _shuffle_moe_weights(w1, w2)
    return torch.ops.vllm.rocm_aiter_fused_moe(
        hidden_states,
        w1_shuffled,
        w2_shuffled,
        topk_weights,
        topk_ids,
        expert_mask=None,
        activation_method=activation_method,
        quant_method=quant_method,
        doweight_stage1=False,
    )


# Custom op tests ---------------------------------------------------------


def test_aiter_fused_moe_custom_op_registered():
    """The main fused-MoE custom op should stay registered for runtime use."""
    _assert_aiter_supported()
    import vllm._aiter_ops as aiter_ops  # noqa: F401

    assert hasattr(torch.ops.vllm, "rocm_aiter_fused_moe")
    assert callable(torch.ops.vllm.rocm_aiter_fused_moe)


def test_aiter_asm_moe_tkw1_custom_op_registered():
    """The tkw1 custom op should stay registered for FP8 apply-router-weight
    paths."""
    _assert_aiter_supported()
    import vllm._aiter_ops as aiter_ops  # noqa: F401

    assert hasattr(torch.ops.vllm, "rocm_aiter_asm_moe_tkw1")
    assert callable(torch.ops.vllm.rocm_aiter_asm_moe_tkw1)


def test_aiter_fused_moe_fake_tensor_support():
    """The fused-MoE op should preserve fake-tensor compatibility for
    torch.compile-style tracing."""
    _assert_aiter_supported()
    import vllm._aiter_ops  # noqa: F401

    num_tokens = 16
    hidden_dim = 1024
    intermediate_dim = 2048
    num_experts = 8
    topk = 2

    hidden_states = torch.randn(
        num_tokens, hidden_dim, dtype=torch.bfloat16, device="cuda"
    )
    w1 = torch.randn(
        num_experts,
        intermediate_dim * 2,
        hidden_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    w2 = torch.randn(
        num_experts,
        hidden_dim,
        intermediate_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    topk_weights = torch.rand(num_tokens, topk, dtype=torch.float32, device="cuda")
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_ids = _make_topk_ids(num_tokens, num_experts, topk, device="cuda")

    torch.library.opcheck(
        torch.ops.vllm.rocm_aiter_fused_moe,
        (hidden_states, w1, w2, topk_weights, topk_ids),
        kwargs={
            "expert_mask": None,
            "activation_method": 0,
            "quant_method": 0,
            "doweight_stage1": False,
        },
        test_utils=("test_faketensor",),
    )


def test_aiter_fused_moe_fake_preserves_packed_fp4_logical_width():
    """The public custom op should infer its output width from w2, not from
    the packed FP4 activation width."""
    from aiter import dtypes
    from torch._subclasses.fake_tensor import FakeTensorMode

    _assert_aiter_supported()
    if getattr(torch, "float4_e2m1fn_x2", None) != dtypes.fp4x2:
        pytest.skip("native torch FP4 dtype is required")

    with FakeTensorMode():
        output = torch.ops.vllm.rocm_aiter_fused_moe(
            torch.empty(0, 32, dtype=dtypes.fp4x2, device="cuda"),
            torch.empty(4, 64, 32, dtype=dtypes.fp4x2, device="cuda"),
            torch.empty(4, 64, 16, dtype=dtypes.fp4x2, device="cuda"),
            torch.empty(0, 2, device="cuda"),
            torch.empty(0, 2, dtype=torch.int32, device="cuda"),
            output_dtype=torch.bfloat16,
        )

    assert output.shape == (0, 64)
    assert output.dtype == torch.bfloat16


# Env gating tests --------------------------------------------------------


@pytest.mark.parametrize(
    ("use_aiter", "use_moe", "expected"),
    [
        (True, True, True),
        (True, False, False),
        (False, True, False),
        (False, False, False),
    ],
)
def test_aiter_moe_enablement_follows_env(
    use_aiter: bool,
    use_moe: bool,
    expected: bool,
    monkeypatch: pytest.MonkeyPatch,
):
    """The fused-MoE gate should depend only on the main AITER toggle and the
    MoE-specific toggle."""
    from vllm._aiter_ops import rocm_aiter_ops

    _assert_aiter_supported()

    with monkeypatch.context() as mp:
        mp.setenv("VLLM_ROCM_USE_AITER", "1" if use_aiter else "0")
        mp.setenv("VLLM_ROCM_USE_AITER_MOE", "1" if use_moe else "0")
        _reload_envs()
        rocm_aiter_ops.refresh_env_variables()

        assert rocm_aiter_ops.is_fused_moe_enabled() is expected


@pytest.mark.parametrize(
    ("use_aiter", "use_moe", "use_shared", "expected"),
    [
        (True, True, True, True),
        (True, True, False, False),
        (True, False, True, False),
        (False, True, True, False),
    ],
)
def test_aiter_moe_shared_experts_enablement_follows_env(
    use_aiter: bool,
    use_moe: bool,
    use_shared: bool,
    expected: bool,
    monkeypatch: pytest.MonkeyPatch,
):
    """Shared-expert fusion should only be enabled when the fused-MoE path is
    enabled too."""
    from vllm._aiter_ops import rocm_aiter_ops

    _assert_aiter_supported()

    with monkeypatch.context() as mp:
        mp.setenv("VLLM_ROCM_USE_AITER", "1" if use_aiter else "0")
        mp.setenv("VLLM_ROCM_USE_AITER_MOE", "1" if use_moe else "0")
        mp.setenv(
            "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS",
            "1" if use_shared else "0",
        )
        _reload_envs()
        rocm_aiter_ops.refresh_env_variables()

        assert rocm_aiter_ops.is_fusion_moe_shared_experts_enabled() is expected


@pytest.mark.parametrize("moe_padding", [True, False])
def test_aiter_moe_padding_env_var(
    moe_padding: bool,
    monkeypatch: pytest.MonkeyPatch,
):
    """The ROCm MoE padding env var should keep its exact parse contract."""
    with monkeypatch.context() as mp:
        mp.setenv("VLLM_ROCM_MOE_PADDING", "1" if moe_padding else "0")
        envs = _reload_envs()
        assert envs.VLLM_ROCM_MOE_PADDING is moe_padding


# Enum tests --------------------------------------------------------------


def test_quant_method_enum_values():
    """The AITER quant-method bridge enum should keep its wire values."""
    from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import QuantMethod

    assert QuantMethod.NO == 0
    assert QuantMethod.PER_TENSOR == 1
    assert QuantMethod.PER_TOKEN == 2
    assert QuantMethod.BLOCK_1X32 == 3
    assert QuantMethod.BLOCK_1X128 == 4
    assert QuantMethod.BLOCK_128x128 == 5


def test_activation_method_enum_values():
    """The AITER activation-method bridge enum should keep its wire values."""
    from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import (
        ActivationMethod,
    )

    assert ActivationMethod.SILU == 0
    assert ActivationMethod.GELU == 1


# MXFP4 kernel tests ------------------------------------------------------


def _make_distributed_mxfp4_config(
    backend: str, all2all_backend: str = "mori_low_latency"
):
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        FusedMoEParallelConfig,
        RoutingMethodType,
    )

    return FusedMoEConfig(
        num_experts=8,
        experts_per_token=2,
        hidden_dim=64,
        intermediate_size=64,
        num_local_experts=4,
        num_logical_experts=8,
        moe_parallel_config=FusedMoEParallelConfig(
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
        ),
        activation=MoEActivation.SILU,
        in_dtype=torch.bfloat16,
        device="cuda",
        routing_method=RoutingMethodType.Renormalize,
        moe_backend=backend,
        max_num_tokens=16,
    )


@pytest.mark.parametrize(
    ("quant_kind", "prequantized", "payload_width", "expected_gate_mode"),
    [
        pytest.param("mxfp8", True, 64, "interleave", id="mxfp8-prequant"),
        pytest.param("mxfp8", False, 64, "interleave", id="mxfp8-dynamic"),
        pytest.param("mxfp4", True, 32, "separated", id="mxfp4-prequant"),
    ],
)
def test_aiter_forwards_mx_quantization_metadata(
    monkeypatch, quant_kind, prequantized, payload_width, expected_gate_mode
):
    from aiter import dtypes
    from aiter.ops.flydsl.moe_common import GateMode

    from tests.kernels.moe.utils import make_dummy_moe_config
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        mxfp4_moe_quant_config,
        mxfp4_w4a8_moe_quant_config,
    )
    from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import (
        QuantMethod,
        rocm_aiter_fused_experts,
    )

    _assert_aiter_supported()
    activation_dtype = (
        dtypes.fp8
        if quant_kind == "mxfp8" and prequantized
        else dtypes.fp4x2
        if quant_kind == "mxfp4"
        else torch.bfloat16
    )
    if quant_kind == "mxfp4" and (
        getattr(torch, "float4_e2m1fn_x2", None) != dtypes.fp4x2
    ):
        pytest.skip("native torch FP4 dtype is required")

    captured: dict[str, Any] = {}

    def fused_moe(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return torch.empty(args[0].shape[0], 64, dtype=torch.bfloat16)

    monkeypatch.setattr(rocm_aiter_ops, "fused_moe", fused_moe)
    hidden_states = torch.empty(3, payload_width, dtype=activation_dtype)
    a1_scale = torch.empty(3, 2, dtype=dtypes.fp8_e8m0) if prequantized else None
    scale = torch.empty(1, dtype=torch.uint8)
    quant_config = (
        mxfp4_w4a8_moe_quant_config(scale, scale)
        if quant_kind == "mxfp8"
        else mxfp4_moe_quant_config(scale, scale)
    )
    moe_config = make_dummy_moe_config(
        num_experts=4,
        experts_per_token=2,
        hidden_dim=64,
        intermediate_size=32,
        activation=MoEActivation.SILU,
    )

    output = rocm_aiter_fused_experts(
        hidden_states,
        torch.empty(4, 64, 32, dtype=dtypes.fp4x2),
        torch.empty(4, 64, 16, dtype=dtypes.fp4x2),
        torch.rand(3, 2, dtype=torch.bfloat16),
        torch.zeros(3, 2, dtype=torch.int64),
        moe_config,
        activation=MoEActivation.SILU,
        quant_config=quant_config,
        a1q_scale=a1_scale,
        output_dtype=torch.bfloat16,
    )

    assert output.shape == (3, 64)
    assert captured["args"][0] is hidden_states
    assert captured["args"][3].dtype == torch.float32
    assert captured["args"][4].dtype == torch.int32
    assert captured["kwargs"]["quant_method"] == QuantMethod.BLOCK_1X32.value
    assert captured["kwargs"]["a1_scale"] is a1_scale
    assert (
        captured["kwargs"]["gate_mode"]
        == getattr(GateMode, expected_gate_mode.upper()).value
    )
    assert captured["kwargs"]["hidden_pad"] == 0


@pytest.mark.parametrize(
    ("quant_kind", "payload_width"),
    [
        pytest.param("mxfp8", 64, id="mxfp8"),
        pytest.param("mxfp4", 32, id="mxfp4"),
    ],
)
@pytest.mark.parametrize("num_tokens", [0, 3])
def test_ag_rs_transports_prequantized_mx_as_bytes(
    monkeypatch, quant_kind, payload_width, num_tokens
):
    from aiter import dtypes

    import vllm.model_executor.layers.fused_moe.all2all_utils as all2all_utils
    import vllm.model_executor.layers.fused_moe.prepare_finalize.naive_dp_ep as ag_rs
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.model_executor.layers.fused_moe.config import (
        mxfp4_moe_quant_config,
        mxfp4_w4a8_moe_quant_config,
    )

    _assert_aiter_supported()
    activation_dtype = dtypes.fp8 if quant_kind == "mxfp8" else dtypes.fp4x2
    if quant_kind == "mxfp4" and (
        getattr(torch, "float4_e2m1fn_x2", None) != dtypes.fp4x2
    ):
        pytest.skip("native torch FP4 dtype is required")

    quantized = torch.empty(num_tokens, payload_width, dtype=activation_dtype)
    scale = torch.empty(num_tokens, 2, dtype=dtypes.fp8_e8m0)
    captured: dict[str, Any] = {}

    def quantize(a1, quant_dtype):
        assert a1.shape == (num_tokens, 64)
        assert quant_dtype == activation_dtype
        return quantized, scale

    class FakeGroup:
        world_size = 2

        def dispatch(
            self,
            hidden_states,
            topk_weights,
            topk_ids,
            *,
            is_sequence_parallel,
            extra_tensors,
        ):
            captured["hidden_states"] = hidden_states
            captured["scales"] = extra_tensors
            assert not is_sequence_parallel
            return hidden_states, topk_weights, topk_ids, extra_tensors

    fake_group = FakeGroup()
    monkeypatch.setattr(ag_rs, "aiter_mx_quantize_input", quantize)
    monkeypatch.setattr(ag_rs, "get_ep_group", lambda: fake_group)
    monkeypatch.setattr(all2all_utils, "get_ep_all2all_manager", lambda _: fake_group)
    monkeypatch.setattr(rocm_aiter_ops, "is_fused_moe_enabled", lambda: True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx950", lambda: True)

    weight_scale = torch.empty(1, dtype=torch.uint8)
    quant_config = (
        mxfp4_w4a8_moe_quant_config(weight_scale, weight_scale)
        if quant_kind == "mxfp8"
        else mxfp4_moe_quant_config(weight_scale, weight_scale)
    )
    quant_config.dispatch_quant_dtype = activation_dtype
    prepare_finalize = all2all_utils.maybe_make_prepare_finalize(
        _make_distributed_mxfp4_config(
            "aiter_mxfp4_fp8" if quant_kind == "mxfp8" else "aiter_mxfp4_mxfp4",
            all2all_backend="allgather_reducescatter",
        ),
        quant_config,
        allow_new_interface=True,
        use_monolithic=False,
    )
    assert isinstance(prepare_finalize, ag_rs.MoEPrepareAndFinalizeNaiveDPEPModular)
    assert prepare_finalize.supports_mx_prequantized_inputs
    lora_mapping = torch.arange(num_tokens, dtype=torch.int32)
    lora_context = types.SimpleNamespace(
        punica_wrapper=types.SimpleNamespace(
            token_mapping_meta=types.SimpleNamespace(token_lora_mapping=lora_mapping)
        ),
        local_token_lora_mapping=None,
    )
    prepare_finalize.set_lora_context(lora_context)
    result = prepare_finalize.prepare(
        torch.empty(num_tokens, 64, dtype=torch.bfloat16),
        torch.rand(num_tokens, 2),
        torch.zeros(num_tokens, 2, dtype=torch.int32),
        num_experts=8,
        expert_map=None,
        apply_router_weight_on_input=False,
        quant_config=quant_config,
    )

    assert captured["hidden_states"].dtype == torch.uint8
    assert captured["scales"][0].dtype == torch.uint8
    torch.testing.assert_close(captured["scales"][1], lora_mapping)
    assert result[0].shape == (num_tokens, payload_width)
    assert result[0].dtype == activation_dtype
    assert result[0].data_ptr() == quantized.data_ptr()
    assert result[1].shape == (num_tokens, 2)
    assert result[1].dtype == dtypes.fp8_e8m0
    assert result[1].data_ptr() == scale.data_ptr()
    torch.testing.assert_close(lora_context.local_token_lora_mapping, lora_mapping)


def test_no_dp_keeps_aiter_mx_quantization_fused():
    from aiter import dtypes

    import vllm.model_executor.layers.fused_moe.modular_kernel as mk
    from vllm.model_executor.layers.fused_moe.config import (
        mxfp4_w4a8_moe_quant_config,
    )
    from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import (
        AiterExperts,
    )
    from vllm.model_executor.layers.fused_moe.prepare_finalize.no_dp_ep import (
        MoEPrepareAndFinalizeNoDPEPModular,
    )

    _assert_aiter_supported()
    moe_config = _make_distributed_mxfp4_config(
        "aiter_mxfp4_fp8", all2all_backend="deepep_high_throughput"
    )
    moe_config.moe_parallel_config.dp_size = 1
    moe_config.moe_parallel_config.ep_size = 1
    moe_config.moe_parallel_config.use_ep = False
    weight_scale = torch.empty(1, dtype=torch.uint8)
    quant_config = mxfp4_w4a8_moe_quant_config(weight_scale, weight_scale)
    quant_config.dispatch_quant_dtype = dtypes.fp8
    prepare_finalize = MoEPrepareAndFinalizeNoDPEPModular()
    kernel = mk.FusedMoEKernel(
        prepare_finalize,
        AiterExperts(moe_config, quant_config),
    )

    assert not prepare_finalize.supports_mx_prequantized_inputs
    assert kernel.impl.defer_input_quant
    hidden_states = torch.empty(3, 64, dtype=torch.bfloat16)
    result = prepare_finalize.prepare(
        hidden_states,
        torch.rand(3, 2),
        torch.zeros(3, 2, dtype=torch.int32),
        num_experts=8,
        expert_map=None,
        apply_router_weight_on_input=False,
        quant_config=quant_config,
        defer_input_quant=kernel.impl.defer_input_quant,
    )
    assert result[0] is hidden_states
    assert result[1] is None


def test_aiter_packed_fp4_problem_size_and_zero_token_output(monkeypatch):
    from aiter import dtypes

    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.model_executor.layers.fused_moe.config import mxfp4_moe_quant_config
    from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import (
        AiterExperts,
    )
    from vllm.model_executor.layers.fused_moe.modular_kernel import (
        FusedMoEKernelModularImpl,
    )

    _assert_aiter_supported()
    if getattr(torch, "float4_e2m1fn_x2", None) != dtypes.fp4x2:
        pytest.skip("native torch FP4 dtype is required")
    monkeypatch.setattr(rocm_aiter_ops, "is_fused_moe_enabled", lambda: True)

    scale = torch.empty(1, dtype=torch.uint8)
    experts = AiterExperts(
        _make_distributed_mxfp4_config("aiter_mxfp4_mxfp4"),
        mxfp4_moe_quant_config(scale, scale),
    )
    hidden_states = torch.empty(0, 32, dtype=dtypes.fp4x2)
    w1 = torch.empty(4, 64, 32, dtype=dtypes.fp4x2)
    w2 = torch.empty(4, 64, 16, dtype=dtypes.fp4x2)
    topk_weights = torch.empty(0, 2)
    topk_ids = torch.empty(0, 2, dtype=torch.int32)

    assert experts.moe_problem_size(hidden_states, w1, w2, topk_ids) == (
        4,
        0,
        64,
        64,
        2,
    )

    kernel = FusedMoEKernelModularImpl(None, experts)  # type: ignore[arg-type]
    output = kernel._fused_experts(
        in_dtype=torch.bfloat16,
        a1q=hidden_states,
        a1q_scale=torch.empty(0, 2, dtype=dtypes.fp8_e8m0),
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation=experts.moe_config.activation,
        global_num_experts=8,
        local_num_experts=4,
        expert_map=None,
        apply_router_weight_on_input=False,
        expert_tokens_meta=None,
    )

    assert output.shape == (0, 64)
    assert output.dtype == torch.bfloat16


def test_aiter_mxfp4_quant_scheme_support_matches_gfx950():
    """AITER MXFP4 MoE support should stay gfx950-only."""
    from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import (
        AiterExperts,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kMxfp4Static,
    )

    assert AiterExperts._supports_quant_scheme(kMxfp4Static, None) is on_gfx950()


@pytest.mark.skipif(not on_gfx950(), reason="gfx950 ROCm only")
@pytest.mark.parametrize(
    "quant_limit,model_limit,expected_limit",
    [
        pytest.param(10.0, 20.0, 10.0, id="quant-limit"),
        pytest.param(None, 10.0, 10.0, id="model-limit"),
        pytest.param(0.0, 10.0, None, id="explicit-zero"),
        pytest.param(None, None, None, id="unclamped"),
    ],
)
def test_aiter_mxfp4_prequantized_moe_applies_clamp(
    quant_limit, model_limit, expected_limit
):
    """Clamp routed gate/up values before the A4W4 intermediate quantization."""
    from aiter import QuantType, dtypes, get_torch_quant
    from aiter.utility.fp4_utils import e8m0_shuffle, e8m0_to_f32, mxfp4_to_f32

    from tests.kernels.moe.utils import make_dummy_moe_config
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import ocp_mx_moe_quant_config
    from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import (
        rocm_aiter_fused_experts,
    )
    from vllm.model_executor.layers.fused_moe.utils import aiter_mx_quantize_input

    _assert_aiter_supported()
    num_tokens, dim, num_experts, topk = 4, 256, 8, 6
    hidden = torch.zeros(num_tokens, dim, device="cuda", dtype=torch.bfloat16)
    hidden[:, 0] = hidden.new_tensor([1.0, 0.25, -1.0, 0.0])
    signs = torch.ones(dim, device="cuda", dtype=torch.bfloat16)
    signs[dim // 2 :] = -1
    # Powers of two survive input/weight MXFP4 quantization exactly.
    w1 = hidden.new_zeros(num_experts, 2 * dim, dim)
    w1[:, :dim, 0] = 32
    w1[:, dim:, 0] = 32 * signs
    w2 = torch.eye(dim, device="cuda", dtype=torch.bfloat16)
    w2 = w2.expand(num_experts, -1, -1).contiguous()
    quantize = get_torch_quant(QuantType.per_1x32)
    w1_q, w1_scale = quantize(w1, quant_dtype=dtypes.fp4x2)
    w2_q, w2_scale = quantize(w2, quant_dtype=dtypes.fp4x2)
    w1_q, w2_q = _shuffle_moe_weights(
        w1_q.view(num_experts, 2 * dim, dim // 2),
        w2_q.view(num_experts, dim, dim // 2),
    )
    quant_config = ocp_mx_moe_quant_config(
        "mxfp4",
        e8m0_shuffle(w1_scale),
        e8m0_shuffle(w2_scale),
        gemm1_clamp_limit=quant_limit,
    )
    moe_config = make_dummy_moe_config(
        num_experts=num_experts,
        experts_per_token=topk,
        hidden_dim=dim,
        intermediate_size=dim,
        activation=MoEActivation.SILU,
    )
    moe_config.swiglu_limit = model_limit
    topk_ids = torch.arange(topk, device="cuda", dtype=torch.int32)
    topk_ids = topk_ids.expand(num_tokens, -1).contiguous()
    topk_weights = torch.full((num_tokens, topk), 0.125, device="cuda")
    hidden_q, hidden_scale = aiter_mx_quantize_input(hidden, dtypes.fp4x2)

    gate = (hidden[:, :1].float() * 32).expand(-1, dim)
    up = gate * signs
    if expected_limit is not None:
        gate = gate.clamp(max=expected_limit)
        up = up.clamp(min=-expected_limit, max=expected_limit)
    intermediate = (F.silu(gate) * up).to(torch.bfloat16)
    intermediate_q, intermediate_scale = quantize(
        intermediate, quant_dtype=dtypes.fp4x2
    )
    expected = mxfp4_to_f32(intermediate_q).view(num_tokens, dim)
    expected *= (
        e8m0_to_f32(intermediate_scale)
        .view(num_tokens, -1)
        .repeat_interleave(32, dim=-1)
    )
    expected *= topk * 0.125

    output = rocm_aiter_fused_experts(
        hidden_q,
        w1_q,
        w2_q,
        topk_weights,
        topk_ids,
        moe_config,
        activation=MoEActivation.SILU,
        quant_config=quant_config,
        a1q_scale=hidden_scale,
        output_dtype=torch.bfloat16,
    )

    torch.testing.assert_close(output.float(), expected, atol=1e-4, rtol=0.01)


@pytest.mark.skipif(not on_gfx950(), reason="gfx950 ROCm only")
def test_aiter_fused_moe_mi350_mxfp4_w4a16_accuracy():
    """The gfx950 AITER MXFP4 W4A16 MoE path should match the dequantized
    MXFP4 reference."""
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import (
        rocm_aiter_fused_experts,
    )

    _assert_aiter_supported()
    case = _make_aiter_mxfp4_moe_case(
        num_tokens=32,
        hidden_dim=512,
        intermediate_dim=1024,
        num_experts=4,
        topk=2,
        seed=11,
    )
    ref_out = ref_moe_forward(
        case.hidden_states,
        case.w1_ref,
        case.w2_ref,
        case.topk_weights,
        case.topk_ids,
    )
    out = rocm_aiter_fused_experts(
        hidden_states=case.hidden_states,
        w1=case.w1_kernel,
        w2=case.w2_kernel,
        topk_weights=case.topk_weights,
        topk_ids=case.topk_ids,
        activation=MoEActivation.SWIGLUOAI,
        quant_config=case.quant_config,
        moe_config=case.moe_config,
        expert_mask=None,
    )

    assert out.shape == case.hidden_states.shape
    _assert_close_budget(
        out.float(),
        ref_out.float(),
        label="mi350_mxfp4_w4a16_accuracy",
        atol=0.1,
        rtol=0.1,
        pass_rate=0.99,
        max_violation_factor=2.0,
    )


@pytest.mark.skipif(not on_gfx950(), reason="gfx950 ROCm only")
def test_aiter_fused_moe_mi350_mxfp4_w4a16_determinism():
    """The gfx950 AITER MXFP4 W4A16 MoE path should stay bitwise
    deterministic."""
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import (
        rocm_aiter_fused_experts,
    )

    _assert_aiter_supported()
    case = _make_aiter_mxfp4_moe_case(
        num_tokens=8,
        hidden_dim=512,
        intermediate_dim=1024,
        num_experts=4,
        topk=2,
        seed=13,
    )

    def run_mxfp4_moe():
        return rocm_aiter_fused_experts(
            hidden_states=case.hidden_states,
            w1=case.w1_kernel,
            w2=case.w2_kernel,
            topk_weights=case.topk_weights,
            topk_ids=case.topk_ids,
            activation=MoEActivation.SWIGLUOAI,
            quant_config=case.quant_config,
            moe_config=case.moe_config,
            expert_mask=None,
        )

    _assert_deterministic(run_mxfp4_moe, n_runs=4)


# FP8 group-quant tests ---------------------------------------------------


@pytest.mark.skipif(
    not (on_gfx942() or on_gfx950()),
    reason="gfx942/gfx950 ROCm only",
)
@pytest.mark.parametrize("num_tokens,hidden_dim", [(16, 2048), (64, 4096), (128, 8192)])
def test_aiter_moe_group_fp8_quant_reconstructs_hidden_states(
    num_tokens: int,
    hidden_dim: int,
):
    """The MoE-facing FP8 group-quant helper should reconstruct hidden states
    within the expected FP8 error budget.

    The raw op's shape and standalone roundtrip contracts are covered in
    ``tests/kernels/core/test_rocm_aiter_ops.py``. This test keeps the
    representative hidden-state sizes that the fused-MoE path actually cares
    about.
    """
    from vllm._aiter_ops import rocm_aiter_ops

    _assert_aiter_supported()
    torch.set_default_device("cuda")
    set_random_seed(1)

    group_size = 128
    hidden_states = torch.randn(num_tokens, hidden_dim, dtype=torch.bfloat16)
    x_fp8, scales = rocm_aiter_ops.group_fp8_quant(hidden_states, group_size)

    assert x_fp8.shape == hidden_states.shape
    assert scales.shape == (num_tokens, (hidden_dim + group_size - 1) // group_size)
    assert scales.dtype == torch.float32

    scales_expanded = scales.repeat_interleave(group_size, dim=1)[:, :hidden_dim]
    dequantized = x_fp8.float() * scales_expanded

    _assert_group_quant_quality(
        dequantized,
        hidden_states,
        label=f"group_fp8_quant shape=({num_tokens}, {hidden_dim})",
    )


# Kernel accuracy tests ---------------------------------------------------


@pytest.mark.parametrize(
    "num_tokens,hidden_dim,intermediate_dim",
    [
        (16, 512, 1024),
        (128, 2048, 4096),
        (2048, 4096, 11008),
    ],
)
def test_aiter_fused_moe_bf16_accuracy(
    num_tokens: int,
    hidden_dim: int,
    intermediate_dim: int,
):
    """The ROCm AITER fused-MoE BF16 path should match the float32 reference
    on representative shapes."""
    from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import (
        ActivationMethod,
        QuantMethod,
    )

    _assert_aiter_supported()
    case = _make_moe_case(
        num_tokens=num_tokens,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        num_experts=8,
        topk=2,
        seed=0,
    )
    ref_out = ref_moe_forward(
        case["hidden_states"],
        case["w1"],
        case["w2"],
        case["topk_weights"],
        case["topk_ids"],
        activation="silu",
    )
    out = _run_fused_moe(
        case["hidden_states"],
        case["w1"],
        case["w2"],
        case["topk_weights"],
        case["topk_ids"],
        activation_method=int(ActivationMethod.SILU),
        quant_method=int(QuantMethod.NO),
    )

    assert out.shape == (num_tokens, hidden_dim)
    _assert_close_budget(
        out.float(),
        ref_out,
        label=f"bf16_accuracy shape=({num_tokens}, {hidden_dim}, {intermediate_dim})",
        atol=0.05,
        rtol=0.0,
    )


def test_aiter_fused_moe_gelu_accuracy():
    """The GELU activation variant should stay aligned with the float32
    reference."""
    from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import (
        ActivationMethod,
        QuantMethod,
    )

    _assert_aiter_supported()
    case = _make_moe_case(
        num_tokens=32,
        hidden_dim=512,
        intermediate_dim=1024,
        num_experts=4,
        topk=2,
        seed=42,
    )
    ref_out = ref_moe_forward(
        case["hidden_states"],
        case["w1"],
        case["w2"],
        case["topk_weights"],
        case["topk_ids"],
        activation="gelu",
    )
    out = _run_fused_moe(
        case["hidden_states"],
        case["w1"],
        case["w2"],
        case["topk_weights"],
        case["topk_ids"],
        activation_method=int(ActivationMethod.GELU),
        quant_method=int(QuantMethod.NO),
    )

    assert out.shape == case["hidden_states"].shape
    _assert_close_budget(
        out.float(),
        ref_out,
        label="gelu_accuracy",
        atol=0.05,
        rtol=0.0,
    )


def test_aiter_fused_moe_determinism():
    """The BF16 fused-MoE kernel should stay bitwise deterministic for the
    same inputs."""
    from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import (
        ActivationMethod,
        QuantMethod,
    )

    _assert_aiter_supported()
    case = _make_moe_case(
        num_tokens=8,
        hidden_dim=256,
        intermediate_dim=512,
        num_experts=4,
        topk=2,
        seed=2,
    )

    def run_moe():
        return _run_fused_moe(
            case["hidden_states"],
            case["w1"],
            case["w2"],
            case["topk_weights"],
            case["topk_ids"],
            activation_method=int(ActivationMethod.SILU),
            quant_method=int(QuantMethod.NO),
        )

    _assert_deterministic(run_moe, n_runs=4)


# Routed end-to-end tests -------------------------------------------------


@pytest.mark.parametrize(
    "num_tokens,num_experts,topk",
    [
        (1, 8, 2),
        (16, 8, 2),
        (64, 16, 4),
    ],
)
def test_aiter_fused_moe_end_to_end(
    num_tokens: int,
    num_experts: int,
    topk: int,
):
    """The full router-logits to top-k to fused-MoE path should stay accurate."""
    from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import (
        ActivationMethod,
        QuantMethod,
    )

    _assert_aiter_supported()
    torch.set_default_device("cuda")
    set_random_seed(7)

    hidden_dim = 512
    intermediate_dim = 1024
    hidden_states = torch.randn(num_tokens, hidden_dim, dtype=torch.bfloat16)
    w1 = torch.randn(
        num_experts,
        intermediate_dim * 2,
        hidden_dim,
        dtype=torch.bfloat16,
    ) / math.sqrt(hidden_dim)
    w2 = torch.randn(
        num_experts,
        hidden_dim,
        intermediate_dim,
        dtype=torch.bfloat16,
    ) / math.sqrt(intermediate_dim)

    router_logits = torch.randn(num_tokens, num_experts, device="cuda")
    router_probs = torch.softmax(router_logits, dim=-1)
    topk_weights, topk_ids = torch.topk(router_probs, k=topk, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights.float()
    topk_ids = topk_ids.to(torch.int32)

    ref_out = ref_moe_forward(hidden_states, w1, w2, topk_weights, topk_ids)
    out = _run_fused_moe(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        activation_method=int(ActivationMethod.SILU),
        quant_method=int(QuantMethod.NO),
    )

    assert out.shape == (num_tokens, hidden_dim)
    assert out.dtype == torch.bfloat16
    _assert_close_budget(
        out.float(),
        ref_out,
        label=f"end_to_end tokens={num_tokens} experts={num_experts} topk={topk}",
        atol=0.05,
        rtol=0.0,
    )


# Arch-specific tests -----------------------------------------------------


@pytest.mark.skipif(
    not (on_gfx942() or on_gfx950()),
    reason="gfx942/gfx950 ROCm only",
)
def test_aiter_fused_moe_mi3xx_bf16_accuracy():
    from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import (
        ActivationMethod,
        QuantMethod,
    )

    _assert_aiter_supported()
    case = _make_moe_case(
        num_tokens=64,
        hidden_dim=4096,
        intermediate_dim=11008,
        num_experts=8,
        topk=2,
        seed=42,
    )
    ref_out = ref_moe_forward(
        case["hidden_states"],
        case["w1"],
        case["w2"],
        case["topk_weights"],
        case["topk_ids"],
    )
    out = _run_fused_moe(
        case["hidden_states"],
        case["w1"],
        case["w2"],
        case["topk_weights"],
        case["topk_ids"],
        activation_method=int(ActivationMethod.SILU),
        quant_method=int(QuantMethod.NO),
    )

    _assert_close_budget(
        out.float(),
        ref_out,
        label=(f"mi3xx_bf16_accuracy arch={'gfx942' if on_gfx942() else 'gfx950'}"),
        atol=0.05,
        rtol=0.0,
    )


@pytest.mark.skipif(
    not (on_gfx942() or on_gfx950()),
    reason="gfx942/gfx950 ROCm only",
)
def test_aiter_fused_moe_mi3xx_fp8_accuracy():
    """The MI3xx FP8 per-tensor MoE path should stay within the measured FP8
    error budget."""
    from tests.kernels.moe.utils import make_test_weights
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import (
        ActivationMethod,
        QuantMethod,
    )

    _assert_aiter_supported()
    torch.set_default_device("cuda")
    set_random_seed(99)

    num_tokens = 32
    hidden_dim = 512
    intermediate_dim = 1024
    num_experts = 4
    topk = 2

    hidden_states = torch.randn(num_tokens, hidden_dim, dtype=torch.bfloat16) / 10
    fp8_dtype = current_platform.fp8_dtype()
    (w1_bf16, w1_fp8, w1_scale, _), (w2_bf16, w2_fp8, w2_scale, _) = make_test_weights(
        num_experts,
        intermediate_dim * 2,
        hidden_dim,
        torch.bfloat16,
        torch.float8_e4m3fn,
        per_out_ch_quant=False,
    )
    _, a1_scale = rocm_aiter_ops.per_tensor_quant(hidden_states, fp8_dtype)
    router_logits = torch.randn(num_tokens, num_experts, device="cuda")
    router_probs = torch.softmax(router_logits, dim=-1)
    topk_weights, topk_ids = torch.topk(router_probs, k=topk, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights.float()
    topk_ids = topk_ids.to(torch.int32)

    ref_out = ref_moe_forward(
        hidden_states,
        w1_bf16,
        w2_bf16,
        topk_weights,
        topk_ids,
    )

    w1_shuffled, w2_shuffled = _shuffle_moe_weights(w1_fp8, w2_fp8)
    out = torch.ops.vllm.rocm_aiter_fused_moe(
        hidden_states,
        w1_shuffled,
        w2_shuffled,
        topk_weights,
        topk_ids,
        expert_mask=None,
        activation_method=int(ActivationMethod.SILU),
        quant_method=int(QuantMethod.PER_TENSOR),
        doweight_stage1=False,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
    )

    assert out.shape == hidden_states.shape
    _assert_close_budget(
        out.float(),
        ref_out,
        label=(f"mi3xx_fp8_accuracy arch={'gfx942' if on_gfx942() else 'gfx950'}"),
        atol=0.02,
        rtol=0.0,
        pass_rate=1.0,
        max_violation_factor=1.5,
    )
