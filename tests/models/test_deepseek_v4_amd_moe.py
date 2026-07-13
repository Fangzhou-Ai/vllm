# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("DeepSeek V4 AMD MoE tests require ROCm", allow_module_level=True)

import vllm.models.deepseek_v4.amd.model as amd_model
from vllm.models.deepseek_v4.amd.model import (
    DeepseekV4FusedSharedRoutedExperts,
    _fp8_block_weight_to_mxfp4,
    _make_deepseek_v4_weights_mapper,
    _remap_shared_expert_to_routed,
    _should_fuse_shared_expert,
)


def _make_fusion_config():
    quant_config = SimpleNamespace(
        get_name=lambda: "deepseek_v4_fp8",
        moe_quant_algo="",
        weight_block_size=[128, 128],
        is_checkpoint_fp8_serialized=True,
        is_scale_e8m0=True,
        ignored_layers=[],
    )
    hf_config = SimpleNamespace(
        n_routed_experts=384,
        num_experts_per_tok=6,
        n_shared_experts=1,
        expert_dtype="fp4",
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf_config),
        quant_config=quant_config,
        parallel_config=SimpleNamespace(
            enable_expert_parallel=False,
            enable_eplb=False,
        ),
        kernel_config=SimpleNamespace(moe_backend="aiter"),
    )


@pytest.mark.parametrize(
    "guard",
    [
        "aiter_moe",
        "gfx950",
        "backend",
        "routed_experts",
        "top_k",
        "shared_experts",
        "expert_dtype",
        "moe_quant_algo",
        "block_size",
        "serialized_fp8",
        "scale_dtype",
        "ignored_layers",
        "expert_parallel",
        "eplb",
    ],
)
def test_deepseek_v4_shared_expert_fusion_guards(monkeypatch, guard):
    config = _make_fusion_config()
    monkeypatch.setattr(
        amd_model.rocm_aiter_ops,
        "is_fused_moe_enabled",
        lambda: guard != "aiter_moe",
    )
    monkeypatch.setattr(amd_model, "on_gfx950", lambda: guard != "gfx950")

    if guard == "backend":
        config.kernel_config.moe_backend = "triton"
    elif guard == "routed_experts":
        config.model_config.hf_config.n_routed_experts = 256
    elif guard == "top_k":
        config.model_config.hf_config.num_experts_per_tok = 8
    elif guard == "shared_experts":
        config.model_config.hf_config.n_shared_experts = 2
    elif guard == "expert_dtype":
        config.model_config.hf_config.expert_dtype = "fp8"
    elif guard == "moe_quant_algo":
        config.quant_config.moe_quant_algo = "NVFP4"
    elif guard == "block_size":
        config.quant_config.weight_block_size = [64, 128]
    elif guard == "serialized_fp8":
        config.quant_config.is_checkpoint_fp8_serialized = False
    elif guard == "scale_dtype":
        config.quant_config.is_scale_e8m0 = False
    elif guard == "ignored_layers":
        config.quant_config.ignored_layers = ["shared_experts"]
    elif guard == "expert_parallel":
        config.parallel_config.enable_expert_parallel = True
    elif guard == "eplb":
        config.parallel_config.enable_eplb = True

    assert not _should_fuse_shared_expert(config)


def test_deepseek_v4_shared_expert_fusion_policy_accepts_supported_config(
    monkeypatch,
):
    config = _make_fusion_config()
    monkeypatch.setattr(
        amd_model.rocm_aiter_ops,
        "is_fused_moe_enabled",
        lambda: True,
    )
    monkeypatch.setattr(amd_model, "on_gfx950", lambda: True)

    assert _should_fuse_shared_expert(config)


@pytest.mark.parametrize("projection", ["w1", "w2", "w3"])
@pytest.mark.parametrize("suffix", ["weight", "scale"])
def test_deepseek_v4_shared_expert_name_is_remapped_before_mapper(projection, suffix):
    name = f"layers.1.ffn.shared_experts.{projection}.{suffix}"
    remapped = _remap_shared_expert_to_routed(name, 384)

    assert remapped == f"layers.1.ffn.experts.384.{projection}.{suffix}"

    weight = torch.empty(0)
    mapped_name, _ = next(
        iter(_make_deepseek_v4_weights_mapper("fp4").apply([(remapped, weight)]))
    )
    mapped_suffix = "weight_scale" if suffix == "scale" else "weight"
    assert mapped_name == (
        f"model.layers.1.ffn.experts.384.{projection}.{mapped_suffix}"
    )


def test_deepseek_v4_unrelated_name_is_not_remapped():
    name = "layers.1.ffn.experts.3.w1.weight"
    assert _remap_shared_expert_to_routed(name, 384) == name


def _boundary_fp8_weight(rows: int = 128) -> torch.Tensor:
    values = torch.tensor(
        [
            0,
            0.25,
            0.75,
            1.25,
            1.75,
            2.5,
            3.5,
            5,
            6,
            -0.25,
            -0.75,
            -1.25,
            -1.75,
            -2.5,
            -3.5,
            -5,
            -6,
            0.5,
            1,
            1.5,
            2,
            3,
            4,
            -0.5,
            -1,
            -1.5,
            -2,
            -3,
            -4,
            0,
            6,
            -6,
        ],
        dtype=torch.bfloat16,
    )
    return values.repeat(rows * 4).view(rows, 128).to(torch.float8_e4m3fn)


def test_deepseek_v4_fp8_to_mxfp4_uses_round_to_nearest_even_bitwise():
    weight = _boundary_fp8_weight()
    scale = torch.ones(1, 1).to(torch.float8_e8m0fnu)

    packed, packed_scale = _fp8_block_weight_to_mxfp4(
        weight, scale, torch.device("cpu")
    )
    packed_from_raw_scale, raw_packed_scale = _fp8_block_weight_to_mxfp4(
        weight, scale.view(torch.uint8), torch.device("cpu")
    )

    expected = torch.tensor(
        [
            0,
            34,
            68,
            102,
            135,
            170,
            204,
            238,
            31,
            50,
            84,
            150,
            186,
            220,
            14,
            247,
        ],
        dtype=torch.uint8,
    ).repeat(4)
    assert packed.dtype == torch.uint8
    assert packed.shape == (128, 64)
    assert torch.equal(packed[0], expected)
    assert packed_scale.dtype == torch.uint8
    assert packed_scale.shape == (128, 4)
    assert torch.all(packed_scale == 127)
    assert torch.equal(packed_from_raw_scale, packed)
    assert torch.equal(raw_packed_scale, packed_scale)


def test_deepseek_v4_fp8_to_mxfp4_uses_reference_zero_scale():
    weight = torch.zeros((128, 128), dtype=torch.float8_e4m3fn)
    scale = torch.ones(1, 1).to(torch.float8_e8m0fnu)

    packed, packed_scale = _fp8_block_weight_to_mxfp4(
        weight, scale, torch.device("cpu")
    )

    assert torch.count_nonzero(packed) == 0
    assert torch.all(packed_scale == 1)


def _make_routed_experts(
    *,
    tp_size: int = 1,
    tp_rank: int = 0,
    intermediate_size_per_partition: int = 128,
    global_num_experts: int = 2,
) -> DeepseekV4FusedSharedRoutedExperts:
    experts = object.__new__(DeepseekV4FusedSharedRoutedExperts)
    nn.Module.__init__(experts)
    experts._pending_shared_expert_weights = {}
    experts._loaded_shared_expert_shards = set()
    experts.global_num_experts = global_num_experts
    experts.local_num_experts = 3
    experts.expert_map_manager = SimpleNamespace(
        num_fused_shared_experts=1,
        map_global_to_local=lambda expert_id: (
            2 if expert_id == global_num_experts else expert_id
        ),
    )
    experts.moe_config = SimpleNamespace(
        tp_rank=tp_rank,
        is_act_and_mul=True,
        moe_parallel_config=SimpleNamespace(tp_size=tp_size),
    )

    hidden_size = 128
    num_experts = 3
    experts.register_parameter(
        "w13_weight",
        nn.Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // 2,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        ),
    )
    experts.register_parameter(
        "w13_weight_scale",
        nn.Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // 32,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        ),
    )
    experts.register_parameter(
        "w2_weight",
        nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        ),
    )
    experts.register_parameter(
        "w2_weight_scale",
        nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // 32,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        ),
    )
    return experts


@pytest.mark.parametrize("scale_first", [False, True])
def test_deepseek_v4_shared_expert_loader_pairs_weight_and_scale(scale_first):
    experts = _make_routed_experts()
    weight = _boundary_fp8_weight()
    scale = torch.ones(1, 1).to(torch.float8_e8m0fnu).view(torch.uint8)
    packed, packed_scale = _fp8_block_weight_to_mxfp4(
        weight, scale, torch.device("cpu")
    )
    entries = [
        (experts.w13_weight, weight),
        (experts.w13_weight_scale, scale),
    ]
    if scale_first:
        entries.reverse()

    param, loaded = entries[0]
    assert experts.weight_loader(
        param,
        loaded,
        "experts.routed_experts.w13_weight",
        shard_id="w1",
        expert_id=2,
        return_success=True,
    )
    assert experts._pending_shared_expert_weights
    assert torch.count_nonzero(experts.w13_weight) == 0

    param, loaded = entries[1]
    assert experts.weight_loader(
        param,
        loaded,
        "experts.routed_experts.w13_weight_scale",
        shard_id="w1",
        expert_id=2,
        return_success=True,
    )
    assert not experts._pending_shared_expert_weights
    assert experts._loaded_shared_expert_shards == {"w1"}
    assert torch.equal(experts.w13_weight[2, :128], packed)
    assert torch.equal(experts.w13_weight_scale[2, :128], packed_scale)
    assert torch.count_nonzero(experts.w13_weight[:2]) == 0
    assert torch.count_nonzero(experts.w13_weight[2, 128:]) == 0


def test_deepseek_v4_shared_expert_loader_uses_padded_tp_sharding():
    experts = _make_routed_experts(
        tp_size=2,
        tp_rank=1,
        intermediate_size_per_partition=256,
    )
    weight = _boundary_fp8_weight(rows=384)
    scale = torch.ones(3, 1).to(torch.float8_e8m0fnu)
    packed, packed_scale = _fp8_block_weight_to_mxfp4(
        weight, scale, torch.device("cpu")
    )

    for param, loaded in (
        (experts.w13_weight_scale, scale),
        (experts.w13_weight, weight),
    ):
        assert experts.weight_loader(
            param,
            loaded,
            "experts.routed_experts.w13_weight",
            shard_id="w1",
            expert_id=2,
            return_success=True,
        )

    assert torch.equal(experts.w13_weight[2, :192], packed[192:])
    assert torch.count_nonzero(experts.w13_weight[2, 192:256]) == 0
    assert torch.equal(experts.w13_weight_scale[2, :192], packed_scale[192:])
    assert torch.count_nonzero(experts.w13_weight_scale[2, 192:256]) == 0


def test_deepseek_v4_shared_expert_loader_places_tp8_padded_projections():
    experts = _make_routed_experts(
        tp_size=8,
        tp_rank=7,
        intermediate_size_per_partition=512,
        global_num_experts=384,
    )
    generator = torch.Generator().manual_seed(0)
    weights = {
        "w1": torch.randn(3072, 128, generator=generator).to(torch.float8_e4m3fn),
        "w3": torch.randn(3072, 128, generator=generator).to(torch.float8_e4m3fn),
        "w2": torch.randn(128, 3072, generator=generator).to(torch.float8_e4m3fn),
    }
    scales = {
        "w1": torch.ones(24, 1).to(torch.float8_e8m0fnu),
        "w3": torch.ones(24, 1).to(torch.float8_e8m0fnu),
        "w2": torch.ones(1, 24).to(torch.float8_e8m0fnu),
    }
    converted = {
        shard_id: _fp8_block_weight_to_mxfp4(
            weight, scales[shard_id], torch.device("cpu")
        )
        for shard_id, weight in weights.items()
    }

    for shard_id in ("w1", "w3", "w2"):
        weight_param = experts.w2_weight if shard_id == "w2" else experts.w13_weight
        scale_param = (
            experts.w2_weight_scale if shard_id == "w2" else experts.w13_weight_scale
        )
        for param, loaded in (
            (scale_param, scales[shard_id]),
            (weight_param, weights[shard_id]),
        ):
            assert experts.weight_loader(
                param,
                loaded,
                f"experts.384.{shard_id}.weight",
                shard_id=shard_id,
                expert_id=384,
                return_success=True,
            )

    w1, w1_scale = converted["w1"]
    w3, w3_scale = converted["w3"]
    w2, w2_scale = converted["w2"]
    assert torch.equal(experts.w13_weight[2, :384], w1[2688:3072])
    assert torch.count_nonzero(experts.w13_weight[2, 384:512]) == 0
    assert torch.equal(experts.w13_weight[2, 512:896], w3[2688:3072])
    assert torch.count_nonzero(experts.w13_weight[2, 896:]) == 0
    assert torch.equal(experts.w13_weight_scale[2, :384], w1_scale[2688:3072])
    assert torch.count_nonzero(experts.w13_weight_scale[2, 384:512]) == 0
    assert torch.equal(experts.w13_weight_scale[2, 512:896], w3_scale[2688:3072])
    assert torch.count_nonzero(experts.w13_weight_scale[2, 896:]) == 0
    assert torch.equal(experts.w2_weight[2, :, :192], w2[:, 1344:1536])
    assert torch.count_nonzero(experts.w2_weight[2, :, 192:]) == 0
    assert torch.equal(experts.w2_weight_scale[2, :, :12], w2_scale[:, 84:96])
    assert torch.count_nonzero(experts.w2_weight_scale[2, :, 12:]) == 0
    assert torch.count_nonzero(experts.w13_weight[:2]) == 0
    assert torch.count_nonzero(experts.w2_weight[:2]) == 0
    experts.validate_fused_shared_expert_weights()


def test_deepseek_v4_shared_expert_loader_validates_pending_pairs():
    experts = _make_routed_experts()
    experts._pending_shared_expert_weights[(2, "w1")] = {
        "weight": _boundary_fp8_weight()
    }
    experts._loaded_shared_expert_shards = {"w2", "w3"}

    with pytest.raises(ValueError, match="pending=.*scale"):
        experts.validate_fused_shared_expert_weights()

    experts._pending_shared_expert_weights.clear()
    experts._loaded_shared_expert_shards.add("w1")
    experts.validate_fused_shared_expert_weights()
