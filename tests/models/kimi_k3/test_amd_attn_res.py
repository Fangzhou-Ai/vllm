# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F

from vllm.models.kimi_k3.amd import linear as kimi_linear
from vllm.models.kimi_k3.amd.ops.attn_res import attn_res
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="AMD AttnRes requires ROCm",
)


def _randn_with_row_padding(*shape: int, padding: int = 0) -> torch.Tensor:
    storage = torch.randn(
        *shape[:-1],
        shape[-1] + padding,
        device="cuda",
        dtype=torch.bfloat16,
    )
    return storage[..., : shape[-1]]


def _reference(
    prefix: torch.Tensor,
    blocks: torch.Tensor,
    norm_weight: torch.Tensor,
    qk_weight: torch.Tensor,
    num_blocks: int,
    eps: float,
) -> torch.Tensor:
    hidden_size = prefix.shape[-1]
    values = torch.cat((blocks[:, :num_blocks], prefix.unsqueeze(1)), dim=1)
    keys = F.rms_norm(values, (hidden_size,), norm_weight, eps)
    probs = (keys @ qk_weight).softmax(dim=-1)
    return torch.matmul(probs.unsqueeze(1), values).squeeze(1)


@pytest.mark.parametrize(
    (
        "num_tokens",
        "num_blocks",
        "block_capacity",
        "hidden_size",
        "row_padding",
    ),
    [
        pytest.param(0, 3, 5, 128, 0, id="empty"),
        pytest.param(1, 1, 2, 128, 0, id="decode-single"),
        pytest.param(17, 4, 6, 1024, 7, id="decode-padded"),
        pytest.param(320, 8, 10, 7168, 0, id="prefill-full"),
    ],
)
def test_amd_attn_res_matches_reference(
    num_tokens: int,
    num_blocks: int,
    block_capacity: int,
    hidden_size: int,
    row_padding: int,
) -> None:
    eps = 1e-5
    prefix = _randn_with_row_padding(num_tokens, hidden_size, padding=row_padding)
    blocks = _randn_with_row_padding(
        num_tokens,
        block_capacity,
        hidden_size,
        padding=row_padding,
    )
    norm_weight = 1 + 0.1 * torch.randn(
        hidden_size, device="cuda", dtype=torch.bfloat16
    )
    qk_weight = (
        torch.randn(hidden_size, device="cuda", dtype=torch.bfloat16) / hidden_size**0.5
    )
    expected = _reference(
        prefix,
        blocks,
        norm_weight,
        qk_weight,
        num_blocks,
        eps,
    )
    original_prefix = prefix.clone()
    original_blocks = blocks.clone()

    actual = attn_res(
        prefix,
        None,
        blocks,
        norm_weight,
        qk_weight,
        None,
        num_blocks,
        -1,
        eps,
        0.0,
    )

    torch.testing.assert_close(actual, expected, atol=8e-2, rtol=3e-2)
    torch.testing.assert_close(prefix, original_prefix, atol=0, rtol=0)
    torch.testing.assert_close(blocks, original_blocks, atol=0, rtol=0)
    assert actual.shape == prefix.shape
    assert actual.is_contiguous()


@pytest.mark.parametrize(
    (
        "num_tokens",
        "num_blocks",
        "hidden_size",
        "has_delta",
        "write_block",
        "apply_output_norm",
    ),
    [
        pytest.param(1, 0, 128, False, True, True, id="empty-write-norm"),
        pytest.param(7, 1, 1024, True, False, True, id="single-add-norm"),
        pytest.param(17, 5, 7168, True, True, True, id="padded-write-add"),
        pytest.param(3, 8, 7168, True, False, True, id="full-add-norm"),
        pytest.param(320, 4, 7168, True, False, False, id="prefill-add"),
    ],
)
def test_amd_attn_res_fused_contract(
    num_tokens: int,
    num_blocks: int,
    hidden_size: int,
    has_delta: bool,
    write_block: bool,
    apply_output_norm: bool,
) -> None:
    torch.manual_seed(42)
    eps = 1e-5
    output_eps = 2e-5
    block_capacity = 9
    prefix = _randn_with_row_padding(num_tokens, hidden_size, padding=7)
    delta = (
        _randn_with_row_padding(num_tokens, hidden_size, padding=11)
        if has_delta
        else None
    )
    blocks = _randn_with_row_padding(
        num_tokens, block_capacity, hidden_size, padding=13
    )
    norm_weight = 1 + 0.1 * torch.randn(
        hidden_size, device="cuda", dtype=torch.bfloat16
    )
    qk_weight = (
        torch.randn(hidden_size, device="cuda", dtype=torch.bfloat16) / hidden_size**0.5
    )
    output_norm_weight = (
        1 + 0.1 * torch.randn(hidden_size, device="cuda", dtype=torch.bfloat16)
        if apply_output_norm
        else None
    )
    expected_prefix = prefix.clone()
    if delta is not None:
        expected_prefix = expected_prefix + delta
    values = torch.cat(
        (blocks[:, :num_blocks].clone(), expected_prefix.unsqueeze(1)), dim=1
    )
    keys = F.rms_norm(values.float(), (hidden_size,), norm_weight.float(), eps)
    probs = (keys @ qk_weight.float()).softmax(dim=-1)
    expected = torch.matmul(probs.unsqueeze(1), values.float()).squeeze(1)
    if output_norm_weight is not None:
        expected = F.rms_norm(
            expected, (hidden_size,), output_norm_weight.float(), output_eps
        )
    expected = expected.to(prefix.dtype)
    original_blocks = blocks.clone()
    block_write_idx = num_blocks if write_block else -1

    actual = attn_res(
        prefix,
        delta,
        blocks,
        norm_weight,
        qk_weight,
        output_norm_weight,
        num_blocks,
        block_write_idx,
        eps,
        output_eps,
    )

    torch.testing.assert_close(actual, expected, atol=8e-2, rtol=3e-2)
    torch.testing.assert_close(prefix, expected_prefix, atol=0, rtol=0)
    if write_block:
        original_blocks[:, block_write_idx].copy_(expected_prefix)
    torch.testing.assert_close(blocks, original_blocks, atol=0, rtol=0)
    assert actual.is_contiguous()


@pytest.mark.parametrize("is_block_write_layer", [False, True])
def test_decoder_defers_mlp_delta_to_next_attn_res(
    monkeypatch: pytest.MonkeyPatch,
    is_block_write_layer: bool,
) -> None:
    attention_delta = torch.tensor([[5.0, 6.0]])
    mlp_delta = torch.tensor([[7.0, 8.0]])
    layer = SimpleNamespace(
        is_block_write_layer=is_block_write_layer,
        block_write_idx=0,
        prev_valid_blocks=1,
        self_attention_res_proj=None,
        self_attention_res_norm=None,
        input_layernorm=None,
        mlp_res_proj=None,
        mlp_res_norm=None,
        post_attention_layernorm=None,
        _run_self_attn=Mock(return_value=attention_delta),
        mlp=Mock(return_value=mlp_delta),
    )
    apply_attn_res = Mock(side_effect=lambda prefix_sum, *args, **kwargs: prefix_sum)
    monkeypatch.setattr(kimi_linear, "_apply_attn_res", apply_attn_res)

    prefix_sum = torch.tensor([[1.0, 2.0]])
    incoming_mlp_delta = torch.tensor([[3.0, 4.0]])
    block_residual = torch.zeros(1, 2, 2)
    hidden_states, updated_prefix, returned_blocks = (
        kimi_linear.KimiDecoderLayer.forward_attn_residual(
            layer,
            positions=torch.tensor([0]),
            hidden_states=incoming_mlp_delta,
            prefix_sum=prefix_sum,
            block_residual=block_residual,
        )
    )

    first_call, second_call = apply_attn_res.call_args_list
    assert first_call.kwargs["delta"] is incoming_mlp_delta
    assert first_call.kwargs["block_write_idx"] == (0 if is_block_write_layer else -1)
    assert second_call.kwargs["delta"] is (
        None if is_block_write_layer else attention_delta
    )
    assert updated_prefix is (attention_delta if is_block_write_layer else prefix_sum)
    assert hidden_states is mlp_delta
    assert returned_blocks is block_residual


@pytest.mark.parametrize("is_last_rank", [False, True])
def test_model_materializes_split_attn_res_state_at_consumers(
    monkeypatch: pytest.MonkeyPatch,
    is_last_rank: bool,
) -> None:
    first_mlp_delta = torch.tensor([[3.0, 4.0]])
    second_mlp_delta = torch.tensor([[5.0, 6.0]])

    def split_layer(expected_delta, output_delta, increment):
        def forward(*, hidden_states, prefix_sum, residual, **kwargs):
            assert hidden_states is expected_delta
            prefix_sum.add_(increment)
            return output_delta, prefix_sum, residual

        return Mock(side_effect=forward)

    model = SimpleNamespace(
        config=SimpleNamespace(attn_res_block_size=12),
        start_layer=0,
        end_layer=2,
        aux_hidden_state_layers=(0, 1),
        output_attn_res_proj=None,
        output_attn_res_norm=None,
        layers=[
            split_layer(None, first_mlp_delta, 10),
            split_layer(first_mlp_delta, second_mlp_delta, 20),
        ],
    )
    monkeypatch.setattr(
        kimi_linear,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=is_last_rank),
    )
    final_attn_res = Mock(
        side_effect=lambda prefix_sum, *args, delta=None, **kwargs: prefix_sum + delta
    )
    monkeypatch.setattr(kimi_linear, "_apply_attn_res", final_attn_res)

    output = kimi_linear.KimiLinearModel.forward(
        model,
        input_ids=None,
        positions=torch.tensor([0]),
        intermediate_tensors=None,
        inputs_embeds=torch.tensor([[1.0, 2.0]]),
    )

    expected_final = torch.tensor([[36.0, 38.0]])
    if not is_last_rank:
        assert isinstance(output, IntermediateTensors)
        torch.testing.assert_close(output["hidden_states"], expected_final)
        final_attn_res.assert_not_called()
        return

    hidden_states, aux_hidden_states = output
    torch.testing.assert_close(hidden_states, expected_final)
    assert final_attn_res.call_args.kwargs["delta"] is second_mlp_delta
    torch.testing.assert_close(aux_hidden_states[0], torch.tensor([[1.0, 2.0]]))
    torch.testing.assert_close(aux_hidden_states[1], torch.tensor([[14.0, 16.0]]))
