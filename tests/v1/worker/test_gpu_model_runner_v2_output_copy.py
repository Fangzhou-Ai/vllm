# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.gpu import async_utils
from vllm.v1.worker.gpu import model_runner as mrv2
from vllm.v1.worker.gpu.sample.output import SamplerOutput

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("use_ready_event", [False, True])
def test_async_output_waits_on_selected_dependency(monkeypatch, use_ready_event):
    events = []
    copy_event = Mock()
    copy_event.record.side_effect = lambda _stream: events.append("copy_done")
    monkeypatch.setattr(async_utils.torch.cuda, "Event", lambda **_: copy_event)
    monkeypatch.setattr(async_utils, "stream", lambda *_: nullcontext())

    def copy_to_np(tensor):
        events.append("copy")
        return tensor.numpy()

    monkeypatch.setattr(
        async_utils,
        "async_copy_to_np",
        copy_to_np,
    )

    main_stream = Mock()
    copy_stream = Mock()
    copy_stream.wait_event.side_effect = lambda _event: events.append("wait_event")
    copy_stream.wait_stream.side_effect = lambda _stream: events.append("wait_stream")
    ready_event = Mock() if use_ready_event else None
    num_sampled = torch.tensor([1], dtype=torch.int32)
    sampler_output = SamplerOutput(
        sampled_token_ids=torch.tensor([[7]], dtype=torch.int64),
        logprobs_tensors=None,
        num_nans=None,
        num_sampled=num_sampled,
        num_rejected=torch.tensor([0], dtype=torch.int32),
    )

    output = async_utils.AsyncOutput(
        model_runner_output=ModelRunnerOutput(req_ids=["req"], req_id_to_index={}),
        sampler_output=sampler_output,
        num_sampled_tokens=num_sampled,
        main_stream=main_stream,
        copy_stream=copy_stream,
        ready_event=ready_event,
    )

    assert output.ready_event is ready_event
    if ready_event is None:
        copy_stream.wait_stream.assert_called_once_with(main_stream)
        copy_stream.wait_event.assert_not_called()
    else:
        copy_stream.wait_event.assert_called_once_with(ready_event)
        copy_stream.wait_stream.assert_not_called()
    copy_event.record.assert_called_once_with(copy_stream)
    wait_name = "wait_event" if use_ready_event else "wait_stream"
    assert events.index(wait_name) < events.index("copy")
    assert events.index("copy_done") > max(
        i for i, event in enumerate(events) if event == "copy"
    )


@pytest.mark.parametrize(
    ("hip", "is_dspark", "check_ep_fault", "expected_order"),
    [
        ("test", True, False, ["record", "postprocess", "propose", "copy"]),
        (None, True, False, ["copy", "postprocess", "propose"]),
        ("test", False, False, ["copy", "postprocess", "propose"]),
        ("test", True, True, ["copy", "postprocess", "propose"]),
    ],
)
def test_output_copy_submission_order(
    monkeypatch,
    hip,
    is_dspark,
    check_ep_fault,
    expected_order,
):
    events = []
    runner = mrv2.GPUModelRunner.__new__(mrv2.GPUModelRunner)
    input_batch = SimpleNamespace(
        req_ids=["req"],
        idx_mapping=torch.tensor([0], dtype=torch.int64),
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        num_draft_tokens=0,
    )
    hidden_states = torch.empty(1, 4)
    runner.execute_model_state = SimpleNamespace(
        input_batch=input_batch,
        attn_metadata=None,
        slot_mappings_by_layer=None,
        hidden_states=hidden_states,
        aux_hidden_states=None,
        finished_req_ids=set(),
        ec_connector_output=None,
        routed_experts=None,
        dp_sync=None,
    )
    runner.is_last_pp_rank = True
    runner.pcp_manager = None
    monkeypatch.setattr(
        mrv2.pcp,
        "maybe_restore_pcp_for_sampling",
        lambda _manager, hidden, batch: (hidden, batch),
    )

    num_sampled = torch.tensor([1], dtype=torch.int32)
    num_rejected = torch.tensor([0], dtype=torch.int32)
    sampler_output = SamplerOutput(
        sampled_token_ids=torch.tensor([[11]], dtype=torch.int64),
        logprobs_tensors=None,
        num_nans=None,
        num_sampled=num_sampled,
        num_rejected=num_rejected,
    )
    runner.sample = lambda *_: (sampler_output, num_sampled, num_rejected)
    runner.pp_handler = None
    runner.prompt_logprobs_worker = SimpleNamespace(
        compute_prompt_logprobs=lambda *_: {}
    )
    runner.model = SimpleNamespace(compute_logits=lambda *_: None)
    runner.req_states = SimpleNamespace(
        all_token_ids=SimpleNamespace(gpu=torch.empty(1, 1, dtype=torch.int64)),
        num_computed_tokens=SimpleNamespace(gpu=torch.zeros(1, dtype=torch.int32)),
        prompt_len=SimpleNamespace(np=None),
        last_sampled_tokens=torch.empty(1, dtype=torch.int64),
        next_prefill_tokens=torch.empty(1, dtype=torch.int64),
        draft_tokens=torch.empty(1, 1, dtype=torch.int64),
    )
    runner.speculative_config = SimpleNamespace(use_dspark=lambda: is_dspark)
    runner.check_ep_fault = check_ep_fault
    runner.main_stream = Mock()
    runner.output_copy_stream = Mock()

    class FakeEvent:
        def record(self, stream):
            assert stream is runner.main_stream
            events.append("record")

    monkeypatch.setattr(mrv2.torch.version, "hip", hip)
    monkeypatch.setattr(mrv2.torch.cuda, "Event", FakeEvent)

    ready_events = []

    def make_async_output(**kwargs):
        events.append("copy")
        ready_events.append(kwargs.get("ready_event"))
        return SimpleNamespace()

    monkeypatch.setattr(mrv2, "AsyncOutput", make_async_output)
    runner.postprocess_sampled = lambda *_: events.append("postprocess")

    def propose(*_args, **_kwargs):
        events.append("propose")
        return torch.tensor([[13]], dtype=torch.int64)

    runner.speculator = SimpleNamespace(
        supports_mm_inputs=False,
        propose=propose,
    )
    runner.sampler = SimpleNamespace(
        sampling_states=SimpleNamespace(
            temperature=SimpleNamespace(gpu=torch.empty(1)),
            seeds=SimpleNamespace(gpu=torch.empty(1, dtype=torch.int64)),
        )
    )
    runner._draft_workspace_lane = None
    monkeypatch.setattr(mrv2, "use_workspace_lane", lambda _lane: nullcontext())
    runner.adaptive_verification = None
    runner.num_speculative_steps = 1
    runner.draft_tokens_handler = SimpleNamespace(set_draft_tokens=lambda *_: None)
    runner.kv_connector = SimpleNamespace(post_forward=lambda _finished: None)
    runner.eplb = SimpleNamespace(step=lambda **_: None)

    output = mrv2.GPUModelRunner.sample_tokens(runner, None)

    assert output is not None
    assert events == expected_order
    assert len(ready_events) == 1
    expect_deferred = hip is not None and is_dspark and not check_ep_fault
    assert (ready_events[0] is not None) is expect_deferred
