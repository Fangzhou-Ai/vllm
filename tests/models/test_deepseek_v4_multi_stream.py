# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from unittest.mock import patch

import pytest
import torch

from vllm.models.deepseek_v4.amd.multi_stream import (
    create_dsv4_rocm_aux_stream_list,
    should_overlap_dsv4_rocm_indexer,
)


@dataclass
class _FakeSWAMetadata:
    num_decodes: int


def test_create_dsv4_rocm_aux_stream_list_disabled():
    with patch("vllm.models.deepseek_v4.amd.multi_stream.envs") as envs:
        envs.VLLM_DSV4_ROCM_MULTI_STREAM = False
        assert create_dsv4_rocm_aux_stream_list() is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm device required")
def test_create_dsv4_rocm_aux_stream_list_enabled():
    with patch("vllm.models.deepseek_v4.amd.multi_stream.envs") as envs:
        envs.VLLM_DSV4_ROCM_MULTI_STREAM = True
        streams = create_dsv4_rocm_aux_stream_list()
    assert streams is not None
    assert len(streams) == 1


def test_should_overlap_dsv4_rocm_indexer_decode_only():
    attn_metadata = {"swa.prefix": _FakeSWAMetadata(num_decodes=1)}
    aux_streams = [object()]
    with patch("vllm.models.deepseek_v4.amd.multi_stream.envs") as envs:
        envs.VLLM_DSV4_ROCM_MULTI_STREAM_DECODE_ONLY = True
        assert should_overlap_dsv4_rocm_indexer(
            aux_streams, attn_metadata, "swa.prefix"
        )
        assert not should_overlap_dsv4_rocm_indexer(
            aux_streams, attn_metadata, "missing.prefix"
        )
        assert not should_overlap_dsv4_rocm_indexer(
            aux_streams,
            {"swa.prefix": _FakeSWAMetadata(num_decodes=0)},
            "swa.prefix",
        )


def test_should_overlap_dsv4_rocm_indexer_no_decode_batch_floor():
    aux_streams = [object()]
    with patch("vllm.models.deepseek_v4.amd.multi_stream.envs") as envs:
        envs.VLLM_DSV4_ROCM_MULTI_STREAM_DECODE_ONLY = True
        assert should_overlap_dsv4_rocm_indexer(
            aux_streams,
            {"swa.prefix": _FakeSWAMetadata(num_decodes=1)},
            "swa.prefix",
        )


def test_should_overlap_dsv4_rocm_indexer_all_steps_when_decode_only_off():
    aux_streams = [object()]
    with patch("vllm.models.deepseek_v4.amd.multi_stream.envs") as envs:
        envs.VLLM_DSV4_ROCM_MULTI_STREAM_DECODE_ONLY = False
        assert should_overlap_dsv4_rocm_indexer(
            aux_streams,
            {"swa.prefix": _FakeSWAMetadata(num_decodes=0)},
            "swa.prefix",
        )


def test_should_overlap_dsv4_rocm_indexer_master_switch():
    attn_metadata = {"swa.prefix": _FakeSWAMetadata(num_decodes=1)}
    assert not should_overlap_dsv4_rocm_indexer(None, attn_metadata, "swa.prefix")
