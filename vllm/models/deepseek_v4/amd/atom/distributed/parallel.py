# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Tensor-parallel group shim for the ported ATOM attention inside vLLM.

ATOM's linear/layernorm/attention query aiter's ``aiter.dist.parallel_state``
for the TP world size, rank, and collectives (all_gather / all_reduce). Inside
vLLM the tensor-parallel group is owned by ``vllm.distributed``, and aiter's own
group is not initialized. This shim delegates every TP query/collective to
vLLM's TP group when it is initialized, and degrades to a single-rank no-op
otherwise (module construction / offline tests). It adapts the small signature
differences ATOM expects (``all_reduce(..., ca_fp8_quant=...)``,
``all_gather(..., dim=...)``).
"""

from typing import Optional

import torch


def _vllm_tp():
    """Return vLLM's TP GroupCoordinator, or None if TP is not initialized."""
    try:
        from vllm.distributed.parallel_state import get_tp_group as _get

        return _get()
    except Exception:
        return None


class _TPGroupAdapter:
    """Wrap vLLM's TP GroupCoordinator with the surface ATOM code expects."""

    def __init__(self, grp):
        self._g = grp

    @property
    def world_size(self) -> int:
        return 1 if self._g is None else int(self._g.world_size)

    @property
    def rank_in_group(self) -> int:
        return 0 if self._g is None else int(self._g.rank_in_group)

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if self._g is None or self._g.world_size == 1:
            return input_
        return self._g.all_gather(input_, dim=dim)

    def all_reduce(self, input_: torch.Tensor, ca_fp8_quant: bool = False, **_):
        # ATOM passes ca_fp8_quant (custom-allreduce fp8 path); vLLM's group
        # does a plain all_reduce. Drop the extra knob (single-node / bf16).
        if self._g is None or self._g.world_size == 1:
            return input_
        return self._g.all_reduce(input_)


def get_tp_group() -> _TPGroupAdapter:
    return _TPGroupAdapter(_vllm_tp())


def get_tensor_model_parallel_world_size() -> int:
    g = _vllm_tp()
    return 1 if g is None else int(g.world_size)


def get_tensor_model_parallel_rank() -> int:
    g = _vllm_tp()
    return 0 if g is None else int(g.rank_in_group)


def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    g = _vllm_tp()
    if g is None or g.world_size == 1:
        return input_
    return g.all_reduce(input_)


def tensor_model_parallel_all_gather(
    input_: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    g = _vllm_tp()
    if g is None or g.world_size == 1:
        return input_
    return g.all_gather(input_, dim=dim)


# ---------------------------------------------------------------------------
# Fused all-reduce + RMSNorm ops. These are aiter compute kernels used only on
# ATOM's TP>1 residual-RMSNorm path (not on the attention q_norm/kv_norm path
# the ported attention exercises). Re-export aiter's originals so the vendored
# ``layernorm`` imports resolve; fall back to no-op stubs if aiter lacks them.
# ---------------------------------------------------------------------------
try:  # pragma: no cover - depends on the installed aiter build
    from aiter.dist.communication_op import (
        tensor_model_parallel_fused_allreduce_rmsnorm,
        tensor_model_parallel_fused_allreduce_rmsnorm_quant,
    )
except Exception:

    def tensor_model_parallel_fused_allreduce_rmsnorm(*args, **kwargs):
        raise RuntimeError(
            "tensor_model_parallel_fused_allreduce_rmsnorm is unavailable in "
            "this aiter build; the ported ATOM attention path does not use it."
        )

    def tensor_model_parallel_fused_allreduce_rmsnorm_quant(*args, **kwargs):
        raise RuntimeError(
            "tensor_model_parallel_fused_allreduce_rmsnorm_quant is unavailable "
            "in this aiter build; the ported ATOM attention path does not use it."
        )


# ---------------------------------------------------------------------------
# Data-parallel / expert-parallel group shims for the ported ATOM MoE.
# The vendored FusedMoE runs single-node (DP=1, non-EP): every DP/EP collective
# is guarded by ``world_size > 1`` / ``use_ep`` and is therefore never reached.
# Only ``get_dp_group().world_size`` (== 1) and ``.rank_in_group`` (== 0) are
# consulted at construction time; the collective methods raise if ever hit so a
# silent-wrong-result can't slip through.
# ---------------------------------------------------------------------------
class _SingleRankGroupAdapter:
    """DP/EP group stand-in for a single-node (world_size==1) deployment."""

    @property
    def world_size(self) -> int:
        return 1

    @property
    def rank_in_group(self) -> int:
        return 0

    @property
    def rank(self) -> int:
        return 0

    @property
    def device_communicator(self):
        raise RuntimeError(
            "ATOM MoE DP/EP device_communicator requested on a single-node "
            "(DP=1) port; this path should be unreachable."
        )

    def _unreachable(self, *args, **kwargs):
        raise RuntimeError(
            "ATOM MoE DP/EP collective invoked on a single-node (DP=1) port; "
            "this path should be unreachable."
        )

    all_gather = broadcast = all_reduce = _unreachable


_DP_GROUP = _SingleRankGroupAdapter()
_EP_GROUP = _SingleRankGroupAdapter()


def get_dp_group() -> _SingleRankGroupAdapter:
    return _DP_GROUP


def get_ep_group() -> _SingleRankGroupAdapter:
    return _EP_GROUP
