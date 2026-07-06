# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""ATOM DeepSeek-V4 MoE (Expert + MoE) vendored verbatim from
ATOM ``atom/models/deepseek_v4.py`` (classes Expert and MoE), with imports
repointed to the vendored ``vllm.models.deepseek_v4.amd.atom`` tree.

Single-node port (TP=8, DP=1, non-EP). Dual-stream shared-expert overlap is
out of scope this session: the ``MoE`` is always built with ``alt_stream=None``
(see ``AtomV4MoE`` in ``amd/atom_integration.py``), so ``_use_dual_stream`` is
False and ``forward`` always takes ``single_stream_moe_forward``. The
``dual_stream_moe_forward`` / ``_gather_ids_for_dp`` methods are kept verbatim
but never invoked at this configuration.
"""

import os
from types import SimpleNamespace
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from aiter import dtypes
from aiter import silu_and_mul as aiter_silu_and_mul

try:  # never called at _V4_USE_TRITON_FUSION=False, import kept for fidelity
    from aiter.ops.triton.fusions.fused_clamp_act_mul import fused_clamp_act_mul
except Exception:  # pragma: no cover
    fused_clamp_act_mul = None

from vllm.models.deepseek_v4.amd.atom.config import get_current_atom_config
from vllm.models.deepseek_v4.amd.atom.distributed.parallel import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.models.deepseek_v4.amd.atom.model_ops.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.models.deepseek_v4.amd.atom.model_ops.moe import FusedMoE
from vllm.models.deepseek_v4.amd.atom.model_ops.topK import (
    is_rocm_aiter_fusion_shared_expert_enabled_for_quant_config,
)
from vllm.models.deepseek_v4.amd.atom.model_ops.triton_hash_topk import hash_topk_triton
from vllm.models.deepseek_v4.amd.atom.model_ops.utils import atom_parameter
from vllm.models.deepseek_v4.amd.atom.models.v4_config import DeepseekV4Args
from vllm.models.deepseek_v4.amd.atom.utils import envs
from vllm.models.deepseek_v4.amd.atom.utils.forward_context import get_forward_context

# Production ATOM V4 keeps the fused-clamp-act-mul FP8 path OFF (matches
# attention_core's _V4_USE_TRITON_FUSION=False); Expert.use_fused_clamp_act_mul
# reads this.
_V4_USE_TRITON_FUSION = False


class _MoEDebug:
    """Env-gated per-stage MoE diagnostics (VLLM_DSV4_MOE_DEBUG=1, eager only)."""

    _counts: dict = {}

    @staticmethod
    def _stat(t):
        if t is None:
            return "None"
        tf = t.float()
        return (
            f"shape={tuple(t.shape)} dtype={t.dtype} "
            f"norm={tf.norm().item():.4f} absmax={tf.abs().max().item():.4f} "
            f"nan={int(torch.isnan(tf).any().item())} inf={int(torch.isinf(tf).any().item())}"
        )

    @classmethod
    def maybe(cls, moe, x):
        n = cls._counts.get(moe.layer_id, 0)
        if n >= 2:
            return
        rl = moe.gate(x)
        vals, ids = torch.topk(rl.float(), moe.n_activated_experts, dim=-1)
        print(
            f"[MOE-DBG L{moe.layer_id} call{n}] x: {cls._stat(x)}",
            flush=True,
        )
        print(
            f"[MOE-DBG L{moe.layer_id} call{n}] router_logits: {cls._stat(rl)} "
            f"| topk_ids[0]={ids[0].tolist()} topk_vals[0]="
            f"{[round(v, 3) for v in vals[0].tolist()]} "
            f"| gate.wt {cls._stat(moe.gate.weight)} "
            f"| bias {cls._stat(getattr(moe.gate, 'e_score_correction_bias', None))} "
            f"| tid2eid {'yes' if getattr(moe.gate, 'tid2eid', None) is not None else 'no'}",
            flush=True,
        )

    @classmethod
    def report(cls, moe, x, shared, routed, out):
        n = cls._counts.get(moe.layer_id, 0)
        if n >= 2:
            return
        cls._counts[moe.layer_id] = n + 1
        print(
            f"[MOE-DBG L{moe.layer_id} call{n}] shared: {cls._stat(shared)} "
            f"|| routed: {cls._stat(routed)} || out: {cls._stat(out)} "
            f"|| tp_size={moe.tp_size} n_shared_fused="
            f"{getattr(moe.experts, 'num_fused_shared_experts', '?')}",
            flush=True,
        )


class Expert(nn.Module):
    """Single MoE expert: SwiGLU FFN (w1, w2, w3). Computation in float32 for stability.

    Port of inference/model.py:587-606. With `swiglu_limit > 0`, clamps both gate
    and up projections (gate clipped above only, up clipped both sides) before
    the SiLU * up product — matches reference behavior exactly.
    """

    def __init__(
        self,
        dim: int,
        inter_dim: int,
        swiglu_limit: float = 0.0,
        quant_config: Optional[Any] = None,
        reduce_results: bool = True,
        prefix: str = "",
    ):
        super().__init__()
        # Fused [w1; w3] (gate_up_proj): both share input x, both ColumnParallel
        # — standard llama/dsv2 fusion. Disk still split; routed via
        # packed_modules_mapping in DeepseekV4ForCausalLM.
        self.gate_up_proj = MergedColumnParallelLinear(
            dim,
            [inter_dim, inter_dim],
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.w2 = RowParallelLinear(
            inter_dim,
            dim,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.w2",
        )
        self.swiglu_limit = swiglu_limit
        # Switch: route clamp + silu(gate)*up [+ weights] + per-token FP8 1x128
        # quant through a single aiter triton kernel. The fused kernel emits
        # FP8 + scale; w2 accepts `x_scale` and skips its own quant step.
        self.use_fused_clamp_act_mul = _V4_USE_TRITON_FUSION

    def forward(
        self,
        x: torch.Tensor,  # [num_tokens, dim]
        weights: Optional[torch.Tensor] = None,  # [num_tokens, 1]  optional gate
    ) -> torch.Tensor:  # [num_tokens, dim]

        dtype = x.dtype
        # Single fused GEMM. Layout is [gate | up] concat on last dim — matches
        # aiter silu_and_mul's split([d, d], dim=-1) contract. The kernel does
        # silu/clamp/mul in fp32 internally regardless of input dtype, so we
        # feed the bf16 GEMM output directly.
        combined = self.gate_up_proj(x)  # [num_tokens, 2*inter_dim_per_tp]
        if self.use_fused_clamp_act_mul:
            x_fp8, x_scale = fused_clamp_act_mul(
                combined,
                swiglu_limit=self.swiglu_limit,
                activation="silu",
                weights=weights,
                dtype_quant=dtypes.fp8,
                transpose_scale=True,
            )
            return self.w2(x_fp8, x_scale=x_scale)
        out = torch.empty(
            (combined.shape[0], combined.shape[-1] // 2),
            dtype=dtype,
            device=combined.device,
        )
        # limit > 0 enables in-kernel clamp (gate≤limit, up∈[-limit,limit]) via
        # ROCm v_med3_f32 — same semantics as the prior torch.clamp pair.
        aiter_silu_and_mul(out, combined, self.swiglu_limit)
        if weights is not None:
            out = weights.to(dtype) * out
        return self.w2(out)  # [num_tokens, dim]


class MoE(nn.Module):
    """Mixture-of-Experts: top-k routed experts (FusedMoE) + 1 shared expert.

    PR3b: replaces the per-expert nn.Linear list with `FusedMoE` so 384 routed
    experts shard across TP/EP ranks and load FP4 weights via the existing
    `gemm_a4w4_quant` aiter kernel.

    Routing math (`sqrtsoftplus(scores) + bias` topk) is delegated to
    `FusedMoE.select_experts(scoring_func="sqrtsoftplus", e_score_correction_bias=...)`,
    which we extended in atom/model_ops/moe.py to add the V4 path.

    Hash routing for `layer_id < n_hash_layers` (first 3 V4 layers) is NOT yet
    wired through FusedMoE — the `tid2eid` buffer is declared so weight loading
    completes, but inference uses the standard sqrtsoftplus path. Hash layers
    will produce incorrect routing; correct hash routing lands in PR3+.
    """

    def __init__(
        self,
        layer_id: int,
        args: DeepseekV4Args,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.prefix = prefix
        self.dim = args.dim
        self.n_routed_experts = args.n_routed_experts
        self.n_activated_experts = args.n_activated_experts
        self.is_hash_layer = layer_id < args.n_hash_layers
        self.routed_scaling_factor = args.route_scale
        self.swiglu_limit = args.swiglu_limit
        self.tp_size = get_tensor_model_parallel_world_size()
        self.alt_stream = alt_stream
        qc = args.quant_config

        self.gate = ReplicatedLinear(
            self.dim,
            self.n_routed_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )
        # V4 hash-routed layers (layer_id < n_hash_layers) use tid2eid lookup,
        # not bias-corrected gate-logit routing — checkpoint has no
        # `gate.bias` for those layers. Only allocate the bias for
        # sqrtsoftplus layers to avoid 3 spurious unloaded-param warnings.
        if not self.is_hash_layer:
            self.gate.e_score_correction_bias = atom_parameter(
                torch.empty(self.n_routed_experts, dtype=torch.float32)
            )
        else:
            # tid2eid: per-token-id top-k expert lookup table (V4 first 3
            # layers use this in lieu of gate-logit routing).
            self.gate.tid2eid = atom_parameter(
                torch.empty(
                    args.vocab_size, args.n_activated_experts, dtype=torch.int32
                ),
            )
            # input_ids for hash routing is read from forward_context.context
            # (set by ModelRunner). torch.compile silently drops NNModule
            # attribute mutation across the compile boundary, so stashing on
            # `self.foo` from inside forward is a no-op at runtime.
        assert args.n_shared_experts == 1
        self._fuse_shared_into_routed = (
            is_rocm_aiter_fusion_shared_expert_enabled_for_quant_config(
                qc,
                shared_expert_prefix=f"{prefix}.shared_experts",
                routed_expert_prefix=f"{prefix}.experts",
            )
        )
        moe_cfg = SimpleNamespace(
            routed_scaling_factor=self.routed_scaling_factor,
            n_shared_experts=(
                args.n_shared_experts if self._fuse_shared_into_routed else 0
            ),
        )
        self.experts = FusedMoE(
            num_experts=self.n_routed_experts,
            top_k=self.n_activated_experts,
            hidden_size=self.dim,
            intermediate_size=args.moe_inter_dim,
            reduce_results=False,
            renormalize=True,
            quant_config=qc,
            use_grouped_topk=False,
            prefix=f"{prefix}.experts",
            scoring_func=args.score_func,  # "sqrtsoftplus"
            e_score_correction_bias=getattr(self.gate, "e_score_correction_bias", None),
            config=moe_cfg,
            shared_expert_prefix=f"{prefix}.shared_experts",
        )
        self.experts.swiglu_limit = args.swiglu_limit

        if not self._fuse_shared_into_routed:
            # self.experts.num_fused_shared_experts = 0
            self.shared_experts = Expert(
                args.dim,
                args.moe_inter_dim,
                swiglu_limit=args.swiglu_limit,
                quant_config=qc,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
            )
        else:
            self.shared_experts = None
        if self.is_hash_layer:
            # Inject hash routing into FusedMoE.select_experts via the
            # custom_routing_function hook (added in atom/model_ops/moe.py).
            self.experts.custom_routing_function = self._hash_topk

        # Dual-stream: run shared_experts on `alt_stream` in parallel with
        # routed experts on the current stream. Mirrors V2's pattern. Only
        # active when shared_experts exist (not fused into routed) AND the
        # env threshold is positive AND we got an alt_stream from the model.
        # Per-call token count gating happens inside the custom op dispatcher
        # — prefill (large batch) skips dual-stream (overhead > benefit).
        self._use_dual_stream = (
            self.shared_experts is not None
            and self.alt_stream is not None
            and envs.ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD > 0
        )
        if self._use_dual_stream:
            # Register self in static_forward_context so the custom op
            # dispatcher can look us up by `layer_name` (= self.prefix).
            get_current_atom_config().compilation_config.static_forward_context[
                prefix
            ] = self

    def _hash_topk(
        self,
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """V4 hash routing for first 3 layers.

        topk_ids = tid2eid[input_ids]  (no gate-based selection)
        topk_weights = sqrtsoftplus(router_logits) gathered at topk_ids
        Then renormalize so weights sum to 1 per token.
        """
        fwd_input_ids = get_forward_context().context.input_ids
        assert (
            fwd_input_ids is not None
        ), "forward_context.context.input_ids is None — caller must invoke DeepseekV4ForCausalLM.forward, not DeepseekV4Model.forward directly."
        ids = fwd_input_ids.flatten()
        num_tokens = gating_output.shape[0]
        assert (
            ids.shape[0] == num_tokens
        ), f"input_ids length {ids.shape[0]} does not match gating_output num_tokens {num_tokens}"
        tid2eid = self.gate.tid2eid

        # Fused-shared expert: the custom_routing_function path bypasses
        # select_experts' shared-expert append, so the shared expert (slot
        # n_routed_experts) would never be routed and its ~40% contribution
        # dropped. When shared is fused, write the routed result into the first
        # `topk` columns of the global topK buffer (shared cols pre-filled) and
        # return the full [N, topk + n_shared] view.
        num_fused_shared = getattr(self.experts, "num_fused_shared_experts", 0)
        if num_fused_shared > 0:
            import vllm.models.deepseek_v4.amd.atom.model_ops.topK as _topK_mod

            assert _topK_mod.aiter_topK_meta_data is not None, (
                "AITER topK meta data is not initialized. "
                "init_aiter_topK_meta_data must run before hash-layer routing."
            )
            total_topk_weights, total_topk_ids = _topK_mod.aiter_topK_meta_data
            assert total_topk_weights.shape[0] >= num_tokens
            hash_topk_triton(
                ids,
                gating_output,
                tid2eid,
                renormalize,
                self.routed_scaling_factor,
                total_topk_ids[:num_tokens, :topk],
                total_topk_weights[:num_tokens, :topk],
            )
            return total_topk_weights[:num_tokens], total_topk_ids[:num_tokens]

        topk_ids = torch.empty(
            (num_tokens, topk), dtype=torch.int32, device=gating_output.device
        )
        topk_weights = torch.empty(
            (num_tokens, topk), dtype=torch.float32, device=gating_output.device
        )
        hash_topk_triton(
            ids,
            gating_output,
            tid2eid,
            renormalize,
            self.routed_scaling_factor,
            topk_ids,
            topk_weights,
        )
        return topk_weights, topk_ids

    def routed_expert_forward(
        self, x: torch.Tensor  # [num_tokens, dim]
    ) -> torch.Tensor:  # [num_tokens, dim]
        """Gate + FusedMoE routed-expert pass.

        For hash layers the gate's `tid2eid` lookup needs `input_ids`;
        `DeepseekV4ForCausalLM.forward` stashes it on
        `forward_context.context.input_ids` before each forward, and
        `_hash_topk` (FusedMoE's custom_routing_function) reads it there.
        """
        router_logits = self.gate(x)  # [num_tokens, n_routed_experts]
        return self.experts(hidden_states=x, router_logits=router_logits)

    @staticmethod
    def _gather_ids_for_dp(ids: torch.Tensor, ctx) -> torch.Tensor:
        """All-gather input_ids across DP ranks to match gathered hidden_states."""
        from vllm.models.deepseek_v4.amd.atom.distributed.parallel import get_dp_group

        ids_2d = ids.unsqueeze(-1)
        dp_eager_mode = (
            not ctx.context.dp_uniform_decode
        ) and ctx.dp_metadata is not None
        if dp_eager_mode:
            from vllm.models.deepseek_v4.amd.atom.model_ops.moe import all_gatherv

            sizes = ctx.dp_metadata.get_sizes_across_dp()
            ids_2d = all_gatherv(ids_2d, sizes, get_dp_group())
        else:
            from vllm.models.deepseek_v4.amd.atom.model_ops.moe import all_gather_with_padding

            ids_2d, _ = all_gather_with_padding(ids_2d, use_cag=False)
        return ids_2d.flatten()

    def combine_outputs(
        self,
        routed: torch.Tensor,  # [num_tokens, dim]
        shared: Optional[torch.Tensor],  # [num_tokens, dim] or None
    ) -> torch.Tensor:  # [num_tokens, dim]
        """Add shared-expert contribution (when not fused into routed) and
        all-reduce across TP ranks.
        """
        if shared is not None:
            routed = routed + shared
        if self.tp_size > 1:
            routed = tensor_model_parallel_all_reduce(routed)
        return routed

    def single_stream_moe_forward(
        self, x: torch.Tensor  # [num_tokens, dim]
    ) -> torch.Tensor:  # [num_tokens, dim]
        """Sequential: shared_experts → routed_experts → combine."""
        import os

        _dbg = os.environ.get("VLLM_DSV4_MOE_DEBUG") == "1" and self.layer_id in (
            0,
            5,
            30,
        )
        if _dbg:
            _MoEDebug.maybe(self, x)
        shared = self.shared_experts(x) if self.shared_experts is not None else None
        routed = self.routed_expert_forward(x)
        out = self.combine_outputs(routed, shared)
        if _dbg:
            _MoEDebug.report(self, x, shared, routed, out)
        return out

    def dual_stream_moe_forward(
        self, x: torch.Tensor  # [num_tokens, dim]
    ) -> torch.Tensor:  # [num_tokens, dim]
        """Run shared_experts on `alt_stream` in parallel with routed_experts
        on the current stream. Mirrors V2's pattern. Both reads of `x` are
        independent; main stream waits on alt_stream's completion before
        combining.
        """
        current_stream = get_forward_context().main_stream
        self.alt_stream.wait_stream(current_stream)
        routed = self.routed_expert_forward(x)
        with torch.cuda.stream(self.alt_stream):
            shared = self.shared_experts.forward(x)
        current_stream.wait_stream(self.alt_stream)
        return self.combine_outputs(routed, shared)

    def forward(
        self,
        x: torch.Tensor,  # [num_tokens, dim]  hidden state (post ffn_norm)
    ) -> torch.Tensor:  # [num_tokens, dim]
        # Hash-layer routing reads `input_ids` from forward_context.context
        # inside `_hash_topk` (FusedMoE.custom_routing_function callback);
        # the MoE call itself doesn't need it as a parameter.
        assert (
            x.dim() == 2 and x.shape[-1] == self.dim
        ), f"MoE expects 2D [num_tokens, {self.dim}], got {tuple(x.shape)}"
        if self._use_dual_stream:
            # Shared custom op (also used by V2). Dispatcher reads
            # `_use_dual_stream` + per-call num_tokens vs threshold to pick
            # dual vs single. Custom op = Dynamo barrier so stream context
            # inside `dual_stream_moe_forward` is opaque to torch.compile.
            return torch.ops.aiter.maybe_dual_stream_forward(x, self.prefix)
        return self.single_stream_moe_forward(x)
