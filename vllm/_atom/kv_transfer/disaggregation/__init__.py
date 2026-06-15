# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
KV cache disaggregation for Prefill/Decode (P/D) separation.

Public API re-exports — engine code should import from this package
rather than reaching into submodules directly.
"""

from vllm._atom.kv_transfer.disaggregation.aggregator import KVOutputAggregator
from vllm._atom.kv_transfer.disaggregation.base import (
    KVConnectorBase,
    KVConnectorSchedulerBase,
)
from vllm._atom.kv_transfer.disaggregation.factory import KVConnectorFactory
from vllm._atom.kv_transfer.disaggregation.types import (
    ConnectorMetadata,
    KVConnectorOutput,
    ReqMeta,
)

__all__ = [
    "KVConnectorBase",
    "KVConnectorSchedulerBase",
    "KVConnectorFactory",
    "KVConnectorOutput",
    "KVOutputAggregator",
    "ConnectorMetadata",
    "ReqMeta",
]
