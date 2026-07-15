# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
A QuantizedActivation is a pre-quantized activation produced by a fused kernel
and consumed directly by a linear layer, letting the layer skip its own input
quantization. A linear advertises the key its kernel can consume via
expose_input_quant_key; the kernel validates and reads the activation via
as_quantized_activation.
"""

from dataclasses import dataclass

import torch

from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey


@dataclass
class QuantizedActivation:
    """A quantized activation paired with its scale and original metadata.

    The quant_key describes how data and scale are to be interpreted (dtype,
    scale granularity, value packing). Details the key does not capture, such
    as blockscale layout or activation padding, must follow the consumer
    kernel's convention.

    TODO(mgoin): Encode layout and padding requirements in the contract so
    producers can match consumer kernels without relying on convention.
    """

    data: torch.Tensor
    scale: torch.Tensor
    orig_dtype: torch.dtype
    orig_shape: torch.Size
    quant_key: QuantKey


def expose_input_quant_key(layer: torch.nn.Module, kernel) -> None:
    """Advertise the kernel's external-input contract on the layer, if any.

    This is the bridge from a kernel's input_quant_key() to the
    layer.input_quant_key attribute that fusion call sites read. The attribute
    is left unset when the kernel quantizes its own input, so non-supporting
    backends never receive a QuantizedActivation.

    Kernels that can also produce their external input may provide an opaque
    input_quantizer_id() and a quantize_input() method. These are exposed
    together so fusion call sites can require identical producer semantics
    without duplicating backend-selection policy.

    TODO(mgoin): Producers also need the consumer's quantization scales (e.g.
    static input scale, global scale). Expose those here as well so producers
    do not reach into kernel-specific layer attributes.
    """
    key = kernel.input_quant_key()
    if key is not None:
        layer.input_quant_key = key

        quantizer_id_fn = getattr(kernel, "input_quantizer_id", None)
        quantize_input = getattr(kernel, "quantize_input", None)
        if quantizer_id_fn is not None and quantize_input is not None:
            quantizer_id = quantizer_id_fn()
            if quantizer_id is not None:
                layer.input_quantizer_id = quantizer_id
                layer.quantize_input = quantize_input


def as_quantized_activation(
    x: "torch.Tensor | QuantizedActivation", expected_key: QuantKey | None
) -> "QuantizedActivation | None":
    """Validate and narrow a pre-quantized activation for a consumer kernel.

    Returns the QuantizedActivation when x is one whose key matches the
    kernel's declared expected_key, and None when x is a plain tensor (the
    caller quantizes in-kernel). Raises on a key mismatch so a wrongly routed
    activation fails loudly instead of being silently re-quantized.
    """
    if not isinstance(x, QuantizedActivation):
        return None
    assert x.quant_key == expected_key, (
        f"QuantizedActivation key {x.quant_key} != consumer kernel "
        f"input_quant_key {expected_key}"
    )
    return x
