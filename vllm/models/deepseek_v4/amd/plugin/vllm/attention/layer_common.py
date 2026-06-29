from vllm.models.deepseek_v4.amd.config import get_current_dsv4_config


def _register_vllm_static_forward_context(layer) -> None:
    dsv4_config = get_current_dsv4_config()
    static_forward_context = (
        dsv4_config.plugin_config.vllm_config.compilation_config.static_forward_context
    )
    if layer.layer_name in static_forward_context:
        raise ValueError(f"Duplicate layer name: {layer.layer_name}")
    static_forward_context[layer.layer_name] = layer
