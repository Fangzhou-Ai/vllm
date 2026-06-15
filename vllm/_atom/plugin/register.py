import logging

from vllm._atom.config import Config
from vllm._atom.plugin.prepare import is_vllm, is_sglang

logger = logging.getLogger("atom")

# NOTE: This vendored port targets DeepSeek-V4 only. ATOM's internal model
# registry (`_ATOM_SUPPORTED_MODELS`) and the SGLang attention hooks were dropped
# along with the other ATOM models and the `sglang` plugin during the vLLM port
# cleanup. Only the vLLM plugin init helpers below are retained.


def set_attn_cls() -> None:
    """Keep compatibility with old plugin init hooks.

    FIXME: This is a legacy no-op after attention construction moved to the
    frontend dispatcher. Remove it once downstream plugin init paths stop
    calling ``set_attn_cls`` for side effects.

    Attention selection now happens in ``atom.model_ops.base_attention.Attention``
    at construction time, so plugin init no longer mutates ``atom.model_ops``.
    """
    if is_vllm():
        logger.info("Use Attention dispatcher for vLLM")
    elif is_sglang():
        logger.info("Use Attention dispatcher for SGLang")


def init_aiter_dist(config: Config) -> None:
    """
    Initialize aiter dist for using aiter custom collective op.

    In vLLM plugin mode, tries to reuse vLLM's TP group and inject aiter's ca_comm
    first (single IPC init, avoids 2x reduce slowdown). Falls back to init_dist_env
    if reuse fails.
    """
    logger.info(
        "Initialize aiter dist for using aiter custom collective op for plugin mode"
    )

    rank = config.plugin_config.rank
    if getattr(config.plugin_config, "is_sglang", False):
        rank = getattr(config.plugin_config, "sglang_aiter_rank_id", rank)
    tensor_parallel_size = config.tensor_parallel_size

    assert (
        config.plugin_config.is_plugin_mode
    ), "Make sure ATOM is running in plugin mode"

    if config.plugin_config.is_vllm:
        from vllm._atom.plugin.vllm.tp_group_reuse import init_aiter_tp_from_vllm

        if init_aiter_tp_from_vllm(tensor_parallel_size):
            return

    # Fallback: create aiter's own groups (vLLM reuse failed or non-vLLM plugin)
    from aiter import init_dist_env
    from aiter.dist.utils import get_distributed_init_method

    if config.plugin_config.is_vllm:
        dp_master_ip = config.parallel_config.data_parallel_master_ip
        dp_master_port = config.parallel_config.data_parallel_master_port
    elif config.plugin_config.is_sglang:
        if config.plugin_config.sglang_dist_init_addr is not None:
            dp_master_ip, dp_master_port = (
                config.plugin_config.sglang_dist_init_addr.split(":")
            )
        else:
            dp_master_ip = "127.0.0.1"
            dp_master_port = config.plugin_config.sglang_port_args.nccl_port

    distributed_init_method = get_distributed_init_method(dp_master_ip, dp_master_port)

    logger.info(
        f"Initialize aiter dist for using aiter custom collective op for plugin mode, rank:{rank}"
    )
    init_dist_env(
        tensor_model_parallel_size=tensor_parallel_size,
        rankID=rank,
        backend="nccl",
        distributed_init_method=distributed_init_method,
        data_parallel_size=config.parallel_config.data_parallel_size,
        data_parallel_rank=config.parallel_config.data_parallel_rank,
    )
