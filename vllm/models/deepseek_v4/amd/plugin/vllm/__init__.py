"""vLLM integration for the DeepSeek-V4 ROCm full-attention path.

Ported from ROCm/ATOM. The model class is resolved through vLLM's registry
via ``vllm/models/deepseek_v4/__init__.py`` (the DSv4 ROCm flag gate); no
platform/model auto-registration happens here.
"""
