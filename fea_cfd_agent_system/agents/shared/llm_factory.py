"""
LLM Factory — single source of truth for all language model clients.

Three-tier provider resolution for dev LLM (first match wins):
  1. NVIDIA_API_KEY   → build.nvidia.com free NIM API (OpenAI-compatible)
  2. OPENROUTER_API_KEY → OpenRouter / Kilo AI (OpenAI-compatible)
  3. fallback          → Direct Anthropic (Claude)

Verifier LLM always uses Anthropic direct for code-gen / physics validation.
"""

import os
from loguru import logger

NVIDIA_BASE_URL     = "https://integrate.api.nvidia.com/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

DEFAULT_NVIDIA_MODEL = "meta/llama-3.3-70b-instruct"
DEFAULT_OR_MODEL     = "anthropic/claude-sonnet-4"
DEFAULT_VER_MODEL    = "claude-sonnet-4-20250514"


def get_dev_llm(max_tokens: int = 2000):
    """
    Returns a dev LLM for planning/analysis/iteration/discovery agents.
    Provider resolution order: NVIDIA → OpenRouter → Anthropic fallback.
    """
    # 1. NVIDIA NIM free API
    nvidia_key = os.getenv("NVIDIA_API_KEY")
    if nvidia_key:
        model = os.getenv("NVIDIA_MODEL", DEFAULT_NVIDIA_MODEL)
        try:
            from langchain_openai import ChatOpenAI
            logger.debug(f"Dev LLM: NVIDIA NIM / {model}")
            return ChatOpenAI(
                openai_api_key=nvidia_key,
                openai_api_base=NVIDIA_BASE_URL,
                model_name=model,
                max_tokens=max_tokens,
            )
        except ImportError:
            logger.warning("langchain-openai not installed — skipping NVIDIA provider")

    # 2. OpenRouter / Kilo AI
    or_key = os.getenv("OPENROUTER_API_KEY")
    if or_key:
        model = os.getenv("OPENROUTER_MODEL", DEFAULT_OR_MODEL)
        try:
            from langchain_openai import ChatOpenAI
            logger.debug(f"Dev LLM: OpenRouter / {model}")
            return ChatOpenAI(
                openai_api_key=or_key,
                openai_api_base=OPENROUTER_BASE_URL,
                model_name=model,
                max_tokens=max_tokens,
                default_headers={
                    "HTTP-Referer": "https://github.com/nsrkoganti/Agents",
                    "X-Title": "FEA-CFD-Agent",
                },
            )
        except ImportError:
            logger.warning("langchain-openai not installed — falling back to ChatAnthropic")

    # 3. Anthropic fallback
    logger.debug(f"Dev LLM: Anthropic fallback / {DEFAULT_VER_MODEL}")
    return _make_anthropic_llm(max_tokens)


def get_verifier_llm(max_tokens: int = 4000):
    """
    Returns a direct Anthropic (Claude) LLM for verification-critical tasks:
    code generation and physics certificate validation.
    """
    model = os.getenv("ANTHROPIC_VERIFIER_MODEL", DEFAULT_VER_MODEL)
    logger.debug(f"Verifier LLM: Anthropic / {model}")
    return _make_anthropic_llm(max_tokens, model=model)


def _make_anthropic_llm(max_tokens: int, model: str = DEFAULT_VER_MODEL):
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(model=model, max_tokens=max_tokens)
