"""
LLM Factory — single source of truth for all language model clients.

Two tiers:
  get_dev_llm()      → OpenRouter (via Kilo AI) for planning/analysis/iteration agents
  get_verifier_llm() → Direct Anthropic (Claude) for code generation and physics verification

Falls back to ChatAnthropic if OPENROUTER_API_KEY is not set, so the system works
with an Anthropic-only setup as well.
"""

import os
from loguru import logger

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_DEV_MODEL   = "anthropic/claude-sonnet-4"
DEFAULT_VER_MODEL   = "claude-sonnet-4-20250514"


def get_dev_llm(max_tokens: int = 2000):
    """
    Returns an OpenRouter-backed LLM for development agents
    (analyst, selector, iteration, architect analysis, discovery).

    Falls back to ChatAnthropic if OPENROUTER_API_KEY is not set.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    model   = os.getenv("OPENROUTER_MODEL", DEFAULT_DEV_MODEL)

    if api_key:
        try:
            from langchain_openai import ChatOpenAI
            logger.debug(f"Dev LLM: OpenRouter / {model}")
            return ChatOpenAI(
                openai_api_key=api_key,
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

    logger.debug(f"Dev LLM: Anthropic fallback / {DEFAULT_VER_MODEL}")
    return _make_anthropic_llm(max_tokens)


def get_verifier_llm(max_tokens: int = 4000):
    """
    Returns a direct Anthropic (Claude) LLM for verification-critical tasks:
    - Code generation (architect's code_generator)
    - Physics certificate validation
    """
    model = os.getenv("ANTHROPIC_VERIFIER_MODEL", DEFAULT_VER_MODEL)
    logger.debug(f"Verifier LLM: Anthropic / {model}")
    return _make_anthropic_llm(max_tokens, model=model)


def _make_anthropic_llm(max_tokens: int, model: str = DEFAULT_VER_MODEL):
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(model=model, max_tokens=max_tokens)
