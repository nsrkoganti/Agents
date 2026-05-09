"""
LLM Factory — single source of truth for all language model clients.

Two-provider system:
  1. GEMINI_API_KEY   → Google Gemini 2.5 Flash (free: 15 RPM / 1500 RPD)
  2. fallback         → Anthropic Claude Sonnet 4.6 (dev tasks)

Verifier always uses Anthropic Claude Opus 4.7 (code-gen / physics validation).
"""

import os
from loguru import logger

GEMINI_MODEL          = "gemini-2.5-flash"
CLAUDE_VERIFIER_MODEL = "claude-opus-4-7"
CLAUDE_FALLBACK_MODEL = "claude-sonnet-4-6"


def get_dev_llm(max_tokens: int = 2000):
    """
    Returns a dev LLM for planning/analysis/iteration/discovery agents.
    Provider order: Gemini 2.5 Flash → Claude Sonnet 4.6 fallback.
    """
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        model = os.getenv("GEMINI_MODEL", GEMINI_MODEL)
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            logger.debug(f"Dev LLM: Gemini / {model}")
            return ChatGoogleGenerativeAI(
                model=model,
                google_api_key=gemini_key,
                max_output_tokens=max_tokens,
            )
        except ImportError:
            logger.warning("langchain-google-genai not installed — falling back to ChatAnthropic")

    logger.debug(f"Dev LLM: Anthropic fallback / {CLAUDE_FALLBACK_MODEL}")
    return _make_anthropic_llm(max_tokens, model=CLAUDE_FALLBACK_MODEL)


def get_verifier_llm(max_tokens: int = 4000):
    """
    Returns Anthropic Claude Opus 4.7 for verification-critical tasks:
    code generation and physics certificate validation.
    """
    model = os.getenv("ANTHROPIC_VERIFIER_MODEL", CLAUDE_VERIFIER_MODEL)
    logger.debug(f"Verifier LLM: Anthropic / {model}")
    return _make_anthropic_llm(max_tokens, model=model)


def _make_anthropic_llm(max_tokens: int, model: str = CLAUDE_VERIFIER_MODEL):
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(model=model, max_tokens=max_tokens)
