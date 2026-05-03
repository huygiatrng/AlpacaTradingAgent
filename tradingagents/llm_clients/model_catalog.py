"""Shared model catalog for provider selections and validation."""

from __future__ import annotations

from typing import Dict, List, Tuple

ModelOption = Tuple[str, str]


MODEL_OPTIONS: Dict[str, Dict[str, List[ModelOption]]] = {
    "google": {
        "quick": [
            ("Gemini 3 Flash - next-gen fast", "gemini-3-flash-preview"),
            ("Gemini 2.5 Flash - balanced, stable", "gemini-2.5-flash"),
            ("Gemini 3.1 Flash Lite - cost efficient", "gemini-3.1-flash-lite-preview"),
            ("Gemini 2.5 Flash Lite - fast, low-cost", "gemini-2.5-flash-lite"),
        ],
        "deep": [
            ("Gemini 3.1 Pro - reasoning-first", "gemini-3.1-pro-preview"),
            ("Gemini 3 Flash - next-gen fast", "gemini-3-flash-preview"),
            ("Gemini 2.5 Pro - stable pro", "gemini-2.5-pro"),
            ("Gemini 2.5 Flash - balanced, stable", "gemini-2.5-flash"),
        ],
    },
    "anthropic": {
        "quick": [
            ("Claude Sonnet 4.6 - speed and intelligence balance", "claude-sonnet-4-6"),
            ("Claude Haiku 4.5 - fast responses", "claude-haiku-4-5"),
            ("Claude Sonnet 4.5 - agents and coding", "claude-sonnet-4-5"),
        ],
        "deep": [
            ("Claude Opus 4.6 - most capable", "claude-opus-4-6"),
            ("Claude Opus 4.5 - premium reasoning", "claude-opus-4-5"),
            ("Claude Sonnet 4.6 - speed and intelligence balance", "claude-sonnet-4-6"),
            ("Claude Sonnet 4.5 - agents and coding", "claude-sonnet-4-5"),
        ],
    },
    "xai": {
        "quick": [
            ("Grok 4.1 Fast non-reasoning - speed optimized", "grok-4-1-fast-non-reasoning"),
            ("Grok 4 Fast non-reasoning - speed optimized", "grok-4-fast-non-reasoning"),
            ("Grok 4.1 Fast reasoning - high performance", "grok-4-1-fast-reasoning"),
        ],
        "deep": [
            ("Grok 4 - flagship", "grok-4-0709"),
            ("Grok 4.1 Fast reasoning - high performance", "grok-4-1-fast-reasoning"),
            ("Grok 4 Fast reasoning - high performance", "grok-4-fast-reasoning"),
            ("Grok 4.1 Fast non-reasoning - speed optimized", "grok-4-1-fast-non-reasoning"),
        ],
    },
    "deepseek": {
        "quick": [
            ("DeepSeek V4 Flash - fast", "deepseek-v4-flash"),
            ("DeepSeek V3.2", "deepseek-chat"),
            ("Custom model ID", "custom"),
        ],
        "deep": [
            ("DeepSeek V4 Pro - flagship", "deepseek-v4-pro"),
            ("DeepSeek V3.2 thinking", "deepseek-reasoner"),
            ("DeepSeek V3.2", "deepseek-chat"),
            ("Custom model ID", "custom"),
        ],
    },
    "qwen": {
        "quick": [
            ("Qwen 3.5 Flash", "qwen3.5-flash"),
            ("Qwen Plus", "qwen-plus"),
            ("Custom model ID", "custom"),
        ],
        "deep": [
            ("Qwen 3.6 Plus", "qwen3.6-plus"),
            ("Qwen 3.5 Plus", "qwen3.5-plus"),
            ("Qwen 3 Max", "qwen3-max"),
            ("Custom model ID", "custom"),
        ],
    },
    "glm": {
        "quick": [
            ("GLM-4.7", "glm-4.7"),
            ("GLM-5", "glm-5"),
            ("Custom model ID", "custom"),
        ],
        "deep": [
            ("GLM-5.1", "glm-5.1"),
            ("GLM-5", "glm-5"),
            ("Custom model ID", "custom"),
        ],
    },
    "openrouter": {"quick": [("Custom OpenRouter model", "custom")], "deep": [("Custom OpenRouter model", "custom")]},
    "ollama": {
        "quick": [
            ("Qwen3:latest - local fast", "qwen3:latest"),
            ("GPT-OSS:latest - local balanced", "gpt-oss:latest"),
            ("GLM-4.7-Flash:latest - local larger", "glm-4.7-flash:latest"),
            ("Custom local model ID", "custom"),
        ],
        "deep": [
            ("GLM-4.7-Flash:latest - local larger", "glm-4.7-flash:latest"),
            ("GPT-OSS:latest - local balanced", "gpt-oss:latest"),
            ("Qwen3:latest - local fast", "qwen3:latest"),
            ("Custom local model ID", "custom"),
        ],
    },
    "azure": {"quick": [("Azure deployment", "custom")], "deep": [("Azure deployment", "custom")]},
}


def get_model_options(provider: str, mode: str) -> List[ModelOption]:
    return MODEL_OPTIONS.get(provider.lower(), {}).get(mode, [])


def get_known_models() -> Dict[str, List[str]]:
    return {
        provider: sorted({value for values in modes.values() for _, value in values if value != "custom"})
        for provider, modes in MODEL_OPTIONS.items()
    }
