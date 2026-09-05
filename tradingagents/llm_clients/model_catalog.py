"""Shared model catalog for provider selections and validation."""

from __future__ import annotations

from typing import Dict, List, Tuple

ModelOption = Tuple[str, str]


MODEL_OPTIONS: Dict[str, Dict[str, List[ModelOption]]] = {
    "google": {
        "quick": [
            ("Gemini 3.8 Flash - latest stable", "gemini-3.8-flash"),
            ("Gemini 3.7 Flash - previous stable", "gemini-3.7-flash"),
            ("Gemini 3.6 Flash", "gemini-3.6-flash"),
            ("Gemini 3.5 Flash-Lite - efficient", "gemini-3.5-flash-lite"),
            ("Gemini 3.1 Flash-Lite - efficient stable", "gemini-3.1-flash-lite"),
            ("Gemini 2.5 Flash - balanced, stable", "gemini-2.5-flash"),
            ("Gemini 2.5 Flash Lite - fast, low-cost", "gemini-2.5-flash-lite"),
        ],
        "deep": [
            ("Gemini 3.8 Flash - latest stable", "gemini-3.8-flash"),
            ("Gemini 3.1 Pro - reasoning-first", "gemini-3.1-pro-preview"),
            ("Gemini 3.7 Flash - previous stable", "gemini-3.7-flash"),
            ("Gemini 3.6 Flash", "gemini-3.6-flash"),
            ("Gemini 2.5 Pro - stable pro", "gemini-2.5-pro"),
            ("Gemini 2.5 Flash - balanced, stable", "gemini-2.5-flash"),
        ],
    },
    "anthropic": {
        "quick": [
            ("Claude Sonnet 5 - latest balanced model", "claude-sonnet-5"),
            ("Claude Haiku 4.5 - fastest current model", "claude-haiku-4-5"),
            ("Claude Sonnet 4.6 - speed and intelligence balance", "claude-sonnet-4-6"),
        ],
        "deep": [
            ("Claude Fable 5.1 - latest long-horizon flagship", "claude-fable-5-1"),
            ("Claude Opus 5 - current premium model", "claude-opus-5"),
            ("Claude Sonnet 5 - latest balanced model", "claude-sonnet-5"),
            ("Claude Fable 5 - previous long-horizon flagship", "claude-fable-5"),
            ("Claude Opus 4.8 - complex agentic work", "claude-opus-4-8"),
            ("Claude Sonnet 4.6 - speed and intelligence balance", "claude-sonnet-4-6"),
        ],
    },
    "xai": {
        "quick": [
            ("Grok 4.6 - latest flagship", "grok-4.6"),
            ("Grok 4.3 - fast current model", "grok-4.3"),
            ("Grok 4.20 Non-Reasoning - low latency", "grok-4.20-non-reasoning"),
        ],
        "deep": [
            ("Grok 4.6 - latest frontier reasoning", "grok-4.6"),
            ("Grok 4.20 - reasoning model", "grok-4.20"),
            ("Grok 4.5 - frontier reasoning", "grok-4.5"),
            ("Grok 4.3 - reliable tool calling", "grok-4.3"),
            ("Grok latest alias", "grok-latest"),
        ],
    },
    "minimax": {
        "quick": [
            ("MiniMax M2.7 Highspeed - fastest current", "MiniMax-M2.7-highspeed"),
            ("MiniMax M2.7 - current flagship", "MiniMax-M2.7"),
            ("MiniMax M2.5 Highspeed", "MiniMax-M2.5-highspeed"),
            ("MiniMax M2.5", "MiniMax-M2.5"),
        ],
        "deep": [
            ("MiniMax M2.7 - current flagship", "MiniMax-M2.7"),
            ("MiniMax M2.7 Highspeed", "MiniMax-M2.7-highspeed"),
            ("MiniMax M2.5", "MiniMax-M2.5"),
            ("MiniMax M2.1", "MiniMax-M2.1"),
        ],
    },
    "deepseek": {
        "quick": [
            ("DeepSeek V4 Flash - fast", "deepseek-v4-flash"),
            ("Custom model ID", "custom"),
        ],
        "deep": [
            ("DeepSeek V4 Pro - flagship", "deepseek-v4-pro"),
            ("DeepSeek V4 Flash - fast", "deepseek-v4-flash"),
            ("Custom model ID", "custom"),
        ],
    },
    "qwen": {
        "quick": [
            ("Qwen 3.8 Flash - latest fast model", "qwen3.8-flash"),
            ("Qwen 3.7 Flash", "qwen3.7-flash"),
            ("Qwen 3.6 Flash", "qwen3.6-flash"),
            ("Qwen 3.5 Flash", "qwen3.5-flash"),
            ("Custom model ID", "custom"),
        ],
        "deep": [
            ("Qwen 3.8 Max - latest flagship", "qwen3.8-max"),
            ("Qwen 3.7 Plus", "qwen3.7-plus"),
            ("Qwen 3.8 Flash - latest fast model", "qwen3.8-flash"),
            ("Qwen 3.6 Plus", "qwen3.6-plus"),
            ("Qwen 3.5 Plus", "qwen3.5-plus"),
            ("Custom model ID", "custom"),
        ],
    },
    "glm": {
        "quick": [
            ("GLM-5.2 - latest flagship", "glm-5.2"),
            ("GLM-5.1", "glm-5.1"),
            ("GLM-5", "glm-5"),
            ("Custom model ID", "custom"),
        ],
        "deep": [
            ("GLM-5.2 - latest flagship", "glm-5.2"),
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
