from openai import OpenAI
import httpx

from .config import get_config


_TRAILING_INTERACTIVE_PATTERNS = (
    "would you like",
    "if you'd like",
    "if you’d like",
    "if you would like",
    "if you want, i can",
    "if you want i can",
    "do you want me to",
    "want me to",
    "which follow-up",
    "which follow‑up",
    "which follow up",
    "should i",
)


def get_openai_client_with_timeout(api_key, timeout_seconds=300):
    """Create OpenAI client with configurable timeout for slow web search operations."""
    return OpenAI(
        api_key=api_key,
        timeout=httpx.Timeout(timeout_seconds, connect=10.0),
    )


def _coerce_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _strip_trailing_interactive_followup(text: str) -> str:
    """Strip trailing interactive follow-up prompts from tool output."""
    cleaned = (text or "").strip()
    if not cleaned:
        return cleaned

    lines = cleaned.splitlines()
    while lines and not lines[-1].strip():
        lines.pop()

    removed_any = False
    while lines:
        tail = (
            lines[-1]
            .strip()
            .lower()
            .replace("’", "'")
            .replace("‘", "'")
            .replace("‑", "-")
            .replace("–", "-")
            .replace("—", "-")
        )
        if any(pattern in tail for pattern in _TRAILING_INTERACTIVE_PATTERNS):
            removed_any = True
            lines.pop()
            while lines and lines[-1].strip().startswith(("- ", "* ")):
                lines.pop()
            while lines and not lines[-1].strip():
                lines.pop()
            continue
        break

    candidate = "\n".join(lines).strip()
    if removed_any and candidate:
        return candidate
    return cleaned


def get_search_context_for_depth(research_depth=None):
    """Get the appropriate search_context_size based on research depth."""
    if research_depth is None:
        config = get_config()
        research_depth = config.get("research_depth", "Medium")

    depth_mapping = {
        "shallow": "low",
        "medium": "medium",
        "deep": "high",
    }
    return depth_mapping.get(
        research_depth.lower() if research_depth else "medium",
        "medium",
    )


def get_llm_params_for_depth(research_depth=None):
    """Get reasoning effort and verbosity matching the research depth."""
    if research_depth is None:
        config = get_config()
        research_depth = config.get("research_depth", "Medium")

    depth = research_depth.lower() if research_depth else "medium"
    mapping = {
        "shallow": {"effort": "low", "verbosity": "low"},
        "medium": {"effort": "medium", "verbosity": "medium"},
        "deep": {"effort": "high", "verbosity": "high"},
    }
    return mapping.get(depth, mapping["medium"])


def get_global_news_profile_for_depth(research_depth=None, fast_profile=True):
    """Get tuned search/LLM settings for global-news web search."""
    if research_depth is None:
        config = get_config()
        research_depth = config.get("research_depth", "Medium")

    depth = research_depth.lower() if research_depth else "medium"

    if fast_profile:
        mapping = {
            "shallow": {
                "search_context": "low",
                "effort": "low",
                "verbosity": "low",
                "lookback_days": 3,
            },
            "medium": {
                "search_context": "low",
                "effort": "low",
                "verbosity": "low",
                "lookback_days": 5,
            },
            # Keep deep mode practical for runtime; broader window without expensive reasoning profile.
            "deep": {
                "search_context": "low",
                "effort": "low",
                "verbosity": "low",
                "lookback_days": 7,
            },
        }
        return mapping.get(depth, mapping["medium"])

    llm_params = get_llm_params_for_depth(research_depth)
    search_context = get_search_context_for_depth(research_depth)
    lookback_map = {"shallow": 3, "medium": 7, "deep": 14}
    return {
        "search_context": search_context,
        "effort": llm_params["effort"],
        "verbosity": llm_params["verbosity"],
        "lookback_days": lookback_map.get(depth, 7),
    }


def get_model_params(model_name, max_tokens_value=3000):
    """Get appropriate parameters for different model types."""
    params = {}

    gpt5_models = ["gpt-5", "gpt-5-mini", "gpt-5-nano"]
    gpt52_models = ["gpt-5.2", "gpt-5.2-pro"]
    gpt41_models = ["gpt-4.1"]

    if any(model_prefix in model_name for model_prefix in gpt52_models):
        params["text"] = {"format": "text"}
        params["summary"] = "auto"

        if "gpt-5.2-pro" in model_name:
            params["store"] = True
        else:
            params["reasoning"] = {"effort": "medium"}
            params["verbosity"] = "medium"
    elif any(model_prefix in model_name for model_prefix in gpt5_models):
        pass
    elif any(model_prefix in model_name for model_prefix in gpt41_models):
        params["temperature"] = 0.2
        params["max_output_tokens"] = max_tokens_value
        params["top_p"] = 1
    else:
        params["temperature"] = 0.2
        params["max_tokens"] = max_tokens_value

    return params
