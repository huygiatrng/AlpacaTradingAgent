"""OpenAI model registry and parameter helpers.

The app lets users choose models from the UI, so the runtime needs one place
that knows which knobs are valid for each model family.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, List, Mapping, Optional

from tradingagents.llm_clients.model_catalog import get_model_options as get_provider_catalog_options


REASONING_SUMMARY_OPTIONS = ["auto", "concise", "detailed", "none"]
TEXT_VERBOSITY_OPTIONS = ["low", "medium", "high"]


PARAMETER_HELP = {
    "reasoning_effort": (
        "Controls how much hidden reasoning budget the model may spend. Lower "
        "values are faster and cheaper; higher values can improve difficult "
        "analysis but use more reasoning tokens."
    ),
    "text_verbosity": (
        "Controls how much detail the model writes. Low is concise and cheaper; "
        "high can produce fuller reports and larger token bills."
    ),
    "reasoning_summary": (
        "Requests a summary of the model reasoning for debugging. It can add "
        "latency and response metadata; set to none for the leanest calls."
    ),
    "temperature": (
        "Controls randomness for non-reasoning chat models. Lower values are "
        "more deterministic; higher values can be more varied but less stable."
    ),
    "top_p": (
        "Nucleus sampling for non-reasoning chat models. Usually leave this at "
        "1.0 unless you are deliberately tuning diversity."
    ),
    "max_output_tokens": (
        "Optional hard cap on generated tokens. Lower caps control cost and "
        "latency, but too low can truncate reports or leave reasoning models "
        "with no visible text output."
    ),
    "store": (
        "Whether OpenAI may store the response for later retrieval. Disabling "
        "keeps requests leaner and avoids extra stored-response state."
    ),
    "parallel_tool_calls": (
        "Allows the model to request multiple tools at once when tool calling "
        "is available. This can reduce latency but may increase bursty calls."
    ),
}


def _reasoning_spec(
    *,
    model_id: str,
    label: str,
    description: str,
    role_defaults: Mapping[str, Mapping[str, Any]],
    reasoning_effort_options: Iterable[str],
    roles: Iterable[str] = ("quick", "deep"),
    visible: bool = True,
    price_hint: str = "",
) -> Dict[str, Any]:
    return {
        "id": model_id,
        "label": label,
        "description": description,
        "price_hint": price_hint,
        "roles": list(roles),
        "visible": visible,
        "api": "responses",
        "supports_reasoning_effort": True,
        "reasoning_effort_options": list(reasoning_effort_options),
        "supports_text_verbosity": True,
        "text_verbosity_options": TEXT_VERBOSITY_OPTIONS,
        "supports_reasoning_summary": True,
        "reasoning_summary_options": REASONING_SUMMARY_OPTIONS,
        "supports_temperature": False,
        "supports_top_p": False,
        "supports_max_output_tokens": True,
        "supports_store": True,
        "supports_parallel_tool_calls": True,
        "role_defaults": {
            role: {
                "reasoning_effort": defaults.get("reasoning_effort", "low"),
                "text_verbosity": defaults.get("text_verbosity", "medium"),
                "reasoning_summary": defaults.get("reasoning_summary", "auto"),
                "max_output_tokens": defaults.get("max_output_tokens"),
                "store": defaults.get("store", False),
                "parallel_tool_calls": defaults.get("parallel_tool_calls", True),
            }
            for role, defaults in role_defaults.items()
        },
    }


MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "gpt-5.4-pro": _reasoning_spec(
        model_id="gpt-5.4-pro",
        label="GPT-5.4 Pro - strongest, expensive",
        description=(
            "Highest quality GPT-5.4 model for tough decisions. Use sparingly "
            "because it is much slower and more expensive."
        ),
        roles=("deep",),
        reasoning_effort_options=["medium", "high", "xhigh"],
        role_defaults={
            "deep": {
                "reasoning_effort": "high",
                "text_verbosity": "medium",
                "reasoning_summary": "auto",
            }
        },
        price_hint="highest cost",
    ),
    "gpt-5.4": _reasoning_spec(
        model_id="gpt-5.4",
        label="GPT-5.4 - flagship current",
        description="Current flagship for complex financial reasoning and agent workflows.",
        reasoning_effort_options=["none", "low", "medium", "high", "xhigh"],
        role_defaults={
            "quick": {"reasoning_effort": "low", "text_verbosity": "low"},
            "deep": {"reasoning_effort": "high", "text_verbosity": "medium"},
        },
        price_hint="frontier",
    ),
    "gpt-5.4-mini": _reasoning_spec(
        model_id="gpt-5.4-mini",
        label="GPT-5.4 Mini - balanced default",
        description="Recent mini model with strong quality, lower latency, and lower cost.",
        reasoning_effort_options=["none", "low", "medium", "high", "xhigh"],
        role_defaults={
            "quick": {"reasoning_effort": "low", "text_verbosity": "low"},
            "deep": {"reasoning_effort": "medium", "text_verbosity": "medium"},
        },
        price_hint="balanced",
    ),
    "gpt-5.4-nano": _reasoning_spec(
        model_id="gpt-5.4-nano",
        label="GPT-5.4 Nano - latest low-cost",
        description="Latest low-cost GPT-5.4 model for quick checks and high-volume runs.",
        reasoning_effort_options=["none", "low", "medium", "high", "xhigh"],
        role_defaults={
            "quick": {"reasoning_effort": "low", "text_verbosity": "low"},
            "deep": {"reasoning_effort": "medium", "text_verbosity": "low"},
        },
        price_hint="low cost",
    ),
    "gpt-5-mini": _reasoning_spec(
        model_id="gpt-5-mini",
        label="GPT-5 Mini - cheaper previous mini",
        description=(
            "Previous mini model. Kept because it is cheaper than GPT-5.4 mini "
            "and useful for cost-sensitive tests."
        ),
        reasoning_effort_options=["minimal", "low", "medium", "high"],
        role_defaults={
            "quick": {"reasoning_effort": "low", "text_verbosity": "low"},
            "deep": {"reasoning_effort": "medium", "text_verbosity": "medium"},
        },
        price_hint="cheap",
    ),
    "gpt-5-nano": _reasoning_spec(
        model_id="gpt-5-nano",
        label="GPT-5 Nano - cheapest test model",
        description="Lowest-cost retained model for smoke tests and development runs.",
        reasoning_effort_options=["minimal", "low", "medium", "high"],
        role_defaults={
            "quick": {"reasoning_effort": "minimal", "text_verbosity": "low"},
            "deep": {"reasoning_effort": "low", "text_verbosity": "low"},
        },
        price_hint="cheapest",
    ),
    "gpt-4.1": {
        "id": "gpt-4.1",
        "label": "GPT-4.1 - non-reasoning control",
        "description": (
            "Current non-reasoning model. Useful when you want temperature and "
            "top_p control instead of reasoning effort."
        ),
        "price_hint": "non-reasoning",
        "roles": ["quick", "deep"],
        "visible": True,
        "api": "chat",
        "supports_reasoning_effort": False,
        "reasoning_effort_options": [],
        "supports_text_verbosity": False,
        "text_verbosity_options": [],
        "supports_reasoning_summary": False,
        "reasoning_summary_options": [],
        "supports_temperature": True,
        "supports_top_p": True,
        "supports_max_output_tokens": True,
        "supports_store": False,
        "supports_parallel_tool_calls": False,
        "role_defaults": {
            "quick": {"temperature": 0.1, "top_p": 1.0, "max_output_tokens": None},
            "deep": {"temperature": 0.2, "top_p": 1.0, "max_output_tokens": None},
        },
    },
    # Backward-compatible hidden entries. They are intentionally not shown in
    # the selector because GPT-5.4 variants are better replacements for new runs.
    "gpt-5.2": _reasoning_spec(
        model_id="gpt-5.2",
        label="GPT-5.2 - previous frontier",
        description="Previous frontier model retained for old saved configs.",
        visible=False,
        reasoning_effort_options=["none", "low", "medium", "high", "xhigh"],
        role_defaults={
            "quick": {"reasoning_effort": "low", "text_verbosity": "low"},
            "deep": {"reasoning_effort": "high", "text_verbosity": "medium"},
        },
    ),
    "gpt-5.2-pro": _reasoning_spec(
        model_id="gpt-5.2-pro",
        label="GPT-5.2 Pro - previous pro",
        description="Previous pro model retained for old saved configs.",
        roles=("deep",),
        visible=False,
        reasoning_effort_options=["medium", "high", "xhigh"],
        role_defaults={
            "deep": {"reasoning_effort": "high", "text_verbosity": "medium"}
        },
    ),
}


def get_model_spec(model_name: str) -> Dict[str, Any]:
    """Return a model spec, falling back by prefix for known families."""
    model = str(model_name or "").strip()
    if model in MODEL_REGISTRY:
        return deepcopy(MODEL_REGISTRY[model])

    # Snapshot names should inherit the alias capabilities.
    for model_id in sorted(MODEL_REGISTRY, key=len, reverse=True):
        if model.startswith(f"{model_id}-"):
            spec = deepcopy(MODEL_REGISTRY[model_id])
            spec["id"] = model
            spec["label"] = model
            return spec

    # Conservative fallback for unknown OpenAI-compatible chat models.
    spec = deepcopy(MODEL_REGISTRY["gpt-4.1"])
    spec["id"] = model or "custom"
    spec["label"] = model or "Custom model ID"
    spec["description"] = (
        "Custom chat-completions-compatible model. Reasoning-only controls are "
        "hidden; sampling and output-token controls remain available."
    )
    spec["price_hint"] = "custom"
    return spec


def get_openai_model_options(role: str) -> List[Dict[str, str]]:
    """Build Dash select options for the given role."""
    role_key = (role or "quick").lower()
    options = []
    for model_id, spec in MODEL_REGISTRY.items():
        if not spec.get("visible", True):
            continue
        if role_key not in spec.get("roles", []):
            continue
        options.append({"label": spec["label"], "value": model_id})
    return options


LLM_PROVIDER_OPTIONS = [
    {"label": "OpenAI", "value": "openai"},
    {"label": "Local OpenAI-compatible", "value": "local_openai"},
    {"label": "Google Gemini", "value": "google"},
    {"label": "Anthropic Claude", "value": "anthropic"},
    {"label": "xAI Grok", "value": "xai"},
    {"label": "DeepSeek", "value": "deepseek"},
    {"label": "Qwen / DashScope", "value": "qwen"},
    {"label": "GLM / Zhipu", "value": "glm"},
    {"label": "OpenRouter", "value": "openrouter"},
    {"label": "Ollama", "value": "ollama"},
    {"label": "Azure OpenAI", "value": "azure"},
]


PROVIDER_UI_METADATA: Dict[str, Dict[str, Any]] = {
    "openai": {
        "title": "OpenAI",
        "api_key": "OPENAI_API_KEY",
        "endpoint": "OpenAI default endpoint",
        "endpoint_placeholder": "Uses OpenAI default endpoint",
        "backend_visible": False,
        "custom_models": False,
        "summary": "Best-supported path for GPT-5.4 reasoning controls and structured output.",
        "pills": ["GPT-5.4 defaults", "reasoning controls"],
    },
    "local_openai": {
        "title": "Local OpenAI-compatible",
        "api_key": "OPENAI_API_KEY optional",
        "endpoint": "http://localhost:11434/v1 or another compatible endpoint",
        "endpoint_placeholder": "http://localhost:11434/v1",
        "backend_visible": True,
        "custom_models": True,
        "summary": "Use Ollama, LM Studio, vLLM, or another local OpenAI-compatible server.",
        "pills": ["custom models", "no cloud key required"],
    },
    "google": {
        "title": "Google Gemini",
        "api_key": "GOOGLE_API_KEY",
        "endpoint": "Google Generative AI default endpoint",
        "endpoint_placeholder": "Usually leave blank",
        "backend_visible": False,
        "custom_models": False,
        "summary": "Gemini models use a provider-level thinking mode instead of OpenAI reasoning controls.",
        "pills": ["thinking mode", "Gemini 2.5/3"],
    },
    "anthropic": {
        "title": "Anthropic Claude",
        "api_key": "ANTHROPIC_API_KEY",
        "endpoint": "Anthropic default endpoint",
        "endpoint_placeholder": "Usually leave blank",
        "backend_visible": False,
        "custom_models": False,
        "summary": "Claude models use a provider-level effort setting for supported 4.5+ and 4.6 models.",
        "pills": ["effort setting", "Claude 4.x"],
    },
    "xai": {
        "title": "xAI Grok",
        "api_key": "XAI_API_KEY",
        "endpoint": "https://api.x.ai/v1",
        "endpoint_placeholder": "https://api.x.ai/v1",
        "backend_visible": True,
        "custom_models": False,
        "summary": "OpenAI-compatible Grok endpoint. Override the endpoint only for proxies.",
        "pills": ["OpenAI-compatible"],
    },
    "deepseek": {
        "title": "DeepSeek",
        "api_key": "DEEPSEEK_API_KEY",
        "endpoint": "https://api.deepseek.com",
        "endpoint_placeholder": "https://api.deepseek.com",
        "backend_visible": True,
        "custom_models": True,
        "summary": "DeepSeek reasoning content is preserved; structured output falls back for reasoner models.",
        "pills": ["reasoning content", "custom allowed"],
    },
    "qwen": {
        "title": "Qwen / DashScope",
        "api_key": "DASHSCOPE_API_KEY",
        "endpoint": "DashScope OpenAI-compatible endpoint",
        "endpoint_placeholder": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "backend_visible": True,
        "custom_models": True,
        "summary": "Qwen runs through DashScope's OpenAI-compatible API.",
        "pills": ["OpenAI-compatible", "custom allowed"],
    },
    "glm": {
        "title": "GLM / Zhipu",
        "api_key": "ZHIPU_API_KEY",
        "endpoint": "https://open.bigmodel.cn/api/paas/v4/",
        "endpoint_placeholder": "https://open.bigmodel.cn/api/paas/v4/",
        "backend_visible": True,
        "custom_models": True,
        "summary": "GLM runs through Zhipu's OpenAI-compatible endpoint.",
        "pills": ["OpenAI-compatible", "custom allowed"],
    },
    "openrouter": {
        "title": "OpenRouter",
        "api_key": "OPENROUTER_API_KEY",
        "endpoint": "https://openrouter.ai/api/v1",
        "endpoint_placeholder": "https://openrouter.ai/api/v1",
        "backend_visible": True,
        "custom_models": True,
        "summary": "Enter the exact OpenRouter model id, for example openai/gpt-5.4-mini.",
        "pills": ["custom required", "router"],
    },
    "ollama": {
        "title": "Ollama",
        "api_key": "not required",
        "endpoint": "http://localhost:11434/v1",
        "endpoint_placeholder": "http://localhost:11434/v1",
        "backend_visible": True,
        "custom_models": True,
        "summary": "Use a local model already pulled into Ollama or another local compatible server.",
        "pills": ["local", "custom allowed"],
    },
    "azure": {
        "title": "Azure OpenAI",
        "api_key": "AZURE_OPENAI_API_KEY",
        "endpoint": "https://<resource>.openai.azure.com/",
        "endpoint_placeholder": "https://<resource>.openai.azure.com/",
        "backend_visible": True,
        "custom_models": True,
        "summary": "Enter your Azure deployment name as the model id and the Azure endpoint as backend URL.",
        "pills": ["deployment name", "endpoint required"],
    },
}


def get_llm_provider_options() -> List[Dict[str, str]]:
    return LLM_PROVIDER_OPTIONS.copy()


def get_provider_ui_metadata(provider: str) -> Dict[str, Any]:
    provider_key = (provider or "openai").lower()
    return deepcopy(PROVIDER_UI_METADATA.get(provider_key, PROVIDER_UI_METADATA["openai"]))


def provider_supports_custom_model(provider: str) -> bool:
    return bool(get_provider_ui_metadata(provider).get("custom_models"))


def get_model_options_for_provider(provider: str, role: str) -> List[Dict[str, str]]:
    provider_key = (provider or "openai").lower()
    if provider_key == "openai":
        return get_openai_model_options(role)
    if provider_key == "local_openai":
        return get_openai_model_options(role) + [{"label": "Custom local model ID", "value": "custom"}]
    options = [
        {"label": label, "value": value}
        for label, value in get_provider_catalog_options(provider_key, role)
    ]
    if provider_supports_custom_model(provider_key) and not any(option["value"] == "custom" for option in options):
        options.append({"label": "Custom model ID", "value": "custom"})
    return options


def get_default_model_for_provider(provider: str, role: str) -> str:
    provider_key = (provider or "openai").lower()
    if provider_key in ("openai", "local_openai"):
        return "gpt-5.4-nano" if role == "quick" else "gpt-5.4-mini"
    options = get_model_options_for_provider(provider, role)
    if options:
        return options[0]["value"]
    return "gpt-5.4-nano" if role == "quick" else "gpt-5.4-mini"


def resolve_model_choice(model_choice: str, custom_model: Optional[str] = None) -> Optional[str]:
    """Resolve UI model selection into the actual runtime model id."""
    choice = str(model_choice or "").strip()
    if choice != "custom":
        return choice or None
    custom = str(custom_model or "").strip()
    return custom or None


def is_responses_model(model_name: str) -> bool:
    """Whether core chat calls should use the Responses API wrapper."""
    return bool(get_model_spec(model_name).get("api") == "responses")


def is_reasoning_model(model_name: str) -> bool:
    return bool(get_model_spec(model_name).get("supports_reasoning_effort"))


def get_default_model_params(model_name: str, role: str) -> Dict[str, Any]:
    spec = get_model_spec(model_name)
    role_key = (role or "quick").lower()
    defaults = spec.get("role_defaults", {}).get(role_key)
    if defaults is None:
        defaults = next(iter(spec.get("role_defaults", {}).values()), {})
    return deepcopy(defaults)


def _coerce_bool(value: Any) -> Optional[bool]:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _coerce_float(value: Any, *, minimum: float, maximum: float) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return min(max(number, minimum), maximum)


def _coerce_int(value: Any, *, minimum: int = 1) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number >= minimum else None


def normalize_model_params(
    model_name: str,
    raw_params: Optional[Mapping[str, Any]] = None,
    role: str = "quick",
) -> Dict[str, Any]:
    """Keep only params supported by the selected model and fill defaults."""
    spec = get_model_spec(model_name)
    merged: Dict[str, Any] = get_default_model_params(model_name, role)
    if raw_params:
        merged.update({k: v for k, v in raw_params.items() if v is not None})

    normalized: Dict[str, Any] = {}

    if spec.get("supports_reasoning_effort"):
        options = spec.get("reasoning_effort_options", [])
        value = str(
            merged.get("reasoning_effort")
            or merged.get("effort")
            or (options[0] if options else "")
        ).lower()
        if value not in options and options:
            value = str(get_default_model_params(model_name, role).get("reasoning_effort", options[0]))
        if value in options:
            normalized["reasoning_effort"] = value

    if spec.get("supports_text_verbosity"):
        options = spec.get("text_verbosity_options", TEXT_VERBOSITY_OPTIONS)
        value = str(
            merged.get("text_verbosity")
            or merged.get("verbosity")
            or "medium"
        ).lower()
        if value not in options:
            value = str(get_default_model_params(model_name, role).get("text_verbosity", "medium"))
        if value in options:
            normalized["text_verbosity"] = value
            normalized["verbosity"] = value

    if spec.get("supports_reasoning_summary"):
        value = str(
            merged.get("reasoning_summary")
            or merged.get("summary")
            or "auto"
        ).lower()
        if value not in REASONING_SUMMARY_OPTIONS:
            value = "auto"
        normalized["reasoning_summary"] = value
        normalized["summary"] = value

    if spec.get("supports_temperature"):
        value = _coerce_float(merged.get("temperature"), minimum=0.0, maximum=2.0)
        if value is not None:
            normalized["temperature"] = value

    if spec.get("supports_top_p"):
        value = _coerce_float(merged.get("top_p"), minimum=0.0, maximum=1.0)
        if value is not None:
            normalized["top_p"] = value

    if spec.get("supports_max_output_tokens"):
        value = _coerce_int(
            merged.get("max_output_tokens") or merged.get("max_tokens"),
            minimum=1,
        )
        if value is not None:
            normalized["max_output_tokens"] = value

    if spec.get("supports_store"):
        value = _coerce_bool(merged.get("store"))
        if value is not None:
            normalized["store"] = value

    if spec.get("supports_parallel_tool_calls"):
        value = _coerce_bool(merged.get("parallel_tool_calls"))
        if value is not None:
            normalized["parallel_tool_calls"] = value

    return normalized


def apply_responses_model_params(
    api_params: Dict[str, Any],
    model_name: str,
    raw_params: Optional[Mapping[str, Any]] = None,
    role: str = "quick",
) -> Dict[str, Any]:
    """Apply normalized model params to a Responses API request payload."""
    params = normalize_model_params(model_name, raw_params, role=role)
    spec = get_model_spec(model_name)

    if spec.get("supports_text_verbosity"):
        text_config = api_params.setdefault("text", {"format": {"type": "text"}})
        text_config.setdefault("format", {"type": "text"})
        verbosity = params.get("text_verbosity") or params.get("verbosity")
        if verbosity:
            text_config["verbosity"] = verbosity

    reasoning: Dict[str, Any] = {}
    if spec.get("supports_reasoning_effort") and params.get("reasoning_effort"):
        reasoning["effort"] = params["reasoning_effort"]

    summary = params.get("reasoning_summary") or params.get("summary")
    if spec.get("supports_reasoning_summary") and summary and summary != "none":
        reasoning["summary"] = summary

    if reasoning:
        api_params["reasoning"] = reasoning

    if "temperature" in params:
        api_params["temperature"] = params["temperature"]
    if "top_p" in params:
        api_params["top_p"] = params["top_p"]
    if "max_output_tokens" in params:
        api_params["max_output_tokens"] = params["max_output_tokens"]
    if "store" in params:
        api_params["store"] = params["store"]
    if "parallel_tool_calls" in params:
        api_params["parallel_tool_calls"] = params["parallel_tool_calls"]

    return api_params


def describe_model_params(model_name: str, params: Optional[Mapping[str, Any]] = None, role: str = "quick") -> str:
    normalized = normalize_model_params(model_name, params, role=role)
    if not normalized:
        return "default params"

    display_order = [
        "reasoning_effort",
        "text_verbosity",
        "reasoning_summary",
        "temperature",
        "top_p",
        "max_output_tokens",
        "store",
        "parallel_tool_calls",
    ]
    labels = {
        "reasoning_effort": "effort",
        "text_verbosity": "verbosity",
        "reasoning_summary": "summary",
        "max_output_tokens": "max_output",
        "parallel_tool_calls": "parallel_tools",
    }
    parts = []
    for key in display_order:
        if key in normalized:
            parts.append(f"{labels.get(key, key)}={normalized[key]}")
    return ", ".join(parts) if parts else "default params"


def get_ui_control_state(model_name: str, role: str, provider: str = "openai") -> Dict[str, Any]:
    """Return model metadata plus supported UI defaults for Dash callbacks."""
    provider_key = (provider or "openai").lower()
    if provider_key not in ("openai", "local_openai"):
        option_label = next(
            (
                label
                for label, value in get_provider_catalog_options(provider_key, role)
                if value == model_name
            ),
            model_name,
        )
        spec = {
            "id": model_name,
            "label": option_label,
            "description": f"{provider_key} model selected. Provider-specific advanced settings are applied from runtime config.",
            "price_hint": provider_key,
            "supports_reasoning_effort": False,
            "reasoning_effort_options": [],
            "supports_text_verbosity": False,
            "text_verbosity_options": [],
            "supports_reasoning_summary": False,
            "reasoning_summary_options": [],
            "supports_temperature": False,
            "supports_top_p": False,
            "supports_max_output_tokens": False,
            "supports_store": False,
            "supports_parallel_tool_calls": False,
        }
        return {
            "spec": spec,
            "defaults": {},
            "description": spec["description"],
            "price_hint": provider_key,
        }

    spec = get_model_spec(model_name)
    defaults = normalize_model_params(model_name, None, role=role)
    return {
        "spec": spec,
        "defaults": defaults,
        "description": spec.get("description", ""),
        "price_hint": spec.get("price_hint", ""),
    }
