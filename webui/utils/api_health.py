"""Read-only API diagnostics with bounded requests and no provider error leakage."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import os
from urllib.parse import urlsplit

import requests


LLM_KEYS = {"qwen": "dashscope", "glm": "zhipu", "azure": "azure-openai"}
LLM_ENDPOINTS = {
    "openai": "https://api.openai.com/v1",
    "local_openai": "http://localhost:1234/v1",
    "google": "https://generativelanguage.googleapis.com/v1beta",
    "anthropic": "https://api.anthropic.com/v1",
    "xai": "https://api.x.ai/v1",
    "minimax": "https://api.minimax.io/v1",
    "deepseek": "https://api.deepseek.com",
    "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "glm": "https://open.bigmodel.cn/api/paas/v4",
    "openrouter": "https://openrouter.ai/api/v1",
    "ollama": "http://localhost:11434/v1",
}
STATUS_LABELS = {"connected": "Connected", "missing": "Not configured", "invalid": "Key rejected",
                 "permission": "Access denied", "limited": "Rate or credit limit", "unavailable": "Unavailable",
                 "configuration": "Check settings", "unverified": "Needs a model test"}


def result(service, status, message, action=""):
    return {"service": service, "status": status, "label": STATUS_LABELS[status],
            "message": message, "action": action}


def valid_key(key):
    return isinstance(key, str) and bool(key.strip()) and not key.strip().lower().startswith("your_")


def safe_endpoint(endpoint):
    parsed = urlsplit(endpoint)
    if (not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.scheme not in {"http", "https"}):
        raise ValueError("Invalid endpoint")
    if parsed.scheme == "http" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("Use HTTPS for remote endpoints")
    return endpoint.rstrip("/")


def probe(service, endpoint, *, headers=None, params=None, expected=None):
    """Never display raw exceptions, URLs, or response text that might echo keys."""
    try:
        safe_endpoint(endpoint)
        response = requests.get(endpoint, headers=headers, params=params, timeout=(4, 10), allow_redirects=False)
        code = response.status_code
        if code == 401:
            return result(service, "invalid", "The service rejected this key.", "Replace the key in API setup and save.")
        if code == 403:
            return result(service, "permission", "The key or account cannot access this service.",
                          "Check the key, subscription, region, and account permissions.")
        if code in {402, 429}:
            return result(service, "limited", "The account has a credit or request limit.",
                          "Check billing or wait before retrying; replace the provider if needed.")
        if 300 <= code < 400:
            return result(service, "configuration", "The endpoint redirected. Credentials were not forwarded.",
                          "Check the endpoint address.")
        if code in {404, 405}:
            return result(service, "unverified", "This endpoint does not support the connection check.",
                          "Check the endpoint, then test the selected models.")
        if code >= 500:
            return result(service, "unavailable", f"The service returned HTTP {code}.", "Retry later or choose another provider.")
        try:
            payload = response.json()
        except ValueError:
            return result(service, "unavailable", "The service returned an unexpected response.", "Check the endpoint and retry.")
        if code >= 400:
            return result(service, "configuration", f"The service rejected the check (HTTP {code}).",
                          "Check the API key and endpoint settings.")
        if not isinstance(payload, dict):
            return result(service, "unavailable", "The response format was not recognized.", "Check the endpoint.")
        if payload.get("error") or payload.get("Error Message") or payload.get("Response") == "Error":
            return result(service, "invalid", "The service reported an authentication or request error.",
                          "Check or replace this key, then retry.")
        if payload.get("Note") or payload.get("Information"):
            return result(service, "limited", "The service reported a usage or subscription limit.",
                          "Check your plan or replace the provider.")
        if expected and not expected(payload):
            return result(service, "unavailable", "The service did not return the expected data.", "Check access and retry.")
        return result(service, "connected", "A read-only API request succeeded.")
    except requests.exceptions.Timeout:
        return result(service, "unavailable", "The service did not respond in time.", "Check your internet connection or retry later.")
    except requests.exceptions.SSLError:
        return result(service, "unavailable", "A secure connection could not be established.", "Check the system clock and network certificate settings.")
    except (requests.exceptions.RequestException, UnicodeError):
        return result(service, "unavailable", "The connection failed.", "Check the network, key format, and endpoint; then retry.")
    except (ValueError, TypeError):
        return result(service, "configuration", "The endpoint or key format is invalid.", "Use an HTTPS endpoint or a local server address.")


def resolved_keys(stored):
    from webui.components.api_config_modal import API_CONFIGS
    from tradingagents.dataflows.config import get_api_key, get_alpaca_use_paper

    values = {}
    for api in API_CONFIGS:
        value = (stored or {}).get(api["id"])
        values[api["id"]] = value if valid_key(value) else get_api_key(api["env_var"].lower(), api["env_var"])
    values["alpaca-paper"] = (stored or {}).get("alpaca-paper", str(get_alpaca_use_paper()).lower() != "false")
    return values


def check_connections(stored=None, provider="openai", backend_url=""):
    """Check configured services plus the selected LLM and essential data sources."""
    keys = resolved_keys(stored)
    jobs = []
    missing = []
    selected = provider or "openai"
    for llm in [*LLM_ENDPOINTS, "azure"]:
        key_id = LLM_KEYS.get(llm, llm)
        key = keys.get(key_id)
        local = llm in {"local_openai", "ollama"}
        if local and llm != selected:
            continue
        if not local and not valid_key(key):
            if llm == selected:
                missing.append(result(llm, "missing", "A key is required for your selected model provider.",
                                      "Open API setup, add a key, and save; or select another provider."))
            continue
        endpoint = (backend_url.strip() if llm == selected and backend_url else "")
        endpoint = endpoint or (os.getenv("AZURE_OPENAI_ENDPOINT", "") if llm == "azure" else LLM_ENDPOINTS[llm])
        headers = {"Authorization": f"Bearer {key.strip()}"} if valid_key(key) else {}
        if llm == "google":
            headers = {"x-goog-api-key": key.strip()}
        elif llm == "anthropic":
            headers = {"x-api-key": key.strip(), "anthropic-version": "2023-06-01"}
        elif llm == "azure":
            missing.append(result("azure", "unverified", "Azure requires a deployment-specific model test.",
                                  "Set the endpoint and deployment, then test the selected models."))
            continue
        # OpenRouter's public model catalog does not authenticate the supplied key.
        path = "/key" if llm == "openrouter" else "/models"
        expected = (lambda data: "data" in data) if llm != "google" else (lambda data: "models" in data)
        jobs.append((llm, endpoint.rstrip("/") + path, headers, None, expected))

    alpaca_key, secret = keys.get("alpaca-key"), keys.get("alpaca-secret")
    if valid_key(alpaca_key) and valid_key(secret):
        paper = keys["alpaca-paper"] not in (False, "False", "false", "0")
        endpoint = "https://paper-api.alpaca.markets" if paper else "https://api.alpaca.markets"
        jobs.append(("Alpaca paper" if paper else "Alpaca live", endpoint + "/v2/account",
                     {"APCA-API-KEY-ID": alpaca_key.strip(), "APCA-API-SECRET-KEY": secret.strip()}, None,
                     lambda data: "status" in data and "id" in data))
    else:
        missing.append(result("Alpaca", "missing", "Both Alpaca keys are required for account access and trading.",
                              "Add an Alpaca key and secret in API setup. Start with paper trading."))
    definitions = [
        ("finnhub", "Finnhub", "https://finnhub.io/api/v1/quote", lambda key: {"X-Finnhub-Token": key},
         lambda key: {"symbol": "SPY"}, lambda data: "c" in data),
        ("fred", "FRED", "https://api.stlouisfed.org/fred/series", lambda key: {},
         lambda key: {"series_id": "FEDFUNDS", "file_type": "json", "api_key": key}, lambda data: "seriess" in data),
        ("coindesk", "CryptoCompare", "https://min-api.cryptocompare.com/data/v2/news/",
         lambda key: {"Authorization": "Apikey " + key}, lambda key: {"lang": "EN"}, lambda data: "Data" in data),
        ("alpha-vantage", "Alpha Vantage", "https://www.alphavantage.co/query", lambda key: {},
         lambda key: {"function": "GLOBAL_QUOTE", "symbol": "SPY", "apikey": key}, lambda data: bool(data.get("Global Quote"))),
        ("fxmacrodata", "FXMacroData", "https://api.fxmacrodata.com/v1/calendar/usd", lambda key: {"X-API-Key": key},
         lambda key: {"start_date": datetime.now(timezone.utc).date().isoformat(),
                      "end_date": datetime.now(timezone.utc).date().isoformat()}, lambda data: "data" in data),
    ]
    for key_id, name, endpoint, headers, params, expected in definitions:
        key = keys.get(key_id)
        if valid_key(key):
            jobs.append((name, endpoint, headers(key.strip()), params(key.strip()), expected))
        elif key_id in {"finnhub", "fred"}:
            missing.append(result(name, "missing", "This data source has no key configured.",
                                  "Add a key in API setup to enable its news or economic data."))
    def run(job):
        name, endpoint, headers, params, expected = job
        return probe(name, endpoint, headers=headers, params=params, expected=expected)
    with ThreadPoolExecutor(max_workers=6) as pool:
        checked = list(pool.map(run, jobs))
    return {"checked_at": datetime.now(timezone.utc).isoformat(), "results": checked + missing}


def check_selected_models(stored, provider, backend_url, models):
    from tradingagents.llm_clients import create_llm_client

    provider = provider or "openai"
    keys = resolved_keys(stored)
    key = keys.get(LLM_KEYS.get(provider, provider))
    local = provider in {"local_openai", "ollama"}

    def run(model):
        if not model:
            return result("AI model", "configuration", "No model was selected.", "Choose a model before testing.")
        if not local and not valid_key(key):
            return result(model, "missing", "The selected provider needs an API key.", "Add a key in API setup.")
        try:
            endpoint = safe_endpoint(backend_url.strip()) if backend_url else None
            if local and not endpoint:
                endpoint = LLM_ENDPOINTS[provider]
            llm = create_llm_client(provider, model, base_url=endpoint, api_key=key or "local-llm",
                                    timeout=15, max_retries=0, max_tokens=256).get_llm()
            for field in ("max_output_tokens", "max_tokens"):
                if field in type(llm).model_fields:
                    setattr(llm, field, 256)
                    break
            # The custom Responses wrapper holds its own SDK client.
            if hasattr(getattr(llm, "_client", None), "with_options"):
                llm._client = llm._client.with_options(timeout=15, max_retries=0)
            reply = llm.invoke("Reply with the single word OK. Do not use tools.")
            if not getattr(reply, "content", ""):
                return result(model, "unverified", "The model returned no text within the test token budget.",
                              "Check model settings or choose another model.")
            return result(model, "connected", "The selected model generated a reply successfully.")
        except Exception as error:
            code = getattr(error, "status_code", None)
            status = {401: "invalid", 403: "permission", 402: "limited", 429: "limited", 404: "configuration"}.get(code, "unavailable")
            return result(model, status, "The model request failed" + (f" (HTTP {code})." if isinstance(code, int) else "."),
                          "Check the key, model ID, endpoint, and available credits; replace the provider if needed.")
    unique_models = list(dict.fromkeys(models))
    with ThreadPoolExecutor(max_workers=2) as pool:
        checked = list(pool.map(run, unique_models))
    return {"checked_at": datetime.now(timezone.utc).isoformat(), "results": checked}
