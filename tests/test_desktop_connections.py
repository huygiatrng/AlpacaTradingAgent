"""Desktop lifecycle and diagnostics must work without real credentials."""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest
import requests

import desktop_launcher as desktop
from webui.utils import api_health as health


@pytest.mark.parametrize("code,status", [(200, "connected"), (301, "configuration"), (307, "configuration"),
                                         (400, "configuration"), (401, "invalid"), (403, "permission"),
                                         (402, "limited"), (429, "limited"), (500, "unavailable"),
                                         (404, "unverified")])
def test_real_http_responses_never_echo_credentials(code, status):
    secret = "SECRET-test-credential"

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(code)
            self.send_header("Location", "/should-not-follow")
            self.end_headers()
            self.wfile.write(json.dumps({"data": [], "detail": secret}).encode())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        report = health.probe("service", f"http://127.0.0.1:{server.server_port}/models",
                              headers={"Authorization": "Bearer " + secret}, expected=lambda data: "data" in data)
        assert report["status"] == status
        assert secret not in json.dumps(report)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


@pytest.mark.parametrize("payload,status", [({"Information": "credit limit"}, "limited"),
                                           ({"error": "echo-SECRET"}, "invalid"),
                                           ({"Response": "Error"}, "invalid"),
                                           ([], "unavailable"), ({"unexpected": True}, "unavailable")])
def test_provider_errors_inside_success_response(monkeypatch, payload, status):
    monkeypatch.setattr(health.requests, "get", lambda *a, **kw: SimpleNamespace(status_code=200, json=lambda: payload))
    report = health.probe("provider", "https://example.com/models", expected=lambda data: "data" in data)
    assert report["status"] == status
    assert "SECRET" not in json.dumps(report)


@pytest.mark.parametrize("error,status", [(requests.Timeout("SECRET"), "unavailable"),
                                         (requests.exceptions.InvalidHeader("SECRET"), "unavailable"),
                                         (requests.exceptions.SSLError("SECRET"), "unavailable"),
                                         (ValueError("SECRET"), "configuration")])
def test_transport_failures_are_actionable_and_secret_free(monkeypatch, error, status):
    def fail(*a, **kw):
        raise error
    monkeypatch.setattr(health.requests, "get", fail)
    report = health.probe("provider", "https://example.com/models")
    assert report["status"] == status
    assert report["action"]
    assert "SECRET" not in json.dumps(report)


@pytest.mark.parametrize("url", ["http://remote.example/v1", "https://user:password@example.com/v1",
                                 "https://example.com/v1?key=secret", "file:///tmp/key", "not a URL"])
def test_unsafe_endpoint_is_rejected_before_request(monkeypatch, url):
    def unexpected(*a, **kw):
        pytest.fail("No network request should be made")
    monkeypatch.setattr(health.requests, "get", unexpected)
    assert health.probe("provider", url)["status"] == "configuration"


def test_only_configured_providers_are_checked_and_alpaca_mode_is_respected(monkeypatch):
    monkeypatch.setattr(health, "resolved_keys", lambda _: {"openrouter": "router-secret", "alpaca-key": "key",
                                                            "alpaca-secret": "secret", "alpaca-paper": False})
    calls = []
    def probe(name, endpoint, **kwargs):
        calls.append((name, endpoint, kwargs))
        return health.result(name, "connected", "OK")
    monkeypatch.setattr(health, "probe", probe)
    report = health.check_connections(provider="openrouter")
    assert len(calls) == 2
    assert any(endpoint == "https://openrouter.ai/api/v1/key" for _, endpoint, _ in calls)
    assert any(endpoint == "https://api.alpaca.markets/v2/account" for _, endpoint, _ in calls)
    assert "router-secret" not in json.dumps(report)
    assert "openai" not in [item["service"] for item in report["results"]]


@pytest.mark.parametrize("provider,key_id", [("google", "google"), ("anthropic", "anthropic"),
                                            ("deepseek", "deepseek"), ("qwen", "dashscope"),
                                            ("glm", "zhipu"), ("local_openai", "openai"), ("ollama", None)])
def test_ai_diagnostics_follow_only_the_selected_provider(monkeypatch, provider, key_id):
    keys = {"openai": "unused-openai-key", "anthropic": "unused-claude-key"}
    if key_id:
        keys[key_id] = "selected-key"
    monkeypatch.setattr(health, "resolved_keys", lambda _: keys)
    calls = []
    def probe(service, endpoint, **kwargs):
        calls.append((service, endpoint))
        return health.result(service, "connected", "OK")
    monkeypatch.setattr(health, "probe", probe)
    report = health.check_connections(provider=provider)
    assert len(calls) == 1
    assert calls[0][1].startswith(health.LLM_ENDPOINTS[provider])
    assert calls[0][0].startswith("AI · ")
    assert all(item["status"] in {"connected", "optional"} for item in report["results"])


def test_unselected_openai_key_does_not_override_missing_gemini_key(monkeypatch):
    monkeypatch.setattr(health, "resolved_keys", lambda _: {"openai": "unused-key"})
    report = health.check_connections(provider="google")
    missing = [item for item in report["results"] if item["status"] == "missing"]
    assert len(missing) == 1
    assert "Gemini" in missing[0]["service"]
    assert not any("OpenAI" in item["service"] for item in report["results"])


def test_partial_alpaca_setup_is_reported_as_incomplete(monkeypatch):
    monkeypatch.setattr(health, "resolved_keys", lambda _: {"alpaca-key": "key-only"})
    report = health.check_connections(provider="google")
    alpaca = next(item for item in report["results"] if item["service"] == "Alpaca")
    assert alpaca["status"] == "configuration"


def test_model_test_deduplicates_and_bounds_output(monkeypatch):
    from tradingagents import llm_clients
    monkeypatch.setattr(health, "resolved_keys", lambda _: {"openai": "SECRET"})
    calls = []
    class Model:
        model_fields = {"max_tokens": object()}
        max_tokens = None
        # Google clients do not offer OpenAI's with_options method.
        _client = object()
        def invoke(self, prompt):
            assert self.max_tokens == 256
            calls.append(prompt)
            return SimpleNamespace(content="OK")
    monkeypatch.setattr(llm_clients, "create_llm_client", lambda *a, **kw: SimpleNamespace(get_llm=lambda: Model()))
    report = health.check_selected_models({}, "openai", "", ["model", "model"])
    assert len(calls) == 1
    assert report["results"][0]["status"] == "connected"
    assert "SECRET" not in json.dumps(report)


def test_model_failure_does_not_display_provider_exception(monkeypatch):
    from tradingagents import llm_clients
    monkeypatch.setattr(health, "resolved_keys", lambda _: {"openai": "SECRET"})
    def fail(*a, **kw):
        error = RuntimeError("SECRET provider echo")
        error.status_code = 401
        raise error
    monkeypatch.setattr(llm_clients, "create_llm_client", fail)
    report = health.check_selected_models({}, "openai", "", ["model"])
    assert report["results"][0]["status"] == "invalid"
    assert "SECRET" not in json.dumps(report)


def test_desktop_environment_uses_writable_home(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADINGAGENTS_DESKTOP_HOME", str(tmp_path))
    for variable in ("TRADINGAGENTS_RESULTS_DIR", "TRADINGAGENTS_CACHE_DIR", "TRADINGAGENTS_MEMORY_LOG_PATH", "TRADINGAGENTS_AGENT_MEMORY_DIR"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.chdir(Path(__file__).parent)
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "0")
    assert desktop.prepare_environment() == tmp_path
    assert Path.cwd() == tmp_path
    assert Path(health.os.environ["TRADINGAGENTS_RESULTS_DIR"]).is_relative_to(tmp_path)


def test_single_instance_lock_and_saved_port(tmp_path):
    first = desktop.acquire_instance(tmp_path)
    assert first is not None
    try:
        # Windows locks are exclusive even for separate handles in one process.
        if health.os.name == "nt":
            assert desktop.acquire_instance(tmp_path) is None
        (tmp_path / "desktop.json").write_text('{"port": 8345}')
        assert desktop.previous_port(tmp_path) == 8345
        (tmp_path / "desktop.json").write_text('{"port": -1}')
        assert desktop.previous_port(tmp_path) == 7860
        (tmp_path / "desktop.json").write_text('broken')
        assert desktop.previous_port(tmp_path) == 7860
    finally:
        first.close()


def test_diagnostic_callback_inputs_exist_in_layout(monkeypatch):
    from webui.app_dash import create_app
    from webui import layout
    from dash import html
    monkeypatch.setattr(layout, "render_alpaca_account_section", lambda: html.Div())
    app = create_app()
    ids = set()
    def walk(component):
        if isinstance(getattr(component, "id", None), str):
            ids.add(component.id)
        children = getattr(component, "children", [])
        for child in children if isinstance(children, list) else [children]:
            if hasattr(child, "to_plotly_json"):
                walk(child)
    walk(app.layout)
    for output, callback in app.callback_map.items():
        if any(name in output for name in ("connection-results", "model-test-status")):
            for item in callback["inputs"] + callback["state"]:
                assert item["id"] in ids
        if output == "connection-results.data":
            assert {"llm-provider", "backend-url"}.issubset({item["id"] for item in callback["inputs"]})
    client = app.server.test_client()
    assert client.get("/_dash-layout").status_code == 200
    assert client.get("/assets/app.ico").status_code == 200
