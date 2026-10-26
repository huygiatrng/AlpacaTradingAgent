"""Credential handling and real HTTP verification of the maintained app collection."""

from pathlib import Path
import shutil
from threading import Thread
from types import SimpleNamespace
from urllib.parse import quote, quote_plus

import pytest
import yaml

from scripts import check_apis


def test_shared_collections_remain_read_only_and_credentials_empty():
    for path in (check_apis.POSTMAN / "collections").rglob("*.request.yaml"):
        request = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert request["method"] == "GET", path
        assert request["settings"]["followRedirects"] is False, path
    environment = yaml.safe_load((check_apis.POSTMAN / "environments/Local.environment.yaml").read_text(encoding="utf-8"))
    assert all(value["value"] == "" for value in environment["values"] if value.get("type") == "secret")


def test_redaction_handles_raw_and_url_encoded_secrets():
    secret = "test-secret/with spaces+?"
    output = " ".join([secret, quote(secret, safe=""), quote_plus(secret)])
    assert check_apis.redact(output, [secret]) == "[REDACTED] [REDACTED] [REDACTED]"


def test_only_requested_provider_credentials_are_required():
    selected, skipped = check_apis.choose_services({"GOOGLE_API_KEY": "test"}, "google", "ai")
    assert selected == ["AI"] and skipped == []
    with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
        check_apis.choose_services({"OPENAI_API_KEY": "test"}, "google", "ai")


def test_local_provider_needs_no_remote_openai_key():
    assert check_apis.choose_services({}, "ollama", "ai") == (["AI"], [])


def test_empty_run_and_partial_alpaca_credentials_are_errors():
    with pytest.raises(ValueError, match="No configured services"):
        check_apis.choose_services({}, "openai")
    with pytest.raises(ValueError, match="ALPACA_SECRET_KEY"):
        check_apis.choose_services({"ALPACA_API_KEY": "test"}, "openai")


def test_credential_loading_ignores_unrelated_secrets_and_expansion(monkeypatch, tmp_path):
    monkeypatch.setenv("FRED_API_KEY", "process-key")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    path = tmp_path / ".env"
    path.write_text("FRED_API_KEY=file-key\nGOOGLE_API_KEY=${DO_NOT_EXPAND}\nUNRELATED_PASSWORD=private\n")
    keys = check_apis.load_credentials(path)
    assert keys["FRED_API_KEY"] == "process-key"
    assert keys["GOOGLE_API_KEY"] == "${DO_NOT_EXPAND}"
    assert "UNRELATED_PASSWORD" not in keys


def test_failure_exit_and_secret_cleanup(monkeypatch, tmp_path, capsys):
    secret = "unique-test-secret/with+spaces"
    monkeypatch.setattr(check_apis, "POSTMAN", tmp_path)
    monkeypatch.setattr(check_apis.shutil, "which", lambda _: "postman")
    monkeypatch.setattr(check_apis, "load_credentials", lambda _: {"FRED_API_KEY": secret})
    environments = []

    def execute(command, **kwargs):
        assert secret not in " ".join(command)
        assert "--no-report-events" in command and "--ignore-redirects" in command
        path = Path(command[command.index("-e") + 1])
        assert secret in path.read_text()
        environments.append(path)
        return SimpleNamespace(returncode=1, stdout=f"GET ?api_key={quote(secret, safe='')}\nfailed assertion", stderr="")

    monkeypatch.setattr(check_apis.subprocess, "run", execute)
    args = SimpleNamespace(suite="providers", env_file=None, provider="openai", only="fred", app_url="http://127.0.0.1:7860", ai_base_url="")
    assert check_apis.run(args) == 1
    assert all(not path.exists() for path in environments)
    report = (tmp_path / "reports/latest-providers.txt").read_text()
    assert "[REDACTED]" in report and secret not in report
    assert secret not in capsys.readouterr().out


@pytest.mark.skipif(not shutil.which("postman"), reason="Postman CLI is installed in the dedicated API CI job")
def test_app_collection_against_running_dashboard(monkeypatch, tmp_path):
    from dash import html
    from werkzeug.serving import make_server
    import webui.layout
    from webui.app_dash import create_app

    # Account reads are outside this collection's contract and covered by Providers.
    monkeypatch.setattr(webui.layout, "render_alpaca_account_section", lambda: html.Div())
    app = create_app()
    server = make_server("127.0.0.1", 0, app.server, threaded=True)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = check_apis.subprocess.run([
            shutil.which("postman"), "collection", "run", str(check_apis.POSTMAN / "collections/App"),
            "--env-var", f"app_url=http://127.0.0.1:{server.server_port}",
            "--no-report-events", "--ignore-redirects", "--timeout-request", "10000", "--timeout", "30000",
        ], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=45)
        assert result.returncode == 0, result.stdout + result.stderr
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
