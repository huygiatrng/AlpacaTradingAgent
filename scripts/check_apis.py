"""Run the maintained Postman checks with temporary credentials and local reports."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from urllib.parse import quote, quote_plus, urlsplit

from dotenv import dotenv_values


ROOT = Path(__file__).resolve().parents[1]
POSTMAN = ROOT / "postman"
AI_KEYS = {
    "openai": "OPENAI_API_KEY", "google": "GOOGLE_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY", "xai": "XAI_API_KEY",
    "minimax": "MINIMAX_API_KEY", "deepseek": "DEEPSEEK_API_KEY",
    "qwen": "DASHSCOPE_API_KEY", "glm": "ZHIPU_API_KEY",
    "openrouter": "OPENROUTER_API_KEY", "azure": "AZURE_OPENAI_API_KEY",
    "local_openai": "LOCAL_LLM_API_KEY", "ollama": "LOCAL_LLM_API_KEY",
}
SERVICES = {
    "ai": ("AI", []),
    "alpaca": ("Alpaca", ["ALPACA_API_KEY", "ALPACA_SECRET_KEY"]),
    "finnhub": ("Finnhub", ["FINNHUB_API_KEY"]),
    "fred": ("FRED", ["FRED_API_KEY"]),
    "crypto": ("CryptoCompare", ["COINDESK_API_KEY"]),
    "fxmacro": ("FXMacroData", ["FXMACRODATA_API_KEY"]),
    "alpha": ("Alpha Vantage", ["ALPHA_VANTAGE_API_KEY"]),
}


def redact(text, secrets):
    for secret in sorted(set(filter(None, secrets)), key=len, reverse=True):
        for value in {secret, quote(secret, safe=""), quote_plus(secret)}:
            text = text.replace(value, "[REDACTED]")
    return re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)


def load_credentials(path=None):
    if path is None:
        desktop = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "AlpacaTradingAgent" / ".env"
        path = desktop if desktop.is_file() else ROOT / ".env"
    elif not path.is_file():
        raise ValueError("The specified environment file does not exist.")
    values = {**dotenv_values(path, interpolate=False), **os.environ}
    names = set(AI_KEYS.values()) | {key for _, keys in SERVICES.values() for key in keys}
    return {
        key: value.strip() for key in names
        if isinstance(value := values.get(key), str) and value.strip()
        and not value.strip().lower().startswith("your_")
    }


def choose_services(keys, provider, only=None):
    selected, skipped = [], []
    names = only.split(",") if only else list(SERVICES)
    for name in names:
        if name not in SERVICES:
            raise ValueError(f"Unknown service: {name}. Choose from {', '.join(SERVICES)}.")
        folder, required = SERVICES[name]
        if name == "ai":
            if provider == "azure":
                reason = "Azure requires the deployment-specific model test in the app."
                if only:
                    raise ValueError(reason)
                skipped.append(f"AI: {reason}")
                continue
            required = [] if provider in {"local_openai", "ollama"} else [AI_KEYS[provider]]
        missing = [key for key in required if not keys.get(key)]
        if missing:
            if only or (name == "alpaca" and len(missing) != len(required)):
                raise ValueError(f"{folder}: missing {', '.join(missing)}.")
            skipped.append(f"{folder}: not configured")
        elif folder not in selected:
            selected.append(folder)
    if not selected:
        raise ValueError("No configured services to test. Add keys in the app or choose a local AI provider.")
    return selected, skipped


def run(args):
    cli = shutil.which("postman")
    if not cli:
        raise ValueError("Postman CLI is missing. Install it with: npm install -g postman-cli@1.77.0")
    if args.suite == "lint":
        commands = [
            [cli, "collection", "lint", str(POSTMAN / "collections" / name), "--fail-severity", "warning"]
            for name in ("App", "Providers")
        ] + [[cli, "environment", "lint", str(POSTMAN / "environments/Local.environment.yaml"), "--fail-severity", "warning"]]
        codes = [subprocess.run(command, cwd=ROOT, check=False).returncode for command in commands]
        return int(any(codes))

    keys = load_credentials(args.env_file) if args.suite == "providers" else {}
    selected, skipped = choose_services(keys, args.provider, args.only) if args.suite == "providers" else ([], [])
    if args.suite == "app":
        endpoint = urlsplit(args.app_url)
        if endpoint.scheme != "http" or endpoint.hostname not in {"localhost", "127.0.0.1", "::1"} or endpoint.username or endpoint.password or endpoint.query or endpoint.fragment:
            raise ValueError("App checks require a local HTTP URL without credentials, query or fragment.")
    values = {**keys, "app_url": args.app_url.rstrip("/"), "ai_provider": args.provider, "ai_base_url": args.ai_base_url}
    collection = "App" if args.suite == "app" else "Providers"
    preface = [f"{collection} checks - {datetime.now(timezone.utc).isoformat()}"]
    if selected:
        preface.append("Selected: " + ", ".join(selected) + f" (AI provider: {args.provider})")
    preface.extend("SKIPPED " + item for item in skipped)
    print("\n".join(preface), flush=True)
    with tempfile.TemporaryDirectory(prefix="alpaca-postman-") as temp:
        environment = Path(temp) / "runtime.environment.yaml"
        environment.write_text(json.dumps({"name": "Runtime", "values": [
            {"key": key, "value": value, "enabled": True, "type": "secret" if key in keys else "default"}
            for key, value in values.items()
        ]}), encoding="utf-8")
        environment.chmod(0o600)
        command = [cli, "collection", "run", str(POSTMAN / "collections" / collection),
                   "-e", str(environment), "--no-report-events", "--ignore-redirects",
                   "--timeout-request", "15000", "--timeout", "180000"]
        for folder in selected:
            command.extend(["-i", folder])
        # No secret command arguments and no raw run exports containing account responses.
        try:
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=200)
            output = redact(result.stdout + result.stderr, keys.values())
            code = result.returncode
        except subprocess.TimeoutExpired:
            output, code = "Postman exceeded the overall timeout. No raw output was retained.\n", 1
    print(output)
    reports = POSTMAN / "reports"
    reports.mkdir(exist_ok=True)
    report = reports / f"latest-{args.suite}.txt"
    report.write_text("\n".join(preface) + "\n" + output + f"\nExit code: {code}\n", encoding="utf-8")
    print(f"Local report: {report}")
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=["lint", "app", "providers"])
    parser.add_argument("--app-url", default="http://127.0.0.1:7860")
    parser.add_argument("--provider", choices=sorted(AI_KEYS), default="openai")
    parser.add_argument("--ai-base-url", default="", help="Optional endpoint for the selected AI provider")
    parser.add_argument("--env-file", type=Path, help="Explicit dotenv file; otherwise use the desktop app's file, then the repo file")
    parser.add_argument("--only", help="Comma-separated services: " + ", ".join(SERVICES))
    args = parser.parse_args()
    try:
        return run(args)
    except (ValueError, OSError) as error:
        print(f"API checks could not start: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
