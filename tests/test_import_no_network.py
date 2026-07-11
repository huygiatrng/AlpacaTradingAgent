"""Importing tradingagents packages must never open a network connection.

Module-level code that calls out to Alpaca (or any other remote API) makes the
package unusable offline, slows every import, and leaks "unauthorized" errors
into unrelated tooling (tests, CLI help, docs builds).  These tests run each
import in a fresh subprocess with ``socket.socket.connect`` instrumented, and
fail if any connection is attempted.
"""

import json
import subprocess
import sys
import textwrap
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_PROBE_TEMPLATE = textwrap.dedent(
    """
    import json
    import socket
    import sys
    import traceback

    attempts = []

    def _spy_connect(self, addr):
        attempts.append(
            {{"addr": str(addr), "stack": traceback.format_stack(limit=15)}}
        )
        raise OSError("network blocked by test_import_no_network")

    socket.socket.connect = _spy_connect

    error = None
    try:
        {imports}
    except Exception as exc:  # noqa: BLE001 - report any import failure
        error = f"{{type(exc).__name__}}: {{exc}}"

    print(json.dumps({{"attempts": attempts, "error": error}}))
    """
)


def _probe_imports(import_lines):
    """Run the given import statements in a clean subprocess; return report."""
    code = _PROBE_TEMPLATE.format(
        imports="\n    ".join(import_lines)
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        timeout=300,
    )
    stdout_lines = [line for line in result.stdout.strip().splitlines() if line]
    assert stdout_lines, (
        f"probe subprocess produced no report; stderr:\n{result.stderr[-2000:]}"
    )
    return json.loads(stdout_lines[-1])


def _assert_no_network(report):
    attempts = report["attempts"]
    if attempts:
        first = attempts[0]
        stack = "".join(first["stack"][-8:])
        raise AssertionError(
            f"import attempted {len(attempts)} network connection(s); "
            f"first to {first['addr']} via:\n{stack}"
        )


def test_importing_agents_package_opens_no_network_connection():
    report = _probe_imports(["import tradingagents.agents"])
    assert report["error"] is None, f"import failed: {report['error']}"
    _assert_no_network(report)


def test_importing_dataflows_package_opens_no_network_connection():
    # NOTE: on some branches importing dataflows first trips a known circular
    # import; importing agents first is the historically safe order and is
    # what the analyst modules do in production.
    report = _probe_imports(
        ["import tradingagents.agents", "import tradingagents.dataflows"]
    )
    assert report["error"] is None, f"import failed: {report['error']}"
    _assert_no_network(report)


def test_importing_prompt_capture_does_not_build_dash_app():
    """webui.utils.prompt_capture is imported by every analyst module; pulling
    it in must not construct the Dash app (which renders account tables and
    calls Alpaca)."""
    report = _probe_imports(
        [
            "import webui.utils.prompt_capture",
            "import sys",
            "assert 'webui.app_dash' not in sys.modules, 'webui import eagerly builds the Dash app'",
        ]
    )
    assert report["error"] is None, f"import failed: {report['error']}"
    _assert_no_network(report)
