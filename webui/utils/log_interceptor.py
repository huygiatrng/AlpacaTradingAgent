"""
log_interceptor.py - Capture stdout log lines tagged with [TAG] patterns

Wraps sys.stdout to intercept lines matching the pattern used throughout the
trading agent codebase (e.g. [RISK MANAGER], [STATE - NVDA], [PRICE VALIDATION]).
Parsed entries are appended to app_state.system_logs for display in the debug panel.
"""

import sys
import re
import threading
import datetime

_TAG_RE = re.compile(r"^\[([^\]]+)\]\s*(.*)", re.DOTALL)
_TAG_SYMBOL_RE = re.compile(r"^([A-Z][A-Z0-9 ]+?)\s*-\s*([A-Z]{1,8}(?:/[A-Z]{1,8})?)$")

_MAX_LOGS = 1000
_TRIM_TO = 800

# Per-thread buffer — prevents writes from different threads interleaving
_thread_local = threading.local()


class LogInterceptor:
    """
    sys.stdout wrapper that parses [TAG] prefixed lines and stores structured
    log entries in app_state.system_logs while passing all output to real stdout.
    """

    def __init__(self, real_stdout, app_state):
        self._real_stdout = real_stdout
        self._app_state = app_state
        self._lock = threading.Lock()
        # Buffer is now per-thread (see write()); no shared self._buffer

    # ── stdout interface ────────────────────────────────────────────────────

    def write(self, text):
        self._real_stdout.write(text)
        # Use a per-thread buffer so concurrent print() calls from different
        # threads can never interleave their two write() calls (text + "\n").
        if not hasattr(_thread_local, "buffer"):
            _thread_local.buffer = ""
        _thread_local.buffer += text
        while "\n" in _thread_local.buffer:
            line, _thread_local.buffer = _thread_local.buffer.split("\n", 1)
            self._parse_line(line)

    def flush(self):
        self._real_stdout.flush()

    def fileno(self):
        return self._real_stdout.fileno()

    def isatty(self):
        return getattr(self._real_stdout, "isatty", lambda: False)()

    # ── parsing ─────────────────────────────────────────────────────────────

    def _parse_line(self, line):
        line = line.rstrip()
        if not line:
            return

        m = _TAG_RE.match(line)
        if not m:
            return

        raw_tag = m.group(1).strip()
        message = m.group(2).strip()

        # Try to extract embedded symbol from tag like "STATE - NVDA"
        sm = _TAG_SYMBOL_RE.match(raw_tag)
        if sm:
            tag = sm.group(1).strip()
            symbol = sm.group(2).strip()
        else:
            tag = raw_tag
            # Fall back to the symbol currently being analyzed
            symbol = getattr(self._app_state, "analyzing_symbol", None) or ""

        entry = {
            "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
            "symbol": symbol,
            "tag": tag,
            "message": message,
        }

        with self._lock:
            logs = self._app_state.system_logs
            logs.append(entry)
            if len(logs) > _MAX_LOGS:
                # Trim oldest entries
                self._app_state.system_logs = logs[-_TRIM_TO:]


def install(app_state):
    """
    Install the LogInterceptor into sys.stdout (idempotent).

    Args:
        app_state: AppState instance that has a system_logs list attribute.
    """
    if isinstance(sys.stdout, LogInterceptor):
        # Already installed — update the app_state reference in case it changed
        sys.stdout._app_state = app_state
        return

    sys.stdout = LogInterceptor(sys.stdout, app_state)
