"""
webui/watchdog.py - Background watchdog for detecting stuck analyses.

The watchdog uses a thread-safe reference counter so that parallel ticker
analyses (run via ThreadPoolExecutor in batch mode) can each call
set_analysis_active() / set_analysis_inactive() independently. The flag
file is created when the count transitions 0 -> 1 and removed when it
transitions 1 -> 0, so it remains present until the last parallel ticker
finishes.
"""

import os
import time
import threading

# ---------------------------------------------------------------------------
# Constants (edit here to tune watchdog behaviour without touching other files)
# ---------------------------------------------------------------------------

ANALYSIS_FLAG_PATH = "/tmp/trading_analysis_active"

# How often the watchdog wakes up to check the flag file (seconds).
WATCHDOG_INTERVAL_SECONDS = 600  # 10 minutes

# How old the flag file must be (no heartbeat update) before an ALERT is
# printed. This is effectively the "stuck analysis" timeout.
STUCK_THRESHOLD_SECONDS = 3600  # 1 hour

# ---------------------------------------------------------------------------
# Module-level private state (all mutations protected by _count_lock)
# ---------------------------------------------------------------------------

_active_count: int = 0
_count_lock: threading.Lock = threading.Lock()
_watchdog_started: bool = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def set_analysis_active() -> None:
    """
    Increment the active-analysis reference counter.

    When the counter transitions from 0 to 1 the flag file is created/touched
    so the watchdog knows an analysis is in progress. Safe to call from
    multiple threads concurrently (e.g., parallel batch mode).
    """
    global _active_count
    with _count_lock:
        _active_count += 1
        if _active_count == 1:
            # First active analysis — create the flag file.
            try:
                with open(ANALYSIS_FLAG_PATH, "a"):
                    pass  # create if absent, leave contents untouched
                os.utime(ANALYSIS_FLAG_PATH, None)
            except OSError as exc:
                print(f"[WATCHDOG] WARNING: could not create flag file {ANALYSIS_FLAG_PATH}: {exc}")


def set_analysis_inactive() -> None:
    """
    Decrement the active-analysis reference counter.

    When the counter transitions from 1 to 0 the flag file is removed so the
    watchdog knows all analyses have finished. The counter is clamped at 0 to
    guard against unexpected extra calls (e.g., from a crash path).
    """
    global _active_count
    with _count_lock:
        if _active_count > 0:
            _active_count -= 1
        if _active_count == 0:
            try:
                if os.path.exists(ANALYSIS_FLAG_PATH):
                    os.remove(ANALYSIS_FLAG_PATH)
            except OSError as exc:
                print(f"[WATCHDOG] WARNING: could not remove flag file {ANALYSIS_FLAG_PATH}: {exc}")


def touch_analysis_flag() -> None:
    """
    Update the mtime of the flag file as a heartbeat.

    Called periodically (once per LangGraph streaming chunk) to signal that
    the analysis is still making forward progress. The watchdog uses the mtime
    to detect analyses that have stopped emitting chunks (i.e., stuck LLM
    calls). If the flag file does not exist when this is called, it will be
    created (open with mode 'a') before the utime call.
    """
    try:
        with open(ANALYSIS_FLAG_PATH, "a"):
            pass  # create if somehow absent
        os.utime(ANALYSIS_FLAG_PATH, None)
    except OSError as exc:
        print(f"[WATCHDOG] WARNING: could not touch flag file {ANALYSIS_FLAG_PATH}: {exc}")


# ---------------------------------------------------------------------------
# Private watchdog loop
# ---------------------------------------------------------------------------

def _watchdog_loop() -> None:
    """
    Daemon loop: sleep for WATCHDOG_INTERVAL_SECONDS, then check the flag
    file and print a status or alert message.
    """
    while True:
        time.sleep(WATCHDOG_INTERVAL_SECONDS)

        if not os.path.exists(ANALYSIS_FLAG_PATH):
            # No flag file — no active analysis; nothing to report.
            continue

        try:
            age = time.time() - os.path.getmtime(ANALYSIS_FLAG_PATH)
        except OSError:
            # File disappeared between the exists() check and getmtime(); safe
            # to ignore — the analysis finished.
            continue

        if age > STUCK_THRESHOLD_SECONDS:
            print(
                f"[WATCHDOG] ALERT: analysis flag is {age:.0f}s old "
                f"— possible stuck analysis (threshold: {STUCK_THRESHOLD_SECONDS}s)"
            )
        else:
            print(f"[WATCHDOG] Analysis active, flag age {age:.0f}s")


# ---------------------------------------------------------------------------
# Public start function
# ---------------------------------------------------------------------------

def start_watchdog() -> None:
    """
    Start the background watchdog daemon thread (idempotent).

    Calling this function more than once (e.g., due to Dash hot-reload
    spawning the worker process twice, or multiple call-sites in tests) is
    safe: only the first call actually creates the thread.
    """
    global _watchdog_started
    with _count_lock:
        if _watchdog_started:
            return
        _watchdog_started = True

    thread = threading.Thread(
        target=_watchdog_loop,
        daemon=True,
        name="watchdog",
    )
    thread.start()
    print("[WATCHDOG] Background watchdog started")
