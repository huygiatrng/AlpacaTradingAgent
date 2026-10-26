"""Windows desktop entry point; the bundled app never needs a system Python."""

import argparse
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import queue
import sys
import threading
import webbrowser


APP_NAME = "AlpacaTradingAgent"


def app_home():
    """Keep writable files outside the installation and temporary bundle."""
    return Path(os.environ.get("TRADINGAGENTS_DESKTOP_HOME") or
                Path(os.environ.get("LOCALAPPDATA", Path.home())) / APP_NAME)


def prepare_environment():
    from dotenv import load_dotenv

    home = app_home()
    home.mkdir(parents=True, exist_ok=True)
    install = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).parent
    load_dotenv(install / ".env", override=False)
    load_dotenv(home / ".env", override=False)
    # Libraries also call load_dotenv(); stop them walking into unrelated
    # parent directories after the desktop has loaded its two explicit files.
    os.environ["PYTHON_DOTENV_DISABLED"] = "1"
    for variable, suffix in {
        "TRADINGAGENTS_RESULTS_DIR": "results",
        "TRADINGAGENTS_CACHE_DIR": "cache",
        "TRADINGAGENTS_MEMORY_LOG_PATH": "memory/trading_memory.md",
        "TRADINGAGENTS_AGENT_MEMORY_DIR": "memory/agents",
    }.items():
        os.environ.setdefault(variable, str(home / suffix))
    # Legacy relative data/checkpoint paths also remain writable after installation.
    os.chdir(home)
    return home


def acquire_instance(home):
    """Hold an OS file lock until exit, including during slow imports."""
    handle = (home / "desktop.lock").open("a+b")
    try:
        if (home / "desktop.lock").stat().st_size == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return None
    return handle


def previous_port(home):
    try:
        port = int(json.loads((home / "desktop.json").read_text())["port"])
        return port if 1024 <= port <= 65535 else 7860
    except (OSError, ValueError, KeyError, TypeError):
        return 7860


def create_server(home):
    from werkzeug.serving import make_server
    from webui.app_dash import create_app

    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    app = create_app()
    # Binding port zero delegates selection to the OS when the saved port is busy.
    try:
        server = make_server("127.0.0.1", previous_port(home), app.server, threaded=True)
    except (OSError, SystemExit):
        server = make_server("127.0.0.1", 0, app.server, threaded=True)
    server.daemon_threads = True
    state = home / "desktop.json"
    temporary = state.with_suffix(".tmp")
    temporary.write_text(json.dumps({"port": server.server_port}), encoding="utf-8")
    temporary.replace(state)
    return server


def main():
    parser = argparse.ArgumentParser(description="Open AlpacaTradingAgent")
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser automatically")
    parser.add_argument("--headless", action="store_true", help="Run the bundled server without a launcher window")
    parser.add_argument("--self-test", type=Path, help="Write an offline packaging verification report and exit")
    args = parser.parse_args()
    if args.self_test:
        args.self_test = args.self_test.resolve()
    home = prepare_environment()
    logs = home / "logs"
    logs.mkdir(exist_ok=True)
    log = (logs / "console.log").open("a", encoding="utf-8", buffering=1)
    if sys.stdout is None:
        sys.stdout = log
    if sys.stderr is None:
        sys.stderr = log
    logging.basicConfig(handlers=[RotatingFileHandler(logs / "desktop.log", maxBytes=2_000_000, backupCount=2, encoding="utf-8")], level=logging.WARNING,
                        format="%(asctime)s %(levelname)s %(message)s")
    lock = acquire_instance(home)
    if lock is None:
        if not args.no_browser:
            webbrowser.open(f"http://127.0.0.1:{previous_port(home)}")
        return 0
    try:
        if args.self_test:
            return self_test(args.self_test.resolve())
        if args.headless:
            server = create_server(home)
            print(f"{APP_NAME} ready at http://127.0.0.1:{server.server_port}", flush=True)
            server.serve_forever()
            return 0
        return run_window(home, args.no_browser)
    finally:
        lock.close()


def self_test(report_path):
    """Exercise resources and imports that static packaging analysis can miss."""
    import chromadb
    from chromadb.config import Settings
    from langgraph.checkpoint.sqlite import SqliteSaver
    from tradingagents.llm_clients import create_llm_client
    from tradingagents.agents.utils.memory import FinancialSituationMemory
    from webui.app_dash import create_app
    # Building the real layout is offline; account requests run in callbacks.
    app = create_app()
    client = app.server.test_client()
    report = {"frozen": bool(getattr(sys, "frozen", False)),
              "layout": client.get("/_dash-layout").status_code,
              "icon": client.get("/assets/app.ico").status_code,
              "bootstrap": client.get("/assets/vendor/bootstrap.min.css").status_code,
              "prompts": bool(list((Path(__file__).parent / "tradingagents/prompts/templates").rglob("*.md")))}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    db = chromadb.PersistentClient(path=str(report_path.parent / "smoke-memory"), settings=Settings(anonymized_telemetry=False))
    collection = db.get_or_create_collection("desktop-smoke", embedding_function=None)
    collection.upsert(ids=["one"], documents=["packaged runtime"], embeddings=[[1.0, 0.0]])
    report["memory"] = collection.query(query_embeddings=[[1.0, 0.0]], n_results=1)["ids"] == [["one"]]
    db.delete_collection("desktop-smoke")
    report["sqlite"] = SqliteSaver is not None
    report["llm_factory"] = create_llm_client is not None and FinancialSituationMemory is not None
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0 if all(report.values()) else 1


def run_window(home, no_browser):
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.title(APP_NAME)
    root.geometry("580x420")
    root.resizable(False, False)
    root.configure(bg="#101b2d")
    icon = Path(__file__).parent / "webui" / "assets" / "app.ico"
    if icon.exists():
        root.iconbitmap(str(icon))
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("Launch.TButton", font=("Segoe UI", 12), padding=12)
    tk.Label(root, text="AlpacaTradingAgent", font=("Segoe UI", 23, "bold"),
             fg="#f1f5f9", bg="#101b2d").pack(pady=(28, 8))
    tk.Label(root, text="Your trading research workspace", font=("Segoe UI", 12),
             fg="#b7c8df", bg="#101b2d").pack()
    status = tk.StringVar(value="Starting your workspace. This can take a moment...")
    tk.Label(root, textvariable=status, wraplength=480, font=("Segoe UI", 11),
             fg="#70e3c0", bg="#101b2d").pack(pady=20)
    messages = queue.Queue()
    url = tk.StringVar()
    open_button = ttk.Button(root, text="Open workspace", state="disabled",
                             style="Launch.TButton", command=lambda: webbrowser.open(url.get()))
    open_button.pack()
    tk.Label(root, text="Start with API setup, check your connections, then run an analysis.\n"
             "Keep this window open while using the app.", font=("Segoe UI", 10),
             fg="#b7c8df", bg="#101b2d").pack(pady=18)
    ttk.Button(root, text="Open app folder", command=lambda: os.startfile(home)).pack()

    def start():
        try:
            server = create_server(home)
            messages.put((True, f"http://127.0.0.1:{server.server_port}"))
            server.serve_forever()
        except BaseException as error:
            logging.error("Desktop startup failed: %s", type(error).__name__)
            messages.put((False, "Could not start the workspace. Open the app folder and check logs/desktop.log."))

    def poll():
        try:
            ready, message = messages.get_nowait()
        except queue.Empty:
            root.after(100, poll)
            return
        if ready:
            url.set(message)
            status.set("Ready. Your workspace opens in your browser.")
            open_button.configure(state="normal")
            if not no_browser:
                webbrowser.open(message)
        else:
            status.set(message)

    def close():
        root.destroy()
        # Analysis workers can be non-daemon threads. Closing the desktop app
        # must stop them as well as the HTTP server, rather than leave trading
        # work running behind a closed window. Windows releases the file lock.
        for stream in (sys.stdout, sys.stderr):
            if stream:
                stream.flush()
        os._exit(0)

    root.protocol("WM_DELETE_WINDOW", close)
    threading.Thread(target=start, name="desktop-server", daemon=True).start()
    root.after(100, poll)
    root.mainloop()
    return 0


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    raise SystemExit(main())
