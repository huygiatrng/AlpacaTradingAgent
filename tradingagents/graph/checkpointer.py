from __future__ import annotations

import hashlib
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

from tradingagents.dataflows.utils import safe_ticker_component


def _db_path(data_dir: str | Path, ticker: str) -> Path:
    safe = safe_ticker_component(ticker).upper()
    cp_dir = Path(data_dir) / "checkpoints"
    cp_dir.mkdir(parents=True, exist_ok=True)
    return cp_dir / f"{safe}.db"


def thread_id(ticker: str, date: str) -> str:
    return hashlib.sha256(f"{ticker.upper()}:{date}".encode("utf-8")).hexdigest()[:16]


def _sqlite_saver_cls():
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver

        return SqliteSaver
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Checkpoint resume requires the optional package "
            "'langgraph-checkpoint-sqlite'. Install it with "
            "`python -m pip install langgraph-checkpoint-sqlite` or leave "
            "`checkpoint_enabled` off."
        ) from exc


@contextmanager
def get_checkpointer(data_dir: str | Path, ticker: str) -> Generator[Any, None, None]:
    conn = sqlite3.connect(str(_db_path(data_dir, ticker)), check_same_thread=False)
    try:
        SqliteSaver = _sqlite_saver_cls()
        saver = SqliteSaver(conn)
        saver.setup()
        yield saver
    finally:
        conn.close()


def checkpoint_step(data_dir: str | Path, ticker: str, date: str) -> int | None:
    db = _db_path(data_dir, ticker)
    if not db.exists():
        return None
    with get_checkpointer(data_dir, ticker) as saver:
        cp = saver.get_tuple({"configurable": {"thread_id": thread_id(ticker, date)}})
        if cp is None:
            return None
        metadata = getattr(cp, "metadata", None) or {}
        return metadata.get("step")


def has_checkpoint(data_dir: str | Path, ticker: str, date: str) -> bool:
    return checkpoint_step(data_dir, ticker, date) is not None


def clear_checkpoint(data_dir: str | Path, ticker: str, date: str) -> None:
    db = _db_path(data_dir, ticker)
    if not db.exists():
        return
    tid = thread_id(ticker, date)
    conn = sqlite3.connect(str(db))
    try:
        for table in ("writes", "checkpoints", "blobs"):
            try:
                conn.execute(f"DELETE FROM {table} WHERE thread_id = ?", (tid,))
            except sqlite3.OperationalError:
                continue
        conn.commit()
    except sqlite3.OperationalError:
        pass
    finally:
        conn.close()


def clear_all_checkpoints(data_dir: str | Path) -> int:
    cp_dir = Path(data_dir) / "checkpoints"
    if not cp_dir.exists():
        return 0
    dbs = list(cp_dir.glob("*.db"))
    for db in dbs:
        db.unlink()
    return len(dbs)
