from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import threading
import uuid
from typing import Any, Dict, Optional


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sanitize_for_path(value: str) -> str:
    sanitized = re.sub(r"[^\w\-.]+", "_", value.strip())
    return sanitized or "unknown"


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    try:
        json.dumps(value)
        return value
    except Exception:
        return str(value)


class RunAuditLogger:
    """
    Persist a complete audit trail for each analysis run.

    Logs are written incrementally so partial runs are still debuggable if a run fails.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._active_runs_by_symbol: Dict[str, str] = {}
        self._active_runs: Dict[str, Dict[str, Any]] = {}

    def start_run(
        self,
        symbol: str,
        trade_date: str,
        config: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        with self._lock:
            safe_symbol = _sanitize_for_path(symbol or "unknown")
            run_uuid = uuid.uuid4().hex[:10]
            run_id = f"{trade_date}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{run_uuid}"

            run_dir = Path("eval_results") / safe_symbol / "TradingAgentsStrategy_logs" / "runs"
            run_dir.mkdir(parents=True, exist_ok=True)
            file_path = run_dir / f"{run_id}.json"

            run_data: Dict[str, Any] = {
                "run_id": run_id,
                "symbol": symbol,
                "trade_date": str(trade_date),
                "file_path": str(file_path),
                "started_at": _utc_now_iso(),
                "ended_at": None,
                "status": "running",
                "config": _json_safe(config or {}),
                "metadata": _json_safe(metadata or {}),
                "events": [],
                "snapshots": {},
                "summary": {
                    "prompt_events": 0,
                    "tool_events": 0,
                    "llm_call_events": 0,
                    "agent_output_events": 0,
                    "node_events": 0,
                    "error_events": 0,
                },
            }

            self._active_runs[run_id] = run_data
            self._active_runs_by_symbol[symbol] = run_id
            self._flush_unlocked(run_id)
            print(f"[RUN_LOG] Started run {run_id} -> {file_path}")
            return run_id

    def _resolve_run_id(self, run_id: Optional[str], symbol: Optional[str]) -> Optional[str]:
        if run_id and run_id in self._active_runs:
            return run_id
        if symbol and symbol in self._active_runs_by_symbol:
            return self._active_runs_by_symbol[symbol]
        if len(self._active_runs) == 1:
            return next(iter(self._active_runs.keys()))
        return None

    def log_event(
        self,
        event_type: str,
        symbol: Optional[str] = None,
        run_id: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._lock:
            resolved_run_id = self._resolve_run_id(run_id, symbol)
            if not resolved_run_id:
                return

            run_data = self._active_runs.get(resolved_run_id)
            if not run_data:
                return

            event = {
                "timestamp": _utc_now_iso(),
                "type": event_type,
                "payload": _json_safe(payload or {}),
            }
            run_data["events"].append(event)

            if event_type == "prompt":
                run_data["summary"]["prompt_events"] += 1
            elif event_type == "tool_call":
                run_data["summary"]["tool_events"] += 1
            elif event_type == "llm_call":
                run_data["summary"]["llm_call_events"] += 1
            elif event_type == "agent_output":
                run_data["summary"]["agent_output_events"] += 1
            elif event_type == "node_execution":
                run_data["summary"]["node_events"] += 1

            if event_type in ("error", "tool_error", "node_error"):
                run_data["summary"]["error_events"] += 1

            self._flush_unlocked(resolved_run_id)

    def log_prompt(
        self,
        report_type: str,
        prompt_text: str,
        symbol: Optional[str] = None,
        run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.log_event(
            event_type="prompt",
            symbol=symbol,
            run_id=run_id,
            payload={
                "report_type": report_type,
                "prompt_text": prompt_text,
                "metadata": metadata or {},
            },
        )

    def log_tool_call(
        self,
        tool_name: str,
        inputs: Dict[str, Any],
        output: Any,
        status: str,
        execution_time_seconds: float,
        agent_type: Optional[str] = None,
        symbol: Optional[str] = None,
        run_id: Optional[str] = None,
        error_details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.log_event(
            event_type="tool_call",
            symbol=symbol,
            run_id=run_id,
            payload={
                "tool_name": tool_name,
                "agent_type": agent_type,
                "inputs": inputs,
                "output": output,
                "status": status,
                "execution_time_seconds": execution_time_seconds,
                "error_details": error_details or {},
            },
        )

    def log_agent_output(
        self,
        output_type: str,
        content: Any,
        symbol: Optional[str] = None,
        run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.log_event(
            event_type="agent_output",
            symbol=symbol,
            run_id=run_id,
            payload={
                "output_type": output_type,
                "content": content,
                "metadata": metadata or {},
            },
        )

    def log_state_snapshot(
        self,
        stage: str,
        snapshot: Dict[str, Any],
        symbol: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> None:
        with self._lock:
            resolved_run_id = self._resolve_run_id(run_id, symbol)
            if not resolved_run_id:
                return

            run_data = self._active_runs.get(resolved_run_id)
            if not run_data:
                return

            run_data["snapshots"][stage] = _json_safe(snapshot)
            self._flush_unlocked(resolved_run_id)

    def finish_run(
        self,
        symbol: Optional[str] = None,
        run_id: Optional[str] = None,
        status: str = "completed",
        final_state: Optional[Dict[str, Any]] = None,
        final_signal: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> None:
        with self._lock:
            resolved_run_id = self._resolve_run_id(run_id, symbol)
            if not resolved_run_id:
                return

            run_data = self._active_runs.get(resolved_run_id)
            if not run_data:
                return

            run_data["ended_at"] = _utc_now_iso()
            run_data["status"] = status

            if final_state is not None:
                run_data["snapshots"]["final_state"] = _json_safe(final_state)
            if final_signal is not None:
                run_data["summary"]["final_signal"] = final_signal
            if error_message:
                run_data["summary"]["error_message"] = error_message
                run_data["summary"]["error_events"] += 1

            self._flush_unlocked(resolved_run_id)
            print(
                f"[RUN_LOG] Finished run {resolved_run_id} ({status}) -> "
                f"{run_data.get('file_path', '')}"
            )

            symbol_key = run_data.get("symbol")
            if symbol_key in self._active_runs_by_symbol:
                if self._active_runs_by_symbol[symbol_key] == resolved_run_id:
                    del self._active_runs_by_symbol[symbol_key]
            del self._active_runs[resolved_run_id]

    def _flush_unlocked(self, run_id: str) -> None:
        run_data = self._active_runs.get(run_id)
        if not run_data:
            return

        safe_symbol = _sanitize_for_path(run_data.get("symbol") or "unknown")
        path = Path("eval_results") / safe_symbol / "TradingAgentsStrategy_logs" / "runs" / f"{run_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(run_data, f, indent=2, ensure_ascii=False)


_RUN_AUDIT_LOGGER = RunAuditLogger()


def get_run_audit_logger() -> RunAuditLogger:
    return _RUN_AUDIT_LOGGER
