"""LLM cost accounting over the persisted run logs.

Turns the token usage the audit logger already records into attributable
dollar estimates: per analysis run, per day, per symbol, and per model.
Attribution — not an aggregate bill — is the point; an unattributed total
cannot answer "which symbol or model made costs jump".

Honesty rules:
- Prices drift and differ by account tier, so the built-in table is an
  estimate and every entry can be overridden via the
  ``llm_pricing_per_million`` config key.
- Unknown models are never guessed: their tokens are surfaced separately
  as ``unpriced_tokens`` instead of silently costing $0.
- Run logs do not preserve every provider's cache, region, promotion, or
  long-context pricing tier, so estimates are attribution aids rather than an
  exact reproduction of the provider invoice.

Everything here is pure file reading and arithmetic — zero network calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

# USD per 1M tokens: {"input": ..., "output": ...}. Prefix-matched
# (longest prefix wins), case-insensitive. Estimates — override via the
# `llm_pricing_per_million` config key when your bill disagrees.
DEFAULT_PRICING_PER_MILLION: Dict[str, Dict[str, float]] = {
    # OpenAI
    "gpt-6-astra": {"input": 10.00, "output": 50.00},
    "gpt-5.4-mini": {"input": 0.25, "output": 2.00},
    "gpt-5.4-nano": {"input": 0.05, "output": 0.40},
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
    "gpt-5-nano": {"input": 0.05, "output": 0.40},
    "gpt-5": {"input": 1.25, "output": 10.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "o4-mini": {"input": 1.10, "output": 4.40},
    "o3": {"input": 2.00, "output": 8.00},
    # Anthropic
    "claude-fable-5-1": {"input": 10.00, "output": 50.00},
    "claude-fable-5": {"input": 10.00, "output": 50.00},
    "claude-opus-5": {"input": 5.00, "output": 25.00},
    "claude-sonnet-5": {"input": 2.00, "output": 10.00},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
    "claude-haiku": {"input": 0.80, "output": 4.00},
    "claude-sonnet": {"input": 3.00, "output": 15.00},
    "claude-opus": {"input": 15.00, "output": 75.00},
    "claude": {"input": 3.00, "output": 15.00},
    # DeepSeek
    "deepseek-v4-pro": {"input": 0.435, "output": 0.87},
    "deepseek-v4-flash": {"input": 0.14, "output": 0.28},
    "deepseek-chat": {"input": 0.27, "output": 1.10},
    "deepseek-reasoner": {"input": 0.55, "output": 2.19},
    # MiniMax
    "minimax-m2.7-highspeed": {"input": 0.60, "output": 2.40},
    "minimax-m2.7": {"input": 0.30, "output": 1.20},
    # Google
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
    # xAI
    "grok-4.6": {"input": 2.00, "output": 6.00},
    "grok-4.20": {"input": 1.25, "output": 2.50},
    "grok-4": {"input": 3.00, "output": 15.00},
    "grok-3-mini": {"input": 0.30, "output": 0.50},
    "grok": {"input": 3.00, "output": 15.00},
}


def resolve_pricing(
    model: str, overrides: Optional[Dict[str, Dict[str, float]]] = None
) -> Optional[Dict[str, float]]:
    """Longest-prefix price lookup; overrides shadow the defaults."""
    table = dict(DEFAULT_PRICING_PER_MILLION)
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and "input" in value and "output" in value:
            table[key.lower()] = {
                "input": float(value["input"]),
                "output": float(value["output"]),
            }
    name = str(model or "").lower()
    best_prefix = None
    for prefix in table:
        if name.startswith(prefix.lower()):
            if best_prefix is None or len(prefix) > len(best_prefix):
                best_prefix = prefix
    return dict(table[best_prefix]) if best_prefix else None


def estimate_cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    overrides: Optional[Dict[str, Dict[str, float]]] = None,
) -> Optional[float]:
    """Dollar estimate for a token count, or None for unknown models."""
    pricing = resolve_pricing(model, overrides=overrides)
    if pricing is None:
        return None
    return (
        int(input_tokens or 0) * pricing["input"]
        + int(output_tokens or 0) * pricing["output"]
    ) / 1_000_000.0


def scan_run_costs(
    eval_results_dir: str = "eval_results",
    overrides: Optional[Dict[str, Dict[str, float]]] = None,
) -> List[dict]:
    """One cost record per persisted run, model-attributed where possible.

    Token counts come from each run's llm_call events (which carry the
    model name and usage); runs without usable events fall back to the
    summary totals as unpriced tokens.
    """
    root = Path(eval_results_dir)
    if not root.is_dir():
        return []

    records: List[dict] = []
    for path in sorted(root.glob("*/TradingAgentsStrategy_logs/runs/*.json")):
        try:
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue

        per_model: Dict[str, Dict[str, int]] = {}
        for event in payload.get("events") or []:
            if event.get("type") != "llm_call":
                continue
            event_payload = event.get("payload") or {}
            usage = event_payload.get("usage") or {}
            model = str(event_payload.get("model") or "").strip()
            input_tokens = int(usage.get("input_tokens", 0) or 0)
            output_tokens = int(usage.get("output_tokens", 0) or 0)
            if not model or (input_tokens <= 0 and output_tokens <= 0):
                continue
            bucket = per_model.setdefault(model, {"input": 0, "output": 0})
            bucket["input"] += input_tokens
            bucket["output"] += output_tokens

        input_total = sum(b["input"] for b in per_model.values())
        output_total = sum(b["output"] for b in per_model.values())
        cost: Optional[float] = None
        unpriced = 0
        models: Dict[str, dict] = {}
        for model, bucket in per_model.items():
            model_cost = estimate_cost_usd(
                model, bucket["input"], bucket["output"], overrides=overrides
            )
            models[model] = {
                "input_tokens": bucket["input"],
                "output_tokens": bucket["output"],
                "cost_usd": model_cost,
            }
            if model_cost is None:
                unpriced += bucket["input"] + bucket["output"]
            else:
                cost = (cost or 0.0) + model_cost

        total_tokens = input_total + output_total
        if not per_model:
            # No attributable events: report the summary total as unpriced.
            total_tokens = int(
                (payload.get("summary") or {}).get("total_llm_tokens", 0) or 0
            )
            unpriced = total_tokens

        records.append(
            {
                "symbol": str(payload.get("symbol") or path.parts[-4]),
                "trade_date": str(payload.get("trade_date") or ""),
                "started_at": str(payload.get("started_at") or ""),
                "status": str(payload.get("status") or ""),
                "run_id": str(payload.get("run_id") or path.stem),
                "input_tokens": input_total,
                "output_tokens": output_total,
                "total_tokens": total_tokens,
                "cost_usd": cost,
                "unpriced_tokens": unpriced,
                "models": models,
            }
        )
    return records


def _bucket(target: Dict[str, dict], key: str) -> dict:
    return target.setdefault(
        key, {"runs": 0, "total_tokens": 0, "cost_usd": 0.0, "unpriced_tokens": 0}
    )


def aggregate_costs(records: List[dict]) -> dict:
    """Roll run records up into per-day, per-symbol, per-model, and totals."""
    per_day: Dict[str, dict] = {}
    per_symbol: Dict[str, dict] = {}
    per_model: Dict[str, dict] = {}
    totals = {"runs": 0, "total_tokens": 0, "cost_usd": 0.0, "unpriced_tokens": 0}

    for record in records:
        day = (record.get("started_at") or "")[:10] or record.get("trade_date") or "?"
        cost = record.get("cost_usd") or 0.0
        for bucket in (_bucket(per_day, day), _bucket(per_symbol, record["symbol"]), totals):
            bucket["runs"] += 1
            bucket["total_tokens"] += int(record.get("total_tokens", 0) or 0)
            bucket["cost_usd"] += cost
            bucket["unpriced_tokens"] += int(record.get("unpriced_tokens", 0) or 0)
        for model, stats in (record.get("models") or {}).items():
            model_bucket = per_model.setdefault(
                model, {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "unpriced": False}
            )
            model_bucket["input_tokens"] += stats.get("input_tokens", 0)
            model_bucket["output_tokens"] += stats.get("output_tokens", 0)
            if stats.get("cost_usd") is None:
                model_bucket["unpriced"] = True
            else:
                model_bucket["cost_usd"] += stats["cost_usd"]

    return {
        "per_day": per_day,
        "per_symbol": per_symbol,
        "per_model": per_model,
        "totals": totals,
    }


def parse_return_pct(raw) -> Optional[float]:
    """'+3.2%' -> 0.032; anything unparseable -> None."""
    if raw is None:
        return None
    text = str(raw).strip().rstrip("%")
    try:
        return float(text) / 100.0
    except ValueError:
        return None


def realized_returns_by_symbol(config: Optional[dict] = None) -> Dict[str, dict]:
    """Average realized return per symbol from the resolved decision log.

    Joins the cost side ("what did analyzing this symbol cost") with the
    outcome side ("what did its decisions actually return").
    """
    from tradingagents.agents.utils.memory import TradingMemoryLog

    log = TradingMemoryLog(config or {})
    per_symbol: Dict[str, dict] = {}
    for entry in log.load_entries():
        if entry.get("pending"):
            continue
        realized = parse_return_pct(entry.get("raw"))
        if realized is None:
            continue
        bucket = per_symbol.setdefault(
            entry["ticker"], {"resolved": 0, "sum_return": 0.0}
        )
        bucket["resolved"] += 1
        bucket["sum_return"] += realized

    for bucket in per_symbol.values():
        bucket["avg_return"] = bucket["sum_return"] / bucket["resolved"]
    return per_symbol
