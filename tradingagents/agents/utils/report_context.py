from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Any, Dict, List, Tuple


REPORT_SPECS: List[Tuple[str, str]] = [
    ("macro_report", "Macro"),
    ("market_report", "Market"),
    ("sentiment_report", "Social Sentiment"),
    ("news_report", "News"),
    ("fundamentals_report", "Fundamentals"),
]


DEFAULT_CONTEXT_CONFIG = {
    "report_context_budget_tokens": 5500,
    "report_context_max_chunks": 16,
    "report_context_min_chunks_per_report": 1,
    "report_context_chunk_chars": 900,
    "report_context_chunk_overlap": 120,
    "report_context_max_points_per_report": 8,
    "report_context_point_chars": 220,
    "report_context_excerpt_chars": 420,
    "report_context_memory_chars": 12000,
    "report_context_compact_points_per_report": 3,
    "report_context_compact_point_chars": 180,
    "report_context_compact_excerpt_chars": 240,
    "report_context_compact_max_excerpts": 8,
    "debate_digest_max_messages": 6,
    "debate_digest_message_chars": 520,
    "debate_digest_total_chars": 2600,
}


ROLE_WEIGHTS: Dict[str, Dict[str, float]] = {
    "bull_researcher": {
        "fundamentals_report": 1.35,
        "market_report": 1.30,
        "news_report": 1.15,
        "macro_report": 1.00,
        "sentiment_report": 1.10,
    },
    "bear_researcher": {
        "macro_report": 1.30,
        "news_report": 1.20,
        "fundamentals_report": 1.20,
        "market_report": 1.10,
        "sentiment_report": 1.10,
    },
    "research_manager": {
        "macro_report": 1.20,
        "market_report": 1.20,
        "sentiment_report": 1.15,
        "news_report": 1.15,
        "fundamentals_report": 1.20,
    },
    "trader": {
        "market_report": 1.45,
        "macro_report": 1.25,
        "news_report": 1.20,
        "sentiment_report": 1.20,
        "fundamentals_report": 1.10,
    },
    "risky_debator": {
        "market_report": 1.35,
        "sentiment_report": 1.25,
        "news_report": 1.20,
        "fundamentals_report": 1.10,
        "macro_report": 1.05,
    },
    "safe_debator": {
        "macro_report": 1.40,
        "fundamentals_report": 1.25,
        "news_report": 1.20,
        "market_report": 1.10,
        "sentiment_report": 1.05,
    },
    "neutral_debator": {
        "macro_report": 1.20,
        "market_report": 1.20,
        "news_report": 1.20,
        "fundamentals_report": 1.20,
        "sentiment_report": 1.15,
    },
    "risk_manager": {
        "macro_report": 1.35,
        "market_report": 1.30,
        "fundamentals_report": 1.25,
        "news_report": 1.20,
        "sentiment_report": 1.10,
    },
    "default": {
        "macro_report": 1.20,
        "market_report": 1.20,
        "sentiment_report": 1.15,
        "news_report": 1.15,
        "fundamentals_report": 1.20,
    },
}


STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "over",
    "under",
    "your",
    "you",
    "are",
    "was",
    "were",
    "have",
    "has",
    "had",
    "about",
    "after",
    "before",
    "also",
    "their",
    "them",
    "they",
    "will",
    "would",
    "should",
    "could",
    "just",
    "than",
    "then",
    "when",
    "where",
    "while",
    "what",
    "which",
    "whose",
    "been",
    "being",
    "into",
    "across",
    "between",
    "current",
    "latest",
    "analysis",
    "report",
    "reports",
    "context",
}


def _get_context_config(config: Dict[str, Any] | None) -> Dict[str, Any]:
    merged = DEFAULT_CONTEXT_CONFIG.copy()
    if config:
        merged.update(config)
    return merged


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def normalize_for_prompt(text: Any) -> str:
    """Normalize any value for prompt injection without truncation."""
    return _normalize_text(text)


def truncate_for_prompt(text: Any, max_chars: int = 1200) -> str:
    """Backward-compatible alias for prompt normalization."""
    _ = max_chars
    return normalize_for_prompt(text)


def _classify_signal(point: str) -> str:
    lower = point.lower()
    bullish_hits = 0
    bearish_hits = 0

    bullish_terms = (
        "bullish",
        "uptrend",
        "breakout",
        "support",
        "beat",
        "upgrade",
        "momentum",
        "long",
        "buy",
        "risk-on",
        "improve",
        "strong",
    )
    bearish_terms = (
        "bearish",
        "downtrend",
        "breakdown",
        "resistance",
        "miss",
        "downgrade",
        "risk-off",
        "volatility",
        "stop",
        "short",
        "sell",
        "weak",
        "headwind",
    )

    for term in bullish_terms:
        if term in lower:
            bullish_hits += 1
    for term in bearish_terms:
        if term in lower:
            bearish_hits += 1

    if bullish_hits > bearish_hits:
        return "Bullish"
    if bearish_hits > bullish_hits:
        return "Bearish"
    return "Mixed"


def _render_decision_claim_matrix(
    context: Dict[str, Any],
    config: Dict[str, Any] | None = None,
) -> str:
    cfg = _get_context_config(config)
    max_points = int(cfg["report_context_compact_points_per_report"])
    point_chars = int(cfg["report_context_compact_point_chars"])

    lines: List[str] = []
    lines.append("Decision Claim Matrix (compressed):")
    for report_key, _ in REPORT_SPECS:
        report_meta = context.get("reports", {}).get(report_key)
        if not report_meta:
            continue

        points = report_meta.get("coverage_points", [])[:max_points]
        if not points:
            lines.append(f"- {report_meta['label']}: No usable claims.")
            continue

        signal_votes = {"Bullish": 0, "Bearish": 0, "Mixed": 0}
        rendered_points: List[str] = []
        for point in points:
            signal = _classify_signal(point)
            signal_votes[signal] += 1
            rendered_points.append(_truncate(point, point_chars))

        dominant = max(signal_votes, key=signal_votes.get)
        joined_points = " | ".join(rendered_points)
        lines.append(f"- {report_meta['label']} [{dominant}]: {joined_points}")

    return "\n".join(lines).strip()


def _split_sections(text: str) -> List[Tuple[str, str]]:
    if not text:
        return []

    sections: List[Tuple[str, str]] = []
    current_title = "Overview"
    current_lines: List[str] = []

    for line in text.splitlines():
        stripped = line.strip()
        if re.match(r"^#{1,6}\s+\S+", stripped):
            body = "\n".join(current_lines).strip()
            if body:
                sections.append((current_title, body))
            current_title = re.sub(r"^#{1,6}\s*", "", stripped).strip()
            current_lines = []
            continue
        current_lines.append(line)

    body = "\n".join(current_lines).strip()
    if body:
        sections.append((current_title, body))

    if not sections:
        sections.append(("Overview", text))

    return sections


def _is_table_separator(line: str) -> bool:
    compact = line.replace("|", "").replace("-", "").replace(":", "").replace(" ", "")
    return compact == ""


def _extract_candidates(text: str) -> List[str]:
    candidates: List[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if len(line) < 20:
            continue
        if _is_table_separator(line):
            continue
        line = line.lstrip("-*0123456789. ").strip()
        if len(line) < 20:
            continue
        candidates.append(line)
    return candidates


def _line_priority(line: str) -> int:
    lower = line.lower()
    score = 0
    if any(ch.isdigit() for ch in line):
        score += 3
    if "%" in line or "$" in line:
        score += 3
    if any(
        keyword in lower
        for keyword in (
            "risk",
            "stop",
            "target",
            "entry",
            "exit",
            "trend",
            "support",
            "resistance",
            "atr",
            "rsi",
            "macd",
            "earnings",
            "guidance",
            "revenue",
            "cpi",
            "fomc",
            "yield",
            "position",
            "volatility",
            "sentiment",
        )
    ):
        score += 3
    if ":" in line:
        score += 1
    return score


def _extract_coverage_points(
    section_title: str,
    section_text: str,
    max_points: int,
    point_chars: int,
) -> List[str]:
    candidates = _extract_candidates(section_text)
    ranked = sorted(
        candidates,
        key=lambda item: (_line_priority(item), len(item)),
        reverse=True,
    )

    seen = set()
    points: List[str] = []

    for candidate in ranked:
        compact = re.sub(r"\W+", "", candidate.lower())
        if not compact or compact in seen:
            continue
        seen.add(compact)
        points.append(f"{section_title}: {_truncate(candidate, point_chars)}")
        if len(points) >= max_points:
            break

    if points:
        return points

    fallback = _truncate(section_text.replace("\n", " ").strip(), point_chars)
    if not fallback:
        return []
    return [f"{section_title}: {fallback}"]


def _chunk_text(text: str, max_chars: int, overlap: int) -> List[str]:
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: List[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(text_len, start + max_chars)
        if end < text_len:
            split_newline = text.rfind("\n", start + int(max_chars * 0.6), end)
            split_space = text.rfind(" ", start + int(max_chars * 0.6), end)
            split = max(split_newline, split_space)
            if split > start:
                end = split

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_len:
            break

        start = max(0, end - overlap)

    return chunks


def _extract_terms(text: str) -> List[str]:
    terms: List[str] = []
    for token in re.findall(r"[A-Za-z][A-Za-z0-9_/%\.-]{2,}", text.lower()):
        if token in STOPWORDS:
            continue
        terms.append(token)
    return terms


def _score_chunk(
    chunk: Dict[str, Any],
    query_terms: List[str],
    role_weights: Dict[str, float],
) -> float:
    score = role_weights.get(chunk["report_key"], 1.0)
    text_lower = chunk["text"].lower()

    if query_terms:
        overlap = 0
        for term in query_terms:
            if term in text_lower:
                overlap += 1
        score += overlap * 2.0
        if overlap == 0:
            score -= 0.3

    if any(k in text_lower for k in ("buy", "sell", "hold", "long", "short", "risk")):
        score += 0.5
    if any(ch.isdigit() for ch in chunk["text"]):
        score += 0.2

    return score


def build_report_context_index(
    state: Dict[str, Any],
    config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    cfg = _get_context_config(config)
    max_points = int(cfg["report_context_max_points_per_report"])
    point_chars = int(cfg["report_context_point_chars"])
    chunk_chars = int(cfg["report_context_chunk_chars"])
    overlap = int(cfg["report_context_chunk_overlap"])

    context: Dict[str, Any] = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reports": {},
        "chunks": [],
        "report_order": [spec[0] for spec in REPORT_SPECS],
        "stats": {
            "reports_with_content": 0,
            "total_chunks": 0,
            "total_tokens_estimate": 0,
            "total_chars": 0,
        },
    }

    global_overview_lines: List[str] = []

    for report_key, report_label in REPORT_SPECS:
        raw_text = _normalize_text(state.get(report_key, ""))
        if not raw_text:
            continue

        sections = _split_sections(raw_text)
        coverage_points: List[str] = []
        report_chunk_ids: List[str] = []

        for sec_idx, (section_title, section_text) in enumerate(sections, start=1):
            coverage_points.extend(
                _extract_coverage_points(
                    section_title,
                    section_text,
                    max_points=max(1, max_points // 3),
                    point_chars=point_chars,
                )
            )
            for chunk_idx, chunk_text in enumerate(
                _chunk_text(section_text, chunk_chars, overlap),
                start=1,
            ):
                chunk_id = (
                    f"{report_key.replace('_report', '')}"
                    f"_s{sec_idx}_c{chunk_idx}"
                )
                chunk_payload = {
                    "id": chunk_id,
                    "report_key": report_key,
                    "report_label": report_label,
                    "section_title": section_title,
                    "text": chunk_text,
                    "token_estimate": _estimate_tokens(chunk_text),
                    "char_count": len(chunk_text),
                }
                context["chunks"].append(chunk_payload)
                report_chunk_ids.append(chunk_id)

        # Keep only the top coverage points per report.
        coverage_points = coverage_points[:max_points]
        summary = "\n".join(f"- {point}" for point in coverage_points)

        context["reports"][report_key] = {
            "label": report_label,
            "char_count": len(raw_text),
            "token_estimate": _estimate_tokens(raw_text),
            "coverage_points": coverage_points,
            "summary": summary,
            "chunk_ids": report_chunk_ids,
        }

        first_point = coverage_points[0] if coverage_points else _truncate(raw_text, 140)
        global_overview_lines.append(f"- {report_label}: {first_point}")

        context["stats"]["reports_with_content"] += 1
        context["stats"]["total_tokens_estimate"] += _estimate_tokens(raw_text)
        context["stats"]["total_chars"] += len(raw_text)

    context["stats"]["total_chunks"] = len(context["chunks"])
    context["global_overview"] = "\n".join(global_overview_lines)

    return context


def _select_chunks_for_agent(
    context: Dict[str, Any],
    agent_role: str,
    objective: str,
    config: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    cfg = _get_context_config(config)
    token_budget = int(cfg["report_context_budget_tokens"])
    max_chunks = int(cfg["report_context_max_chunks"])
    min_chunks_per_report = int(cfg["report_context_min_chunks_per_report"])

    chunks = context.get("chunks", [])
    if not chunks:
        return []

    query_terms = _extract_terms(objective)
    role_weights = ROLE_WEIGHTS.get(agent_role, ROLE_WEIGHTS["default"])

    scored: List[Tuple[float, Dict[str, Any]]] = []
    for chunk in chunks:
        scored.append((_score_chunk(chunk, query_terms, role_weights), chunk))
    scored.sort(key=lambda item: item[0], reverse=True)

    selected: List[Dict[str, Any]] = []
    selected_ids = set()
    used_tokens = 0

    # Coverage pass: force representation from each report first.
    for report_key, _ in REPORT_SPECS:
        added_for_report = 0
        for score, chunk in scored:
            if chunk["report_key"] != report_key:
                continue
            if chunk["id"] in selected_ids:
                continue
            token_cost = chunk["token_estimate"]
            if used_tokens + token_cost > token_budget:
                continue
            selected.append(chunk)
            selected_ids.add(chunk["id"])
            used_tokens += token_cost
            added_for_report += 1
            if added_for_report >= min_chunks_per_report:
                break

    # Relevance pass: fill remaining budget by score.
    for score, chunk in scored:
        if len(selected) >= max_chunks:
            break
        if chunk["id"] in selected_ids:
            continue
        token_cost = chunk["token_estimate"]
        if used_tokens + token_cost > token_budget:
            continue
        selected.append(chunk)
        selected_ids.add(chunk["id"])
        used_tokens += token_cost

    return selected


def _render_analysis_context(
    context: Dict[str, Any],
    selected_chunks: List[Dict[str, Any]],
    config: Dict[str, Any] | None = None,
) -> str:
    cfg = _get_context_config(config)
    excerpt_chars = int(cfg["report_context_excerpt_chars"])

    lines: List[str] = []
    lines.append("Cross-Analyst Context Packet")
    lines.append("Use this packet as canonical evidence synthesized from all analyst reports.")
    lines.append("")

    global_overview = context.get("global_overview", "").strip()
    if global_overview:
        lines.append("Topline Overview:")
        lines.append(global_overview)
        lines.append("")

    lines.append("Full-Coverage Highlights:")
    for report_key, _ in REPORT_SPECS:
        report_meta = context.get("reports", {}).get(report_key)
        if not report_meta:
            continue
        lines.append(f"{report_meta['label']}:")
        points = report_meta.get("coverage_points", [])
        if not points:
            lines.append("- No structured highlights available.")
            continue
        for point in points:
            lines.append(f"- {point}")
    lines.append("")

    if selected_chunks:
        lines.append("Role-Relevant Evidence Excerpts (retrieved by objective):")
        for chunk in selected_chunks:
            excerpt = _truncate(chunk["text"].replace("\n", " "), excerpt_chars)
            lines.append(
                f"[{chunk['id']} | {chunk['report_label']} | {chunk['section_title']}] "
                f"{excerpt}"
            )

    return "\n".join(lines).strip()


def _render_analysis_context_compact(
    context: Dict[str, Any],
    selected_chunks: List[Dict[str, Any]],
    config: Dict[str, Any] | None = None,
) -> str:
    cfg = _get_context_config(config)
    excerpt_chars = int(cfg["report_context_compact_excerpt_chars"])
    max_excerpts = int(cfg["report_context_compact_max_excerpts"])

    lines: List[str] = []
    lines.append("Cross-Analyst Context Packet (Compact)")
    lines.append("Use this compact packet for fast decisioning with full report coverage preserved.")
    lines.append("")

    global_overview = context.get("global_overview", "").strip()
    if global_overview:
        lines.append("Topline Overview:")
        lines.append(global_overview)
        lines.append("")

    lines.append(_render_decision_claim_matrix(context, config=config))
    lines.append("")

    if selected_chunks:
        lines.append("Top Evidence Excerpts:")
        for chunk in selected_chunks[:max_excerpts]:
            excerpt = _truncate(chunk["text"].replace("\n", " "), excerpt_chars)
            lines.append(
                f"[{chunk['id']} | {chunk['report_label']} | {chunk['section_title']}] {excerpt}"
            )

    return "\n".join(lines).strip()


def _render_memory_context(
    context: Dict[str, Any],
    config: Dict[str, Any] | None = None,
) -> str:
    cfg = _get_context_config(config)
    max_chars = int(cfg["report_context_memory_chars"])

    lines: List[str] = []
    lines.append("Condensed cross-report situation context:")

    global_overview = context.get("global_overview", "").strip()
    if global_overview:
        lines.append(global_overview)

    for report_key, _ in REPORT_SPECS:
        report_meta = context.get("reports", {}).get(report_key)
        if not report_meta:
            continue
        lines.append(f"{report_meta['label']} highlights:")
        for point in report_meta.get("coverage_points", [])[:4]:
            lines.append(f"- {point}")

    rendered = "\n".join(lines).strip()
    return _truncate(rendered, max_chars)


def _render_all_reports_text(state: Dict[str, Any]) -> str:
    """Render full analyst reports without truncation so downstream agents can access all data."""
    lines: List[str] = []
    lines.append("Full Analyst Reports (Untruncated):")
    for report_key, report_label in REPORT_SPECS:
        raw_text = _normalize_text(state.get(report_key, ""))
        if not raw_text:
            continue
        lines.append(f"### {report_label}")
        lines.append(raw_text)
        lines.append("")
    return "\n".join(lines).strip()


def _short_message_line(message: str, label: str, max_chars: int) -> str:
    msg = _normalize_text(message)
    if msg.lower().startswith(label.lower() + ":"):
        msg = msg.split(":", 1)[1].strip()
    return _truncate(msg, max_chars)


def build_debate_digest(
    debate_state: Dict[str, Any] | None,
    debate_type: str,
    config: Dict[str, Any] | None = None,
) -> str:
    """Build a compact digest for either investment or risk debate states."""
    if not isinstance(debate_state, dict):
        return ""

    cfg = _get_context_config(config)
    max_messages = int(cfg["debate_digest_max_messages"])
    msg_chars = int(cfg["debate_digest_message_chars"])
    total_chars = int(cfg["debate_digest_total_chars"])

    lines: List[str] = []
    if debate_type == "investment":
        lines.append("Investment Debate Digest:")
        lines.append(f"- Turn count: {debate_state.get('count', 0)}")
        current = _truncate(normalize_for_prompt(debate_state.get("current_response", "")), msg_chars)
        if current:
            lines.append(f"- Latest response: {current}")

        bull_messages = list(debate_state.get("bull_messages", []))[-max_messages:]
        bear_messages = list(debate_state.get("bear_messages", []))[-max_messages:]
        for message in bull_messages[-max_messages // 2 :]:
            lines.append(f"- Bull: {_short_message_line(message, 'Bull Analyst', msg_chars)}")
        for message in bear_messages[-max_messages // 2 :]:
            lines.append(f"- Bear: {_short_message_line(message, 'Bear Analyst', msg_chars)}")
    else:
        lines.append("Risk Debate Digest:")
        lines.append(f"- Turn count: {debate_state.get('count', 0)}")
        lines.append(f"- Latest speaker: {debate_state.get('latest_speaker', 'Unknown')}")

        latest_map = [
            ("Risky", debate_state.get("current_risky_response", "")),
            ("Safe", debate_state.get("current_safe_response", "")),
            ("Neutral", debate_state.get("current_neutral_response", "")),
        ]
        for label, content in latest_map:
            compact = _truncate(normalize_for_prompt(content), msg_chars)
            if compact:
                lines.append(f"- {label} latest: {compact}")

        msg_sources = [
            ("Risky", list(debate_state.get("risky_messages", []))),
            ("Safe", list(debate_state.get("safe_messages", []))),
            ("Neutral", list(debate_state.get("neutral_messages", []))),
        ]
        per_agent = max(1, max_messages // 3)
        for label, messages in msg_sources:
            for message in messages[-per_agent:]:
                lines.append(f"- {label}: {_short_message_line(message, f'{label} Analyst', msg_chars)}")

    return _truncate("\n".join(lines).strip(), total_chars)


def ensure_report_context(
    state: Dict[str, Any],
    config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    context = state.get("report_context")
    if isinstance(context, dict) and context.get("chunks") is not None:
        return context
    context = build_report_context_index(state, config=config)
    # Mutating state here is intentional to avoid rebuilding the index in each node.
    state["report_context"] = context
    return context


def get_agent_context_bundle(
    state: Dict[str, Any],
    agent_role: str,
    objective: str,
    config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    context = ensure_report_context(state, config=config)
    selected_chunks = _select_chunks_for_agent(
        context,
        agent_role=agent_role,
        objective=objective,
        config=config,
    )

    analysis_context = _render_analysis_context(context, selected_chunks, config=config)
    analysis_context_compact = _render_analysis_context_compact(
        context, selected_chunks, config=config
    )
    decision_claim_matrix = _render_decision_claim_matrix(context, config=config)
    memory_context = _render_memory_context(context, config=config)

    return {
        "analysis_context": analysis_context,
        "analysis_context_compact": analysis_context_compact,
        "decision_claim_matrix": decision_claim_matrix,
        "memory_context": memory_context,
        "all_reports_text": _render_all_reports_text(state),
        "selected_chunk_ids": [chunk["id"] for chunk in selected_chunks],
        "context_stats": context.get("stats", {}),
    }


def create_report_context_node(config: Dict[str, Any] | None = None):
    def report_context_node(state: Dict[str, Any]) -> Dict[str, Any]:
        return {"report_context": build_report_context_index(state, config=config)}

    return report_context_node
