import chromadb
from chromadb.config import Settings
from openai import OpenAI
import numpy as np
from pathlib import Path
import re
from tradingagents.dataflows.config import get_openai_client_config, get_openai_embedding_model
from tradingagents.agents.utils.agent_trading_modes import extract_recommendation


class FinancialSituationMemory:
    def __init__(self, name):
        client_config = get_openai_client_config()
        self.client = OpenAI(**client_config) if client_config else None
        self.embedding_model = get_openai_embedding_model()
        self.embeddings_enabled = self.client is not None
        self._warned_embedding_failure = False
        self.chroma_client = chromadb.Client(Settings(allow_reset=True))
        self.situation_collection = self.chroma_client.get_or_create_collection(name=name)

    def get_embedding(self, text):
        """Get OpenAI embedding for a text"""
        if not self.embeddings_enabled or self.client is None:
            return None

        # Truncate text if it exceeds the model's token limit
        # text-embedding-ada-002 has a max context length of 8192 tokens
        # Conservative estimate: ~3 characters per token for safety margin
        max_chars = 24000  # ~8000 tokens * 3 chars/token
        if len(text) > max_chars:
            # Take first part and last part to preserve both beginning and end context
            half_chars = max_chars // 2
            text = text[:half_chars] + "\n...[TRUNCATED]...\n" + text[-half_chars:]
            print(f"[MEMORY] Warning: Text truncated to ~{max_chars} characters for embedding")
        
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model, input=text
            )
            return response.data[0].embedding
        except Exception as exc:
            self.embeddings_enabled = False
            if not self._warned_embedding_failure:
                print(
                    "[MEMORY] Embeddings unavailable; reflection memory will be skipped "
                    f"for this run. ({exc})"
                )
                self._warned_embedding_failure = True
            return None

    def add_situations(self, situations_and_advice):
        """Add financial situations and their corresponding advice. Parameter is a list of tuples (situation, rec)"""
        if not self.embeddings_enabled:
            return

        situations = []
        advice = []
        ids = []
        embeddings = []

        offset = self.situation_collection.count()

        for i, (situation, recommendation) in enumerate(situations_and_advice):
            embedding = self.get_embedding(situation)
            if embedding is None:
                continue
            situations.append(situation)
            advice.append(recommendation)
            ids.append(str(offset + i))
            embeddings.append(embedding)

        if not embeddings:
            return

        self.situation_collection.add(
            documents=situations,
            metadatas=[{"recommendation": rec} for rec in advice],
            embeddings=embeddings,
            ids=ids,
        )

    def get_memories(self, current_situation, n_matches=1):
        """Find matching recommendations using OpenAI embeddings"""
        if not self.embeddings_enabled:
            return []

        query_embedding = self.get_embedding(current_situation)
        if query_embedding is None:
            return []

        results = self.situation_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_matches,
            include=["metadatas", "documents", "distances"],
        )

        matched_results = []
        for i in range(len(results["documents"][0])):
            matched_results.append(
                {
                    "matched_situation": results["documents"][0][i],
                    "recommendation": results["metadatas"][0][i]["recommendation"],
                    "similarity_score": 1 - results["distances"][0][i],
                }
            )

        return matched_results


if __name__ == "__main__":
    # Example usage
    matcher = FinancialSituationMemory()

    # Example data
    example_data = [
        (
            "High inflation rate with rising interest rates and declining consumer spending",
            "Consider defensive sectors like consumer staples and utilities. Review fixed-income portfolio duration.",
        ),
        (
            "Tech sector showing high volatility with increasing institutional selling pressure",
            "Reduce exposure to high-growth tech stocks. Look for value opportunities in established tech companies with strong cash flows.",
        ),
        (
            "Strong dollar affecting emerging markets with increasing forex volatility",
            "Hedge currency exposure in international positions. Consider reducing allocation to emerging market debt.",
        ),
        (
            "Market showing signs of sector rotation with rising yields",
            "Rebalance portfolio to maintain target allocations. Consider increasing exposure to sectors benefiting from higher rates.",
        ),
    ]

    # Add the example situations and recommendations
    matcher.add_situations(example_data)

    # Example query
    current_situation = """
    Market showing increased volatility in tech sector, with institutional investors 
    reducing positions and rising interest rates affecting growth stock valuations
    """

    try:
        recommendations = matcher.get_memories(current_situation, n_matches=2)

        for i, rec in enumerate(recommendations, 1):
            print(f"\nMatch {i}:")
            print(f"Similarity Score: {rec['similarity_score']:.2f}")
            print(f"Matched Situation: {rec['matched_situation']}")
            print(f"Recommendation: {rec['recommendation']}")

    except Exception as e:
        print(f"Error during recommendation: {str(e)}")


class TradingMemoryLog:
    """Append-only markdown decision log with pending outcome resolution."""

    _SEPARATOR = "\n\n<!-- ENTRY_END -->\n\n"
    _DECISION_RE = re.compile(r"DECISION:\n(.*?)(?=\nREFLECTION:|\Z)", re.DOTALL)
    _REFLECTION_RE = re.compile(r"REFLECTION:\n(.*?)$", re.DOTALL)

    def __init__(self, config: dict = None):
        cfg = config or {}
        path = cfg.get("memory_log_path")
        self._log_path = Path(path).expanduser() if path else None
        if self._log_path:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._max_entries = cfg.get("memory_log_max_entries")

    def store_decision(self, ticker: str, trade_date: str, final_trade_decision: str, trading_mode: str = "investment") -> None:
        if not self._log_path:
            return
        if self._log_path.exists():
            raw = self._log_path.read_text(encoding="utf-8")
            for line in raw.splitlines():
                if line.startswith(f"[{trade_date} | {ticker} |") and line.endswith("| pending]"):
                    return
        action = extract_recommendation(final_trade_decision, trading_mode) or "UNKNOWN"
        rating = self._extract_advisory_rating(final_trade_decision)
        tag = f"[{trade_date} | {ticker} | {action} | {rating or 'n/a'} | pending]"
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(f"{tag}\n\nDECISION:\n{final_trade_decision}{self._SEPARATOR}")

    def load_entries(self) -> list[dict]:
        if not self._log_path or not self._log_path.exists():
            return []
        text = self._log_path.read_text(encoding="utf-8")
        entries = []
        for raw in [e.strip() for e in text.split(self._SEPARATOR) if e.strip()]:
            parsed = self._parse_entry(raw)
            if parsed:
                entries.append(parsed)
        return entries

    def get_pending_entries(self, ticker: str | None = None) -> list[dict]:
        entries = [e for e in self.load_entries() if e.get("pending")]
        if ticker is None:
            return entries
        return [e for e in entries if e.get("ticker") == ticker]

    def get_past_context(self, ticker: str, n_same: int = 5, n_cross: int = 3) -> str:
        entries = [e for e in self.load_entries() if not e.get("pending")]
        same, cross = [], []
        for entry in reversed(entries):
            if entry["ticker"] == ticker and len(same) < n_same:
                same.append(entry)
            elif entry["ticker"] != ticker and len(cross) < n_cross:
                cross.append(entry)
            if len(same) >= n_same and len(cross) >= n_cross:
                break
        parts = []
        if same:
            parts.append(f"Past analyses of {ticker} (most recent first):")
            parts.extend(self._format_full(e) for e in same)
        if cross:
            parts.append("Recent cross-ticker lessons:")
            parts.extend(self._format_reflection_only(e) for e in cross)
        return "\n\n".join(parts)

    def update_with_outcome(
        self,
        ticker: str,
        trade_date: str,
        raw_return: float,
        alpha_return: float | None,
        holding_days: int,
        reflection: str,
    ) -> None:
        self.batch_update_with_outcomes([
            {
                "ticker": ticker,
                "trade_date": trade_date,
                "raw_return": raw_return,
                "alpha_return": alpha_return,
                "holding_days": holding_days,
                "reflection": reflection,
            }
        ])

    def batch_update_with_outcomes(self, updates: list[dict]) -> None:
        if not self._log_path or not self._log_path.exists() or not updates:
            return
        text = self._log_path.read_text(encoding="utf-8")
        blocks = text.split(self._SEPARATOR)
        update_map = {(u["trade_date"], u["ticker"]): u for u in updates}
        new_blocks = []
        for block in blocks:
            stripped = block.strip()
            if not stripped:
                new_blocks.append(block)
                continue
            lines = stripped.splitlines()
            tag_line = lines[0].strip()
            matched = False
            for (trade_date, ticker), upd in list(update_map.items()):
                if tag_line.startswith(f"[{trade_date} | {ticker} |") and tag_line.endswith("| pending]"):
                    fields = [f.strip() for f in tag_line[1:-1].split("|")]
                    raw_pct = f"{upd['raw_return']:+.1%}"
                    alpha_value = upd.get("alpha_return")
                    alpha_pct = f"{alpha_value:+.1%}" if alpha_value is not None else "n/a"
                    new_tag = (
                        f"[{trade_date} | {ticker} | {fields[2]} | {fields[3]} "
                        f"| {raw_pct} | {alpha_pct} | {upd['holding_days']}d]"
                    )
                    rest = "\n".join(lines[1:]).lstrip()
                    new_blocks.append(f"{new_tag}\n\n{rest}\n\nREFLECTION:\n{upd['reflection']}")
                    del update_map[(trade_date, ticker)]
                    matched = True
                    break
            if not matched:
                new_blocks.append(block)
        new_blocks = self._apply_rotation(new_blocks)
        tmp = self._log_path.with_suffix(".tmp")
        tmp.write_text(self._SEPARATOR.join(new_blocks), encoding="utf-8")
        tmp.replace(self._log_path)

    def _apply_rotation(self, blocks: list[str]) -> list[str]:
        if not self._max_entries or self._max_entries <= 0:
            return blocks
        resolved = [b for b in blocks if b.strip() and not b.strip().splitlines()[0].endswith("| pending]")]
        to_drop = max(0, len(resolved) - int(self._max_entries))
        kept = []
        for block in blocks:
            if to_drop and block.strip() and not block.strip().splitlines()[0].endswith("| pending]"):
                to_drop -= 1
                continue
            kept.append(block)
        return kept

    def _parse_entry(self, raw: str) -> dict | None:
        lines = raw.strip().splitlines()
        if not lines or not lines[0].startswith("[") or not lines[0].endswith("]"):
            return None
        fields = [f.strip() for f in lines[0][1:-1].split("|")]
        if len(fields) < 5:
            return None
        body = "\n".join(lines[1:]).strip()
        decision = self._DECISION_RE.search(body)
        reflection = self._REFLECTION_RE.search(body)
        return {
            "date": fields[0],
            "ticker": fields[1],
            "action": fields[2],
            "rating": fields[3],
            "pending": fields[4] == "pending",
            "raw": fields[4] if len(fields) > 5 else None,
            "alpha": fields[5] if len(fields) > 6 else None,
            "holding": fields[6] if len(fields) > 6 else None,
            "decision": decision.group(1).strip() if decision else "",
            "reflection": reflection.group(1).strip() if reflection else "",
        }

    def _format_full(self, entry: dict) -> str:
        tag = (
            f"[{entry['date']} | {entry['ticker']} | {entry['action']} | {entry['rating']} "
            f"| {entry.get('raw') or 'n/a'} | {entry.get('alpha') or 'n/a'} | {entry.get('holding') or 'n/a'}]"
        )
        parts = [tag, f"DECISION:\n{entry['decision']}"]
        if entry.get("reflection"):
            parts.append(f"REFLECTION:\n{entry['reflection']}")
        return "\n\n".join(parts)

    def _format_reflection_only(self, entry: dict) -> str:
        tag = f"[{entry['date']} | {entry['ticker']} | {entry['action']} | {entry.get('raw') or 'n/a'}]"
        return f"{tag}\n{entry.get('reflection') or entry.get('decision', '')[:300]}"

    @staticmethod
    def _extract_advisory_rating(text: str) -> str | None:
        for line in str(text).splitlines():
            if "advisory rating" in line.lower():
                return line.split(":", 1)[-1].strip(" *") or None
        return None
