import chromadb
from chromadb.config import Settings
from openai import OpenAI
import numpy as np
from tradingagents.dataflows.config import get_openai_client_config, get_openai_embedding_model


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
