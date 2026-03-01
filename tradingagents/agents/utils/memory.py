import chromadb
import threading
from openai import OpenAI
import numpy as np
from tradingagents.dataflows.config import get_api_key


class FinancialSituationMemory:
    # chromadb.EphemeralClient() is not thread-safe in 1.x — serialize instantiation
    _init_lock = threading.Lock()

    def __init__(self, name):
        # Get API key from environment variables or config
        api_key = get_api_key("openai_api_key", "OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)
        with FinancialSituationMemory._init_lock:
            self.chroma_client = chromadb.EphemeralClient()
        self.situation_collection = self.chroma_client.get_or_create_collection(name=name)

    def _summarize_text(self, text, target_chars=20000):
        """Use AI to intelligently summarize text that exceeds character limit"""
        print(f"[MEMORY] Summarizing text ({len(text)} chars) to fit embedding limits...")

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Fast and cost-effective for summarization
                messages=[
                    {
                        "role": "system",
                        "content": "You are a financial analysis summarizer. Condense the given text while preserving all key insights, data points, decisions, and reasoning. Maintain technical accuracy and important numerical details. Target length: approximately 20,000 characters or less."
                    },
                    {
                        "role": "user",
                        "content": f"Summarize this financial analysis while preserving all critical information:\n\n{text}"
                    }
                ],
                temperature=0.3,  # Lower temperature for more focused summarization
            )
            summarized = response.choices[0].message.content
            print(f"[MEMORY] Successfully summarized to {len(summarized)} characters")
            return summarized
        except Exception as e:
            # Fallback to truncation if summarization fails
            print(f"[MEMORY] Warning: Summarization failed ({str(e)}), falling back to truncation")
            half_chars = target_chars // 2
            return text[:half_chars] + "\n...[TRUNCATED]...\n" + text[-half_chars:]

    def get_embedding(self, text):
        """Get OpenAI embedding for a text"""
        # text-embedding-ada-002 has a max context length of 8192 tokens
        # Conservative estimate: ~3 characters per token for safety margin
        max_chars = 24000  # ~8000 tokens * 3 chars/token

        if len(text) > max_chars:
            # Use AI to intelligently summarize instead of simple truncation
            text = self._summarize_text(text, target_chars=20000)

        response = self.client.embeddings.create(
            model="text-embedding-ada-002", input=text
        )
        return response.data[0].embedding

    def add_situations(self, situations_and_advice):
        """Add financial situations and their corresponding advice. Parameter is a list of tuples (situation, rec)"""

        situations = []
        advice = []
        ids = []
        embeddings = []

        offset = self.situation_collection.count()

        for i, (situation, recommendation) in enumerate(situations_and_advice):
            situations.append(situation)
            advice.append(recommendation)
            ids.append(str(offset + i))
            embeddings.append(self.get_embedding(situation))

        self.situation_collection.add(
            documents=situations,
            metadatas=[{"recommendation": rec} for rec in advice],
            embeddings=embeddings,
            ids=ids,
        )

    def get_memories(self, current_situation, n_matches=1):
        """Find matching recommendations using OpenAI embeddings"""
        query_embedding = self.get_embedding(current_situation)

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
