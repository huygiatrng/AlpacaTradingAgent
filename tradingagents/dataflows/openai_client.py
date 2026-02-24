"""Shared OpenAI client for thread-safe concurrent API calls."""
from openai import OpenAI
import threading
from typing import Optional
from .config import get_api_key

_client: Optional[OpenAI] = None
_lock = threading.Lock()


def get_openai_client(timeout: float = 300.0) -> OpenAI:
    """
    Get or create the shared OpenAI client instance.

    Thread-safe singleton pattern ensures all threads share the same
    HTTP connection pool for optimal parallel performance.

    Args:
        timeout: Request timeout in seconds (default: 300.0)

    Returns:
        Shared OpenAI client instance
    """
    global _client

    if _client is None:
        with _lock:
            # Double-check locking pattern
            if _client is None:
                api_key = get_api_key("openai_api_key", "OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")

                _client = OpenAI(
                    api_key=api_key,
                    timeout=timeout,
                    max_retries=3
                )

    return _client
