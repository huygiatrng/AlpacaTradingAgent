import re
import unittest
from pathlib import Path


REQUIRED_ENV_VARS = {
    "TRADINGAGENTS_RESULTS_DIR",
    "TRADINGAGENTS_CACHE_DIR",
    "TRADINGAGENTS_MEMORY_LOG_PATH",
    "LLM_PROVIDER",
    "OPENAI_API_KEY",
    "OPENAI_USE_LOCAL",
    "OPENAI_BASE_URL",
    "OPENAI_EMBEDDING_MODEL",
    "GOOGLE_API_KEY",
    "ANTHROPIC_API_KEY",
    "XAI_API_KEY",
    "DEEPSEEK_API_KEY",
    "DASHSCOPE_API_KEY",
    "ZHIPU_API_KEY",
    "OPENROUTER_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_DEPLOYMENT_NAME",
    "AZURE_OPENAI_API_VERSION",
    "ALPACA_API_KEY",
    "ALPACA_SECRET_KEY",
    "ALPACA_USE_PAPER",
    "FINNHUB_API_KEY",
    "FRED_API_KEY",
    "COINDESK_API_KEY",
    "ALPHA_VANTAGE_API_KEY",
}


class EnvSampleTests(unittest.TestCase):
    def test_env_sample_documents_required_runtime_keys(self):
        sample = Path("env.sample").read_text(encoding="utf-8")
        documented = set(re.findall(r"^([A-Z][A-Z0-9_]*)=", sample, flags=re.MULTILINE))

        self.assertFalse(
            REQUIRED_ENV_VARS - documented,
            f"env.sample is missing: {sorted(REQUIRED_ENV_VARS - documented)}",
        )


if __name__ == "__main__":
    unittest.main()
