import os
import pathlib
import importlib.util
import unittest

MODULE_PATH = pathlib.Path(__file__).resolve().parents[1] / "tradingagents" / "dataflows" / "config.py"
SPEC = importlib.util.spec_from_file_location("config_under_test", MODULE_PATH)
config_module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(config_module)


class LocalLLMConfigTests(unittest.TestCase):
    def setUp(self):
        self.original_config = config_module.get_config()
        self.original_runtime_keys = config_module.get_runtime_api_keys()
        self.original_env = {
            "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
            "OPENAI_USE_LOCAL": os.environ.get("OPENAI_USE_LOCAL"),
            "OPENAI_BASE_URL": os.environ.get("OPENAI_BASE_URL"),
            "OPENAI_EMBEDDING_MODEL": os.environ.get("OPENAI_EMBEDDING_MODEL"),
        }

        config_module.clear_runtime_api_keys()
        config_module.set_config(
            {
                "openai_api_key": None,
                "openai_use_local": False,
                "openai_base_url": None,
                "openai_embedding_model": "text-embedding-ada-002",
            }
        )
        for key in self.original_env:
            os.environ.pop(key, None)

    def tearDown(self):
        config_module.clear_runtime_api_keys()
        if self.original_runtime_keys:
            config_module.set_runtime_api_keys(self.original_runtime_keys)
        config_module.set_config(self.original_config)

        for key, value in self.original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_local_client_config_uses_base_url_and_default_api_key(self):
        config_module.set_config(
            {
                "openai_use_local": True,
                "openai_base_url": "http://localhost:1234/v1",
                "openai_api_key": None,
            }
        )

        client_config = config_module.get_openai_client_config()

        self.assertEqual(client_config["base_url"], "http://localhost:1234/v1")
        self.assertEqual(client_config["api_key"], "local-llm")

    def test_cloud_client_config_stays_on_openai_key_when_local_disabled(self):
        config_module.set_config(
            {
                "openai_use_local": False,
                "openai_base_url": "http://localhost:1234/v1",
                "openai_api_key": "sk-test",
            }
        )

        client_config = config_module.get_openai_client_config()

        self.assertEqual(client_config, {"api_key": "sk-test"})

    def test_environment_variables_enable_local_mode(self):
        os.environ["OPENAI_USE_LOCAL"] = "true"
        os.environ["OPENAI_BASE_URL"] = "http://localhost:11434/v1"

        self.assertTrue(config_module.is_local_openai_enabled())
        self.assertEqual(config_module.get_openai_base_url(), "http://localhost:11434/v1")
        self.assertEqual(
            config_module.get_openai_client_config(),
            {"api_key": "local-llm", "base_url": "http://localhost:11434/v1"},
        )

    def test_embedding_model_can_be_overridden(self):
        os.environ["OPENAI_EMBEDDING_MODEL"] = "nomic-embed-text"
        self.assertEqual(config_module.get_openai_embedding_model(), "nomic-embed-text")


if __name__ == "__main__":
    unittest.main()
