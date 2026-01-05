"""
Example: How to Use Local LLMs with AlpacaTradingAgent

This file demonstrates how to configure the AlpacaTradingAgent to work with local LLMs
like LMStudio, Ollama, or vLLM instead of OpenAI's cloud API.

## Method 1: Environment Variables (.env file)

Create a .env file with:

```
# Enable local LLM mode
OPENAI_USE_LOCAL=true

# Set your local LLM server URL
OPENAI_BASE_URL=http://localhost:1234/v1

# API key can be anything for local LLMs
OPENAI_API_KEY=local-llm
```

## Method 2: Configuration Code

```python
from tradingagents.dataflows.config import set_config

# Configure for local LLM
config = {
    "openai_use_local": True,
    "openai_base_url": "http://192.168.1.100:1234/v1",  # Your local server
    "openai_api_key": "local-llm"  # Any value works for local
}
set_config(config)
```

## Supported Local LLM Servers:

1. **LMStudio** (default port 1234):
   - Base URL: http://localhost:1234/v1
   - Load a model and start the local server

2. **Ollama** (with OpenAI compatibility):
   - Base URL: http://localhost:11434/v1
   - Run: `OLLAMA_HOST=0.0.0.0:11434 ollama serve`

3. **vLLM** (with OpenAI API server):
   - Base URL: http://localhost:8000/v1
   - Run: `vllm serve model_name --api-key local-llm`

4. **Any OpenAI-compatible API server**

## Model Selection:

The agent will use whatever model your local server provides. Make sure to:
1. Load a capable model (7B+ parameters recommended)
2. Configure your server to use the model you want
3. Ensure the model supports function calling if using advanced features

## Testing Your Setup:

Here is a simple test:

```python
from tradingagents.dataflows.config import set_config, get_openai_client_config
from openai import OpenAI

# Configure for local LLM
set_config({
    "openai_use_local": True,
    "openai_base_url": "http://localhost:1234/v1"
})

# Test the connection
client_config = get_openai_client_config()
client = OpenAI(**client_config)

response = client.chat.completions.create(
    model="local-model",  # Your local model name
    messages=[{"role": "user", "content": "Hello, are you working?"}]
)
print(response.choices[0].message.content)
```
"""