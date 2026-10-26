# API checks

Two collections cover the HTTP boundaries used by this project. Requests are stored as editable Postman v3 YAML; there is no second JSON export to maintain. The existing [Postman workspace](https://go.postman.co/workspace/f5d1eccc-8671-48d8-8214-5220544a0f27/overview) holds the shared copies.

| Collection | Checks | Target |
| --- | --- | --- |
| App | Dashboard HTML, required controls, provider-change callback registration | Running local app |
| Providers / AI | Authenticated model catalog or OpenRouter key details for the selected provider | Selected AI service |
| Providers / Alpaca | Account status and risk inputs; daily OHLCV bars | Paper account and IEX data |
| Providers / Finnhub | Quote price/timestamp; company news fields | Finnhub |
| Providers / FRED | Dated observations with usable numeric values | FRED |
| Providers / CryptoCompare | News titles, timestamps and source links | CryptoCompare |
| Providers / FXMacroData | Calendar response envelope; empty periods are valid | Authenticated USD calendar |
| Providers / Alpha Vantage | Fallback quote, symbol and positive price | Alpha Vantage |

## Run from the repository

Use the project's Python environment. Install the CLI once if it is missing:

```powershell
npm install -g postman-cli@1.77.0
.venv/Scripts/python.exe scripts/check_apis.py lint
.venv/Scripts/python.exe scripts/check_apis.py app
.venv/Scripts/python.exe scripts/check_apis.py providers --provider openai
```

Open the desktop app before the `app` check. If it uses another port, pass `--app-url http://127.0.0.1:PORT`. The commands also work on Linux with the appropriate Python executable. Local runs do not need Postman login; CLI 1.77.0 may still print a login notice before running successfully.

The provider runner loads credentials from the desktop app's `%LOCALAPPDATA%/AlpacaTradingAgent/.env` when present, otherwise the repository `.env`. Process environment values override that file. Use `--env-file PATH` to choose a different file. It prints which services are selected and which are skipped. An explicitly selected service with missing credentials fails before making requests; a partially configured Alpaca pair also fails.

```powershell
.venv/Scripts/python.exe scripts/check_apis.py providers --provider google --only ai
.venv/Scripts/python.exe scripts/check_apis.py providers --only alpaca,finnhub,fred
.venv/Scripts/python.exe scripts/check_apis.py providers --provider ollama --only ai
```

Provider names match the app: `openai`, `google`, `anthropic`, `xai`, `minimax`, `deepseek`, `qwen`, `glm`, `openrouter`, `local_openai`, `ollama`. Use `--ai-base-url` for a custom endpoint. Local servers use optional `LOCAL_LLM_API_KEY`, so a remote OpenAI key is not sent to a local server by default. Azure requires a deployment-specific inference request; use **Test selected models** in the app instead. A catalog/key check does not prove that a selected model can generate text or that credit is available for inference.

## Use the Postman app

Open the linked workspace and choose **AlpacaTradingAgent - Local**. The App collection is ready to run while the desktop app is open. For provider checks, set `ai_provider` and supply only the credentials you use as local values or Postman Vault references. Run the relevant service folder. Missing optional keys produce a **SKIPPED** console message; skipped requests are not successful checks. `symbol` and `fred_series` are collection variables you can change.

Shared environment values for credentials are intentionally empty. Do not use **Share value** for a key or paste keys into request URLs. The command-line runner can reuse the app's existing credentials without saving them in Postman: it writes a temporary environment outside the repository, removes it after the run, disables cloud run events and redacts keys from console reports. Do not use verbose/raw exports for authenticated runs; FRED and Alpha Vantage require keys in query parameters. Never push a credential-filled local environment to the workspace.

## Results and maintenance

The latest redacted console reports are `postman/reports/latest-app.txt` and `latest-providers.txt`, both ignored by Git. Exit code `0` means the executed assertions passed, `1` means a run/lint failure, and `2` means configuration prevented a run. Check the skipped list as well as the exit code. HTTP errors, provider errors inside HTTP 200, missing fields and unusable numeric data fail checks. No orders are placed, positions closed, settings changed or paid inference requests sent.

CI lints both collections and the shared environment, then runs the App collection against an isolated instance on Python 3.12. Live providers are checked manually so third-party outages, quotas and credentials do not control pull-request results. Existing Python tests cover rejected orders, duplicate submissions, kill switches, timeouts and provider failures; keep those tests rather than adding a second mock trading system here. These checks establish connectivity and response usability, not trading profitability or model quality.

Edit the YAML files, run lint and the affected checks, then review the diff before syncing the two collections and blank environment to the existing workspace. Use create/update sync, never force-sync deletion. Keep generated skill copies, reports, credentials and vendor SDK/spec dumps out of commits.

The response checks follow the [Alpaca Trading API](https://docs.alpaca.markets/us/reference/getaccount-1), [Alpaca bars API](https://docs.alpaca.markets/us/reference/stockbars), [Finnhub API](https://finnhub.io/docs/api), [FRED observations API](https://fred.stlouisfed.org/docs/api/fred/series_observations.html), [CryptoCompare API](https://min-api.cryptocompare.com/documentation), [FXMacroData reference](https://fxmacrodata.com/documentation/reference) and [Alpha Vantage documentation](https://www.alphavantage.co/documentation/), together with the app's provider configuration.
