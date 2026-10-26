"""Build a self-contained Windows executable without credentials or user data."""

from pathlib import Path
import os
from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules, copy_metadata

root = Path(SPECPATH).parent
datas = [(str(root / "webui" / "assets"), "webui/assets"),
         (str(root / "tradingagents" / "prompts" / "templates"), "tradingagents/prompts/templates")]
binaries = []
hiddenimports = collect_submodules("tradingagents") + collect_submodules("webui")
for package in ("dash", "dash_bootstrap_components", "chromadb", "tiktoken_ext", "langgraph.checkpoint.sqlite"):
    data, binary, hidden = collect_all(package)
    datas += data
    binaries += binary
    hiddenimports += hidden
for package in ("exchange_calendars", "curl_cffi", "certifi"):
    datas += collect_data_files(package)
for package in ("chromadb", "langchain_core", "langchain_openai", "langchain_anthropic",
                "langchain_google_genai", "langgraph", "langgraph-checkpoint", "langgraph-checkpoint-sqlite",
                "openai", "pydantic", "dash", "dash_bootstrap_components"):
    datas += copy_metadata(package)

a = Analysis([str(root / "desktop_launcher.py")], pathex=[str(root)], binaries=binaries,
             datas=datas, hiddenimports=hiddenimports,
             excludes=["torch", "tensorflow", "transformers", "sentence_transformers", "gradio",
                       "IPython", "jupyter", "pytest", "matplotlib", "scipy", "sympy", "notebook"],
             noarchive=False)
pyz = PYZ(a.pure)
options = dict(name="AlpacaTradingAgent", debug=False, strip=False, upx=False,
               console=False, icon=str(root / "webui" / "assets" / "app.ico"))
if os.environ.get("TRADINGAGENTS_BUILD_ONEFILE") == "1":
    exe = EXE(pyz, a.scripts, a.binaries, a.datas, [], **options)
else:
    # Installed libraries stay beside the launcher instead of being extracted
    # into a temporary directory on every launch.
    exe = EXE(pyz, a.scripts, [], exclude_binaries=True, **options)
    bundle = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="AlpacaTradingAgent")
