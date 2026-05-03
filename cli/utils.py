import questionary
from typing import List, Optional, Tuple, Dict
from rich import console
from cli.models import AnalystType
from tradingagents.openai_model_registry import (
    get_llm_provider_options,
    get_model_options_for_provider,
)

ANALYST_ORDER = [
    ("Market Analyst", AnalystType.MARKET),
    ("Social Media Analyst", AnalystType.SOCIAL),
    ("News Analyst", AnalystType.NEWS),
    ("Fundamentals Analyst", AnalystType.FUNDAMENTALS),
    ("Macro Analyst", AnalystType.MACRO),
]


def get_ticker() -> str:
    """Prompt the user to enter a ticker symbol."""
    ticker = questionary.text(
        "Enter the ticker symbol to analyze:",
        validate=lambda x: len(x.strip()) > 0 or "Please enter a valid ticker symbol.",
        style=questionary.Style(
            [
                ("text", "fg:green"),
                ("highlighted", "noinherit"),
            ]
        ),
    ).ask()

    if not ticker:
        console.print("\n[red]No ticker symbol provided. Exiting...[/red]")
        exit(1)

    return ticker.strip().upper()


def get_analysis_date() -> str:
    """Prompt the user to enter a date in YYYY-MM-DD format."""
    import re
    from datetime import datetime

    def validate_date(date_str: str) -> bool:
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
            return False
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
            return True
        except ValueError:
            return False

    date = questionary.text(
        "Enter the analysis date (YYYY-MM-DD):",
        validate=lambda x: validate_date(x.strip())
        or "Please enter a valid date in YYYY-MM-DD format.",
        style=questionary.Style(
            [
                ("text", "fg:green"),
                ("highlighted", "noinherit"),
            ]
        ),
    ).ask()

    if not date:
        console.print("\n[red]No date provided. Exiting...[/red]")
        exit(1)

    return date.strip()


def select_analysts() -> List[AnalystType]:
    """Select analysts using an interactive checkbox."""
    choices = questionary.checkbox(
        "Select Your [Analysts Team]:",
        choices=[
            questionary.Choice(display, value=value) for display, value in ANALYST_ORDER
        ],
        instruction="\n- Press Space to select/unselect analysts\n- Press 'a' to select/unselect all\n- Press Enter when done",
        validate=lambda x: len(x) > 0 or "You must select at least one analyst.",
        style=questionary.Style(
            [
                ("checkbox-selected", "fg:green"),
                ("selected", "fg:green noinherit"),
                ("highlighted", "noinherit"),
                ("pointer", "noinherit"),
            ]
        ),
    ).ask()

    if not choices:
        console.print("\n[red]No analysts selected. Exiting...[/red]")
        exit(1)

    return choices


def select_research_depth() -> int:
    """Select research depth using an interactive selection."""

    # Define research depth options with their corresponding values
    DEPTH_OPTIONS = [
        ("Shallow - Quick research, few debate and strategy discussion rounds", 1),
        ("Medium - Middle ground, moderate debate rounds and strategy discussion", 3),
        ("Deep - Comprehensive research, in depth debate and strategy discussion", 5),
    ]

    choice = questionary.select(
        "Select Your [Research Depth]:",
        choices=[
            questionary.Choice(display, value=value) for display, value in DEPTH_OPTIONS
        ],
        instruction="\n- Use arrow keys to navigate\n- Press Enter to select",
        style=questionary.Style(
            [
                ("selected", "fg:yellow noinherit"),
                ("highlighted", "fg:yellow noinherit"),
                ("pointer", "fg:yellow noinherit"),
            ]
        ),
    ).ask()

    if choice is None:
        console.print("\n[red]No research depth selected. Exiting...[/red]")
        exit(1)

    return choice


def select_llm_provider() -> str:
    provider_options = [
        (option["label"], option["value"])
        for option in get_llm_provider_options()
    ]
    choice = questionary.select(
        "Select Your [LLM Provider]:",
        choices=[questionary.Choice(display, value=value) for display, value in provider_options],
        instruction="\n- Use arrow keys to navigate\n- Press Enter to select",
        style=questionary.Style(
            [
                ("selected", "fg:cyan noinherit"),
                ("highlighted", "fg:cyan noinherit"),
                ("pointer", "fg:cyan noinherit"),
            ]
        ),
    ).ask()
    if choice is None:
        console.print("\n[red]No LLM provider selected. Exiting...[/red]")
        exit(1)
    return choice


def select_checkpoint_enabled() -> bool:
    return bool(
        questionary.confirm(
            "Enable checkpoint resume for failed runs?",
            default=False,
            style=questionary.Style([("question", "fg:cyan")]),
        ).ask()
    )


def get_output_language() -> str:
    language = questionary.text(
        "Output language:",
        default="English",
        style=questionary.Style([("text", "fg:green"), ("highlighted", "noinherit")]),
    ).ask()
    return (language or "English").strip()


def ask_gemini_thinking_config() -> str:
    choice = questionary.select(
        "Gemini thinking mode:",
        choices=[
            questionary.Choice("Provider default", ""),
            questionary.Choice("High thinking", "high"),
            questionary.Choice("Minimal / disabled", "minimal"),
        ],
        style=questionary.Style([("selected", "fg:cyan noinherit"), ("highlighted", "fg:cyan noinherit")]),
    ).ask()
    return choice or ""


def ask_anthropic_effort() -> str:
    choice = questionary.select(
        "Claude effort:",
        choices=[
            questionary.Choice("Provider default", ""),
            questionary.Choice("High", "high"),
            questionary.Choice("Medium", "medium"),
            questionary.Choice("Low", "low"),
        ],
        style=questionary.Style([("selected", "fg:cyan noinherit"), ("highlighted", "fg:cyan noinherit")]),
    ).ask()
    return choice or ""


def get_backend_url() -> str:
    url = questionary.text(
        "Optional backend URL (leave blank for provider default):",
        default="",
        style=questionary.Style([("text", "fg:green"), ("highlighted", "noinherit")]),
    ).ask()
    return (url or "").strip()


def _prompt_custom_model_id(provider: str, mode: str) -> str:
    provider_key = (provider or "").lower()
    label = "deployment name" if provider_key == "azure" else "model ID"
    model = questionary.text(
        f"Enter {provider_key or 'provider'} {mode}-thinking {label}:",
        validate=lambda x: len(x.strip()) > 0 or f"Please enter a {label}.",
        style=questionary.Style([("text", "fg:green"), ("highlighted", "noinherit")]),
    ).ask()
    if not model:
        console.print(f"\n[red]No {label} provided. Exiting...[/red]")
        exit(1)
    return model.strip()


def select_shallow_thinking_agent(provider: str = "openai") -> str:
    """Select shallow thinking llm engine using an interactive selection."""

    SHALLOW_AGENT_OPTIONS = [
        (option["label"], option["value"])
        for option in get_model_options_for_provider(provider, "quick")
    ]

    choice = questionary.select(
        "Select Your [Quick-Thinking LLM Engine]:",
        choices=[
            questionary.Choice(display, value=value)
            for display, value in SHALLOW_AGENT_OPTIONS
        ],
        instruction="\n- Use arrow keys to navigate\n- Press Enter to select",
        style=questionary.Style(
            [
                ("selected", "fg:magenta noinherit"),
                ("highlighted", "fg:magenta noinherit"),
                ("pointer", "fg:magenta noinherit"),
            ]
        ),
    ).ask()

    if choice is None:
        console.print(
            "\n[red]No shallow thinking llm engine selected. Exiting...[/red]"
        )
        exit(1)

    if choice == "custom":
        return _prompt_custom_model_id(provider, "quick")

    return choice


def select_deep_thinking_agent(provider: str = "openai") -> str:
    """Select deep thinking llm engine using an interactive selection."""

    DEEP_AGENT_OPTIONS = [
        (option["label"], option["value"])
        for option in get_model_options_for_provider(provider, "deep")
    ]

    choice = questionary.select(
        "Select Your [Deep-Thinking LLM Engine]:",
        choices=[
            questionary.Choice(display, value=value)
            for display, value in DEEP_AGENT_OPTIONS
        ],
        instruction="\n- Use arrow keys to navigate\n- Press Enter to select",
        style=questionary.Style(
            [
                ("selected", "fg:magenta noinherit"),
                ("highlighted", "fg:magenta noinherit"),
                ("pointer", "fg:magenta noinherit"),
            ]
        ),
    ).ask()

    if choice is None:
        console.print("\n[red]No deep thinking llm engine selected. Exiting...[/red]")
        exit(1)

    if choice == "custom":
        return _prompt_custom_model_id(provider, "deep")

    return choice
