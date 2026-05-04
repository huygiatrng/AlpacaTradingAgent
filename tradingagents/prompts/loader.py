from __future__ import annotations

import os
from pathlib import Path
from typing import Any


PROMPT_DIR_ENV = "TRADINGAGENTS_PROMPT_DIR"
_DEFAULT_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


class PromptTemplateError(ValueError):
    """Raised when a prompt template cannot be loaded or rendered."""


def _template_roots() -> list[Path]:
    configured = os.getenv(PROMPT_DIR_ENV)
    roots = []
    if configured:
        roots.append(Path(configured).expanduser())
    roots.append(_DEFAULT_TEMPLATE_DIR)
    return roots


def _safe_template_path(name: str) -> Path:
    if not name or "\x00" in name:
        raise PromptTemplateError("Prompt template name is empty or invalid")

    template_name = name if name.endswith(".md") else f"{name}.md"
    relative_path = Path(template_name)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise PromptTemplateError(f"Unsafe prompt template path: {name}")
    return relative_path


def _resolve_template_path(root: Path, relative_path: Path) -> Path:
    root = root.resolve()
    path = (root / relative_path).resolve()
    if root != path and root not in path.parents:
        raise PromptTemplateError(f"Prompt template path escapes root: {relative_path}")
    return path


def load_prompt(name: str) -> str:
    """Load a prompt template by name from the configured prompt directory."""

    relative_path = _safe_template_path(name)
    for root in _template_roots():
        path = _resolve_template_path(root, relative_path)
        if path.exists():
            return path.read_text(encoding="utf-8")
    raise PromptTemplateError(f"Prompt template not found: {relative_path.as_posix()}")


def render_prompt(name: str, **values: Any) -> str:
    """Load and render a prompt template using Python format placeholders."""

    try:
        return load_prompt(name).format_map(values)
    except KeyError as exc:
        raise PromptTemplateError(
            f"Missing prompt value '{exc.args[0]}' for template '{name}'"
        ) from exc


def list_prompt_templates() -> list[str]:
    """Return available prompt template names relative to the active prompt root."""

    templates = set()
    for root in _template_roots():
        if root.exists():
            templates.update(
                path.relative_to(root).as_posix()
                for path in root.rglob("*.md")
                if path.is_file()
            )
    return sorted(templates)
