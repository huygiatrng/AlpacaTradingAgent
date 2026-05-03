import json
import os
import re
import pandas as pd
from datetime import date, timedelta, datetime
from typing import Annotated

SavePathType = Annotated[str, "File path to save data. If None, data is not saved."]


_SAFE_TICKER_RE = re.compile(r"^[A-Za-z0-9._\-/\^]+$")


def safe_ticker_component(value: str, *, max_len: int = 64) -> str:
    """Return a filesystem-safe ticker component.

    Stocks and exchange-qualified tickers are preserved. Crypto pairs are
    normalized for filenames by replacing "/" with "_", e.g. BTC/USD -> BTC_USD.
    """
    if not isinstance(value, str) or not value:
        raise ValueError(f"ticker must be a non-empty string, got {value!r}")
    if value != value.strip():
        raise ValueError(f"ticker cannot contain leading/trailing whitespace: {value!r}")
    if len(value) > max_len:
        raise ValueError(f"ticker exceeds {max_len} chars: {value!r}")
    if "\\" in value or "\x00" in value or any(ch.isspace() for ch in value):
        raise ValueError(f"ticker contains invalid path characters: {value!r}")
    if ".." in value or set(value) == {"."}:
        raise ValueError(f"ticker cannot traverse directories: {value!r}")
    if not _SAFE_TICKER_RE.fullmatch(value):
        raise ValueError(f"ticker contains characters not allowed in a path: {value!r}")

    safe = value.replace("/", "_")
    if not safe or set(safe) <= {".", "_"}:
        raise ValueError(f"ticker cannot be used as a path component: {value!r}")
    return safe


def save_output(data: pd.DataFrame, tag: str, save_path: SavePathType = None) -> None:
    if save_path:
        data.to_csv(save_path, encoding="utf-8")
        print(f"{tag} saved to {save_path}")


def get_current_date():
    return date.today().strftime("%Y-%m-%d")


def decorate_all_methods(decorator):
    def class_decorator(cls):
        for attr_name, attr_value in cls.__dict__.items():
            if callable(attr_value):
                setattr(cls, attr_name, decorator(attr_value))
        return cls

    return class_decorator


def get_next_weekday(date):

    if not isinstance(date, datetime):
        date = datetime.strptime(date, "%Y-%m-%d")

    if date.weekday() >= 5:
        days_to_add = 7 - date.weekday()
        next_weekday = date + timedelta(days=days_to_add)
        return next_weekday
    else:
        return date
