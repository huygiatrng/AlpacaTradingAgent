"""
cache_utils.py - Generic caching utilities for data fetching tools

This module provides a smart caching system that automatically:
1. Checks for cached data first
2. Falls back to API calls if cache doesn't exist
3. Saves API responses to cache for future use

This makes the offline/online distinction automatic and transparent.
"""

import os
import json
import hashlib
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Any, Optional, Union
from functools import wraps

from .config import get_config


def get_cache_dir() -> str:
    """Get the data cache directory from config."""
    config = get_config()
    cache_dir = config.get("data_cache_dir", "tradingagents/dataflows/data_cache")

    # Ensure directory exists
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    return cache_dir


def generate_cache_key(*args, **kwargs) -> str:
    """
    Generate a unique cache key from function arguments.

    Args:
        *args: Positional arguments
        **kwargs: Keyword arguments

    Returns:
        str: MD5 hash of the arguments
    """
    # Create a string representation of all arguments
    key_parts = [str(arg) for arg in args]
    key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
    key_string = "|".join(key_parts)

    # Return MD5 hash for a clean filename
    return hashlib.md5(key_string.encode()).hexdigest()


def get_cache_path(cache_category: str, cache_key: str, extension: str = "json") -> str:
    """
    Get the full path for a cache file.

    Args:
        cache_category: Category/subdirectory for this type of cache (e.g., 'alpaca_data', 'news', 'macro')
        cache_key: Unique identifier for this cached item
        extension: File extension (json, csv, etc.)

    Returns:
        str: Full path to cache file
    """
    cache_dir = get_cache_dir()
    category_dir = os.path.join(cache_dir, cache_category)
    Path(category_dir).mkdir(parents=True, exist_ok=True)

    return os.path.join(category_dir, f"{cache_key}.{extension}")


def save_to_cache(
    data: Any,
    cache_category: str,
    cache_key: str,
    extension: str = "json",
    metadata: Optional[dict] = None
) -> bool:
    """
    Save data to cache.

    Args:
        data: Data to cache (dict, DataFrame, or string)
        cache_category: Category/subdirectory for this type of cache
        cache_key: Unique identifier for this cached item
        extension: File extension (json, csv, etc.)
        metadata: Optional metadata to store alongside data (timestamps, parameters, etc.)

    Returns:
        bool: True if successfully saved, False otherwise
    """
    try:
        cache_path = get_cache_path(cache_category, cache_key, extension)

        # Add timestamp to metadata
        if metadata is None:
            metadata = {}
        metadata['cached_at'] = datetime.now().isoformat()

        if extension == "json":
            # Save as JSON
            cache_data = {
                "metadata": metadata,
                "data": data
            }
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f, indent=2, default=str)

        elif extension == "csv":
            # Save as CSV (for DataFrames)
            if isinstance(data, pd.DataFrame):
                data.to_csv(cache_path, index=False)
                # Save metadata separately
                metadata_path = cache_path.replace('.csv', '_metadata.json')
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
            else:
                raise ValueError("CSV format requires pandas DataFrame")

        else:
            # Save as plain text
            with open(cache_path, 'w') as f:
                f.write(str(data))
            # Save metadata separately
            metadata_path = cache_path.replace(f'.{extension}', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

        print(f"[CACHE] Saved to {cache_path}")
        return True

    except Exception as e:
        print(f"[CACHE] Error saving to cache: {e}")
        return False


def load_from_cache(
    cache_category: str,
    cache_key: str,
    extension: str = "json",
    max_age_hours: Optional[int] = None
) -> Optional[Any]:
    """
    Load data from cache if it exists and is not too old.

    Args:
        cache_category: Category/subdirectory for this type of cache
        cache_key: Unique identifier for this cached item
        extension: File extension (json, csv, etc.)
        max_age_hours: Maximum age of cache in hours (None = no expiration)

    Returns:
        Cached data if found and valid, None otherwise
    """
    try:
        cache_path = get_cache_path(cache_category, cache_key, extension)

        # Check if cache file exists
        if not os.path.exists(cache_path):
            return None

        # Check cache age if max_age_hours is specified
        if max_age_hours is not None:
            file_mtime = datetime.fromtimestamp(os.path.getmtime(cache_path))
            age = datetime.now() - file_mtime
            if age > timedelta(hours=max_age_hours):
                print(f"[CACHE] Cache expired (age: {age.total_seconds() / 3600:.1f} hours)")
                return None

        # Load data based on extension
        if extension == "json":
            with open(cache_path, 'r') as f:
                cache_data = json.load(f)
            print(f"[CACHE] Loaded from {cache_path}")
            return cache_data.get("data")

        elif extension == "csv":
            data = pd.read_csv(cache_path)
            print(f"[CACHE] Loaded from {cache_path}")
            return data

        else:
            with open(cache_path, 'r') as f:
                data = f.read()
            print(f"[CACHE] Loaded from {cache_path}")
            return data

    except Exception as e:
        print(f"[CACHE] Error loading from cache: {e}")
        return None


def with_cache(
    cache_category: str,
    extension: str = "json",
    max_age_hours: Optional[int] = None,
    key_params: Optional[list] = None
):
    """
    Decorator that adds automatic caching to a function.

    The decorator will:
    1. Generate a cache key from function arguments
    2. Check if cached data exists
    3. Return cached data if valid
    4. Otherwise, call the function and cache the result

    Args:
        cache_category: Category/subdirectory for this type of cache
        extension: File extension for cache files
        max_age_hours: Maximum age of cache in hours (None = no expiration)
        key_params: List of parameter names to use for cache key (None = use all params)

    Usage:
        @with_cache(cache_category="news", max_age_hours=24)
        def get_news(ticker: str, date: str) -> str:
            # Fetch news from API
            return news_data
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key from specified parameters or all parameters
            if key_params:
                # Extract only specified parameters
                import inspect
                sig = inspect.signature(func)
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()

                cache_args = []
                cache_kwargs = {}
                for param_name in key_params:
                    if param_name in bound_args.arguments:
                        cache_kwargs[param_name] = bound_args.arguments[param_name]

                cache_key = generate_cache_key(*cache_args, **cache_kwargs)
            else:
                # Use all parameters
                cache_key = generate_cache_key(*args, **kwargs)

            # Try to load from cache
            cached_data = load_from_cache(
                cache_category=cache_category,
                cache_key=cache_key,
                extension=extension,
                max_age_hours=max_age_hours
            )

            if cached_data is not None:
                return cached_data

            # Cache miss - call the function
            print(f"[CACHE] Cache miss for {func.__name__}, calling API...")
            result = func(*args, **kwargs)

            # Check if result is an error message
            is_error = False
            if isinstance(result, str):
                error_indicators = [
                    "Error", "ERROR:", "error:",
                    "No earnings data found", "No data found",
                    "Failed to fetch", "Invalid API response"
                ]
                is_error = any(indicator in result for indicator in error_indicators)

            # Save result to cache (only if not empty/error)
            if result and not is_error:
                # Extract metadata from function arguments for better cache management
                metadata = {
                    "function": func.__name__,
                    "args": str(args)[:200],  # Truncate long args
                    "kwargs": str(kwargs)[:200]
                }
                save_to_cache(
                    data=result,
                    cache_category=cache_category,
                    cache_key=cache_key,
                    extension=extension,
                    metadata=metadata
                )
            else:
                if is_error:
                    print(f"[CACHE] Not caching error response from {func.__name__}")

            return result

        return wrapper
    return decorator


def clear_cache(cache_category: Optional[str] = None, older_than_hours: Optional[int] = None):
    """
    Clear cached data.

    Args:
        cache_category: Specific category to clear (None = clear all)
        older_than_hours: Only clear cache older than this many hours (None = clear all)
    """
    cache_dir = get_cache_dir()

    if cache_category:
        target_dir = os.path.join(cache_dir, cache_category)
    else:
        target_dir = cache_dir

    if not os.path.exists(target_dir):
        print(f"[CACHE] No cache found at {target_dir}")
        return

    deleted_count = 0
    cutoff_time = datetime.now() - timedelta(hours=older_than_hours) if older_than_hours else None

    for root, dirs, files in os.walk(target_dir):
        for file in files:
            file_path = os.path.join(root, file)

            # Check age if specified
            if cutoff_time:
                file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                if file_mtime > cutoff_time:
                    continue

            try:
                os.remove(file_path)
                deleted_count += 1
            except Exception as e:
                print(f"[CACHE] Error deleting {file_path}: {e}")

    print(f"[CACHE] Deleted {deleted_count} cache files from {target_dir}")
