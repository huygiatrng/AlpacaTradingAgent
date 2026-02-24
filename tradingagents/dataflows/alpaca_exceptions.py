"""Custom exceptions for Alpaca API errors"""


class AlpacaAuthError(Exception):
    """Raised when Alpaca API authentication fails (401/403 errors)"""
    pass
