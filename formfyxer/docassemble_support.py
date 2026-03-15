"""Support for docassemble configuration integration.

This module provides centralized support for accessing docassemble configuration
values, with graceful fallback when docassemble is not available.
"""

from typing import Optional, Dict, Any
import os

# Try to import docassemble config function at module level for performance
try:
    from docassemble.base.util import get_config  # type: ignore[import-not-found, import-untyped]

    _DOCASSEMBLE_AVAILABLE = True
    _da_get_config = get_config
except ImportError:
    get_config = None
    _DOCASSEMBLE_AVAILABLE = False
    _da_get_config = None


def get_openai_api_key(explicit_key: Optional[str] = None) -> Optional[str]:
    """Get OpenAI API key with fallback to docassemble config if available.

    This function implements the following priority order:
    1. Explicit API key parameter (highest priority)
    2. Docassemble get_config("openai api key")
    3. Docassemble get_config("open ai", {}).get("key")
    4. OPENAI_API_KEY environment variable (lowest priority)

    Args:
        explicit_key: Explicitly provided API key (takes highest precedence)

    Returns:
        The API key to use, or None if none found
    """
    if explicit_key:
        return explicit_key

    # Try docassemble config if available (cached import at module level)
    if _DOCASSEMBLE_AVAILABLE and _da_get_config:
        # Try the direct key first
        da_key = _da_get_config("openai api key")
        if da_key:
            return da_key
        # Try the nested config
        openai_config = _da_get_config("open ai", {})
        if isinstance(openai_config, dict) and "key" in openai_config:
            return openai_config["key"]

    # Fall back to environment variable
    return os.getenv("OPENAI_API_KEY")


def get_openai_api_key_from_sources(
    explicit_key: Optional[str] = None, creds: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """Get OpenAI API key with fallback through multiple sources.

    This function implements the following priority order:
    1. Explicit API key parameter (highest priority)
    2. Credentials dict "key" field
    3. Docassemble get_config("openai api key")
    4. Docassemble get_config("open ai", {}).get("key")
    5. OPENAI_API_KEY environment variable (lowest priority)

    Args:
        explicit_key: Explicitly provided API key (takes highest precedence)
        creds: Credentials dict that may contain a "key" field

    Returns:
        The API key to use, or None if none found
    """
    if explicit_key:
        return explicit_key

    if creds and "key" in creds:
        return creds["key"]

    # Delegate to the main function for docassemble and env var fallback
    return get_openai_api_key()


def get_openai_base_url(explicit_base_url: Optional[str] = None) -> Optional[str]:
    """Get OpenAI base URL with fallback through multiple sources.

    This function implements the following priority order:
    1. Explicit base URL parameter (highest priority)
    2. Docassemble get_config("openai base url")
    3. Docassemble get_config("open ai", {}).get("base url")
    4. Docassemble get_config("open ai", {}).get("base_url")
    5. OPENAI_BASE_URL environment variable (lowest priority)

    Args:
        explicit_base_url: Explicitly provided base URL (takes highest precedence)

    Returns:
        The base URL to use, or None if none found
    """
    if explicit_base_url:
        return explicit_base_url

    # Try docassemble config if available (cached import at module level)
    if _DOCASSEMBLE_AVAILABLE and _da_get_config:
        # Try the direct key first
        da_base_url = _da_get_config("openai base url")
        if da_base_url:
            return da_base_url
        # Try the nested config
        openai_config = _da_get_config("open ai", {})
        if isinstance(openai_config, dict):
            if "base url" in openai_config:
                return openai_config["base url"]
            if "base_url" in openai_config:
                return openai_config["base_url"]

    # Fall back to environment variable
    return os.getenv("OPENAI_BASE_URL")


def is_docassemble_available() -> bool:
    """Check if docassemble is available in the current environment.

    Returns:
        True if docassemble.base.util can be imported, False otherwise
    """
    return _DOCASSEMBLE_AVAILABLE
