"""Central configuration, read from environment variables with safe defaults."""
from __future__ import annotations

import os
from dataclasses import dataclass


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    model: str
    api_key: str | None
    max_rows_returned: int = 200      # rows kept for display / charts
    max_rows_to_llm: int = 40         # rows echoed back to the LLM per tool call
    max_upload_mb: int = 50
    max_agent_steps: int = 8          # tool-calling loop guard
    query_timeout_s: int = 20
    cache_size: int = 128
    max_history_turns: int = 10
    max_output_tokens: int = 2048
    log_dir: str = "logs"
    app_password: str | None = None   # optional password gate for the UI


def get_settings() -> Settings:
    return Settings(
        model=os.getenv("ANALYST_MODEL", "claude-sonnet-5-5"),
        api_key=os.getenv("ANTHROPIC_API_KEY") or None,
        max_rows_returned=_int("ANALYST_MAX_ROWS", 200),
        max_rows_to_llm=_int("ANALYST_MAX_ROWS_TO_LLM", 40),
        max_upload_mb=_int("ANALYST_MAX_UPLOAD_MB", 50),
        max_agent_steps=_int("ANALYST_MAX_STEPS", 8),
        query_timeout_s=_int("ANALYST_QUERY_TIMEOUT", 20),
        cache_size=_int("ANALYST_CACHE_SIZE", 128),
        max_history_turns=_int("ANALYST_HISTORY_TURNS", 10),
        max_output_tokens=_int("ANALYST_MAX_TOKENS", 2048),
        log_dir=os.getenv("ANALYST_LOG_DIR", "logs"),
        app_password=os.getenv("APP_PASSWORD") or None,
    )
