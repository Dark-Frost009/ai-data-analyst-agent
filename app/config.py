"""
Centralized application configuration.

All configuration values are loaded from environment variables (via a
local .env file in development, or real environment variables in
production/deployment). No secrets are read, stored, or hardcoded here.

"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load variables from a local .env file if present. In real deployments
# (containers, cloud platforms, CI) environment variables are injected
# directly, so a missing .env file here is expected and not an error.
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _get_int(name: str, default: int) -> int:
    """Read an integer environment variable, falling back to a default."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(
            f"Environment variable {name!r} must be an integer, got {raw!r}"
        ) from exc


@dataclass(frozen=True)
class AppConfig:
    """Immutable, type-safe application configuration."""

    # --- Application ---
    app_name: str = "AI Data Analyst Agent"
    app_env: str = "development"
    log_level: str = "INFO"

    # --- Language model (credentials are resolved only by the adapter) ---
    llm_provider: str = "groq"
    llm_model_id: str = "openai/gpt-oss-20b"
    llm_connect_timeout_seconds: int = 5
    llm_read_timeout_seconds: int = 45

    # --- Data & query limits ---
    # Shared limits enforced by the loader, executor and capacity guard.
    max_upload_size_mb: int = 200
    max_query_result_rows: int = 10_000
    query_timeout_seconds: int = 30
    max_concurrent_analyses: int = 1

    # --- SQL execution (DuckDB) resource limits ---
    # Defense-in-depth alongside utils/security.py's SQL-level checks —
    # these apply at the DuckDB connection itself, so they hold even if
    # a validation bug ever let something bad through.
    duckdb_memory_limit: str = "512MB"
    duckdb_thread_limit: int = 2

    def __post_init__(self) -> None:
        """Fail fast when deployment configuration is invalid."""
        positive_values = {
            "llm_connect_timeout_seconds": self.llm_connect_timeout_seconds,
            "llm_read_timeout_seconds": self.llm_read_timeout_seconds,
            "max_upload_size_mb": self.max_upload_size_mb,
            "max_query_result_rows": self.max_query_result_rows,
            "query_timeout_seconds": self.query_timeout_seconds,
            "max_concurrent_analyses": self.max_concurrent_analyses,
            "duckdb_thread_limit": self.duckdb_thread_limit,
        }

        for name, value in positive_values.items():
            if value <= 0:
                raise ValueError(
                    f"Configuration value {name!r} must be greater than zero."
                )

        required_text_values = {
            "app_name": self.app_name,
            "app_env": self.app_env,
            "llm_provider": self.llm_provider,
            "llm_model_id": self.llm_model_id,
            "duckdb_memory_limit": self.duckdb_memory_limit,
        }

        for name, value in required_text_values.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"Configuration value {name!r} must be a non-empty string."
                )

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Build config from environment variables, using dataclass defaults as fallbacks."""
        return cls(
            app_env=os.getenv("APP_ENV", cls.app_env),
            log_level=os.getenv("LOG_LEVEL", cls.log_level).upper(),
            llm_provider=os.getenv("LLM_PROVIDER", cls.llm_provider),
            llm_model_id=os.getenv("LLM_MODEL_ID", cls.llm_model_id),
            llm_connect_timeout_seconds=_get_int(
                "LLM_CONNECT_TIMEOUT_SECONDS",
                cls.llm_connect_timeout_seconds,
            ),
            llm_read_timeout_seconds=_get_int(
                "LLM_READ_TIMEOUT_SECONDS",
                cls.llm_read_timeout_seconds,
            ),
            max_upload_size_mb=_get_int("MAX_UPLOAD_SIZE_MB", cls.max_upload_size_mb),
            max_query_result_rows=_get_int(
                "MAX_QUERY_RESULT_ROWS", cls.max_query_result_rows
            ),
            query_timeout_seconds=_get_int(
                "QUERY_TIMEOUT_SECONDS", cls.query_timeout_seconds
            ),
            max_concurrent_analyses=_get_int(
                "MAX_CONCURRENT_ANALYSES", cls.max_concurrent_analyses
            ),
            duckdb_memory_limit=os.getenv("DUCKDB_MEMORY_LIMIT", cls.duckdb_memory_limit),
            duckdb_thread_limit=_get_int("DUCKDB_THREAD_LIMIT", cls.duckdb_thread_limit),
        )

    @property
    def is_production(self) -> bool:
        return self.app_env.lower() == "production"


# Module-level singleton. Import this from anywhere in the app:
#   from app.config import config
config = AppConfig.from_env()
