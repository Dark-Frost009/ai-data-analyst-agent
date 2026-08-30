"""
Centralized application configuration.

All configuration values are loaded from environment variables (via a
local .env file in development, or real environment variables in
production/deployment). No secrets are read, stored, or hardcoded here.

AWS credentials specifically are NEVER handled by this module — boto3
resolves them on its own via the standard AWS credential chain
(`aws configure`, an IAM role, or AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
environment variables set outside of this project).
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

    # --- AWS / Amazon Bedrock ---
    # Only non-secret settings live here (region, model id). Credentials
    # are intentionally out of scope for this class.
    #
    # NOTE: many Bedrock models cannot be invoked on-demand using the bare
    # model ID — they require an *inference profile* ID instead
    # (regionally prefixed, e.g. "us.", "eu.", "apac.", or account-wide
    # "global."), or they raise ValidationException: "... with on-demand
    # throughput isn't supported". The values below are the region +
    # inference profile ID confirmed working via a live Bedrock smoke
    # test. Verify under Bedrock -> Cross-region inference in the AWS
    # console if this ever needs to change.
    aws_region: str = "ap-south-1"
    bedrock_model_id: str = "apac.amazon.nova-lite-v1:0"

    # --- Data & query limits ---
    # Not enforced yet (added in the security/execution milestones), but
    # centralized here now so every future component reads limits from a
    # single source of truth instead of redefining its own constants.
    max_upload_size_mb: int = 200
    max_query_result_rows: int = 10_000
    query_timeout_seconds: int = 30

    # --- SQL execution (DuckDB) resource limits ---
    # Defense-in-depth alongside utils/security.py's SQL-level checks —
    # these apply at the DuckDB connection itself, so they hold even if
    # a validation bug ever let something bad through.
    duckdb_memory_limit: str = "512MB"
    duckdb_thread_limit: int = 2

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Build config from environment variables, using dataclass defaults as fallbacks."""
        return cls(
            app_env=os.getenv("APP_ENV", cls.app_env),
            log_level=os.getenv("LOG_LEVEL", cls.log_level).upper(),
            aws_region=os.getenv("AWS_REGION", cls.aws_region),
            bedrock_model_id=os.getenv("BEDROCK_MODEL_ID", cls.bedrock_model_id),
            max_upload_size_mb=_get_int("MAX_UPLOAD_SIZE_MB", cls.max_upload_size_mb),
            max_query_result_rows=_get_int(
                "MAX_QUERY_RESULT_ROWS", cls.max_query_result_rows
            ),
            query_timeout_seconds=_get_int(
                "QUERY_TIMEOUT_SECONDS", cls.query_timeout_seconds
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