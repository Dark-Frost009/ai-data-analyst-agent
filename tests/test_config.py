import pytest

from app.config import AppConfig


@pytest.mark.parametrize(
    "setting",
    [
        "bedrock_connect_timeout_seconds",
        "bedrock_read_timeout_seconds",
        "max_upload_size_mb",
        "max_query_result_rows",
        "query_timeout_seconds",
        "max_concurrent_analyses",
        "duckdb_thread_limit",
    ],
)
@pytest.mark.parametrize("invalid_value", [0, -1])
def test_config_rejects_non_positive_operational_limits(
    setting,
    invalid_value,
):
    with pytest.raises(ValueError, match="must be greater than zero"):
        AppConfig(**{setting: invalid_value})


@pytest.mark.parametrize(
    "setting",
    [
        "app_name",
        "app_env",
        "aws_region",
        "bedrock_model_id",
        "duckdb_memory_limit",
    ],
)
@pytest.mark.parametrize("invalid_value", ["", "   "])
def test_config_rejects_blank_required_text_values(
    setting,
    invalid_value,
):
    with pytest.raises(ValueError, match="must be a non-empty string"):
        AppConfig(**{setting: invalid_value})


def test_config_accepts_defaults():
    config = AppConfig()

    assert config.max_concurrent_analyses == 1
    assert config.bedrock_connect_timeout_seconds == 5
    assert config.bedrock_read_timeout_seconds == 45