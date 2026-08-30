"""
Tests for app.core.llm_client.BedrockClient.

Two kinds of tests live here on purpose:

1. Mocked unit tests (run by default, no network/AWS credentials needed).
   These verify our request building, response parsing, and error-mapping
   logic in isolation — pytest will run these every time.

2. ONE live integration smoke test, skipped by default, that makes a real
   call to Bedrock. This is what actually proves your AWS setup (credentials,
   region, and model access) works end-to-end.

   Run it explicitly with:

       RUN_LIVE_AWS_TESTS=1 pytest tests/test_llm_client.py -v -s
"""

import os
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError, NoCredentialsError

from app.config import config
from app.core.llm_client import (
    BedrockAccessDeniedError,
    BedrockAPIError,
    BedrockClient,
    BedrockCredentialsError,
    BedrockResponseFormatError,
    BedrockThrottlingError,
)


def _client_error(code: str, message: str = "error") -> ClientError:
    """Build a fake botocore ClientError for unit tests."""
    return ClientError(
        error_response={
            "Error": {
                "Code": code,
                "Message": message,
            }
        },
        operation_name="Converse",
    )


# --------------------------------------------------------------------------
# Mocked unit tests — no network, no AWS credentials required
# --------------------------------------------------------------------------


@patch("app.core.llm_client.boto3.client")
def test_client_uses_config_defaults_for_region_and_model(mock_boto_client):
    """
    BedrockClient(), called with no explicit args, must build its boto3
    client from app.config.config rather than using hardcoded values.

    This guards against the config and Bedrock client silently drifting
    apart.
    """
    mock_runtime = MagicMock()

    mock_runtime.converse.return_value = {
        "output": {
            "message": {
                "content": [
                    {
                        "text": "ok"
                    }
                ]
            }
        },
        "stopReason": "end_turn",
    }

    mock_boto_client.return_value = mock_runtime

    client = BedrockClient()
    client.generate_text("Say hello")

    mock_boto_client.assert_called_once_with(
        "bedrock-runtime",
        region_name=config.aws_region,
    )

    called_kwargs = mock_runtime.converse.call_args.kwargs

    assert called_kwargs["modelId"] == config.bedrock_model_id


@patch("app.core.llm_client.boto3.client")
def test_generate_text_success(mock_boto_client):
    """A valid Bedrock response should return the generated text."""
    mock_runtime = MagicMock()

    mock_runtime.converse.return_value = {
        "output": {
            "message": {
                "content": [
                    {
                        "text": "Hello from Bedrock"
                    }
                ]
            }
        },
        "stopReason": "end_turn",
        "usage": {
            "inputTokens": 5,
            "outputTokens": 4,
            "totalTokens": 9,
        },
    }

    mock_boto_client.return_value = mock_runtime

    client = BedrockClient(
        region_name="us-east-1",
        model_id="fake-model-id",
    )

    result = client.generate_text("Say hello")

    assert result == "Hello from Bedrock"

    mock_runtime.converse.assert_called_once()

    called_kwargs = mock_runtime.converse.call_args.kwargs

    assert called_kwargs["modelId"] == "fake-model-id"
    assert called_kwargs["messages"][0]["role"] == "user"
    assert called_kwargs["messages"][0]["content"][0]["text"] == "Say hello"


@patch("app.core.llm_client.boto3.client")
def test_generate_text_with_system_prompt(mock_boto_client):
    """A system prompt should be included in the Converse request."""
    mock_runtime = MagicMock()

    mock_runtime.converse.return_value = {
        "output": {
            "message": {
                "content": [
                    {
                        "text": "ok"
                    }
                ]
            }
        },
        "stopReason": "end_turn",
    }

    mock_boto_client.return_value = mock_runtime

    client = BedrockClient()

    client.generate_text(
        "Hi",
        system_prompt="You are terse.",
    )

    called_kwargs = mock_runtime.converse.call_args.kwargs

    assert called_kwargs["system"] == [
        {
            "text": "You are terse."
        }
    ]


def test_generate_text_rejects_empty_prompt():
    """
    An empty or whitespace-only prompt should be rejected before any
    AWS client is needed.
    """
    client = BedrockClient.__new__(BedrockClient)

    with pytest.raises(ValueError):
        BedrockClient.generate_text(client, "   ")


@patch("app.core.llm_client.boto3.client")
def test_generate_text_missing_credentials(mock_boto_client):
    """Missing AWS credentials should map to BedrockCredentialsError."""
    mock_runtime = MagicMock()

    mock_runtime.converse.side_effect = NoCredentialsError()

    mock_boto_client.return_value = mock_runtime

    client = BedrockClient()

    with pytest.raises(BedrockCredentialsError):
        client.generate_text("Say hello")


@patch("app.core.llm_client.boto3.client")
def test_generate_text_access_denied(mock_boto_client):
    """AccessDeniedException should map to BedrockAccessDeniedError."""
    mock_runtime = MagicMock()

    mock_runtime.converse.side_effect = _client_error(
        "AccessDeniedException"
    )

    mock_boto_client.return_value = mock_runtime

    client = BedrockClient()

    with pytest.raises(BedrockAccessDeniedError):
        client.generate_text("Say hello")


@patch("app.core.llm_client.boto3.client")
def test_generate_text_throttling(mock_boto_client):
    """ThrottlingException should map to BedrockThrottlingError."""
    mock_runtime = MagicMock()

    mock_runtime.converse.side_effect = _client_error(
        "ThrottlingException"
    )

    mock_boto_client.return_value = mock_runtime

    client = BedrockClient()

    with pytest.raises(BedrockThrottlingError):
        client.generate_text("Say hello")


@patch("app.core.llm_client.boto3.client")
def test_generate_text_generic_client_error(mock_boto_client):
    """
    Unknown ClientError codes should map to BedrockAPIError while
    preserving the original AWS error code.
    """
    mock_runtime = MagicMock()

    mock_runtime.converse.side_effect = _client_error(
        "ValidationException",
        "bad model id",
    )

    mock_boto_client.return_value = mock_runtime

    client = BedrockClient()

    with pytest.raises(BedrockAPIError) as excinfo:
        client.generate_text("Say hello")

    assert excinfo.value.error_code == "ValidationException"


@patch("app.core.llm_client.boto3.client")
def test_generate_text_malformed_response_missing_content(mock_boto_client):
    """
    A Bedrock response missing output.message.content should raise
    BedrockResponseFormatError.
    """
    mock_runtime = MagicMock()

    mock_runtime.converse.return_value = {
        "output": {
            "message": {}
        }
    }

    mock_boto_client.return_value = mock_runtime

    client = BedrockClient()

    with pytest.raises(BedrockResponseFormatError):
        client.generate_text("Say hello")


@patch("app.core.llm_client.boto3.client")
def test_generate_text_malformed_response_empty_content(mock_boto_client):
    """
    A Bedrock response with an empty content list should raise
    BedrockResponseFormatError.
    """
    mock_runtime = MagicMock()

    mock_runtime.converse.return_value = {
        "output": {
            "message": {
                "content": []
            }
        }
    }

    mock_boto_client.return_value = mock_runtime

    client = BedrockClient()

    with pytest.raises(BedrockResponseFormatError):
        client.generate_text("Say hello")


# --------------------------------------------------------------------------
# Live integration smoke test — skipped by default
# --------------------------------------------------------------------------

RUN_LIVE = os.getenv("RUN_LIVE_AWS_TESTS", "").strip() == "1"


@pytest.mark.skipif(
    not RUN_LIVE,
    reason=(
        "Live AWS test skipped by default. Set RUN_LIVE_AWS_TESTS=1 to run it "
        "against a real Bedrock endpoint (requires valid AWS credentials and "
        "model access). Example:\n"
        "  RUN_LIVE_AWS_TESTS=1 pytest tests/test_llm_client.py -v -s"
    ),
)
def test_generate_text_live_smoke_test():
    """
    Sends one real request to Bedrock using app.config.config and checks
    that a non-empty text response comes back.

    This is the test that actually proves your AWS credentials, configured
    region, and configured Bedrock model access are working end-to-end.
    """
    client = BedrockClient()

    result = client.generate_text(
        prompt="Reply with exactly the single word: OK",
        max_tokens=10,
    )

    print(f"\nBedrock response: {result!r}")

    assert isinstance(result, str)
    assert len(result.strip()) > 0