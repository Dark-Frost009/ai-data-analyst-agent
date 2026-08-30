"""
Amazon Bedrock client.

A thin, isolated wrapper around the Bedrock Runtime API. This is the ONLY
module in the application allowed to import boto3 or talk to Bedrock
directly — every other module (query planner, explainer, UI, etc.) must
depend on `BedrockClient` / `get_bedrock_client()` instead.

Design notes
------------
- Uses the Bedrock **Converse API** (`client.converse(...)`) rather than
  the older, per-model `invoke_model` API. Different model providers on
  Bedrock (Anthropic, Amazon Nova/Titan, Meta Llama, Cohere, Mistral,
  AI21, ...) each expect a different JSON request/response payload under
  `invoke_model`. Converse is AWS's unified interface: the same
  request/response shape works across every Bedrock model that supports
  messages, so this wrapper does not need to special-case each provider's
  payload format.
- Credentials are never read, stored, or passed explicitly. boto3 resolves
  them via the standard AWS credential chain (`aws configure`, an IAM
  role, or AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY environment
  variables set outside of this project).
- All boto3/botocore exceptions are caught and translated into a small,
  typed exception hierarchy (`BedrockClientError` and subclasses) so
  callers never need to know about botocore internals.
"""

from functools import lru_cache
from typing import Any, Optional

import boto3
from botocore.exceptions import (
    BotoCoreError,
    ClientError,
    NoCredentialsError,
    PartialCredentialsError,
)

from app.config import config
from app.utils.logger import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------
# Exceptions
# --------------------------------------------------------------------------
class BedrockClientError(Exception):
    """Base class for all errors raised by BedrockClient."""


class BedrockCredentialsError(BedrockClientError):
    """AWS credentials are missing, incomplete, invalid, or expired."""


class BedrockAccessDeniedError(BedrockClientError):
    """The caller lacks permission, or model access hasn't been granted."""


class BedrockThrottlingError(BedrockClientError):
    """The request was throttled or a quota/limit was exceeded."""


class BedrockAPIError(BedrockClientError):
    """Any other Bedrock/AWS API-level error not covered above."""

    def __init__(self, message: str, error_code: Optional[str] = None):
        super().__init__(message)
        self.error_code = error_code


class BedrockResponseFormatError(BedrockClientError):
    """Bedrock returned a response that doesn't match the expected shape."""


# botocore ClientError codes mapped to the exceptions above. Anything not
# listed here still raises BedrockAPIError with the original code attached,
# so no error is ever silently swallowed.
_CREDENTIAL_ERROR_CODES = {
    "UnrecognizedClientException",
    "InvalidSignatureException",
    "ExpiredTokenException",
}
_ACCESS_DENIED_ERROR_CODES = {
    "AccessDeniedException",
}
_THROTTLING_ERROR_CODES = {
    "ThrottlingException",
    "TooManyRequestsException",
    "ServiceQuotaExceededException",
}


class BedrockClient:
    """Thin wrapper around the Amazon Bedrock Runtime `converse` API."""

    def __init__(self, region_name: Optional[str] = None, model_id: Optional[str] = None):
        self._region_name = region_name or config.aws_region
        self._model_id = model_id or config.bedrock_model_id

        # Building the client is a local operation — it does not make a
        # network call and does not require valid credentials yet. No
        # credentials are passed here; boto3 resolves them itself.
        self._client = boto3.client("bedrock-runtime", region_name=self._region_name)

        logger.info(
            "BedrockClient initialized | region=%s | model_id=%s",
            self._region_name,
            self._model_id,
        )

    def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        """
        Send a single text prompt to the configured Bedrock model and
        return the generated text.

        Raises
        ------
        ValueError
            If `prompt` is empty.
        BedrockCredentialsError
            AWS credentials are missing, invalid, or expired.
        BedrockAccessDeniedError
            The IAM identity lacks permission, or model access hasn't
            been granted for this model in the Bedrock console.
        BedrockThrottlingError
            The request was throttled or a quota was exceeded.
        BedrockAPIError
            Any other Bedrock/AWS API error.
        BedrockResponseFormatError
            Bedrock returned a response that doesn't match the expected
            Converse API shape.
        """
        if not prompt or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        request_kwargs: dict[str, Any] = {
            "modelId": self._model_id,
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": {
                "maxTokens": max_tokens,
                "temperature": temperature,
            },
        }
        if system_prompt:
            request_kwargs["system"] = [{"text": system_prompt}]

        logger.debug(
            "Sending Bedrock request | model_id=%s | max_tokens=%s | temperature=%s",
            self._model_id,
            max_tokens,
            temperature,
        )

        try:
            response = self._client.converse(**request_kwargs)
        except (NoCredentialsError, PartialCredentialsError) as exc:
            raise BedrockCredentialsError(
                "No valid AWS credentials were found. Configure credentials via "
                "`aws configure`, an IAM role, or AWS_ACCESS_KEY_ID / "
                "AWS_SECRET_ACCESS_KEY environment variables."
            ) from exc
        except ClientError as exc:
            raise self._map_client_error(exc) from exc
        except BotoCoreError as exc:
            raise BedrockAPIError(
                f"A network or SDK-level error occurred calling Bedrock: {exc}"
            ) from exc

        return self._extract_text(response)

    def _map_client_error(self, exc: ClientError) -> BedrockClientError:
        """Translate a botocore ClientError into our own exception hierarchy."""
        error = exc.response.get("Error", {})
        code = error.get("Code", "UnknownError")
        message = error.get("Message", str(exc))

        if code in _CREDENTIAL_ERROR_CODES:
            return BedrockCredentialsError(
                f"AWS credentials were rejected ({code}): {message}"
            )
        if code in _ACCESS_DENIED_ERROR_CODES:
            return BedrockAccessDeniedError(
                f"Access denied for model '{self._model_id}' ({code}): {message}. "
                "Check that model access has been granted for this model in the "
                "Bedrock console (Model access page) and that your IAM identity "
                "has bedrock:InvokeModel permission."
            )
        if code in _THROTTLING_ERROR_CODES:
            return BedrockThrottlingError(
                f"Bedrock request was throttled ({code}): {message}. "
                "Retry with backoff, or request a quota increase."
            )
        return BedrockAPIError(
            f"Bedrock returned an error ({code}): {message}", error_code=code
        )

    @staticmethod
    def _extract_text(response: dict) -> str:
        """Pull the generated text out of a Converse API response."""
        try:
            content_blocks = response["output"]["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise BedrockResponseFormatError(
                "Unexpected Bedrock response shape — missing "
                f"output.message.content: {response!r}"
            ) from exc

        for block in content_blocks:
            text = block.get("text")
            if text:
                if response.get("stopReason") == "max_tokens":
                    logger.warning(
                        "Bedrock response may be truncated (stopReason=max_tokens); "
                        "consider raising max_tokens."
                    )
                return text

        raise BedrockResponseFormatError(
            f"Bedrock response contained no text content block: {content_blocks!r}"
        )


@lru_cache(maxsize=1)
def get_bedrock_client() -> BedrockClient:
    """Return a process-wide singleton BedrockClient built from app config."""
    return BedrockClient()
