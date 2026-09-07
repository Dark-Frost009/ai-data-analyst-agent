"""Provider-independent text interface and initial Groq adapter.

Client creation is lazy: startup and uploads do not require credentials.
Provider response bodies and keys are never included in application errors.
"""
import os
from functools import lru_cache
from typing import Optional, Protocol

import groq
import httpx

from app.config import config


class LLMClientError(Exception):
    """Base error exposed to application callers."""


class LLMCredentialsError(LLMClientError):
    """Missing or rejected credentials."""


class LLMAccessDeniedError(LLMClientError):
    """Provider or model access denied."""


class LLMThrottlingError(LLMClientError):
    """Provider quota exhausted; retry later."""


class LLMAPIError(LLMClientError):
    """Provider unavailable or request rejected."""


class LLMResponseFormatError(LLMClientError):
    """Empty, malformed or truncated completion."""


class LLMClient(Protocol):
    """Implement this boundary to add providers without changing callers."""

    def generate_text(self, prompt: str, system_prompt: Optional[str] = None,
                      max_tokens: int = 1024, temperature: float = 0.0) -> str: ...


class GroqClient:
    """Text-only completions, with no tools or provider fallback."""

    def __init__(self, model_id: Optional[str] = None):
        self._model_id = model_id or config.llm_model_id

    def generate_text(self, prompt: str, system_prompt: Optional[str] = None,
                      max_tokens: int = 1024, temperature: float = 0.0) -> str:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        key = os.getenv("GROQ_API_KEY", "").strip()
        if not key:
            raise LLMCredentialsError("Configure GROQ_API_KEY privately in deployment secrets.")
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        options = {}
        if self._model_id in {"openai/gpt-oss-20b", "openai/gpt-oss-120b"}:
            options = {"include_reasoning": False, "reasoning_effort": "low"}
        try:
            # No automatic retries: quota errors and timeouts must not multiply
            # token use. A new user request is the retry boundary.
            with groq.Groq(api_key=key, max_retries=0,
                           timeout=httpx.Timeout(config.llm_read_timeout_seconds,
                                                 connect=config.llm_connect_timeout_seconds)) as client:
                response = client.chat.completions.create(
                    model=self._model_id, messages=messages,
                    max_completion_tokens=max_tokens, temperature=temperature, **options,
                )
        except groq.AuthenticationError:
            raise LLMCredentialsError("The provider rejected the API key.") from None
        except groq.PermissionDeniedError:
            raise LLMAccessDeniedError("The configured model is not permitted for this account.") from None
        except groq.RateLimitError:
            raise LLMThrottlingError("The provider quota was reached. Try again later.") from None
        except groq.APIError:
            raise LLMAPIError("The provider request failed. Check service status and model configuration.") from None
        try:
            choice = response.choices[0]
            content = choice.message.content
            if choice.finish_reason != "stop" or not isinstance(content, str) or not content.strip():
                raise LLMResponseFormatError("The provider returned no complete text answer.")
            return content.strip()
        except (AttributeError, IndexError, TypeError):
            raise LLMResponseFormatError("The provider returned an invalid response.") from None


@lru_cache(maxsize=1)
def get_llm_client() -> LLMClient:
    """Select explicitly; unsupported providers never fall back silently."""
    if config.llm_provider != "groq":
        raise LLMAPIError("Unsupported LLM_PROVIDER. Configure groq.")
    return GroqClient()
