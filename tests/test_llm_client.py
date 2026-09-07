"""Adapter contract tests use the real SDK with an offline HTTP transport."""
import json
from types import SimpleNamespace
from unittest.mock import Mock

import groq
import httpx
import pytest

from app.core import llm_client as llm


@pytest.fixture
def transport(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-only-not-a-real-key")
    calls = []
    responses = []
    original = groq.Groq
    def handler(request):
        calls.append(request)
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response
    def factory(**kwargs):
        assert kwargs['max_retries'] == 0
        assert kwargs['timeout'].connect == 5
        return original(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    monkeypatch.setattr(llm.groq, 'Groq', factory)
    return calls, responses


def completion(content=" SELECT 1 ", reason="stop"):
    return httpx.Response(200, json={"id":"offline", "object":"chat.completion",
        "created":0, "model":"offline", "choices":[{"index":0,
        "finish_reason":reason,"message":{"role":"assistant","content":content}}]})


def test_request_and_response_contract(transport):
    calls, responses = transport
    responses.append(completion())
    assert llm.GroqClient(model_id="configured-model").generate_text("question", "rules", 128, 0) == "SELECT 1"
    body = json.loads(calls[0].content)
    assert body == {"model":"configured-model", "messages":[
        {"role":"system","content":"rules"},{"role":"user","content":"question"}],
        "max_completion_tokens":128,"temperature":0}


@pytest.mark.parametrize('status,expected', [(401,llm.LLMCredentialsError),
    (403,llm.LLMAccessDeniedError),(429,llm.LLMThrottlingError),
    (400,llm.LLMAPIError),(404,llm.LLMAPIError),(500,llm.LLMAPIError)])
def test_safe_errors_without_retries(transport, status, expected):
    calls, responses = transport
    responses.append(httpx.Response(status,json={"error":{"message":"private data secret-value"}}))
    with pytest.raises(expected) as error:
        llm.GroqClient().generate_text("question")
    assert 'secret-value' not in str(error.value)
    assert len(calls) == 1
    assert error.value.__suppress_context__


def test_timeout_without_retry(transport):
    calls, responses = transport
    responses.append(httpx.ReadTimeout("private details"))
    with pytest.raises(llm.LLMAPIError, match="provider request failed"):
        llm.GroqClient().generate_text("question")
    assert len(calls) == 1


@pytest.mark.parametrize('content,reason', [(None,'stop'),('','stop'),('partial','length'),('x','tool_calls')])
def test_rejects_incomplete_responses(transport, content, reason):
    transport[1].append(completion(content,reason))
    with pytest.raises(llm.LLMResponseFormatError):
        llm.GroqClient().generate_text("question")


def test_missing_key_is_lazy_and_local(monkeypatch):
    monkeypatch.delenv('GROQ_API_KEY',raising=False)
    client = llm.GroqClient()
    sdk = Mock()
    monkeypatch.setattr(llm.groq,'Groq',sdk)
    with pytest.raises(llm.LLMCredentialsError):
        client.generate_text("question")
    sdk.assert_not_called()


@pytest.mark.parametrize('prompt',[None,'','   '])
def test_empty_prompt_is_local(prompt):
    with pytest.raises(ValueError):
        llm.GroqClient().generate_text(prompt)


def test_unknown_provider_never_falls_back(monkeypatch):
    llm.get_llm_client.cache_clear()
    monkeypatch.setattr(llm,'config',SimpleNamespace(llm_provider='unknown'))
    with pytest.raises(llm.LLMAPIError):
        llm.get_llm_client()
    llm.get_llm_client.cache_clear()


def test_default_model_reasoning_is_separate(transport):
    calls, responses = transport
    responses.append(completion())
    llm.GroqClient().generate_text('question')
    body = json.loads(calls[0].content)
    assert body['include_reasoning'] is False
    assert body['reasoning_effort'] == 'low'


def test_malformed_response_is_safe(transport):
    transport[1].append(httpx.Response(200,json={'choices':[]}))
    with pytest.raises(llm.LLMResponseFormatError):
        llm.GroqClient().generate_text('question')
