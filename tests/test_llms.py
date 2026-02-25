"""Tests for LLM provider implementations."""

import pytest

from quickrag.llms.base import BaseLLM, LLMResponse
from quickrag.llms.ollama import OllamaLLM
from quickrag.llms.openai import OpenAILLM


class TestLLMResponse:
    """Tests for the LLMResponse dataclass."""

    def test_basic(self):
        resp = LLMResponse(content="Hello", model="test")
        assert resp.content == "Hello"
        assert resp.model == "test"
        assert resp.usage is None

    def test_with_usage(self):
        resp = LLMResponse(
            content="Hello",
            model="test",
            usage={"prompt_tokens": 10, "completion_tokens": 5},
        )
        assert resp.usage["prompt_tokens"] == 10


class TestOllamaLLM:
    """Tests for Ollama LLM configuration (no server needed)."""

    def test_defaults(self):
        llm = OllamaLLM()
        assert llm.model == "llama3.2"
        assert llm.temperature == 0.7
        assert llm.timeout == 120.0
        assert "localhost:11434" in llm.host

    def test_custom_config(self):
        llm = OllamaLLM(model="mistral", temperature=0.3, timeout=60.0)
        assert llm.model == "mistral"
        assert llm.temperature == 0.3
        assert llm.timeout == 60.0

    def test_build_messages_no_system(self):
        llm = OllamaLLM()
        msgs = llm._build_messages("hello")
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "hello"

    def test_build_messages_with_system(self):
        llm = OllamaLLM()
        msgs = llm._build_messages("hello", system="Be brief.")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_host_trailing_slash_stripped(self):
        llm = OllamaLLM(host="http://localhost:11434/")
        assert not llm.host.endswith("/")


class TestOpenAILLM:
    """Tests for OpenAI LLM configuration."""

    def test_requires_api_key(self):
        with pytest.raises(ValueError, match="API key required"):
            OpenAILLM(api_key=None)

    def test_custom_config(self):
        llm = OpenAILLM(model="gpt-4o", api_key="test-key", temperature=0.5)
        assert llm.model == "gpt-4o"
        assert llm.api_key == "test-key"
        assert llm.temperature == 0.5

    def test_build_messages(self):
        llm = OpenAILLM(api_key="test-key")
        msgs = llm._build_messages("query", system="System prompt")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["content"] == "query"
