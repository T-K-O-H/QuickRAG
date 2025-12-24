"""OpenAI LLM provider."""

from typing import AsyncIterator, Iterator

from quickrag.llms.base import BaseLLM, LLMResponse
from quickrag.config import settings


class OpenAILLM(BaseLLM):
    """OpenAI LLM via API.

    Recommended models:
    - gpt-4o-mini: Fast, cheap, great quality
    - gpt-4o: Best quality
    - gpt-3.5-turbo: Fastest, cheapest
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        temperature: float = 0.7,
    ):
        """Initialize OpenAI LLM.

        Args:
            model: OpenAI model name.
            api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
            temperature: Sampling temperature.
        """
        self.model = model
        self.api_key = api_key or settings.openai_api_key
        self.temperature = temperature

        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key."
            )

    @property
    def _client(self):
        """Get sync OpenAI client."""
        from openai import OpenAI

        return OpenAI(api_key=self.api_key)

    @property
    def _async_client(self):
        """Get async OpenAI client."""
        from openai import AsyncOpenAI

        return AsyncOpenAI(api_key=self.api_key)

    def _build_messages(self, prompt: str, system: str | None = None) -> list[dict]:
        """Build message list for OpenAI API."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return messages

    def generate(self, prompt: str, system: str | None = None) -> LLMResponse:
        """Generate a response using OpenAI."""
        messages = self._build_messages(prompt, system)

        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )

        return LLMResponse(
            content=response.choices[0].message.content or "",
            model=self.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            },
        )

    def stream(self, prompt: str, system: str | None = None) -> Iterator[str]:
        """Stream a response using OpenAI."""
        messages = self._build_messages(prompt, system)

        stream = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            stream=True,
        )

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def agenerate(self, prompt: str, system: str | None = None) -> LLMResponse:
        """Async generate a response using OpenAI."""
        messages = self._build_messages(prompt, system)

        response = await self._async_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )

        return LLMResponse(
            content=response.choices[0].message.content or "",
            model=self.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            },
        )

    async def astream(self, prompt: str, system: str | None = None) -> AsyncIterator[str]:
        """Async stream a response using OpenAI."""
        messages = self._build_messages(prompt, system)

        stream = await self._async_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

