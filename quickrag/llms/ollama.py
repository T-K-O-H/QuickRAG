"""Ollama LLM provider for local inference."""

from typing import AsyncIterator, Iterator

import httpx

from quickrag.llms.base import BaseLLM, LLMResponse
from quickrag.config import settings


class OllamaLLM(BaseLLM):
    """Ollama LLM for local inference.

    Recommended models:
    - llama3.2: Fast, good quality
    - mistral: Great for RAG
    - phi3: Smaller, faster
    """

    def __init__(
        self,
        model: str = "llama3.2",
        host: str | None = None,
        temperature: float = 0.7,
        timeout: float = 120.0,
    ):
        """Initialize Ollama LLM.

        Args:
            model: Ollama model name.
            host: Ollama host URL. Falls back to OLLAMA_HOST env var.
            temperature: Sampling temperature.
            timeout: Request timeout in seconds.
        """
        self.model = model
        self.host = (host or settings.ollama_host).rstrip("/")
        self.temperature = temperature
        self.timeout = timeout

    def _build_messages(self, prompt: str, system: str | None = None) -> list[dict]:
        """Build message list for Ollama API."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return messages

    def generate(self, prompt: str, system: str | None = None) -> LLMResponse:
        """Generate a response using Ollama."""
        messages = self._build_messages(prompt, system)

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.host}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": self.temperature},
                },
            )
            response.raise_for_status()
            data = response.json()

        return LLMResponse(
            content=data["message"]["content"],
            model=self.model,
            usage={
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
            },
        )

    def stream(self, prompt: str, system: str | None = None) -> Iterator[str]:
        """Stream a response using Ollama."""
        messages = self._build_messages(prompt, system)

        with httpx.Client(timeout=self.timeout) as client:
            with client.stream(
                "POST",
                f"{self.host}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": True,
                    "options": {"temperature": self.temperature},
                },
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        import json

                        data = json.loads(line)
                        if "message" in data and "content" in data["message"]:
                            yield data["message"]["content"]

    async def agenerate(self, prompt: str, system: str | None = None) -> LLMResponse:
        """Async generate a response using Ollama."""
        messages = self._build_messages(prompt, system)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.host}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": self.temperature},
                },
            )
            response.raise_for_status()
            data = response.json()

        return LLMResponse(
            content=data["message"]["content"],
            model=self.model,
            usage={
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
            },
        )

    async def astream(self, prompt: str, system: str | None = None) -> AsyncIterator[str]:
        """Async stream a response using Ollama."""
        messages = self._build_messages(prompt, system)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                f"{self.host}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": True,
                    "options": {"temperature": self.temperature},
                },
            ) as response:
                response.raise_for_status()
                import json

                async for line in response.aiter_lines():
                    if line:
                        data = json.loads(line)
                        if "message" in data and "content" in data["message"]:
                            yield data["message"]["content"]

