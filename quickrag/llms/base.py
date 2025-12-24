"""Base LLM class for QuickRAG."""

from abc import ABC, abstractmethod
from typing import AsyncIterator, Iterator
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """Response from LLM."""

    content: str
    model: str
    usage: dict | None = None


class BaseLLM(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate(self, prompt: str, system: str | None = None) -> LLMResponse:
        """Generate a response.

        Args:
            prompt: User prompt.
            system: Optional system message.

        Returns:
            LLMResponse with generated content.
        """
        ...

    @abstractmethod
    def stream(self, prompt: str, system: str | None = None) -> Iterator[str]:
        """Stream a response.

        Args:
            prompt: User prompt.
            system: Optional system message.

        Yields:
            Response chunks as strings.
        """
        ...

    @abstractmethod
    async def agenerate(self, prompt: str, system: str | None = None) -> LLMResponse:
        """Async generate a response.

        Args:
            prompt: User prompt.
            system: Optional system message.

        Returns:
            LLMResponse with generated content.
        """
        ...

    @abstractmethod
    async def astream(self, prompt: str, system: str | None = None) -> AsyncIterator[str]:
        """Async stream a response.

        Args:
            prompt: User prompt.
            system: Optional system message.

        Yields:
            Response chunks as strings.
        """
        ...

