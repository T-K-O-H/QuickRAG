"""Configuration management for QuickRAG."""

from dataclasses import dataclass, field
from typing import Literal

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Global settings loaded from environment variables."""

    # OpenAI
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")

    # Qdrant
    qdrant_host: str = Field(default="localhost", alias="QDRANT_HOST")
    qdrant_port: int = Field(default=6333, alias="QDRANT_PORT")

    # Ollama
    ollama_host: str = Field(default="http://localhost:11434", alias="OLLAMA_HOST")

    # Default models
    default_embedding_model: str = Field(
        default="bge-small-en-v1.5", alias="DEFAULT_EMBEDDING_MODEL"
    )
    default_llm_model: str = Field(default="llama3.2", alias="DEFAULT_LLM_MODEL")
    default_collection: str = Field(default="documents", alias="DEFAULT_COLLECTION")

    # Retrieval settings
    retrieval_top_k: int = Field(default=5, alias="RETRIEVAL_TOP_K")
    chunk_size: int = Field(default=512, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=50, alias="CHUNK_OVERLAP")

    # Feature toggle defaults
    search_mode: str = Field(default="hybrid", alias="QUICKRAG_SEARCH_MODE")
    citations_enabled: bool = Field(default=True, alias="QUICKRAG_CITATIONS")
    debug: bool = Field(default=False, alias="QUICKRAG_DEBUG")
    chunking_strategy: str = Field(default="recursive", alias="QUICKRAG_CHUNKING")
    score_threshold: float = Field(default=0.0, alias="QUICKRAG_SCORE_THRESHOLD")
    routing_enabled: bool = Field(default=True, alias="QUICKRAG_ROUTING")

    # API authentication (comma-separated keys, empty = no auth required)
    api_keys: str | None = Field(default=None, alias="QUICKRAG_API_KEYS")

    # CORS origins (comma-separated)
    cors_origins: str = Field(
        default="http://localhost:3000,http://127.0.0.1:3000",
        alias="CORS_ORIGINS",
    )

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    def get_api_keys(self) -> list[str]:
        """Parse comma-separated API keys."""
        if not self.api_keys:
            return []
        return [k.strip() for k in self.api_keys.split(",") if k.strip()]

    def get_cors_origins(self) -> list[str]:
        """Parse comma-separated CORS origins."""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()


@dataclass
class FeatureToggles:
    """Runtime feature toggles for controlling pipeline behavior.

    These can be set at pipeline creation time or overridden per-query
    via the API.

    Attributes:
        search_mode: How retrieval works. "hybrid" combines semantic + keyword,
            "dense" uses only semantic embeddings, "keyword" uses only BM25
            keyword matching.
        citations: Whether to generate citation references in responses.
        debug: Enable verbose debug logging throughout the pipeline.
        chunking_strategy: "recursive" splits on natural boundaries (paragraphs,
            sentences), "fixed" uses simple fixed-size windows.
        score_threshold: Minimum similarity score (0.0-1.0) to include a result.
            Results below this are dropped. 0.0 means keep everything.
        routing: Whether the router can skip retrieval for simple queries
            (e.g. greetings). When False, every query goes through retrieval.
    """

    search_mode: Literal["hybrid", "dense", "keyword"] = "hybrid"
    citations: bool = True
    debug: bool = False
    chunking_strategy: Literal["recursive", "fixed"] = "recursive"
    score_threshold: float = 0.0
    routing: bool = True

    @classmethod
    def from_settings(cls) -> "FeatureToggles":
        """Create toggles from the global Settings / environment variables."""
        return cls(
            search_mode=settings.search_mode,  # type: ignore[arg-type]
            citations=settings.citations_enabled,
            debug=settings.debug,
            chunking_strategy=settings.chunking_strategy,  # type: ignore[arg-type]
            score_threshold=settings.score_threshold,
            routing=settings.routing_enabled,
        )

    def merge(self, overrides: dict) -> "FeatureToggles":
        """Return a new FeatureToggles with the given overrides applied.

        Unknown keys are silently ignored so callers can pass arbitrary dicts
        (e.g. from a JSON request body).
        """
        known = {f.name for f in self.__dataclass_fields__.values()}
        changes = {k: v for k, v in overrides.items() if k in known and v is not None}
        if not changes:
            return self
        return FeatureToggles(**{**self.__dict__, **changes})

