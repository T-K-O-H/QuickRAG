"""Configuration management for QuickRAG."""

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

