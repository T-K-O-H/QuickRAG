"""Tests for configuration management."""

import pytest

from quickrag.config import Settings


class TestSettings:
    """Tests for the Settings class."""

    def test_defaults(self):
        s = Settings(
            _env_file=None,
        )
        assert s.qdrant_host == "localhost"
        assert s.qdrant_port == 6333
        assert s.ollama_host == "http://localhost:11434"
        assert s.retrieval_top_k == 5
        assert s.chunk_size == 512
        assert s.chunk_overlap == 50
        assert s.default_collection == "documents"

    def test_api_keys_none(self):
        s = Settings(_env_file=None)
        assert s.get_api_keys() == []

    def test_api_keys_parsing(self):
        s = Settings(_env_file=None, QUICKRAG_API_KEYS="key1,key2,key3")
        keys = s.get_api_keys()
        assert keys == ["key1", "key2", "key3"]

    def test_api_keys_with_whitespace(self):
        s = Settings(_env_file=None, QUICKRAG_API_KEYS=" key1 , key2 ")
        keys = s.get_api_keys()
        assert keys == ["key1", "key2"]

    def test_cors_origins_default(self):
        s = Settings(_env_file=None)
        origins = s.get_cors_origins()
        assert "http://localhost:3000" in origins
        assert "http://127.0.0.1:3000" in origins

    def test_cors_origins_custom(self):
        s = Settings(_env_file=None, CORS_ORIGINS="https://app.example.com,https://admin.example.com")
        origins = s.get_cors_origins()
        assert origins == ["https://app.example.com", "https://admin.example.com"]
