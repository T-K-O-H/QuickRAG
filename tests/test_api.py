"""Tests for the FastAPI API endpoints."""

from unittest.mock import patch, MagicMock, AsyncMock

import pytest
from fastapi.testclient import TestClient

from quickrag.pipeline import RAGResponse
from quickrag.stores.base import Document, SearchResult


def _make_mock_pipeline():
    """Create a properly configured mock pipeline."""
    pipeline = MagicMock()
    pipeline.store = MagicMock()
    pipeline.store.collection = "test_collection"
    pipeline.store._client = MagicMock()
    pipeline.store._client.scroll.return_value = ([], None)
    pipeline.embeddings = MagicMock()
    pipeline.embeddings.dimension = 384
    pipeline.count.return_value = 42
    pipeline.top_k = 5

    # aquery is async, needs AsyncMock
    pipeline.aquery = AsyncMock(
        return_value=RAGResponse(answer="mocked answer", sources=[], query="q")
    )
    pipeline.query_conversational = MagicMock(
        return_value=RAGResponse(answer="conversational answer", sources=[], query="q")
    )

    return pipeline


@pytest.fixture
def mock_pipeline():
    return _make_mock_pipeline()


@pytest.fixture
def client(mock_pipeline):
    """Create a test client patching pipeline at every import location."""
    patches = [
        patch("api.dependencies.get_pipeline", return_value=mock_pipeline),
        patch("api.dependencies.get_pipeline_for_collection", return_value=mock_pipeline),
        patch("api.routes.query.get_pipeline", return_value=mock_pipeline),
        patch("api.routes.query.get_pipeline_for_collection", return_value=mock_pipeline),
        patch("api.routes.documents.get_pipeline", return_value=mock_pipeline),
        patch("api.routes.documents.get_pipeline_for_collection", return_value=mock_pipeline),
        patch("api.routes.ingest.get_pipeline", return_value=mock_pipeline),
        patch("api.routes.ingest.get_pipeline_for_collection", return_value=mock_pipeline),
        patch("api.routes.collections.get_pipeline", return_value=mock_pipeline),
        patch("api.routes.collections.get_pipeline_for_collection", return_value=mock_pipeline),
    ]

    for p in patches:
        p.start()

    from api.main import app

    yield TestClient(app, raise_server_exceptions=False)

    for p in patches:
        p.stop()


class TestRootAndHealth:
    """Tests for root and health endpoints."""

    def test_root(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "QuickRAG API"

    def test_health(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["document_count"] == 42


class TestAuthMiddleware:
    """Tests for API key authentication."""

    def test_no_auth_when_disabled(self, client):
        """When no API keys are configured, requests should pass through."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_auth_rejects_without_key(self):
        """When API keys are set, missing key should yield 401."""
        mp = _make_mock_pipeline()

        with patch("api.auth.settings") as mock_settings:
            mock_settings.get_api_keys.return_value = ["valid-key-123"]

            patches = [
                patch("api.routes.query.get_pipeline", return_value=mp),
                patch("api.routes.query.get_pipeline_for_collection", return_value=mp),
                patch("api.dependencies.get_pipeline", return_value=mp),
            ]
            for p in patches:
                p.start()

            from api.main import app
            client = TestClient(app, raise_server_exceptions=False)

            response = client.post("/api/query", json={"query": "test"})
            assert response.status_code == 401

            for p in patches:
                p.stop()

    def test_auth_accepts_valid_key(self):
        """When API keys are set, valid key should work."""
        mp = _make_mock_pipeline()

        with patch("api.auth.settings") as mock_settings:
            mock_settings.get_api_keys.return_value = ["valid-key-123"]

            patches = [
                patch("api.routes.query.get_pipeline", return_value=mp),
                patch("api.routes.query.get_pipeline_for_collection", return_value=mp),
                patch("api.dependencies.get_pipeline", return_value=mp),
            ]
            for p in patches:
                p.start()

            from api.main import app
            client = TestClient(app, raise_server_exceptions=False)

            response = client.post(
                "/api/query",
                json={"query": "test"},
                headers={"X-API-Key": "valid-key-123"},
            )
            assert response.status_code == 200

            for p in patches:
                p.stop()


class TestQueryEndpoints:
    """Tests for query-related endpoints."""

    def test_query_success(self, client, mock_pipeline):
        response = client.post("/api/query", json={"query": "test question"})
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "mocked answer"

    def test_query_with_collection(self, client, mock_pipeline):
        response = client.post(
            "/api/query",
            json={"query": "test", "collection": "custom_collection"},
        )
        assert response.status_code == 200

    def test_clear_history(self, client, mock_pipeline):
        response = client.post("/api/query/clear-history")
        assert response.status_code == 200
        mock_pipeline.clear_history.assert_called_once()


class TestDocumentEndpoints:
    """Tests for document management endpoints."""

    def test_list_documents_empty(self, client, mock_pipeline):
        response = client.get("/api/documents")
        assert response.status_code == 200
        data = response.json()
        assert data["documents"] == []
        assert data["total_documents"] == 0
        assert data["page"] == 1

    def test_list_documents_pagination_params(self, client, mock_pipeline):
        response = client.get("/api/documents?page=2&page_size=10")
        assert response.status_code == 200
        data = response.json()
        assert data["page"] == 2
        assert data["page_size"] == 10

    def test_delete_document_not_found(self, client, mock_pipeline):
        response = client.delete("/api/documents/nonexistent-id")
        assert response.status_code == 404


class TestCollectionEndpoints:
    """Tests for collection management endpoints."""

    def test_list_collections(self, client, mock_pipeline):
        col = MagicMock()
        col.name = "test_collection"
        mock_pipeline.store._client.get_collections.return_value = MagicMock(collections=[col])

        response = client.get("/api/collections")
        assert response.status_code == 200
        data = response.json()
        assert data["current"] == "test_collection"
        assert len(data["collections"]) >= 1

    def test_switch_collection(self, client, mock_pipeline):
        response = client.post("/api/collections/new_collection/switch")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
