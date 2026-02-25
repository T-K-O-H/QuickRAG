"""Tests for the RAG pipeline (unit-level, mocked dependencies)."""

from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from quickrag.pipeline import RAGPipeline, RAGResponse
from quickrag.stores.base import Document, SearchResult
from quickrag.llms.base import LLMResponse


class TestRAGPipelineInit:
    """Tests for pipeline initialization."""

    @patch("quickrag.pipeline.OllamaLLM")
    @patch("quickrag.pipeline.QdrantStore")
    @patch("quickrag.pipeline.LocalEmbeddings")
    def test_defaults(self, mock_emb, mock_store, mock_llm):
        mock_emb_instance = MagicMock()
        mock_emb_instance.dimension = 384
        mock_emb.return_value = mock_emb_instance

        pipeline = RAGPipeline()
        assert pipeline.top_k == 5
        assert pipeline.use_graph is False
        assert pipeline._conversation_history == []

    @patch("quickrag.pipeline.OllamaLLM")
    @patch("quickrag.pipeline.QdrantStore")
    @patch("quickrag.pipeline.LocalEmbeddings")
    def test_custom_top_k(self, mock_emb, mock_store, mock_llm):
        mock_emb_instance = MagicMock()
        mock_emb_instance.dimension = 384
        mock_emb.return_value = mock_emb_instance

        pipeline = RAGPipeline(top_k=10)
        assert pipeline.top_k == 10


class TestRAGPipelineDecorators:
    """Tests for decorator-based customization."""

    @patch("quickrag.pipeline.OllamaLLM")
    @patch("quickrag.pipeline.QdrantStore")
    @patch("quickrag.pipeline.LocalEmbeddings")
    def test_custom_router(self, mock_emb, mock_store, mock_llm):
        mock_emb_instance = MagicMock()
        mock_emb_instance.dimension = 384
        mock_emb.return_value = mock_emb_instance

        pipeline = RAGPipeline()

        @pipeline.router
        def my_router(query):
            return "direct"

        assert pipeline._route("anything") == "direct"

    @patch("quickrag.pipeline.OllamaLLM")
    @patch("quickrag.pipeline.QdrantStore")
    @patch("quickrag.pipeline.LocalEmbeddings")
    def test_custom_generator(self, mock_emb, mock_store, mock_llm):
        mock_emb_instance = MagicMock()
        mock_emb_instance.dimension = 384
        mock_emb.return_value = mock_emb_instance

        pipeline = RAGPipeline()

        @pipeline.generator
        def my_gen(query, context):
            return f"Custom: {query}"

        assert pipeline._custom_generator is not None


class TestRAGPipelineQuery:
    """Tests for query methods with mocked backends."""

    def _make_pipeline(self):
        pipeline = RAGPipeline.__new__(RAGPipeline)
        pipeline.embeddings = MagicMock()
        pipeline.embeddings.embed_query.return_value = [0.1] * 384
        pipeline.store = MagicMock()
        pipeline.llm = MagicMock()
        pipeline.top_k = 5
        pipeline.use_graph = False
        pipeline._custom_chunker = None
        pipeline._custom_retriever = None
        pipeline._custom_router = None
        pipeline._custom_generator = None
        pipeline._conversation_history = []
        pipeline._conversational_graph = None
        pipeline._graph = None
        pipeline.system_prompt = "Context:\n{context}\n\nAnswer:"
        return pipeline

    def test_query_retrieval(self):
        pipeline = self._make_pipeline()

        doc = Document(content="Test doc", metadata={"source": "test"})
        pipeline.store.search.return_value = [SearchResult(document=doc, score=0.9)]
        pipeline.llm.generate.return_value = LLMResponse(content="Answer", model="test")

        response = pipeline.query("What is test?")

        assert isinstance(response, RAGResponse)
        assert response.answer == "Answer"
        assert len(response.sources) == 1
        assert response.metadata["route"] == "retrieval"

    def test_query_direct_route(self):
        pipeline = self._make_pipeline()
        pipeline._custom_router = lambda q: "direct"
        pipeline.llm.generate.return_value = LLMResponse(content="Direct answer", model="test")

        response = pipeline.query("Hello")

        assert response.answer == "Direct answer"
        assert response.sources == []
        assert response.metadata["route"] == "direct"

    def test_build_context_empty(self):
        pipeline = self._make_pipeline()
        ctx = pipeline._build_context([])
        assert ctx == "No relevant context found."

    def test_build_context_with_results(self):
        pipeline = self._make_pipeline()
        doc = Document(content="Content here", metadata={"source": "file.txt"})
        results = [SearchResult(document=doc, score=0.9)]
        ctx = pipeline._build_context(results)
        assert "Content here" in ctx
        assert "file.txt" in ctx

    def test_retrieve_with_filter(self):
        pipeline = self._make_pipeline()
        pipeline.store.search.return_value = []

        pipeline._retrieve("test query", filter={"source_type": "file"})

        pipeline.store.search.assert_called_once()
        call_kwargs = pipeline.store.search.call_args[1]
        assert call_kwargs["filter"] == {"source_type": "file"}

    def test_clear_history(self):
        pipeline = self._make_pipeline()
        pipeline._conversation_history = [{"role": "user", "content": "hi"}]
        pipeline.clear_history()
        assert pipeline._conversation_history == []


class TestRAGResponse:
    """Tests for the RAGResponse dataclass."""

    def test_basic(self):
        resp = RAGResponse(answer="42", sources=[], query="What?")
        assert resp.answer == "42"
        assert resp.query == "What?"
        assert resp.metadata == {}

    def test_with_metadata(self):
        resp = RAGResponse(
            answer="yes", sources=[], query="q", metadata={"route": "retrieval"}
        )
        assert resp.metadata["route"] == "retrieval"
