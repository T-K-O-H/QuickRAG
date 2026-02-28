"""Tests for the RAG pipeline (unit-level, mocked dependencies)."""

from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from quickrag.pipeline import RAGPipeline, RAGResponse, Citation
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
        pipeline.system_prompt = "Context:\n{context}\n\nAnswer with [n] citations:"
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
        assert len(response.citations) == 1
        assert response.citations[0].ref == "[1]"
        assert response.metadata["route"] == "retrieval"

    def test_query_direct_route(self):
        pipeline = self._make_pipeline()
        pipeline._custom_router = lambda q: "direct"
        pipeline.llm.generate.return_value = LLMResponse(content="Direct answer", model="test")

        response = pipeline.query("Hello")

        assert response.answer == "Direct answer"
        assert response.sources == []
        assert response.citations == []
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


class TestCitation:
    """Tests for the Citation dataclass."""

    def test_basic(self):
        cit = Citation(ref="[1]", source="report.pdf")
        assert cit.ref == "[1]"
        assert cit.source == "report.pdf"
        assert cit.page is None
        assert cit.chunk_index is None
        assert cit.score == 0.0

    def test_with_page(self):
        cit = Citation(ref="[2]", source="manual.pdf", page=6, score=0.85)
        assert cit.page == 6
        assert cit.score == 0.85

    def test_with_all_fields(self):
        cit = Citation(
            ref="[3]",
            source="data.csv",
            page=None,
            chunk_index=2,
            score=0.7,
            content_preview="Some preview text",
            document_id="abc-123",
        )
        assert cit.chunk_index == 2
        assert cit.content_preview == "Some preview text"
        assert cit.document_id == "abc-123"


class TestBuildCitations:
    """Tests for _build_citations."""

    def _make_pipeline(self):
        pipeline = RAGPipeline.__new__(RAGPipeline)
        return pipeline

    def test_empty_results(self):
        pipeline = self._make_pipeline()
        citations = pipeline._build_citations([])
        assert citations == []

    def test_single_result_basic(self):
        pipeline = self._make_pipeline()
        doc = Document(
            content="Hello world",
            metadata={"source": "test.txt", "filename": "test.txt"},
        )
        results = [SearchResult(document=doc, score=0.9)]

        citations = pipeline._build_citations(results)
        assert len(citations) == 1
        assert citations[0].ref == "[1]"
        assert citations[0].source == "test.txt"
        assert citations[0].score == 0.9
        assert "Hello world" in citations[0].content_preview

    def test_pdf_with_page(self):
        pipeline = self._make_pipeline()
        doc = Document(
            content="PDF content here",
            metadata={
                "source": "/path/to/report.pdf",
                "filename": "report.pdf",
                "page": 6,
                "chunk_index": 3,
                "document_id": "doc-abc",
            },
        )
        results = [SearchResult(document=doc, score=0.85)]

        citations = pipeline._build_citations(results)
        assert citations[0].source == "report.pdf"
        assert citations[0].page == 6
        assert citations[0].chunk_index == 3
        assert citations[0].document_id == "doc-abc"

    def test_multiple_results_numbered(self):
        pipeline = self._make_pipeline()
        docs = [
            Document(content=f"Doc {i}", metadata={"source": f"file{i}.txt"})
            for i in range(3)
        ]
        results = [SearchResult(document=d, score=0.9 - i * 0.1) for i, d in enumerate(docs)]

        citations = pipeline._build_citations(results)
        assert len(citations) == 3
        assert citations[0].ref == "[1]"
        assert citations[1].ref == "[2]"
        assert citations[2].ref == "[3]"

    def test_original_filename_preferred(self):
        pipeline = self._make_pipeline()
        doc = Document(
            content="Content",
            metadata={
                "source": "/tmp/abc123.pdf",
                "filename": "abc123.pdf",
                "original_filename": "Annual Report 2024.pdf",
            },
        )
        results = [SearchResult(document=doc, score=0.8)]

        citations = pipeline._build_citations(results)
        assert citations[0].source == "Annual Report 2024.pdf"

    def test_page_number_alias(self):
        """Some loaders use 'page_number' instead of 'page'."""
        pipeline = self._make_pipeline()
        doc = Document(
            content="Content",
            metadata={"source": "doc.pdf", "page_number": 12},
        )
        results = [SearchResult(document=doc, score=0.8)]

        citations = pipeline._build_citations(results)
        assert citations[0].page == 12

    def test_content_preview_truncated(self):
        pipeline = self._make_pipeline()
        long_content = "A" * 300
        doc = Document(content=long_content, metadata={"source": "big.txt"})
        results = [SearchResult(document=doc, score=0.5)]

        citations = pipeline._build_citations(results)
        assert len(citations[0].content_preview) == 120


class TestFormatSourceLabel:
    """Tests for _format_source_label static method."""

    def test_basic_source(self):
        doc = Document(content="x", metadata={"source": "file.txt"})
        result = SearchResult(document=doc, score=0.5)

        label = RAGPipeline._format_source_label(result, 1)
        assert "file.txt" in label

    def test_with_page(self):
        doc = Document(content="x", metadata={"source": "doc.pdf", "page": 5})
        result = SearchResult(document=doc, score=0.5)

        label = RAGPipeline._format_source_label(result, 1)
        assert "Page 5" in label

    def test_with_chunk_index(self):
        doc = Document(content="x", metadata={"source": "doc.txt", "chunk_index": 3})
        result = SearchResult(document=doc, score=0.5)

        label = RAGPipeline._format_source_label(result, 1)
        assert "Chunk 3" in label

    def test_with_row_index(self):
        doc = Document(content="x", metadata={"source": "data.csv", "row_index": 7})
        result = SearchResult(document=doc, score=0.5)

        label = RAGPipeline._format_source_label(result, 1)
        assert "Row 7" in label

    def test_original_filename_preferred(self):
        doc = Document(
            content="x",
            metadata={
                "source": "/tmp/xyz.pdf",
                "original_filename": "Report.pdf",
            },
        )
        result = SearchResult(document=doc, score=0.5)

        label = RAGPipeline._format_source_label(result, 1)
        assert "Report.pdf" in label


class TestRAGResponse:
    """Tests for the RAGResponse dataclass."""

    def test_basic(self):
        resp = RAGResponse(answer="42", sources=[], query="What?")
        assert resp.answer == "42"
        assert resp.query == "What?"
        assert resp.citations == []
        assert resp.metadata == {}

    def test_with_metadata(self):
        resp = RAGResponse(
            answer="yes", sources=[], query="q", metadata={"route": "retrieval"}
        )
        assert resp.metadata["route"] == "retrieval"

    def test_with_citations(self):
        cit = Citation(ref="[1]", source="file.pdf", page=3, score=0.9)
        resp = RAGResponse(answer="answer", sources=[], query="q", citations=[cit])
        assert len(resp.citations) == 1
        assert resp.citations[0].page == 3
