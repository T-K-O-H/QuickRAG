"""Main RAG Pipeline for QuickRAG."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, AsyncIterator, Union

from quickrag.stores.base import BaseStore, Document, SearchResult
from quickrag.stores.qdrant import QdrantStore
from quickrag.embeddings.base import BaseEmbeddings
from quickrag.embeddings.local import LocalEmbeddings
from quickrag.embeddings.openai import OpenAIEmbeddings
from quickrag.llms.base import BaseLLM
from quickrag.llms.ollama import OllamaLLM
from quickrag.llms.openai import OpenAILLM
from quickrag.loaders.auto import load as auto_load
from quickrag.chunkers.text import RecursiveChunker
from quickrag.config import settings
from quickrag.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Citation:
    """A reference citation linking an answer back to a source chunk."""

    ref: str
    source: str
    page: int | None = None
    chunk_index: int | None = None
    score: float = 0.0
    content_preview: str = ""
    document_id: str | None = None


@dataclass
class RAGResponse:
    """Response from RAG pipeline."""

    answer: str
    sources: list[SearchResult]
    query: str
    citations: list[Citation] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class RAGPipeline:
    """Main RAG pipeline with LangGraph-based flow.

    Provides a plug-and-play interface for building RAG applications.
    Supports decorator-based customization for advanced users.

    Example:
        >>> pipeline = RAGPipeline.local()  # Full local setup
        >>> pipeline.ingest("./documents")
        >>> response = pipeline.query("What is the refund policy?")
        >>> print(response.answer)
    """

    def __init__(
        self,
        store: BaseStore | None = None,
        embeddings: BaseEmbeddings | None = None,
        llm: BaseLLM | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        top_k: int | None = None,
        use_graph: bool = False,
    ):
        """Initialize RAG pipeline.

        Args:
            store: Vector store for documents. Defaults to QdrantStore.
            embeddings: Embedding model. Defaults to LocalEmbeddings.
            llm: Language model. Defaults to OllamaLLM.
            chunk_size: Chunk size for splitting documents.
            chunk_overlap: Overlap between chunks.
            top_k: Number of documents to retrieve.
            use_graph: Whether to use LangGraph workflow for queries.
        """
        self.embeddings = embeddings or LocalEmbeddings()
        self.store = store or QdrantStore(embedding_dim=self.embeddings.dimension)
        self.llm = llm or OllamaLLM()
        self.top_k = top_k or settings.retrieval_top_k
        self.use_graph = use_graph

        self.chunker = RecursiveChunker(
            chunk_size=chunk_size or settings.chunk_size,
            chunk_overlap=chunk_overlap or settings.chunk_overlap,
        )

        # Custom hooks (set via decorators)
        self._custom_chunker: Callable | None = None
        self._custom_retriever: Callable | None = None
        self._custom_router: Callable | None = None
        self._custom_generator: Callable | None = None

        # Conversation memory: list of {"role": ..., "content": ...} dicts
        self._conversation_history: list[dict[str, str]] = []
        self._conversational_graph = None

        # LangGraph compiled graph (lazy-built)
        self._graph = None

        # System prompt — instructs the LLM to cite sources using [n] markers
        self.system_prompt = """You are a helpful assistant that answers questions based on the provided context.

Use the context below to answer the question. If the context doesn't contain enough information to answer the question, say so clearly.

IMPORTANT: When you use information from the context, cite the source by including the reference number in square brackets, e.g. [1], [2]. You may cite multiple sources for the same claim, e.g. [1][3]. Always include at least one citation for each factual claim.

Context:
{context}

Answer the question accurately and concisely based on the context above, citing sources with [n] markers."""

        logger.info("Pipeline initialized (graph=%s)", use_graph)

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def local(
        cls,
        collection: str | None = None,
        embedding_model: str = "bge-small-en-v1.5",
        llm_model: str = "llama3.2",
        use_graph: bool = False,
    ) -> "RAGPipeline":
        """Create a fully local pipeline.

        Uses local embeddings (sentence-transformers) and Ollama for LLM.
        Fast, free, and works offline.

        Args:
            collection: Qdrant collection name.
            embedding_model: Local embedding model name.
            llm_model: Ollama model name.
            use_graph: Whether to use LangGraph workflow.

        Returns:
            Configured RAGPipeline.
        """
        embeddings = LocalEmbeddings(embedding_model)
        return cls(
            store=QdrantStore(
                collection=collection,
                embedding_dim=embeddings.dimension,
            ),
            embeddings=embeddings,
            llm=OllamaLLM(llm_model),
            use_graph=use_graph,
        )

    @classmethod
    def cloud(
        cls,
        collection: str | None = None,
        embedding_model: str = "text-embedding-3-small",
        llm_model: str = "gpt-4o-mini",
        api_key: str | None = None,
        use_graph: bool = False,
    ) -> "RAGPipeline":
        """Create a cloud-based pipeline using OpenAI.

        Higher quality but requires API key and has latency.

        Args:
            collection: Qdrant collection name.
            embedding_model: OpenAI embedding model.
            llm_model: OpenAI LLM model.
            api_key: OpenAI API key.
            use_graph: Whether to use LangGraph workflow.

        Returns:
            Configured RAGPipeline.
        """
        embeddings = OpenAIEmbeddings(embedding_model, api_key=api_key)
        return cls(
            store=QdrantStore(
                collection=collection,
                embedding_dim=embeddings.dimension,
            ),
            embeddings=embeddings,
            llm=OpenAILLM(llm_model, api_key=api_key),
            use_graph=use_graph,
        )

    @classmethod
    def hybrid(
        cls,
        collection: str | None = None,
        embedding_model: str = "bge-small-en-v1.5",
        llm_model: str = "gpt-4o-mini",
        api_key: str | None = None,
        use_graph: bool = False,
    ) -> "RAGPipeline":
        """Create a hybrid pipeline (local embeddings + cloud LLM).

        Best of both worlds: fast local retrieval with high-quality generation.

        Args:
            collection: Qdrant collection name.
            embedding_model: Local embedding model.
            llm_model: OpenAI LLM model.
            api_key: OpenAI API key.
            use_graph: Whether to use LangGraph workflow.

        Returns:
            Configured RAGPipeline.
        """
        embeddings = LocalEmbeddings(embedding_model)
        return cls(
            store=QdrantStore(
                collection=collection,
                embedding_dim=embeddings.dimension,
            ),
            embeddings=embeddings,
            llm=OpenAILLM(llm_model, api_key=api_key),
            use_graph=use_graph,
        )

    # ------------------------------------------------------------------
    # LangGraph integration
    # ------------------------------------------------------------------

    def _get_graph(self):
        """Lazily build and return the compiled LangGraph RAG workflow."""
        if self._graph is None:
            from quickrag.graph import create_rag_graph

            self._graph = create_rag_graph(self)
            logger.info("LangGraph RAG workflow compiled")
        return self._graph

    def _get_conversational_graph(self):
        """Lazily build and return the conversational RAG graph."""
        if self._conversational_graph is None:
            from quickrag.graph import create_conversational_rag_graph

            self._conversational_graph = create_conversational_rag_graph(self)
            logger.info("LangGraph conversational workflow compiled")
        return self._conversational_graph

    # ------------------------------------------------------------------
    # Decorator methods for customization
    # ------------------------------------------------------------------

    def chunker(self, func: Callable) -> Callable:
        """Decorator to set a custom chunker function.

        Example:
            @pipeline.chunker
            def my_chunker(text: str, metadata: dict) -> list[Chunk]:
                return custom_chunk_logic(text)
        """
        self._custom_chunker = func
        return func

    def retriever(self, func: Callable) -> Callable:
        """Decorator to set a custom retriever function.

        Example:
            @pipeline.retriever
            def my_retriever(query: str, top_k: int) -> list[SearchResult]:
                return custom_retrieval_logic(query)
        """
        self._custom_retriever = func
        return func

    def router(self, func: Callable) -> Callable:
        """Decorator to set a custom router function.

        Example:
            @pipeline.router
            def my_router(query: str) -> str:
                if is_greeting(query):
                    return "direct"
                return "retrieval"
        """
        self._custom_router = func
        return func

    def generator(self, func: Callable) -> Callable:
        """Decorator to set a custom generator function.

        Example:
            @pipeline.generator
            def my_generator(query: str, context: str) -> str:
                return custom_generation_logic(query, context)
        """
        self._custom_generator = func
        return func

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(
        self,
        source: Union[str, Path, list],
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Ingest documents from a source.

        Automatically detects file types and handles:
        - Single files (PDF, TXT, MD, CSV, JSON, DOCX, etc.)
        - Directories (recursively)
        - URLs (web pages)
        - Lists of any of the above

        Args:
            source: Path, URL, directory, or list of sources.
            metadata: Optional metadata to attach to all documents.

        Returns:
            Number of chunks indexed.
        """
        logger.info("Ingesting from: %s", source)
        loaded_docs = auto_load(source)

        all_chunks = []
        for doc in loaded_docs:
            chunk_metadata = {**(metadata or {}), **doc.metadata}

            if self._custom_chunker:
                chunks = self._custom_chunker(doc.content, chunk_metadata)
            else:
                chunks = self.chunker.chunk(doc.content, chunk_metadata)

            all_chunks.extend(chunks)

        if not all_chunks:
            logger.warning("No chunks produced from source: %s", source)
            return 0

        documents = [
            Document(content=chunk.content, metadata=chunk.metadata)
            for chunk in all_chunks
        ]

        texts = [chunk.content for chunk in all_chunks]
        embeddings = self.embeddings.embed_documents(texts)

        self.store.add(documents, embeddings)

        logger.info("Indexed %d chunks from %s", len(documents), source)
        return len(documents)

    # ------------------------------------------------------------------
    # Retrieval helpers
    # ------------------------------------------------------------------

    def _retrieve(self, query: str, filter: dict[str, Any] | None = None) -> list[SearchResult]:
        """Retrieve relevant documents for a query."""
        if self._custom_retriever:
            return self._custom_retriever(query, self.top_k)

        query_embedding = self.embeddings.embed_query(query)

        results = self.store.search(
            query_embedding=query_embedding,
            top_k=self.top_k,
            query_text=query,
            filter=filter,
        )

        return results

    def _build_context(self, results: list[SearchResult]) -> str:
        """Build context string from search results with citation labels."""
        if not results:
            return "No relevant context found."

        context_parts = []
        for i, result in enumerate(results, 1):
            label = self._format_source_label(result, i)
            context_parts.append(
                f"[{i}] {result.document.content}\n({label})"
            )

        return "\n\n".join(context_parts)

    @staticmethod
    def _format_source_label(result: SearchResult, index: int) -> str:
        """Build a human-readable source label like 'report.pdf, Page 6'."""
        meta = result.document.metadata
        parts: list[str] = []

        # Document name — prefer original_filename > filename > source
        name = (
            meta.get("original_filename")
            or meta.get("filename")
            or meta.get("source", "Unknown")
        )
        parts.append(f"Source: {name}")

        page = meta.get("page") or meta.get("page_number")
        if page is not None:
            parts.append(f"Page {page}")

        row = meta.get("row_index")
        if row is not None:
            parts.append(f"Row {row}")

        chunk_idx = meta.get("chunk_index")
        if chunk_idx is not None:
            parts.append(f"Chunk {chunk_idx}")

        return ", ".join(parts)

    def _build_citations(self, results: list[SearchResult]) -> list[Citation]:
        """Build structured citations from search results."""
        citations: list[Citation] = []
        for i, result in enumerate(results, 1):
            meta = result.document.metadata
            name = (
                meta.get("original_filename")
                or meta.get("filename")
                or meta.get("source", "Unknown")
            )
            page = meta.get("page") or meta.get("page_number")
            if page is not None:
                page = int(page)

            chunk_idx = meta.get("chunk_index")
            if chunk_idx is not None:
                chunk_idx = int(chunk_idx)

            preview = result.document.content[:120].replace("\n", " ")

            citations.append(
                Citation(
                    ref=f"[{i}]",
                    source=name,
                    page=page,
                    chunk_index=chunk_idx,
                    score=result.score,
                    content_preview=preview,
                    document_id=meta.get("document_id"),
                )
            )
        return citations

    def _route(self, query: str) -> str:
        """Route the query to appropriate handler."""
        if self._custom_router:
            return self._custom_router(query)

        return "retrieval"

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def query(
        self,
        query: str,
        filter: dict[str, Any] | None = None,
    ) -> RAGResponse:
        """Query the RAG pipeline.

        If ``use_graph`` is enabled, the query is executed via the compiled
        LangGraph workflow.  Otherwise the default inline logic is used.

        Args:
            query: User question.
            filter: Optional metadata filter for retrieval.

        Returns:
            RAGResponse with answer and sources.
        """
        if self.use_graph:
            return self._query_via_graph(query)

        route = self._route(query)

        if route == "direct":
            response = self.llm.generate(query)
            return RAGResponse(
                answer=response.content,
                sources=[],
                query=query,
                citations=[],
                metadata={"route": "direct"},
            )

        results = self._retrieve(query, filter=filter)
        context = self._build_context(results)
        citations = self._build_citations(results)

        if self._custom_generator:
            answer = self._custom_generator(query, context)
        else:
            prompt = self.system_prompt.format(context=context) + f"\n\nQuestion: {query}"
            response = self.llm.generate(prompt)
            answer = response.content

        return RAGResponse(
            answer=answer,
            sources=results,
            query=query,
            citations=citations,
            metadata={"route": route, "num_sources": len(results)},
        )

    def _query_via_graph(self, query: str) -> RAGResponse:
        """Execute a query through the LangGraph workflow."""
        graph = self._get_graph()
        result = graph.invoke(
            {
                "query": query,
                "route": "",
                "context": "",
                "sources": [],
                "answer": "",
                "messages": [],
            }
        )
        sources = result.get("sources", [])
        return RAGResponse(
            answer=result["answer"],
            sources=sources,
            query=query,
            citations=self._build_citations(sources),
            metadata={"route": result.get("route", ""), "graph": True},
        )

    async def aquery(
        self,
        query: str,
        filter: dict[str, Any] | None = None,
    ) -> RAGResponse:
        """Async query the RAG pipeline.

        Args:
            query: User question.
            filter: Optional metadata filter for retrieval.

        Returns:
            RAGResponse with answer and sources.
        """
        route = self._route(query)

        if route == "direct":
            response = await self.llm.agenerate(query)
            return RAGResponse(
                answer=response.content,
                sources=[],
                query=query,
                citations=[],
                metadata={"route": "direct"},
            )

        results = self._retrieve(query, filter=filter)
        context = self._build_context(results)
        citations = self._build_citations(results)

        if self._custom_generator:
            answer = self._custom_generator(query, context)
        else:
            prompt = self.system_prompt.format(context=context) + f"\n\nQuestion: {query}"
            response = await self.llm.agenerate(prompt)
            answer = response.content

        return RAGResponse(
            answer=answer,
            sources=results,
            query=query,
            citations=citations,
            metadata={"route": route, "num_sources": len(results)},
        )

    async def astream(
        self,
        query: str,
        filter: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        """Stream a response from the RAG pipeline.

        Args:
            query: User question.
            filter: Optional metadata filter for retrieval.

        Yields:
            Response chunks as strings.
        """
        route = self._route(query)

        if route == "direct":
            async for chunk in self.llm.astream(query):
                yield chunk
            return

        results = self._retrieve(query, filter=filter)
        context = self._build_context(results)

        prompt = self.system_prompt.format(context=context) + f"\n\nQuestion: {query}"
        async for chunk in self.llm.astream(prompt):
            yield chunk

    # ------------------------------------------------------------------
    # Conversational query (uses LangGraph with memory)
    # ------------------------------------------------------------------

    def query_conversational(self, query: str) -> RAGResponse:
        """Query with conversation memory.

        Uses the conversational LangGraph workflow that rewrites follow-up
        questions using prior context and maintains chat history.

        Args:
            query: User question.

        Returns:
            RAGResponse with answer and sources.
        """
        graph = self._get_conversational_graph()
        result = graph.invoke(
            {
                "query": query,
                "route": "",
                "context": "",
                "sources": [],
                "answer": "",
                "messages": list(self._conversation_history),
            }
        )

        self._conversation_history = result.get("messages", [])
        sources = result.get("sources", [])

        return RAGResponse(
            answer=result["answer"],
            sources=sources,
            query=query,
            citations=self._build_citations(sources),
            metadata={
                "route": result.get("route", ""),
                "conversational": True,
                "history_length": len(self._conversation_history),
            },
        )

    def clear_history(self) -> None:
        """Clear conversation memory."""
        self._conversation_history.clear()
        logger.info("Conversation history cleared")

    # ------------------------------------------------------------------
    # Store operations
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Clear all documents from the store."""
        self.store.clear()
        logger.info("Store cleared")

    def count(self) -> int:
        """Return the number of documents in the store."""
        return self.store.count()
