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


@dataclass
class RAGResponse:
    """Response from RAG pipeline."""

    answer: str
    sources: list[SearchResult]
    query: str
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
    ):
        """Initialize RAG pipeline.

        Args:
            store: Vector store for documents. Defaults to QdrantStore.
            embeddings: Embedding model. Defaults to LocalEmbeddings.
            llm: Language model. Defaults to OllamaLLM.
            chunk_size: Chunk size for splitting documents.
            chunk_overlap: Overlap between chunks.
            top_k: Number of documents to retrieve.
        """
        self.embeddings = embeddings or LocalEmbeddings()
        self.store = store or QdrantStore(embedding_dim=self.embeddings.dimension)
        self.llm = llm or OllamaLLM()
        self.top_k = top_k or settings.retrieval_top_k

        self.chunker = RecursiveChunker(
            chunk_size=chunk_size or settings.chunk_size,
            chunk_overlap=chunk_overlap or settings.chunk_overlap,
        )

        # Custom hooks (set via decorators)
        self._custom_chunker: Callable | None = None
        self._custom_retriever: Callable | None = None
        self._custom_router: Callable | None = None
        self._custom_generator: Callable | None = None

        # System prompt
        self.system_prompt = """You are a helpful assistant that answers questions based on the provided context.

Use the context below to answer the question. If the context doesn't contain enough information to answer the question, say so clearly.

Context:
{context}

Answer the question accurately and concisely based on the context above."""

    @classmethod
    def local(
        cls,
        collection: str | None = None,
        embedding_model: str = "bge-small-en-v1.5",
        llm_model: str = "llama3.2",
    ) -> "RAGPipeline":
        """Create a fully local pipeline.

        Uses local embeddings (sentence-transformers) and Ollama for LLM.
        Fast, free, and works offline.

        Args:
            collection: Qdrant collection name.
            embedding_model: Local embedding model name.
            llm_model: Ollama model name.

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
        )

    @classmethod
    def cloud(
        cls,
        collection: str | None = None,
        embedding_model: str = "text-embedding-3-small",
        llm_model: str = "gpt-4o-mini",
        api_key: str | None = None,
    ) -> "RAGPipeline":
        """Create a cloud-based pipeline using OpenAI.

        Higher quality but requires API key and has latency.

        Args:
            collection: Qdrant collection name.
            embedding_model: OpenAI embedding model.
            llm_model: OpenAI LLM model.
            api_key: OpenAI API key.

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
        )

    @classmethod
    def hybrid(
        cls,
        collection: str | None = None,
        embedding_model: str = "bge-small-en-v1.5",
        llm_model: str = "gpt-4o-mini",
        api_key: str | None = None,
    ) -> "RAGPipeline":
        """Create a hybrid pipeline (local embeddings + cloud LLM).

        Best of both worlds: fast local retrieval with high-quality generation.

        Args:
            collection: Qdrant collection name.
            embedding_model: Local embedding model.
            llm_model: OpenAI LLM model.
            api_key: OpenAI API key.

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
        )

    # Decorator methods for customization
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

    def ingest(
        self,
        source: Union[str, Path, list],
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Ingest documents from a source.

        Automatically detects file types and handles:
        - Single files (PDF, TXT, MD, etc.)
        - Directories (recursively)
        - URLs (web pages)
        - Lists of any of the above

        Args:
            source: Path, URL, directory, or list of sources.
            metadata: Optional metadata to attach to all documents.

        Returns:
            Number of chunks indexed.
        """
        # Load documents
        loaded_docs = auto_load(source)

        # Chunk documents
        all_chunks = []
        for doc in loaded_docs:
            chunk_metadata = {**(metadata or {}), **doc.metadata}

            if self._custom_chunker:
                chunks = self._custom_chunker(doc.content, chunk_metadata)
            else:
                chunks = self.chunker.chunk(doc.content, chunk_metadata)

            all_chunks.extend(chunks)

        if not all_chunks:
            return 0

        # Create Document objects
        documents = [
            Document(content=chunk.content, metadata=chunk.metadata)
            for chunk in all_chunks
        ]

        # Generate embeddings
        texts = [chunk.content for chunk in all_chunks]
        embeddings = self.embeddings.embed_documents(texts)

        # Store in vector database
        self.store.add(documents, embeddings)

        return len(documents)

    def _retrieve(self, query: str) -> list[SearchResult]:
        """Retrieve relevant documents for a query."""
        if self._custom_retriever:
            return self._custom_retriever(query, self.top_k)

        # Embed query
        query_embedding = self.embeddings.embed_query(query)

        # Search with hybrid (semantic + BM25)
        results = self.store.search(
            query_embedding=query_embedding,
            top_k=self.top_k,
            query_text=query,  # For BM25
        )

        return results

    def _build_context(self, results: list[SearchResult]) -> str:
        """Build context string from search results."""
        if not results:
            return "No relevant context found."

        context_parts = []
        for i, result in enumerate(results, 1):
            source = result.document.metadata.get("source", "Unknown")
            context_parts.append(
                f"[{i}] {result.document.content}\n(Source: {source})"
            )

        return "\n\n".join(context_parts)

    def _route(self, query: str) -> str:
        """Route the query to appropriate handler."""
        if self._custom_router:
            return self._custom_router(query)

        # Default: always use retrieval
        return "retrieval"

    def query(self, query: str) -> RAGResponse:
        """Query the RAG pipeline.

        Args:
            query: User question.

        Returns:
            RAGResponse with answer and sources.
        """
        # Route query
        route = self._route(query)

        if route == "direct":
            # Direct LLM response without retrieval
            response = self.llm.generate(query)
            return RAGResponse(
                answer=response.content,
                sources=[],
                query=query,
                metadata={"route": "direct"},
            )

        # Retrieve relevant documents
        results = self._retrieve(query)

        # Build context
        context = self._build_context(results)

        # Generate response
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
            metadata={"route": route, "num_sources": len(results)},
        )

    async def aquery(self, query: str) -> RAGResponse:
        """Async query the RAG pipeline.

        Args:
            query: User question.

        Returns:
            RAGResponse with answer and sources.
        """
        # Route query
        route = self._route(query)

        if route == "direct":
            response = await self.llm.agenerate(query)
            return RAGResponse(
                answer=response.content,
                sources=[],
                query=query,
                metadata={"route": "direct"},
            )

        # Retrieve
        results = self._retrieve(query)
        context = self._build_context(results)

        # Generate
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
            metadata={"route": route, "num_sources": len(results)},
        )

    async def astream(self, query: str) -> AsyncIterator[str]:
        """Stream a response from the RAG pipeline.

        Args:
            query: User question.

        Yields:
            Response chunks as strings.
        """
        # Route query
        route = self._route(query)

        if route == "direct":
            async for chunk in self.llm.astream(query):
                yield chunk
            return

        # Retrieve
        results = self._retrieve(query)
        context = self._build_context(results)

        # Stream generation
        prompt = self.system_prompt.format(context=context) + f"\n\nQuestion: {query}"
        async for chunk in self.llm.astream(prompt):
            yield chunk

    def clear(self) -> None:
        """Clear all documents from the store."""
        self.store.clear()

    def count(self) -> int:
        """Return the number of documents in the store."""
        return self.store.count()

