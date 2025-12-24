"""Example demonstrating hybrid search (semantic + BM25)."""

from quickrag import RAGPipeline, QdrantStore, LocalEmbeddings

# Create a pipeline with hybrid search enabled
embeddings = LocalEmbeddings("bge-small-en-v1.5")
store = QdrantStore(
    collection="hybrid_demo",
    embedding_dim=embeddings.dimension,
    enable_hybrid=True,  # Enable BM25 + semantic fusion
)

pipeline = RAGPipeline(
    store=store,
    embeddings=embeddings,
    top_k=5,
)

# Sample documents to demonstrate hybrid search
documents = [
    """
    The Python programming language was created by Guido van Rossum and
    first released in 1991. Python emphasizes code readability and allows
    programmers to express concepts in fewer lines of code.
    """,
    """
    Anaconda is a distribution of Python and R for scientific computing.
    It simplifies package management and deployment. The Anaconda distribution
    includes data science packages like NumPy, SciPy, and Pandas.
    """,
    """
    Machine learning with Python typically involves libraries like scikit-learn,
    TensorFlow, and PyTorch. These frameworks provide tools for building
    neural networks and training models on data.
    """,
    """
    The anaconda is a large snake found in South America. It is one of the
    heaviest snakes in the world. Anacondas are non-venomous and kill their
    prey by constriction.
    """,
]

if __name__ == "__main__":
    # Ingest documents
    print("Ingesting documents...")
    for doc in documents:
        from tempfile import NamedTemporaryFile
        with NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(doc.strip())
            pipeline.ingest(f.name)
    
    print(f"Total chunks: {pipeline.count()}")
    
    # Query 1: Semantic search should find related concepts
    print("\n" + "=" * 60)
    print("Query 1: 'What programming language is good for data science?'")
    print("(Semantic search should match Python/ML content)")
    print("=" * 60)
    
    response = pipeline.query("What programming language is good for data science?")
    print(f"\nAnswer: {response.answer[:200]}...")
    print(f"\nTop sources:")
    for s in response.sources[:2]:
        print(f"  - Score: {s.score:.2f}: {s.document.content[:80]}...")
    
    # Query 2: Keyword search should distinguish between anaconda meanings
    print("\n" + "=" * 60)
    print("Query 2: 'Tell me about anaconda the snake'")
    print("(BM25 keyword match should prefer snake content)")
    print("=" * 60)
    
    response = pipeline.query("Tell me about anaconda the snake")
    print(f"\nAnswer: {response.answer[:200]}...")
    print(f"\nTop sources:")
    for s in response.sources[:2]:
        print(f"  - Score: {s.score:.2f}: {s.document.content[:80]}...")
    
    # Query 3: Technical term that benefits from keyword matching
    print("\n" + "=" * 60)
    print("Query 3: 'How do I use scikit-learn?'")
    print("(Keyword 'scikit-learn' should boost ML document)")
    print("=" * 60)
    
    response = pipeline.query("How do I use scikit-learn?")
    print(f"\nAnswer: {response.answer[:200]}...")
    print(f"\nTop sources:")
    for s in response.sources[:2]:
        print(f"  - Score: {s.score:.2f}: {s.document.content[:80]}...")

