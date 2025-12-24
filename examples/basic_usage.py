"""Basic usage example for QuickRAG."""

from quickrag import RAGPipeline

# Create a local pipeline (uses Ollama + local embeddings)
# This requires:
# 1. Qdrant running: docker run -d -p 6333:6333 qdrant/qdrant
# 2. Ollama running with a model: ollama run llama3.2

pipeline = RAGPipeline.local(collection="example_docs")

# Ingest some documents
# Supports: PDF, TXT, MD, directories, URLs
docs_ingested = pipeline.ingest("./documents")
print(f"Ingested {docs_ingested} chunks")

# Query the pipeline
response = pipeline.query("What are the main topics in the documents?")

print("\n" + "=" * 50)
print("ANSWER:")
print("=" * 50)
print(response.answer)

print("\n" + "=" * 50)
print("SOURCES:")
print("=" * 50)
for i, source in enumerate(response.sources, 1):
    print(f"\n[{i}] Score: {source.score:.2f}")
    print(f"    {source.document.content[:200]}...")

