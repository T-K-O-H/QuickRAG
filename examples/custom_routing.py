"""Example of custom routing with decorators."""

from quickrag import RAGPipeline

pipeline = RAGPipeline.local(collection="routing_demo")

# Custom router to handle different query types
@pipeline.router
def smart_router(query: str) -> str:
    """Route queries based on their type.
    
    Returns:
        - "direct": Answer without retrieval (greetings, simple questions)
        - "retrieval": Normal RAG flow
    """
    query_lower = query.lower().strip()
    
    # Handle greetings
    greetings = ["hello", "hi", "hey", "good morning", "good afternoon"]
    if any(query_lower.startswith(g) for g in greetings):
        return "direct"
    
    # Handle simple questions
    simple_questions = [
        "what time is it",
        "what's the date",
        "who are you",
        "what can you do",
    ]
    if any(q in query_lower for q in simple_questions):
        return "direct"
    
    # Default to retrieval
    return "retrieval"


# Custom retriever with logging
@pipeline.retriever
def logging_retriever(query: str, top_k: int):
    """Custom retriever that logs retrieval."""
    print(f"🔍 Retrieving for: {query}")
    
    # Use default retrieval
    query_embedding = pipeline.embeddings.embed_query(query)
    results = pipeline.store.search(
        query_embedding=query_embedding,
        top_k=top_k,
        query_text=query,
    )
    
    print(f"📄 Found {len(results)} results")
    for i, r in enumerate(results):
        print(f"   [{i+1}] Score: {r.score:.2f}")
    
    return results


# Test different query types
if __name__ == "__main__":
    # First, ingest some documents
    pipeline.ingest("./documents")
    
    print("\n" + "=" * 50)
    print("Testing greeting (should skip retrieval):")
    print("=" * 50)
    response = pipeline.query("Hello, how are you?")
    print(f"Answer: {response.answer}")
    print(f"Route: {response.metadata.get('route')}")
    
    print("\n" + "=" * 50)
    print("Testing document query (should use retrieval):")
    print("=" * 50)
    response = pipeline.query("What is the main topic of the documents?")
    print(f"Answer: {response.answer}")
    print(f"Route: {response.metadata.get('route')}")
    print(f"Sources: {len(response.sources)}")

