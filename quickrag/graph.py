"""LangGraph-based RAG workflow for QuickRAG."""

from typing import TypedDict, Annotated, Literal
from dataclasses import dataclass

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from quickrag.stores.base import SearchResult


class RAGState(TypedDict):
    """State for the RAG graph."""

    query: str
    route: str
    context: str
    sources: list[SearchResult]
    answer: str
    messages: Annotated[list, add_messages]


def create_rag_graph(pipeline: "RAGPipeline") -> StateGraph:
    """Create a LangGraph workflow for RAG.

    This creates a graph with the following flow:
    1. Router: Decide if we need retrieval or can answer directly
    2. Retriever: Get relevant documents (if needed)
    3. Generator: Generate the answer

    Args:
        pipeline: The RAGPipeline instance to use.

    Returns:
        Compiled StateGraph.
    """
    from quickrag.pipeline import RAGPipeline

    def route_query(state: RAGState) -> RAGState:
        """Route the query based on type."""
        query = state["query"]
        route = pipeline._route(query)
        return {**state, "route": route}

    def retrieve(state: RAGState) -> RAGState:
        """Retrieve relevant documents."""
        query = state["query"]
        results = pipeline._retrieve(query)
        context = pipeline._build_context(results)
        return {**state, "sources": results, "context": context}

    def generate(state: RAGState) -> RAGState:
        """Generate the answer."""
        query = state["query"]
        context = state.get("context", "")

        if state["route"] == "direct":
            response = pipeline.llm.generate(query)
        else:
            prompt = pipeline.system_prompt.format(context=context) + f"\n\nQuestion: {query}"
            response = pipeline.llm.generate(prompt)

        return {**state, "answer": response.content}

    def should_retrieve(state: RAGState) -> Literal["retrieve", "generate"]:
        """Decide whether to retrieve or generate directly."""
        if state["route"] == "direct":
            return "generate"
        return "retrieve"

    # Build the graph
    graph = StateGraph(RAGState)

    # Add nodes
    graph.add_node("router", route_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)

    # Add edges
    graph.set_entry_point("router")
    graph.add_conditional_edges(
        "router",
        should_retrieve,
        {
            "retrieve": "retrieve",
            "generate": "generate",
        },
    )
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


def create_conversational_rag_graph(pipeline: "RAGPipeline") -> StateGraph:
    """Create a conversational RAG graph with memory.

    This extends the basic RAG graph with:
    - Conversation history
    - Query rewriting for context
    - Follow-up handling

    Args:
        pipeline: The RAGPipeline instance to use.

    Returns:
        Compiled StateGraph.
    """
    from quickrag.pipeline import RAGPipeline

    def rewrite_query(state: RAGState) -> RAGState:
        """Rewrite query with conversation context."""
        messages = state.get("messages", [])
        query = state["query"]

        if not messages:
            return state

        # Use LLM to rewrite query with context
        history = "\n".join([f"{m['role']}: {m['content']}" for m in messages[-4:]])
        rewrite_prompt = f"""Given the conversation history and the latest question, rewrite the question to be self-contained.

Conversation history:
{history}

Latest question: {query}

Rewritten question (self-contained):"""

        response = pipeline.llm.generate(rewrite_prompt)
        rewritten = response.content.strip()

        return {**state, "query": rewritten}

    def route_query(state: RAGState) -> RAGState:
        """Route the query."""
        route = pipeline._route(state["query"])
        return {**state, "route": route}

    def retrieve(state: RAGState) -> RAGState:
        """Retrieve documents."""
        results = pipeline._retrieve(state["query"])
        context = pipeline._build_context(results)
        return {**state, "sources": results, "context": context}

    def generate(state: RAGState) -> RAGState:
        """Generate answer."""
        query = state["query"]
        context = state.get("context", "")
        messages = state.get("messages", [])

        # Build conversation-aware prompt
        if state["route"] == "direct":
            prompt = query
        else:
            prompt = pipeline.system_prompt.format(context=context)
            if messages:
                prompt += "\n\nPrevious conversation:\n"
                prompt += "\n".join([f"{m['role']}: {m['content']}" for m in messages[-4:]])
            prompt += f"\n\nQuestion: {query}"

        response = pipeline.llm.generate(prompt)

        # Add to messages
        new_messages = messages + [
            {"role": "user", "content": state["query"]},
            {"role": "assistant", "content": response.content},
        ]

        return {**state, "answer": response.content, "messages": new_messages}

    def should_retrieve(state: RAGState) -> Literal["retrieve", "generate"]:
        if state["route"] == "direct":
            return "generate"
        return "retrieve"

    # Build graph
    graph = StateGraph(RAGState)

    graph.add_node("rewrite", rewrite_query)
    graph.add_node("router", route_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)

    graph.set_entry_point("rewrite")
    graph.add_edge("rewrite", "router")
    graph.add_conditional_edges(
        "router",
        should_retrieve,
        {
            "retrieve": "retrieve",
            "generate": "generate",
        },
    )
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()

