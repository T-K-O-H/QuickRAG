"""CLI for QuickRAG."""

import argparse
import sys

from quickrag.logging import get_logger

logger = get_logger(__name__)


def main():
    """QuickRAG CLI entry point."""
    parser = argparse.ArgumentParser(
        description="QuickRAG - Fast, plug-and-play RAG framework"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start the API server")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    serve_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents")
    ingest_parser.add_argument("source", help="File, directory, or URL to ingest")
    ingest_parser.add_argument("--collection", default="documents", help="Collection name")
    ingest_parser.add_argument("--mode", choices=["local", "cloud", "hybrid"], default="local")

    # Query command
    query_parser = subparsers.add_parser("query", help="Query the RAG pipeline")
    query_parser.add_argument("question", help="Question to ask")
    query_parser.add_argument("--collection", default="documents", help="Collection name")
    query_parser.add_argument("--mode", choices=["local", "cloud", "hybrid"], default="local")
    query_parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    query_parser.add_argument(
        "--conversational", action="store_true", help="Enable conversational mode"
    )

    # Info command
    info_parser = subparsers.add_parser("info", help="Show collection info")
    info_parser.add_argument("--collection", default="documents", help="Collection name")

    args = parser.parse_args()

    if args.command == "serve":
        import uvicorn

        uvicorn.run(
            "api.main:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
        )

    elif args.command == "ingest":
        from quickrag import RAGPipeline

        pipeline = _make_pipeline(args.mode, args.collection)

        logger.info("Ingesting from: %s", args.source)
        count = pipeline.ingest(args.source)
        logger.info("Ingested %d chunks into '%s'", count, args.collection)

    elif args.command == "query":
        from quickrag import RAGPipeline

        pipeline = _make_pipeline(args.mode, args.collection)
        pipeline.top_k = args.top_k

        logger.info("Querying: %s", args.question)

        if args.conversational:
            response = pipeline.query_conversational(args.question)
        else:
            response = pipeline.query(args.question)

        print("=" * 50)
        print("ANSWER:")
        print("=" * 50)
        print(response.answer)

        if response.sources:
            print("\n" + "=" * 50)
            print(f"SOURCES ({len(response.sources)}):")
            print("=" * 50)
            for i, source in enumerate(response.sources, 1):
                print(f"\n[{i}] Score: {source.score:.2f}")
                preview = source.document.content[:150].replace("\n", " ")
                print(f"    {preview}...")

    elif args.command == "info":
        from quickrag import RAGPipeline

        pipeline = _make_pipeline("local", args.collection)

        print(f"Collection: {args.collection}")
        print(f"  Documents: {pipeline.count()}")
        print(f"  Embedding dim: {pipeline.embeddings.dimension}")

    else:
        parser.print_help()
        sys.exit(1)


def _make_pipeline(mode: str, collection: str):
    """Create a pipeline with the given mode and collection."""
    from quickrag import RAGPipeline

    if mode == "cloud":
        return RAGPipeline.cloud(collection=collection)
    elif mode == "hybrid":
        return RAGPipeline.hybrid(collection=collection)
    else:
        return RAGPipeline.local(collection=collection)


if __name__ == "__main__":
    main()
