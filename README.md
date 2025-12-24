# ⚡ QuickRAG

**Fast, plug-and-play RAG framework built on LangGraph**

From zero to RAG in 5 minutes. QuickRAG provides a batteries-included framework for building Retrieval-Augmented Generation applications with hybrid search, conditional routing, and a beautiful web interface.

## ✨ Features

- **🚀 Plug & Play** - Get started with 3 lines of code
- **🔍 Hybrid Search** - Semantic + BM25 keyword search via Qdrant
- **⚡ Fast Local Mode** - Local embeddings + Ollama for zero-latency retrieval
- **☁️ Cloud Mode** - OpenAI embeddings + GPT for highest quality
- **🎯 Smart Routing** - Conditional query routing via LangGraph
- **🎨 Decorator Pattern** - Pythonic customization for power users
- **🖥️ Beautiful UI** - Next.js chat interface with document management
- **📚 Document Management** - Upload, view, and delete individual documents

## 🏃 Quick Start

### Prerequisites

- **Python 3.10+**
- **Docker** (for Qdrant)
- **Node.js 18+** (for the web UI)

### 1. Clone & Install

```bash
git clone https://github.com/T-K-O-H/QuickRAG
cd QuickRAG

# Install Python package
pip install -e ".[api]"
```

### 2. Start Qdrant

```bash
docker run -d -p 6333:6333 --name qdrant qdrant/qdrant
```

### 3. Configure Environment

```bash
cp env.example .env
# Edit .env and add your OpenAI API key (for hybrid/cloud mode)
```

### 4. Start the API

```bash
# Set mode: local (Ollama), hybrid (local embeddings + OpenAI), or cloud (all OpenAI)
export QUICKRAG_MODE=hybrid  # or set in .env

uvicorn api.main:app --host 127.0.0.1 --port 8000
```

### 5. Start the Web UI

```bash
cd web
npm install
npm run dev
```

Open **http://localhost:3000** and start chatting!

## 💻 Python Usage

```python
from quickrag import RAGPipeline

# Create a pipeline (choose your mode)
pipeline = RAGPipeline.local()   # Ollama + local embeddings
pipeline = RAGPipeline.hybrid()  # Local embeddings + OpenAI GPT
pipeline = RAGPipeline.cloud()   # All OpenAI

# Ingest documents
pipeline.ingest("./documents")       # Directory
pipeline.ingest("document.pdf")      # Single file
pipeline.ingest("https://example.com") # Web page

# Query
response = pipeline.query("What is the refund policy?")
print(response.answer)
print(response.sources)  # Retrieved chunks with scores
```

## 🔧 Pipeline Modes

| Mode | Embeddings | LLM | Best For |
|------|------------|-----|----------|
| `local()` | sentence-transformers | Ollama | Offline, fast, free |
| `hybrid()` | sentence-transformers | OpenAI GPT | Best balance |
| `cloud()` | OpenAI | OpenAI GPT | Highest quality |

## 🎨 Decorator Customization

```python
pipeline = RAGPipeline.local()

# Custom query routing
@pipeline.router
def my_router(query: str) -> str:
    if "hello" in query.lower():
        return "direct"  # Skip retrieval
    return "retrieval"   # Normal RAG

# Custom retrieval logic
@pipeline.retriever
def my_retriever(query: str, top_k: int):
    # Your custom retrieval logic
    return results

# Custom generation
@pipeline.generator
def my_generator(query: str, context: str) -> str:
    # Your custom generation logic
    return answer
```

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/query` | Query the RAG pipeline |
| POST | `/api/query/stream` | Stream response (SSE) |
| POST | `/api/ingest/file` | Upload a file |
| POST | `/api/ingest/files` | Upload multiple files |
| POST | `/api/ingest/url` | Ingest from URL |
| POST | `/api/ingest/text` | Ingest raw text |
| GET | `/api/documents` | List all documents |
| DELETE | `/api/documents/{id}` | Delete a document |
| GET | `/api/collections` | List collections |

## 🖥️ Web Interface

The web UI provides:

- **💬 Chat Interface** - Ask questions with streaming responses
- **📄 Source Citations** - See which documents informed the answer
- **📁 Knowledge Base** - Manage documents and collections
  - Upload files (PDF, TXT, MD)
  - Add URLs
  - Paste text
  - Delete individual documents

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        QuickRAG                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────┐  ┌─────────┐  ┌──────────┐  ┌─────────┐       │
│  │ Loaders │  │Chunkers │  │Embeddings│  │  LLMs   │       │
│  │PDF, TXT │  │Recursive│  │  Local   │  │ Ollama  │       │
│  │Web, MD  │  │  Text   │  │ OpenAI   │  │ OpenAI  │       │
│  └────┬────┘  └────┬────┘  └────┬─────┘  └────┬────┘       │
│       │            │            │             │             │
│       ▼            ▼            ▼             ▼             │
│  ┌──────────────────────────────────────────────────┐      │
│  │                  RAGPipeline                      │      │
│  │  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐ │      │
│  │  │ Ingest │→ │ Router │→ │Retrieve│→ │Generate│ │      │
│  │  └────────┘  └────────┘  └────────┘  └────────┘ │      │
│  └──────────────────────────────────────────────────┘      │
│                           │                                 │
│                           ▼                                 │
│  ┌──────────────────────────────────────────────────┐      │
│  │           Qdrant Vector Database                  │      │
│  │        Hybrid Search (Semantic + BM25)            │      │
│  └──────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
QuickRAG/
├── quickrag/              # Core Python framework
│   ├── pipeline.py        # Main RAGPipeline class
│   ├── graph.py           # LangGraph workflows
│   ├── embeddings/        # Local & OpenAI embeddings
│   ├── llms/              # Ollama & OpenAI LLMs
│   ├── stores/            # Qdrant with hybrid search
│   ├── loaders/           # PDF, text, web loaders
│   └── chunkers/          # Text chunking
├── api/                   # FastAPI backend
│   ├── main.py
│   └── routes/            # API endpoints
├── web/                   # Next.js frontend
│   ├── app/               # Pages
│   └── components/        # React components
├── examples/              # Usage examples
├── docker-compose.yml     # Full stack deployment
└── pyproject.toml         # Python dependencies
```

## 🐳 Docker Deployment

```bash
# Start all services
docker-compose up -d

# Services:
# - Qdrant: http://localhost:6333
# - API: http://localhost:8000
# - Web: http://localhost:3000
```

## 📝 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | - | OpenAI API key (for hybrid/cloud) |
| `QDRANT_HOST` | localhost | Qdrant host |
| `QDRANT_PORT` | 6333 | Qdrant port |
| `OLLAMA_HOST` | http://localhost:11434 | Ollama host |
| `QUICKRAG_MODE` | local | Pipeline mode |
| `QUICKRAG_COLLECTION` | documents | Default collection |

## 🔬 Hybrid Search

QuickRAG uses Qdrant's native hybrid search combining:

- **Semantic Search** - Dense vectors from embedding models
- **Keyword Search** - Sparse BM25-style vectors
- **RRF Fusion** - Reciprocal Rank Fusion for optimal results

This catches both semantic meaning and exact keyword matches.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Built With

- [LangGraph](https://github.com/langchain-ai/langgraph) - Stateful LLM workflows
- [Qdrant](https://qdrant.tech) - Vector database with hybrid search
- [FastEmbed](https://github.com/qdrant/fastembed) - Fast local embeddings
- [FastAPI](https://fastapi.tiangolo.com) - Python API framework
- [Next.js](https://nextjs.org) - React framework
