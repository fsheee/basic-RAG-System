# basic-rag

A Retrieval-Augmented Generation (RAG) system that loads PDF documents, generates vector embeddings, stores them in ChromaDB, and answers queries using Groq-hosted LLMs.

> **Status:** Scaffolding phase — dependencies and configuration are in place. The core RAG pipeline is yet to be implemented.

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.11 |
| Package manager | `uv` |
| LLM provider | Groq (`llama-3.3-70b-versatile`) |
| Vector database | ChromaDB (persistent, local) |
| Embedding model | `all-MiniLM-L6-v2` (HuggingFace) |
| Document loader | PyPDF |
| Framework | LangChain |

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)

## Setup

```bash
# Clone the repository
git clone <repo-url>
cd basic-rag

# Create and activate virtual environment
uv venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # macOS/Linux

# Install dependencies
uv sync
```

Configure environment variables in `.env` (a working Groq API key is already provided).

## Usage

```bash
uv run main.py
```

Currently the entry point is a placeholder. The intended pipeline: load a PDF → split into chunks → embed → store in ChromaDB → retrieve relevant chunks → generate answers via Groq.

## Project Structure

```
basic-rag/
├── main.py              # Entry point (WIP)
├── rag/
│   └── config.py        # Environment config loader
├── data/
│   └── healthcare_long.pdf  # Sample PDF document
├── chroma_db/           # Auto-generated vector store
├── pyproject.toml       # Project metadata & dependencies
├── uv.lock              # Dependency lockfile
└── .env                 # Environment variables
```

## Customisation

- **Replace the PDF:** Drop your own PDFs into `data/` and update the loader path in `main.py`.
- **Change the LLM:** Update `MODEL_NAME` in `.env`.
- **Change the embedding model:** Update `EMBEDDING_MODEL` in `.env`.
