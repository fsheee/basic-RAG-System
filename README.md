# Basic-RAG-SYSTEM

A Retrieval-Augmented Generation (RAG) system that loads PDF documents, generates vector embeddings, stores them in ChromaDB, and answers queries using Groq-hosted LLMs.

## How RAG Works

### Ingestion Phase

```
PDF Documents
     │
     ▼
 ┌─────────────┐
 │   Loader    │  PyPDFLoader
 └──────┬──────┘
        │
        ▼
 ┌─────────────┐
 │  Splitter   │  RecursiveCharacterTextSplitter (chunk_size=500, overlap=100)
 └──────┬──────┘
        │
        ▼
 ┌─────────────┐
 │ Embeddings  │  all-MiniLM-L6-v2 → vector embeddings
 └──────┬──────┘
        │
        ▼
 ┌─────────────┐
 │  ChromaDB   │  Persistent vector store
 └─────────────┘
```

### Query Phase

```
User Question
     │
     ▼
 ┌─────────────┐
 │  Retriever  │  Find top-3 similar chunks in ChromaDB
 └──────┬──────┘
        │
        ▼
 ┌─────────────┐
 │  Context    │  Retrieved chunks + original question
 └──────┬──────┘
        │
        ▼
 ┌─────────────┐
 │  LLM (Groq) │  llama-3.3-70b-versatile generates answer
 └──────┬──────┘
        │
        ▼
    Answer
```

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
git clone <repo-url>
cd basic-rag

uv venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # macOS/Linux

uv sync
```

Configure environment variables in `.env`.

## Usage

### 1. Ingest documents

```bash
uv run ingest.py
```

Loads PDFs from `data/`, splits into chunks, and creates a ChromaDB vector store.

### 2. Run the chatbot

**CLI:**
```bash
uv run main.py
```

**Web UI (Streamlit):**
```bash
uv run streamlit run app.py
```

## Project Structure

```
basic-rag/
├── app.py                # Streamlit web UI
├── ingest.py             # Vector database builder
├── main.py               # CLI chatbot
├── rag/
│   ├── config.py         # Environment config loader
│   ├── embeddings.py     # HuggingFace embedding model
│   ├── llm.py            # Groq LLM setup
│   ├── loader.py         # PDF document loader
│   ├── prompt.py         # RAG prompt template
│   ├── rag_chain.py      # Retriever + LLM chain
│   ├── retriever.py      # Vector store retriever
│   ├── splitter.py       # Document chunking
│   └── vector_store.py   # ChromaDB integration
├── data/
│   ├── healthcare_long.pdf
│   └── hospital_policy.pdf
├── chroma_db/            # Auto-generated vector store
├── pyproject.toml
├── uv.lock
└── .env
```

## Customisation

- **Replace PDFs:** Drop your own PDFs into `data/`.
- **Change the LLM:** Update `MODEL_NAME` in `.env`.
- **Change the embedding model:** Update `EMBEDDING_MODEL` in `.env`.
- **Adjust chunk size:** Edit `chunk_size` and `chunk_overlap` in `rag/splitter.py`.
- **Retrieval count:** Change `k` in `rag/retriever.py`.
