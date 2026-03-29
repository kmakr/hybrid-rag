# How to Run the Contextual RAG System

## Prerequisites

- [OrbStack](https://orbstack.dev) or Docker Desktop
- [uv](https://github.com/astral-sh/uv)

## Setup

### 1. Install dependencies

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 2. Configure environment

Copy the example and fill in your credentials:

```bash
cp .env.example .env
```

`.env` should contain:

```
OPENROUTER_API_KEY=sk-or-...
DATABASE_URL=postgresql://rag:rag@localhost:5432/contextual_rag
```

### 3. Start the database

```bash
docker compose up -d
```

This starts a PostgreSQL 16 instance with the `pgvector` extension on port `5432`.

## Ingesting Documents

Drop your files into the `data/` folder. Supported formats:

- `.pdf`
- `.doc` / `.docx`
- `.txt`
- `.md`

Then run:

```bash
python scripts/ingest.py
```

Options:

```
--data-dir PATH   Directory containing documents (default: ./data)
--threads INT     Parallel threads for contextualization (default: 4)
```

The ingest pipeline will:

1. Initialize the database schema (pgvector extension, tables, indexes)
2. Chunk documents into 500-character segments with 50-character overlap
3. Contextualize each chunk in parallel using **Claude 3.5 Haiku** via OpenRouter
4. Generate embeddings using `sentence-transformers` (local, no API cost)
5. Store everything in PostgreSQL

## Querying

```bash
python scripts/query.py "Your question here"
```

Options:

```
positional arguments:
  query           The question to ask

options:
  -k INT          Number of chunks to retrieve (default: 5)
  --rerank        Rerank with a cross-encoder before answering (improves accuracy)
```

Examples:

```bash
# Basic query
python scripts/query.py "What are the race entry requirements?" -k 10

# With reranking (higher accuracy, slightly slower)
python scripts/query.py "What are the race entry requirements?" -k 10 --rerank
```

The query pipeline:
1. Hybrid search — dense (pgvector cosine) + sparse (Postgres full-text / BM25) with Reciprocal Rank Fusion
2. Optional cross-encoder reranking (local model, no API cost)
3. Answer generation using **Claude Sonnet 4** via OpenRouter

## Running the Frontend

```bash
uvicorn src.api:app --reload --port 8000
```

Then open **http://localhost:8000** in your browser.

Features:
- Upload PDF/DOCX/TXT/MD files directly from the UI and ingest them
- Adjust Top-K chunks with a slider
- Toggle reranking on/off per query
- Sources displayed under each answer

## Retrieval Techniques (from Anthropic Contextual Retrieval cookbook)

| Technique | This project | Notes |
|---|---|---|
| Contextual embeddings | ✓ | Claude situates each chunk before embedding |
| Hybrid search (dense + BM25) | ✓ | Postgres tsvector as BM25 equivalent |
| Parallel contextualization | ✓ | `--threads` flag |
| Reranking | ✓ | `--rerank` flag, local cross-encoder |
| Prompt caching | — | Requires direct Anthropic API (not OpenRouter) |
