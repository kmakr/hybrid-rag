import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.pipeline import query as pipeline_query, ingest
from src.log import get_logger

log = get_logger("api")

app = FastAPI(title="HKJC Contextual RAG")

STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "static")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


# ── Models ────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str
    k: int = 5
    rerank: bool = False


class Source(BaseModel):
    document: str
    chunk_index: int


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model")
def get_model():
    from src.config import models
    return {"model": models["generator"]}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    log.info("Chat request: %r (k=%d, rerank=%s)", req.question, req.k, req.rerank)
    try:
        fetch_k = req.k * 10 if req.rerank else req.k
        from src.retriever import search_hybrid
        chunks = search_hybrid(req.question, k=fetch_k)

        if req.rerank:
            from src.reranker import rerank
            chunks = rerank(req.question, chunks, k=req.k)

        from src.generator import generate_response
        answer = generate_response(req.question, chunks)

        sources = [
            Source(
                document=c["source_document"] or "",
                chunk_index=c["chunk_index"] or 0,
            )
            for c in chunks
            if c
        ]
        # Deduplicate sources
        seen = set()
        unique_sources = []
        for s in sources:
            key = (s.document, s.chunk_index)
            if key not in seen:
                seen.add(key)
                unique_sources.append(s)

        return ChatResponse(answer=answer, sources=unique_sources)
    except Exception as e:
        log.error("Chat error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    """Stream the LLM response token-by-token via SSE."""
    log.info("Stream chat request: %r (k=%d, rerank=%s)", req.question, req.k, req.rerank)
    try:
        fetch_k = req.k * 10 if req.rerank else req.k
        from src.retriever import search_hybrid
        chunks = search_hybrid(req.question, k=fetch_k)

        if req.rerank:
            from src.reranker import rerank
            chunks = rerank(req.question, chunks, k=req.k)

        sources = []
        seen = set()
        for c in chunks:
            if c:
                key = (c.get("source_document", ""), c.get("chunk_index", 0))
                if key not in seen:
                    seen.add(key)
                    sources.append({"document": key[0], "chunk_index": key[1]})

        from src.generator import generate_response_stream

        def event_stream():
            for token in generate_response_stream(req.question, chunks):
                yield f"data: {json.dumps({'token': token})}\n\n"
            yield f"data: {json.dumps({'sources': sources, 'done': True})}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")
    except Exception as e:
        log.error("Stream chat error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest")
async def ingest_files(files: list[UploadFile] = File(...)):
    """Upload and ingest documents."""
    os.makedirs(DATA_DIR, exist_ok=True)
    saved = []
    for file in files:
        dest = os.path.join(DATA_DIR, file.filename)
        content = await file.read()
        with open(dest, "wb") as f:
            f.write(content)
        saved.append(file.filename)
        log.info("Saved uploaded file: %s", file.filename)

    log.info("Starting ingestion for: %s", saved)
    try:
        ingest(DATA_DIR)
    except Exception as e:
        log.error("Ingestion error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    return {"ingested": saved}


# ── Static frontend ───────────────────────────────────────────────────────────

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))
