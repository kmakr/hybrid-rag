import os
from contextlib import contextmanager

import psycopg2
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv

from src.log import get_logger

load_dotenv()

log = get_logger("db")
DATABASE_URL = os.getenv("DATABASE_URL")


@contextmanager
def get_conn():
    conn = psycopg2.connect(DATABASE_URL)
    register_vector(conn)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Create extensions, tables, and indexes if they don't exist."""
    log.info("Connecting to database: %s", DATABASE_URL)

    # Use a plain connection first to create the extension before register_vector
    conn = psycopg2.connect(DATABASE_URL)
    try:
        cur = conn.cursor()
        log.info("Creating pgvector extension...")
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()
    finally:
        conn.close()

    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        log.info("Creating chunks table...")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id SERIAL PRIMARY KEY,
                content TEXT NOT NULL,
                contextualized_content TEXT,
                full_text TEXT NOT NULL,
                source_document TEXT,
                chunk_index INTEGER,
                embedding vector(1536),
                tsv tsvector GENERATED ALWAYS AS (
                    to_tsvector('english', full_text)
                ) STORED
            );
        """)
        log.info("Creating vector index (hnsw)...")
        cur.execute("""
            CREATE INDEX IF NOT EXISTS chunks_embedding_idx
            ON chunks USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64);
        """)
        log.info("Creating full-text search index...")
        cur.execute("""
            CREATE INDEX IF NOT EXISTS chunks_tsv_idx
            ON chunks USING gin (tsv);
        """)
        log.info("Database initialized.")


def insert_chunk(content, context, full_text, source, chunk_index, embedding):
    """Insert a single chunk."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO chunks (content, contextualized_content, full_text,
                                source_document, chunk_index, embedding)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (content, context, full_text, source, chunk_index, embedding),
        )
        return cur.fetchone()[0]


def insert_chunks_batch(chunks):
    """Batch insert chunks. Each chunk is a dict with keys:
    content, context, full_text, source, chunk_index, embedding
    """
    log.info("Inserting %d chunks into database...", len(chunks))
    with get_conn() as conn:
        cur = conn.cursor()
        args = [
            (
                c["content"],
                c["context"],
                c["full_text"],
                c["source"],
                c["chunk_index"],
                c["embedding"],
            )
            for c in chunks
        ]
        cur.executemany(
            """
            INSERT INTO chunks (content, contextualized_content, full_text,
                                source_document, chunk_index, embedding)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            args,
        )
        log.info("Batch insert complete.")


def search_dense(query_embedding, k=150):
    """Cosine similarity search via pgvector."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, content, contextualized_content, full_text,
                   source_document, chunk_index,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (query_embedding, query_embedding, k),
        )
        columns = [
            "id", "content", "contextualized_content", "full_text",
            "source_document", "chunk_index", "similarity",
        ]
        return [dict(zip(columns, row)) for row in cur.fetchall()]


def search_sparse(query_text, k=150):
    """Full-text search via tsvector."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, content, contextualized_content, full_text,
                   source_document, chunk_index,
                   ts_rank(tsv, plainto_tsquery('english', %s)) AS rank
            FROM chunks
            WHERE tsv @@ plainto_tsquery('english', %s)
            ORDER BY rank DESC
            LIMIT %s
            """,
            (query_text, query_text, k),
        )
        columns = [
            "id", "content", "contextualized_content", "full_text",
            "source_document", "chunk_index", "rank",
        ]
        return [dict(zip(columns, row)) for row in cur.fetchall()]


def get_chunk_by_id(chunk_id):
    """Fetch a single chunk by ID."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, content, contextualized_content, full_text,
                   source_document, chunk_index
            FROM chunks WHERE id = %s
            """,
            (chunk_id,),
        )
        columns = [
            "id", "content", "contextualized_content", "full_text",
            "source_document", "chunk_index",
        ]
        row = cur.fetchone()
        return dict(zip(columns, row)) if row else None


def get_chunks_by_ids(chunk_ids: list[int]) -> list[dict]:
    """Fetch multiple chunks in a single query, preserving input order."""
    if not chunk_ids:
        return []
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, content, contextualized_content, full_text,
                   source_document, chunk_index
            FROM chunks WHERE id = ANY(%s)
            """,
            (chunk_ids,),
        )
        columns = ["id", "content", "contextualized_content", "full_text",
                   "source_document", "chunk_index"]
        rows = {row[0]: dict(zip(columns, row)) for row in cur.fetchall()}
    return [rows[cid] for cid in chunk_ids if cid in rows]


def get_all_chunks():
    """Return all chunks (for debugging/eval)."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, content, contextualized_content, full_text,
                   source_document, chunk_index
            FROM chunks ORDER BY id
            """
        )
        columns = [
            "id", "content", "contextualized_content", "full_text",
            "source_document", "chunk_index",
        ]
        return [dict(zip(columns, row)) for row in cur.fetchall()]
