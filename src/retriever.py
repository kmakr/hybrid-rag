import time

from src import db
from src.embedder import embed_query
from src.log import get_logger

log = get_logger("retriever")


def search_hybrid(
    query: str,
    k: int = 20,
    semantic_weight: float = 0.8,
    bm25_weight: float = 0.2,
) -> list[dict]:
    """Hybrid search using dense (pgvector) + sparse (tsvector) with RRF."""
    t0 = time.perf_counter()

    query_embedding = embed_query(query)
    t1 = time.perf_counter()
    print(f"[BENCH] embed_query:    {t1 - t0:.3f}s")

    dense_results = db.search_dense(query_embedding, k=150)
    t2 = time.perf_counter()
    print(f"[BENCH] search_dense:   {t2 - t1:.3f}s")

    sparse_results = db.search_sparse(query, k=150)
    t3 = time.perf_counter()
    print(f"[BENCH] search_sparse:  {t3 - t2:.3f}s")

    # Reciprocal Rank Fusion
    scores = {}
    for rank, result in enumerate(dense_results):
        chunk_id = result["id"]
        scores[chunk_id] = scores.get(chunk_id, 0) + semantic_weight / (60 + rank)
    for rank, result in enumerate(sparse_results):
        chunk_id = result["id"]
        scores[chunk_id] = scores.get(chunk_id, 0) + bm25_weight / (60 + rank)

    sorted_ids = sorted(scores, key=scores.get, reverse=True)[:k]
    t4 = time.perf_counter()
    print(f"[BENCH] rrf_fusion:     {t4 - t3:.3f}s")

    results = db.get_chunks_by_ids(sorted_ids)
    t5 = time.perf_counter()
    print(f"[BENCH] get_chunks:     {t5 - t4:.3f}s")
    print(f"[BENCH] TOTAL:          {t5 - t0:.3f}s")

    return results
