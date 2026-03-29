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
    query_embedding = embed_query(query)

    log.info("Running dense (vector) search...")
    dense_results = db.search_dense(query_embedding, k=150)
    log.info("Running sparse (full-text) search...")
    sparse_results = db.search_sparse(query, k=150)
    log.info("Fusing results with RRF (dense=%d, sparse=%d)...", len(dense_results), len(sparse_results))

    # Reciprocal Rank Fusion
    scores = {}
    for rank, result in enumerate(dense_results):
        chunk_id = result["id"]
        scores[chunk_id] = scores.get(chunk_id, 0) + semantic_weight / (60 + rank)
    for rank, result in enumerate(sparse_results):
        chunk_id = result["id"]
        scores[chunk_id] = scores.get(chunk_id, 0) + bm25_weight / (60 + rank)

    # Sort by fused score, return top-k
    sorted_ids = sorted(scores, key=scores.get, reverse=True)[:k]
    log.info("Returning top %d chunks.", len(sorted_ids))
    return [db.get_chunk_by_id(cid) for cid in sorted_ids]
