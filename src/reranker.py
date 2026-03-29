from sentence_transformers import CrossEncoder

from src.log import get_logger

log = get_logger("reranker")

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_model = None


def _get_model():
    global _model
    if _model is None:
        log.info("Loading reranker model '%s' (first run downloads from HuggingFace)...", MODEL_NAME)
        _model = CrossEncoder(MODEL_NAME)
        log.info("Reranker model loaded.")
    return _model


def rerank(query: str, chunks: list[dict], k: int) -> list[dict]:
    """Rerank chunks using a cross-encoder model.

    Retrieves k*10 candidates from the input list and returns the top-k
    after scoring with the cross-encoder.

    Each chunk must have a 'full_text' or 'contextualized_content' field.
    """
    model = _get_model()
    candidates = chunks[: k * 10]
    log.info("Reranking %d candidates -> top %d...", len(candidates), k)

    texts = [c.get("full_text") or c.get("contextualized_content") or c.get("content", "") for c in candidates]
    pairs = [(query, t) for t in texts]
    scores = model.predict(pairs)

    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    log.info("Reranking complete.")
    return [chunk for _, chunk in ranked[:k]]
