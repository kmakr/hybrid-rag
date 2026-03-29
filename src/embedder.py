from sentence_transformers import SentenceTransformer

from src.log import get_logger

log = get_logger("embedder")

MODEL_NAME = "all-MiniLM-L6-v2"  # 384 dimensions

_model = None


def _get_model():
    global _model
    if _model is None:
        log.info("Loading embedding model '%s' (first run downloads from HuggingFace)...", MODEL_NAME)
        _model = SentenceTransformer(MODEL_NAME)
        log.info("Embedding model loaded.")
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts. Returns list of 384-dim vectors."""
    model = _get_model()
    log.info("Embedding %d texts...", len(texts))
    result = model.encode(texts, show_progress_bar=True, batch_size=128).tolist()
    log.info("Embedding complete.")
    return result


def embed_query(query: str) -> list[float]:
    """Embed a single query string."""
    model = _get_model()
    return model.encode(query).tolist()
