import os

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from src.config import models
from src.log import get_logger

load_dotenv()

log = get_logger("embedder")

MODEL = models["embedder"]
DIMS = 1536
BATCH_SIZE = 64

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts via OpenRouter. Returns list of 2560-dim vectors."""
    log.info("Embedding %d texts via OpenRouter (%s)...", len(texts), MODEL)
    embeddings = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding"):
        batch = texts[i : i + BATCH_SIZE]
        resp = client.embeddings.create(model=MODEL, input=batch, dimensions=DIMS)
        embeddings.extend([d.embedding for d in resp.data])
    log.info("Embedding complete.")
    return embeddings


def embed_query(query: str) -> list[float]:
    """Embed a single query string."""
    resp = client.embeddings.create(model=MODEL, input=query, dimensions=DIMS)
    return resp.data[0].embedding
