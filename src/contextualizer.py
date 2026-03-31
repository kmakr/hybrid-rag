import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from src.config import models
from src.log import get_logger

load_dotenv()

log = get_logger("contextualizer")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

MODEL = models["contextualizer"]

DOCUMENT_CONTEXT_PROMPT = """<document>
{doc_content}
</document>
Here is the chunk we want to situate within the whole document
<chunk>
{chunk_content}
</chunk>
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""


def situate_context(doc_content: str, chunk_content: str) -> str:
    """Generate a short context snippet for a chunk within its parent document."""
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=1000,
        temperature=0.0,
        messages=[
            {
                "role": "user",
                "content": DOCUMENT_CONTEXT_PROMPT.format(
                    doc_content=doc_content,
                    chunk_content=chunk_content,
                ),
            }
        ],
    )
    return response.choices[0].message.content


def contextualize_chunks(
    chunks: list[dict],
    parallel_threads: int = 4,
    delay: float = 0.1,
) -> list[dict]:
    """Add contextualized content to each chunk in parallel.

    Each chunk must have 'content' and 'metadata.parent_document'.
    Returns chunks with added 'context' and 'full_text' fields.
    """

    def process(chunk):
        context = situate_context(
            chunk["metadata"]["parent_document"],
            chunk["content"],
        )
        time.sleep(delay)
        return context

    log.info(
        "Contextualizing %d chunks with %d threads via OpenRouter (%s)...",
        len(chunks),
        parallel_threads,
        MODEL,
    )
    with ThreadPoolExecutor(max_workers=parallel_threads) as executor:
        futures = {executor.submit(process, chunk): i for i, chunk in enumerate(chunks)}
        results = [None] * len(chunks)
        for future in tqdm(
            as_completed(futures), total=len(chunks), desc="Contextualizing"
        ):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                log.error("Chunk %d failed contextualization: %s", idx, exc)
                results[idx] = ""

    for chunk, context in zip(chunks, results):
        chunk["context"] = context
        chunk["full_text"] = f"{chunk['content']}\n\n{context}"

    log.info("Contextualization complete. %d chunks processed.", len(chunks))
    return chunks
