import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from src.log import get_logger

load_dotenv()

log = get_logger("contextualizer")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

MODEL = "minimax/minimax-m2.7"

DOCUMENT_SUMMARY_PROMPT = """Please write a detailed summary of the following document. Cover the main topics, purpose, key arguments, section structure, and any important details that would help identify what a given passage is about. Aim for a thorough overview proportional to the document's length.

<document>
{doc_content}
</document>

Answer only with the summary and nothing else."""

DOCUMENT_CONTEXT_PROMPT = """<document_summary>
{doc_summary}
</document_summary>
Here is the chunk we want to situate within the whole document
<chunk>
{chunk_content}
</chunk>
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""


def summarize_document(doc_content: str) -> str:
    """Generate a concise summary of a document."""
    log.debug("Sending summarization request (doc length: %d chars)", len(doc_content))
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=1500,
        temperature=0.0,
        messages=[
            {
                "role": "user",
                "content": DOCUMENT_SUMMARY_PROMPT.format(doc_content=doc_content),
            }
        ],
    )
    log.debug("Summary response — finish_reason: %s, content: %r", response.choices[0].finish_reason, response.choices[0].message.content)
    result = response.choices[0].message.content
    if not result:
        log.warning("summarize_document returned empty/None content. Full response: %s", response)
    return result


def situate_context(doc_summary: str, chunk_content: str) -> str:
    """Generate a short context snippet for a chunk given a document summary."""
    log.debug("Sending contextualization request (summary length: %d chars, chunk length: %d chars)", len(doc_summary), len(chunk_content))
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=600,
        temperature=0.0,
        messages=[
            {
                "role": "user",
                "content": DOCUMENT_CONTEXT_PROMPT.format(
                    doc_summary=doc_summary,
                    chunk_content=chunk_content,
                ),
            }
        ],
    )
    log.debug("Context response — finish_reason: %s, content: %r", response.choices[0].finish_reason, response.choices[0].message.content)
    result = response.choices[0].message.content
    if not result:
        log.warning("situate_context returned empty/None content. Full response: %s", response)
    return result


def contextualize_chunks(
    chunks: list[dict],
    parallel_threads: int = 1,
    delay: float = 0.1,
) -> list[dict]:
    """Add contextualized content to each chunk in parallel.

    Each chunk must have 'content' and 'metadata.parent_document'.
    Returns chunks with added 'context' and 'full_text' fields.
    """
    # Summarize each unique document once
    unique_docs = {c["metadata"]["parent_document"] for c in chunks}
    log.info("Summarizing %d unique document(s)...", len(unique_docs))
    doc_summaries: dict[str, str] = {}
    for doc in tqdm(unique_docs, desc="Summarizing documents"):
        try:
            summary = summarize_document(doc)
            log.info("Document summarized (%d chars → %d char summary)", len(doc), len(summary or ""))
            doc_summaries[doc] = summary
        except Exception as exc:
            log.error("Failed to summarize document: %s", exc, exc_info=True)
            doc_summaries[doc] = ""

    def process(chunk):
        summary = doc_summaries[chunk["metadata"]["parent_document"]]
        context = situate_context(summary, chunk["content"])
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
