import os
import time

from openai import OpenAI
from dotenv import load_dotenv

from src.log import get_logger

load_dotenv()

log = get_logger("generator")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

MODEL = "minimax/minimax-m2.7"


def generate_response(query: str, chunks: list[dict]) -> str:
    """Generate a grounded response using retrieved chunks as context."""
    log.info("Generating response using %s with %d context chunks...", MODEL, len(chunks))
    context = "\n---\n".join([c["full_text"] for c in chunks])

    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=1024,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Answer the user's question "
                    "based only on the provided context. If the context doesn't "
                    "contain the answer, say so."
                ),
            },
            {
                "role": "user",
                "content": f"""<context>
{context}
</context>

Question: {query}""",
            },
        ],
    )
    t1 = time.perf_counter()
    print(f"[BENCH] llm_generate:   {t1 - t0:.3f}s")
    answer = response.choices[0].message.content or ""
    log.info("Response received.")
    return answer
