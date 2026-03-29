from src import db
from src.chunker import chunk_documents
from src.contextualizer import contextualize_chunks
from src.embedder import embed_texts
from src.retriever import search_hybrid
from src.generator import generate_response


def ingest(data_dir: str, parallel_threads: int = 4):
    """Full ingestion pipeline: chunk -> contextualize -> embed -> store."""
    # 1. Initialize database
    print("Initializing database...")
    db.init_db()

    # 2. Chunk documents
    print(f"Loading and chunking documents from {data_dir}...")
    chunks = chunk_documents(data_dir)
    if not chunks:
        print("No chunks found. Make sure data/ contains supported files (.pdf, .doc, .docx, .txt, .md).")
        return

    print(f"Total chunks: {len(chunks)}")

    # 3. Contextualize each chunk (parallelized)
    print("Contextualizing chunks via OpenRouter...")
    chunks = contextualize_chunks(chunks, parallel_threads=parallel_threads)

    # 4. Embed all contextualized chunks
    print("Generating embeddings...")
    full_texts = [c["full_text"] for c in chunks]
    embeddings = embed_texts(full_texts)

    # 5. Insert into database
    print("Inserting into PostgreSQL...")
    batch = [
        {
            "content": c["content"],
            "context": c["context"],
            "full_text": c["full_text"],
            "source": c["metadata"]["source"],
            "chunk_index": c["metadata"]["chunk_index"],
            "embedding": embeddings[i],
        }
        for i, c in enumerate(chunks)
    ]
    db.insert_chunks_batch(batch)

    print(f"Ingestion complete. {len(chunks)} chunks stored.")


def query(query_text: str, k: int = 5, rerank: bool = False) -> str:
    """Full query pipeline: search -> (optional rerank) -> generate."""
    # 1. Hybrid search — over-retrieve if reranking
    fetch_k = k * 10 if rerank else k
    print(f"Searching for: {query_text}")
    results = search_hybrid(query_text, k=fetch_k)
    print(f"Found {len(results)} relevant chunks")

    # 2. Optional reranking
    if rerank:
        from src.reranker import rerank as rerank_fn
        print("Reranking...")
        results = rerank_fn(query_text, results, k=k)
        print(f"Reranked to top {len(results)}")

    # 3. Generate response
    print("Generating response...")
    answer = generate_response(query_text, results)

    # 4. Print sources
    sources = set()
    for r in results:
        sources.add(f"{r['source_document']} (chunk {r['chunk_index']})")
    print(f"Sources: {', '.join(sorted(sources))}")

    return answer
