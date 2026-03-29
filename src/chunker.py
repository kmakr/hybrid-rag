import os
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.log import get_logger

log = get_logger("chunker")

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".doc", ".docx"}

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
)


def _extract_text(file_path: Path) -> str:
    ext = file_path.suffix.lower()
    if ext in {".txt", ".md"}:
        return file_path.read_text(encoding="utf-8")
    elif ext == ".pdf":
        import pdfplumber
        with pdfplumber.open(file_path) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    elif ext in {".doc", ".docx"}:
        import docx2txt
        return docx2txt.process(str(file_path))
    return ""


def load_documents(data_dir: str) -> list[dict]:
    """Load all supported files from a directory.

    Returns list of {"content": str, "source": str}.
    """
    documents = []
    data_path = Path(data_dir)
    files = [f for f in sorted(data_path.iterdir()) if f.suffix.lower() in SUPPORTED_EXTENSIONS]
    log.info("Found %d supported file(s) in %s", len(files), data_dir)
    for file_path in files:
        log.info("Loading %s...", file_path.name)
        text = _extract_text(file_path)
        documents.append({
            "content": text,
            "source": file_path.name,
        })
    return documents


def chunk_document(document: dict) -> list[dict]:
    """Split a single document into chunks.

    Input: {"content": str, "source": str}
    Output: list of {"content": str, "metadata": {"source", "chunk_index", "parent_document"}}
    """
    texts = splitter.split_text(document["content"])
    return [
        {
            "content": text,
            "metadata": {
                "source": document["source"],
                "chunk_index": i,
                "parent_document": document["content"],
            },
        }
        for i, text in enumerate(texts)
    ]


def chunk_documents(data_dir: str) -> list[dict]:
    """Load and chunk all documents from a directory."""
    documents = load_documents(data_dir)
    all_chunks = []
    for doc in documents:
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)
        log.info("%s → %d chunks", doc["source"], len(chunks))
    log.info("Total chunks: %d", len(all_chunks))
    return all_chunks
