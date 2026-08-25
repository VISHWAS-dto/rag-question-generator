"""Fetch the Startup Science article, chunk it, embed it, and store it in ChromaDB.

Run this once (or whenever the source article changes) before generating questions.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.chunking import chunk_article
from app.rag.ingest import ExtractionError, FetchError, load_article
from app.rag.vectorstore import build_vectorstore


def main() -> None:
    print("Fetching and extracting article...")
    try:
        article = load_article()
    except (FetchError, ExtractionError) as exc:
        print(f"ERROR: {exc}")
        print(
            "Stopping: the webpage could not be fetched or parsed correctly. "
            "Not falling back to another source."
        )
        sys.exit(1)

    print(f"  Title: {article.title}")
    print(f"  Sections extracted: {len(article.sections)}")
    for section in article.sections:
        print(f"    - [{section.level}] {section.heading}")

    print("\nChunking article...")
    documents = chunk_article(article)
    print(f"  Chunks created: {len(documents)}")

    print("\nEmbedding and storing in ChromaDB (this may take a moment on first run)...")
    build_vectorstore(documents)
    print("  Done. Vector store persisted to ./data/chroma")


if __name__ == "__main__":
    main()
