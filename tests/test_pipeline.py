"""Simple smoke test for the Phase 1 pipeline: ingestion, chunking, retrieval,
and top-10 question generation.

Run with: python tests/test_pipeline.py
(Requires scripts/build_index.py to have been run first so ChromaDB is populated.)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import NUM_QUESTIONS
from app.question_engine.generator import generate_top_questions
from app.rag.chunking import chunk_article
from app.rag.ingest import load_article
from app.rag.retriever import retrieve_context

STARTUP_INFO = "We're a B2B SaaS startup, 15 employees, ₹2 crore annual revenue"


def test_ingestion() -> None:
    print("[1/4] Testing ingestion...")
    article = load_article()
    assert article.title, "Article title should not be empty"
    assert len(article.sections) > 0, "Article should have at least one section"
    print(f"      OK - title='{article.title}', sections={len(article.sections)}")


def test_chunking() -> None:
    print("[2/4] Testing chunking...")
    article = load_article()
    documents = chunk_article(article)
    assert len(documents) > 0, "Should produce at least one chunk"
    for doc in documents[:3]:
        assert doc.metadata.get("source_url"), "Chunk missing source_url metadata"
        assert doc.metadata.get("title"), "Chunk missing title metadata"
        assert doc.metadata.get("section"), "Chunk missing section metadata"
    print(f"      OK - {len(documents)} chunks, metadata present")


def test_retrieval() -> None:
    print("[3/4] Testing retrieval...")
    documents = retrieve_context(STARTUP_INFO, k=10)
    assert len(documents) > 0, "Retrieval should return at least one document"
    print(f"      OK - retrieved {len(documents)} chunks")
    for doc in documents:
        print(f"        - [{doc.metadata.get('section')}]")


def test_question_generation() -> None:
    print("[4/4] Testing top-10 question generation...")
    questions = generate_top_questions(STARTUP_INFO)

    assert len(questions) == NUM_QUESTIONS, (
        f"Expected exactly {NUM_QUESTIONS} questions, got {len(questions)}"
    )

    seen = set()
    for q in questions:
        assert q.question.strip(), "Question text should not be empty"
        assert q.category.strip(), "Category should not be empty"
        assert q.priority in ("High", "Medium", "Low"), f"Unexpected priority: {q.priority}"
        assert q.reason.strip(), "Reason should not be empty"
        assert q.source_context.strip(), "Source context should not be empty"

        normalized = " ".join(q.question.lower().split())
        assert normalized not in seen, f"Duplicate question detected: {q.question}"
        seen.add(normalized)

    print(f"      OK - {len(questions)} unique, fully-populated questions generated")
    for i, q in enumerate(questions, start=1):
        print(f"        {i}. [{q.priority}/{q.category}] {q.question}")


def main() -> None:
    test_ingestion()
    test_chunking()
    test_retrieval()
    test_question_generation()
    print("\nAll Phase 1 smoke tests passed.")


if __name__ == "__main__":
    main()
