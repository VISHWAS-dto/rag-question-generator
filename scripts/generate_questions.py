"""One-shot CLI: startup info in, top-N due-diligence questions out.

Uses the same engines the API does, wired with clients built from the current
environment (.env / env vars). With APP_LLM_PROVIDER=echo and a running or
in-process RAG index this works fully offline.

Usage:
    python scripts/generate_questions.py "We're a B2B SaaS startup, 15 staff, ..."
"""

from __future__ import annotations

import sys

from app.clients.llm import build_llm_client
from app.clients.rag import build_rag_client
from app.engines.question_generator import QuestionGenerationError, generate_top_questions
from shared.config import get_app_settings
from shared.logging import configure_logging

DEFAULT = "We have 10,000 customers and about 2 crore INR annual revenue."


def main() -> int:
    configure_logging(json_output=False, service="cli")
    settings = get_app_settings()
    startup_info = " ".join(sys.argv[1:]).strip() or DEFAULT

    llm = build_llm_client(settings)
    rag = build_rag_client(settings)

    print(f"Startup information:\n  {startup_info}\n")
    print("Retrieving context and generating questions...\n")
    try:
        questions = generate_top_questions(
            llm=llm,
            rag=rag,
            startup_info=startup_info,
            startup_stage=None,
            num_questions=settings.num_questions,
            top_k=settings.rag_retrieval_top_k,
            collection=settings.rag_collection,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            max_repair_attempts=settings.llm_max_repair_attempts,
        )
    except QuestionGenerationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"TOP {len(questions)} QUESTIONS\n")
    for i, q in enumerate(questions, start=1):
        print(f"{i}. [{q.priority}/{q.category}] {q.question}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
