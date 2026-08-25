"""Thin orchestrator for Phase 1, plus the Phase 2 FastAPI app.

For the indexing step, use scripts/build_index.py.
For Phase 1 CLI usage, use scripts/generate_questions.py.
For the Phase 2 API server, run: uvicorn app.main:app --reload
"""

from app.api import create_app
from app.question_engine.generator import DueDiligenceQuestion, generate_top_questions

app = create_app()


def generate_due_diligence_questions(startup_info: str) -> list[DueDiligenceQuestion]:
    """Public entry point: startup info in, top 10 ranked questions out."""
    return generate_top_questions(startup_info)


if __name__ == "__main__":
    example = "We're a B2B SaaS startup, 15 employees, ₹2 crore annual revenue"
    for i, q in enumerate(generate_due_diligence_questions(example), start=1):
        print(f"{i}. {q.question}")
