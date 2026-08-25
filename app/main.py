"""Thin orchestrator for Phase 1: given startup info, return the top 10
ranked due-diligence questions.

For the indexing step, use scripts/build_index.py.
For CLI usage, use scripts/generate_questions.py.
"""

from app.question_engine.generator import DueDiligenceQuestion, generate_top_questions


def generate_due_diligence_questions(startup_info: str) -> list[DueDiligenceQuestion]:
    """Public entry point: startup info in, top 10 ranked questions out."""
    return generate_top_questions(startup_info)


if __name__ == "__main__":
    example = "We're a B2B SaaS startup, 15 employees, ₹2 crore annual revenue"
    for i, q in enumerate(generate_due_diligence_questions(example), start=1):
        print(f"{i}. {q.question}")
