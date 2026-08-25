"""Generate the top 10 highest-value due-diligence questions for a piece of
startup information.

Usage:
    python scripts/generate_questions.py "We're a B2B SaaS startup, 15 employees, ₹2 crore annual revenue"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.question_engine.generator import generate_top_questions

DEFAULT_STARTUP_INFO = "We have 10,000 customers and ₹2 crore annual revenue."


def print_questions(questions) -> None:
    print("\nTOP 10 QUESTIONS\n")
    for i, q in enumerate(questions, start=1):
        print(f"{i}. {q.question}")


def main() -> None:
    startup_info = " ".join(sys.argv[1:]).strip() or DEFAULT_STARTUP_INFO

    print(f"Startup information:\n  {startup_info}")
    print("\nRetrieving due-diligence context and generating questions...")

    questions = generate_top_questions(startup_info)
    print_questions(questions)


if __name__ == "__main__":
    main()
