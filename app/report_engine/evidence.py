"""Retrieves RAG evidence from the existing ChromaDB knowledge base for a
completed interview, so the Phase 3 analysis is grounded in the same
due-diligence reference material Phase 1/2 already use.

Reuses app/rag/retriever.py unchanged — no new retrieval logic.
"""

from dataclasses import dataclass

from app.config import RETRIEVAL_TOP_K
from app.models import AssessmentSession
from app.rag.retriever import format_context, retrieve_context


@dataclass
class InterviewTurn:
    question: str
    category: str | None
    is_followup: bool
    answer: str


def collect_interview_turns(session: AssessmentSession) -> list[InterviewTurn]:
    """All answered questions in the session, in creation order (Top-10
    interleaved with their follow-ups, in the order they were actually
    asked).
    """
    turns: list[InterviewTurn] = []
    for q in session.questions:
        if q.answer is None:
            continue
        turns.append(
            InterviewTurn(
                question=q.question,
                category=q.category,
                is_followup=q.is_followup,
                answer=q.answer.answer,
            )
        )
    return turns


def retrieve_interview_evidence(session: AssessmentSession, k: int = RETRIEVAL_TOP_K) -> str:
    """Retrieve due-diligence reference material relevant to the whole
    interview, formatted as one context block for the analysis prompt.

    The retrieval query combines the startup's self-description with every
    answer given, so the retrieved chunks cover the actual topics the
    founder raised (financials, team, market, etc.) rather than just the
    startup's initial pitch.
    """
    turns = collect_interview_turns(session)
    query_parts = [session.startup_info] + [t.answer for t in turns]
    query = "\n".join(query_parts)

    documents = retrieve_context(query, k=k)
    return format_context(documents)
