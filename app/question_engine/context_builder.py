"""Builds the exact context sent to the LLM when deciding whether a follow-up
question is needed.

Keeps the prompt minimal: only what's needed to judge *this* answer, not the
entire session transcript verbatim beyond what's relevant.
"""

from dataclasses import dataclass

from app.models import AssessmentSession, Question
from app.rag.retriever import format_context, retrieve_context


@dataclass
class PriorTurn:
    question: str
    answer: str


@dataclass
class FollowupContext:
    startup_info: str
    startup_stage: str | None
    rag_context: str
    top_level_question: str
    current_question: str
    current_answer: str
    previous_turns: list[PriorTurn]
    existing_followup_questions: list[str]


def build_followup_context(
    session: AssessmentSession,
    current_question: Question,
    current_answer_text: str,
) -> FollowupContext:
    """Assemble everything the LLM needs to judge one answer, and nothing more.

    RAG context is retrieved fresh, scoped to the current question + answer
    (not the whole session), so retrieval stays relevant to what's actually
    being evaluated right now.
    """
    retrieval_query = f"{current_question.question}\n{current_answer_text}"
    documents = retrieve_context(retrieval_query)
    rag_context = format_context(documents)

    top_level = _top_level_question(session, current_question)

    previous_turns = _answered_turns_before(session, current_question)

    existing_followups = [
        q.question for q in top_level.follow_ups if q.question_id != current_question.question_id
    ]

    return FollowupContext(
        startup_info=session.startup_info,
        startup_stage=session.startup_stage,
        rag_context=rag_context,
        top_level_question=top_level.question,
        current_question=current_question.question,
        current_answer=current_answer_text,
        previous_turns=previous_turns,
        existing_followup_questions=existing_followups,
    )


def _top_level_question(session: AssessmentSession, question: Question) -> Question:
    if not question.is_followup:
        return question
    for q in session.questions:
        if q.question_id == question.parent_question_id:
            return q
    raise ValueError(f"Parent question {question.parent_question_id} not found in session")


def _answered_turns_before(session: AssessmentSession, question: Question) -> list[PriorTurn]:
    """All previously answered questions in this session, in creation order,
    excluding the current one.
    """
    turns: list[PriorTurn] = []
    for q in session.questions:
        if q.question_id == question.question_id:
            continue
        if q.status == "ANSWERED" and q.answer is not None:
            turns.append(PriorTurn(question=q.question, answer=q.answer.answer))
    return turns


def render_followup_prompt_context(ctx: FollowupContext) -> str:
    """Render the FollowupContext into the text block sent to the LLM."""
    lines: list[str] = []

    lines.append("STARTUP INFORMATION:")
    lines.append(ctx.startup_info)
    lines.append("")

    lines.append(f"STARTUP STAGE: {ctx.startup_stage or 'Not specified'}")
    lines.append("")

    lines.append("RELEVANT DUE-DILIGENCE REFERENCE MATERIAL:")
    lines.append(ctx.rag_context or "(none retrieved)")
    lines.append("")

    if ctx.previous_turns:
        lines.append("PREVIOUS QUESTIONS AND ANSWERS IN THIS SESSION:")
        for i, turn in enumerate(ctx.previous_turns, start=1):
            lines.append(f"{i}. Q: {turn.question}")
            lines.append(f"   A: {turn.answer}")
        lines.append("")

    if ctx.existing_followup_questions:
        lines.append("FOLLOW-UP QUESTIONS ALREADY ASKED FOR THE CURRENT TOP-10 QUESTION:")
        for i, q in enumerate(ctx.existing_followup_questions, start=1):
            lines.append(f"{i}. {q}")
        lines.append("")

    lines.append("ORIGINAL TOP-10 QUESTION THIS TURN BELONGS TO:")
    lines.append(ctx.top_level_question)
    lines.append("")

    lines.append("CURRENT QUESTION:")
    lines.append(ctx.current_question)
    lines.append("")

    lines.append("CURRENT ANSWER:")
    lines.append(ctx.current_answer)

    return "\n".join(lines)
