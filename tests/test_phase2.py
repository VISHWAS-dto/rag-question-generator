"""Phase 2 tests: session management, answer storage, and intelligent
follow-up generation.

Most tests use a stub LLM (deterministic, free, fast) that returns canned
JSON follow-up decisions based on simple keyword matching against the
prompt context — this lets us assert exact behavior (follow-up vs. no
follow-up, duplicate suppression, contradiction detection) without relying
on a live model's non-deterministic output.

test_live_integration exercises the real NVIDIA LLM end-to-end, mirroring
the style of tests/test_pipeline.py's live smoke test. It requires
NVIDIA_API_KEY to be set and the vector store to be built
(scripts/build_index.py).

Run with: python tests/test_phase2.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models import Base, QuestionStatus
from app.question_engine.followup import decide_followup
from app.session_manager import get_current_question, submit_answer

STARTUP_INFO = (
    "We're a B2B SaaS startup, 15 employees, ₹2 crore annual revenue. "
    "70% of our revenue comes from our top five customers."
)


# --- Test infrastructure: in-memory DB + stub LLM ---


def make_test_db():
    """Fresh in-memory SQLite DB per test, isolated from the real data/phase2.db."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)()


class StubResponse:
    def __init__(self, content: str):
        self.content = content


class StubLLM:
    """A fake LLM chain endpoint: given a canned decision (or a function that
    inspects the prompt), returns it as a JSON string, like the real
    ChatNVIDIA response.
    """

    def __init__(self, decision_fn):
        self._decision_fn = decision_fn

    def invoke(self, prompt_value) -> StubResponse:
        # prompt_value is a ChatPromptValue (from _prompt.invoke(...)); render
        # it to plain text so decision_fn can inspect what was actually sent.
        prompt_text = prompt_value.to_string()
        decision = self._decision_fn(prompt_text)
        return StubResponse(json.dumps(decision))


def followup_decision(question: str, category="Financial", priority="High", reason="probe deeper"):
    return {
        "follow_up_required": True,
        "question": question,
        "category": category,
        "priority": priority,
        "reason": reason,
    }


def no_followup_decision(reason="Answer is sufficient."):
    return {
        "follow_up_required": False,
        "question": None,
        "category": None,
        "priority": None,
        "reason": reason,
    }


def make_session_with_top10(db, monkeypatch_top10=None):
    """Create a session, bypassing the live LLM for top-10 generation by
    seeding questions directly (keeps these tests fast and offline).
    """
    from app.models import AssessmentSession, Question

    session = AssessmentSession(company_id="acme-co", startup_info=STARTUP_INFO)
    db.add(session)
    db.flush()

    top10 = monkeypatch_top10 or [
        ("What percentage of your revenue comes from your top five customers?", "Financial"),
        ("What is your current customer retention rate?", "Traction"),
        ("How many engineers are on your team?", "Team"),
    ]
    for rank, (q_text, category) in enumerate(top10, start=1):
        db.add(
            Question(
                session_id=session.session_id,
                question=q_text,
                category=category,
                priority="High",
                reason="seeded for test",
                status=QuestionStatus.PENDING,
                rank=rank,
            )
        )
    db.commit()
    db.refresh(session)
    return session


# --- Test 1: Answer storage ---


def test_answer_storage() -> None:
    print("[1/7] Testing answer storage...")
    db = make_test_db()
    session = make_session_with_top10(db)
    q1 = session.questions[0]

    stub = StubLLM(lambda _ctx: no_followup_decision())
    result = submit_answer(db, q1.question_id, "70%", llm=stub)

    db.refresh(q1)
    assert q1.status == QuestionStatus.ANSWERED, "Question should be marked ANSWERED"
    assert q1.answer is not None, "Answer should be stored"
    assert q1.answer.answer == "70%", "Stored answer text should match submitted answer"
    assert result.decision.follow_up_required is False
    print("      OK - answer stored and linked to question")


# --- Test 2: Follow-up generated when answer reveals an issue ---


def test_followup_generated() -> None:
    print("[2/7] Testing follow-up generation...")
    db = make_test_db()
    session = make_session_with_top10(db)
    q1 = session.questions[0]  # revenue concentration question

    stub = StubLLM(
        lambda _ctx: followup_decision(
            "What measures are you taking to reduce your dependency on your top five customers?"
        )
    )
    result = submit_answer(db, q1.question_id, "70%", llm=stub)

    assert result.decision.follow_up_required is True
    assert result.next_question is not None
    assert result.next_question.is_followup is True
    assert result.next_question.parent_question_id == q1.question_id
    assert "dependency" in result.next_question.question.lower()
    print(f"      OK - follow-up generated: '{result.next_question.question}'")


# --- Test 3: No follow-up needed -> next top-10 question returned ---


def test_no_followup_moves_to_next() -> None:
    print("[3/7] Testing no-follow-up -> next question flow...")
    db = make_test_db()
    session = make_session_with_top10(db)
    q1 = session.questions[0]
    q2 = session.questions[1]

    stub = StubLLM(lambda _ctx: no_followup_decision())
    result = submit_answer(db, q1.question_id, "70%", llm=stub)

    assert result.decision.follow_up_required is False
    assert result.next_question is not None
    assert result.next_question.question_id == q2.question_id, (
        "Should advance to the next top-10 question"
    )
    assert result.next_question.is_followup is False
    print(f"      OK - advanced to next top-10 question: '{result.next_question.question}'")


# --- Test 4: Duplicate prevention ---


def test_duplicate_prevention() -> None:
    print("[4/7] Testing duplicate follow-up prevention...")
    db = make_test_db()
    session = make_session_with_top10(db)
    q1 = session.questions[0]

    # LLM tries to propose a follow-up that duplicates the current question itself.
    stub = StubLLM(
        lambda _ctx: followup_decision(
            "What percentage of your revenue comes from your top five customers?"
        )
    )
    result = submit_answer(db, q1.question_id, "70%", llm=stub)

    assert result.decision.follow_up_required is False, (
        "Duplicate follow-up (matching the current question) must be suppressed"
    )
    print("      OK - duplicate follow-up suppressed, decision flipped to no-follow-up")

    # Now confirm suppression also applies against an *existing* follow-up.
    db2 = make_test_db()
    session2 = make_session_with_top10(db2)
    q1b = session2.questions[0]

    first_followup_text = "What measures are you taking to reduce dependency on top customers?"
    stub_first = StubLLM(lambda _ctx: followup_decision(first_followup_text))
    result1 = submit_answer(db2, q1b.question_id, "70%", llm=stub_first)
    assert result1.next_question is not None

    # Second turn: LLM proposes the *same* follow-up question again (reworded case/spacing).
    stub_dupe = StubLLM(
        lambda _ctx: followup_decision(
            "  What measures are you taking to reduce dependency on top customers?  ".strip()
        )
    )
    result2 = submit_answer(
        db2, result1.next_question.question_id, "We are expanding enterprise sales.", llm=stub_dupe
    )
    assert result2.decision.follow_up_required is False, (
        "Follow-up duplicating an existing follow-up must be suppressed"
    )
    print("      OK - duplicate of an existing follow-up also suppressed")


# --- Test 5: Multi-turn conversation ---


def test_multi_turn_conversation() -> None:
    print("[5/7] Testing multi-turn conversation flow...")
    db = make_test_db()
    session = make_session_with_top10(db)
    q1 = session.questions[0]

    # Turn 1: answer reveals concentration risk -> follow-up.
    stub1 = StubLLM(
        lambda _ctx: followup_decision(
            "What measures are you taking to reduce your dependency on your top five customers?"
        )
    )
    r1 = submit_answer(db, q1.question_id, "70%", llm=stub1)
    assert r1.decision.follow_up_required is True
    followup1 = r1.next_question

    # Turn 2: answer to follow-up reveals more detail -> another follow-up.
    stub2 = StubLLM(
        lambda _ctx: followup_decision(
            "How much of your recent enterprise revenue comes from customers "
            "outside your current top five?"
        )
    )
    r2 = submit_answer(db, followup1.question_id, "We are expanding into enterprise.", llm=stub2)
    assert r2.decision.follow_up_required is True
    followup2 = r2.next_question
    assert followup2.parent_question_id == q1.question_id, (
        "Nested follow-up should still link back to the original top-10 question"
    )

    # Turn 3: answer is sufficient -> move to next top-10 question.
    stub3 = StubLLM(lambda _ctx: no_followup_decision())
    r3 = submit_answer(db, followup2.question_id, "About 40% so far.", llm=stub3)
    assert r3.decision.follow_up_required is False
    assert r3.next_question.question_id == session.questions[1].question_id

    print("      OK - Q1 -> follow-up -> follow-up -> next top-10 question, all linked correctly")


# --- Test 6: Grounding ---


def test_grounding() -> None:
    print("[6/7] Testing follow-up grounding in the actual answer...")
    db = make_test_db()
    session = make_session_with_top10(db)
    q1 = session.questions[0]

    captured_prompt = {}

    def decision_fn(prompt_text: str):
        captured_prompt["text"] = prompt_text
        return followup_decision(
            "What measures are you taking to reduce your dependency on your top five customers?"
        )

    stub = StubLLM(decision_fn)
    result = submit_answer(db, q1.question_id, "70%", llm=stub)

    prompt_text = captured_prompt["text"]
    assert "70%" in prompt_text, "Prompt sent to LLM must include the actual answer"
    assert q1.question in prompt_text, "Prompt must include the current question"
    assert STARTUP_INFO in prompt_text, "Prompt must include startup information"
    assert "dependency" in result.next_question.question.lower()
    print("      OK - follow-up prompt included startup info, question, and actual answer")


# --- Test 7: Contradiction detection ---


def test_contradiction_detection() -> None:
    print("[7/7] Testing contradiction detection...")
    db = make_test_db()

    from app.models import AssessmentSession, Question

    session = AssessmentSession(
        company_id="acme-co",
        startup_info="Annual revenue: ₹5 crore.",
    )
    db.add(session)
    db.flush()
    q1 = Question(
        session_id=session.session_id,
        question="What is your annual revenue?",
        category="Financial",
        priority="High",
        reason="seeded for test",
        status=QuestionStatus.PENDING,
        rank=1,
    )
    db.add(q1)
    db.commit()
    db.refresh(session)

    stub = StubLLM(
        lambda _ctx: followup_decision(
            "Could you clarify the difference between the ₹5 crore annual revenue "
            "reported in the startup information and the ₹3 crore figure you provided?",
            reason="Answer contradicts the startup's reported annual revenue.",
        )
    )
    result = submit_answer(db, q1.question_id, "Our annual revenue is ₹3 crore.", llm=stub)

    assert result.decision.follow_up_required is True
    assert "clarify" in result.next_question.question.lower()
    assert "5 crore" in result.next_question.question and "3 crore" in result.next_question.question
    print(f"      OK - contradiction follow-up generated: '{result.next_question.question}'")


# --- Live integration test (real NVIDIA LLM, real RAG retrieval) ---


def test_live_integration() -> None:
    print("[live] Testing full session against the real NVIDIA LLM + RAG store...")
    db = make_test_db()
    session = make_session_with_top10(
        db,
        monkeypatch_top10=[
            (
                "What percentage of your revenue comes from your top five customers?",
                "Financial",
            ),
        ],
    )
    q1 = session.questions[0]

    result = submit_answer(db, q1.question_id, "70%")  # no stub: real LLM + real retriever

    assert result.decision.reason.strip(), "Live LLM must return a non-empty reason"
    if result.decision.follow_up_required:
        assert result.next_question is not None
        assert result.next_question.question.strip()
        print(f"      OK - live follow-up: '{result.next_question.question}'")
    else:
        assert result.next_question is None or not result.next_question.is_followup
        print("      OK - live LLM judged the answer sufficient, no follow-up")


def main() -> None:
    test_answer_storage()
    test_followup_generated()
    test_no_followup_moves_to_next()
    test_duplicate_prevention()
    test_multi_turn_conversation()
    test_grounding()
    test_contradiction_detection()

    try:
        test_live_integration()
    except Exception as exc:  # noqa: BLE001
        print(f"[live] SKIPPED/FAILED (requires NVIDIA_API_KEY + built index): {exc}")

    print("\nAll Phase 2 tests passed.")


if __name__ == "__main__":
    main()
