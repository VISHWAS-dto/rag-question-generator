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

from app.config import MAX_FOLLOWUPS_PER_QUESTION
from app.models import Base, QuestionStatus
from app.question_engine.followup import decide_followup
from app.session_manager import ValidationError, get_current_question, submit_answer

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


class RepairingStubLLM:
    """Like StubLLM, but the first `.invoke()` returns deliberately malformed
    output — exercising the generate -> parse -> repair LangGraph. Subsequent
    invocations return the canned decision, so a caller that repairs correctly
    still gets a valid FollowupDecision instead of a RuntimeError.
    """

    def __init__(self, decision_fn, bad_responses=1):
        self._decision_fn = decision_fn
        self._bad_left = bad_responses
        self.calls = 0

    def invoke(self, prompt_value) -> StubResponse:
        self.calls += 1
        if self._bad_left > 0:
            self._bad_left -= 1
            return StubResponse("sorry, here is the answer: {follow_up_required: true")
        prompt_text = prompt_value.to_string()
        return StubResponse(json.dumps(self._decision_fn(prompt_text)))


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
    print("[1/10] Testing answer storage...")
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
    print("[2/10] Testing follow-up generation...")
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
    print("[3/10] Testing no-follow-up -> next question flow...")
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
    print("[4/10] Testing duplicate follow-up prevention...")
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
    print("[5/10] Testing multi-turn conversation flow...")
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
    print("[6/10] Testing follow-up grounding in the actual answer...")
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
    print("[7/10] Testing contradiction detection...")
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


# --- Test 8: current-question sequencing (a pending follow-up on Q1 must be
# answered before Q2 is offered, even though all Top-10 questions were
# created before the follow-up) ---


def test_current_question_sequencing() -> None:
    print("[8/10] Testing current-question sequencing (follow-up before next Top-10)...")
    db = make_test_db()
    session = make_session_with_top10(db)
    q1 = session.questions[0]
    q2_id = session.questions[1].question_id

    # Q1 answered -> follow-up required.
    stub = StubLLM(
        lambda _ctx: followup_decision(
            "What measures are you taking to reduce your dependency on your top five customers?"
        )
    )
    result = submit_answer(db, q1.question_id, "70%", llm=stub)
    followup = result.next_question
    assert followup is not None and followup.is_followup

    # The current question must be the pending follow-up, NOT Q2 — even
    # though Q2's created_at is earlier than the follow-up's, which is
    # exactly the case "oldest pending" gets wrong. Expire the session's
    # identity-mapped objects first so this read sees the follow-up
    # submit_answer just committed, matching a real request: each FastAPI
    # request gets its own fresh `db` session via Depends(get_db), so this
    # staleness only arises here because the test reuses one `db` across
    # multiple calls in-process.
    db.expire_all()
    current = get_current_question(db, session.session_id)
    assert current is not None, "Expected the pending follow-up, got None"
    assert current.question_id == followup.question_id, (
        f"Expected the pending follow-up to be current, got '{current.question}' "
        f"(is_followup={current.is_followup}) instead — Q2 must not be offered "
        "before the follow-up on Q1 is answered"
    )
    assert current.question_id != q2_id
    print("      OK - pending follow-up correctly precedes Q2, despite Q2's earlier created_at")

    # Answer the follow-up with no further follow-up needed -> Q2 should now be current.
    stub2 = StubLLM(lambda _ctx: no_followup_decision())
    result2 = submit_answer(db, followup.question_id, "Diversifying our customer base.", llm=stub2)
    assert result2.next_question is not None
    assert result2.next_question.question_id == q2_id, "Should now advance to Q2"
    print("      OK - advanced to Q2 only after the follow-up was resolved")


# --- Test 9: follow-up depth cap ---
#
# Discovered by running the live system end-to-end: a chain of vague/evasive
# answers kept opening new, individually-reasonable follow-up threads with
# no sign of converging (11+ follow-ups deep on a single Top-10 question in
# that live run). Without a cap, a session could never reach COMPLETED.


def test_followup_depth_cap() -> None:
    print(f"[9/10] Testing follow-up depth cap (max {MAX_FOLLOWUPS_PER_QUESTION})...")
    db = make_test_db()
    session = make_session_with_top10(db)
    q1 = session.questions[0]
    q2_id = session.questions[1].question_id

    # An LLM that ALWAYS wants another follow-up, with a distinct question
    # each time so the duplicate-suppression backstop never kicks in and
    # only the depth cap can stop the chain.
    call_count = {"n": 0}

    def always_followup(_ctx):
        call_count["n"] += 1
        return followup_decision(f"Follow-up question number {call_count['n']}?")

    stub = StubLLM(always_followup)

    result = submit_answer(db, q1.question_id, "70%", llm=stub)
    current_id = q1.question_id
    followups_seen = 0
    while result.decision.follow_up_required:
        followups_seen += 1
        assert followups_seen <= MAX_FOLLOWUPS_PER_QUESTION, (
            f"Follow-up chain exceeded the configured cap of {MAX_FOLLOWUPS_PER_QUESTION} "
            f"(seen {followups_seen} so far) — the depth cap did not stop it"
        )
        current_id = result.next_question.question_id
        result = submit_answer(db, current_id, "Still no concrete data.", llm=stub)

    assert followups_seen == MAX_FOLLOWUPS_PER_QUESTION, (
        f"Expected exactly {MAX_FOLLOWUPS_PER_QUESTION} follow-ups before the cap forced a "
        f"move on, got {followups_seen}"
    )
    assert result.next_question is not None
    assert result.next_question.question_id == q2_id, (
        "After hitting the cap, the session must advance to the next Top-10 question"
    )
    assert "maximum" in result.decision.reason.lower()
    print(
        f"      OK - chain capped at exactly {MAX_FOLLOWUPS_PER_QUESTION} follow-ups, "
        "then advanced to Q2"
    )


# --- Test 10: concurrent duplicate-answer race is handled gracefully ---
#
# Discovered live: two submissions landing close together on the same
# question can both read status=PENDING before either commits, so the
# in-Python ANSWERED check doesn't catch it — the answers.question_id
# UNIQUE constraint is the real guard. Before the fix, that constraint
# violation surfaced as a raw, unhandled IntegrityError (500) instead of the
# same ValidationError (422) a non-concurrent duplicate submission raises.


def test_concurrent_duplicate_answer_race() -> None:
    print("[10/10] Testing concurrent duplicate-answer submissions raise ValidationError...")
    db = make_test_db()
    session = make_session_with_top10(db)
    q1 = session.questions[0]

    stub = StubLLM(lambda _ctx: no_followup_decision())
    submit_answer(db, q1.question_id, "70%", llm=stub)

    # Simulate the race directly: bypass the in-Python status check that a
    # second concurrent request's read would also have passed, by inserting
    # a duplicate Answer row the same way submit_answer does, and confirm
    # the DB-level guard converts the resulting IntegrityError into the same
    # ValidationError a normal duplicate-answer call raises.
    from app.models import Answer
    from sqlalchemy.exc import IntegrityError

    db.add(Answer(question_id=q1.question_id, session_id=session.session_id, answer="duplicate"))
    try:
        db.flush()
        raise AssertionError("Expected IntegrityError from the UNIQUE constraint")
    except IntegrityError:
        db.rollback()

    # And the actual code path: submit_answer on an already-ANSWERED question
    # must raise ValidationError (422), not leak a raw IntegrityError.
    try:
        submit_answer(db, q1.question_id, "duplicate via submit_answer", llm=stub)
        raise AssertionError("Expected ValidationError")
    except ValidationError:
        print("      OK - duplicate submission raises ValidationError, not a raw IntegrityError")


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


def test_followup_repairs_malformed_llm_output() -> None:
    print("[repair] Testing follow-up recovers from malformed LLM JSON...")
    db = make_test_db()
    session = make_session_with_top10(db)
    q1 = session.questions[0]

    stub = RepairingStubLLM(
        lambda _ctx: followup_decision(
            "What measures are you taking to reduce your dependency on your top five customers?"
        )
    )
    result = submit_answer(db, q1.question_id, "70%", llm=stub)

    assert stub.calls == 2, "Graph should re-invoke once to repair the bad first response"
    assert result.decision.follow_up_required is True
    assert result.next_question is not None
    assert "dependency" in result.next_question.question.lower()
    print("      OK - malformed first response repaired, valid follow-up produced")


def main() -> None:
    test_answer_storage()
    test_followup_generated()
    test_followup_repairs_malformed_llm_output()
    test_no_followup_moves_to_next()
    test_duplicate_prevention()
    test_multi_turn_conversation()
    test_grounding()
    test_contradiction_detection()
    test_current_question_sequencing()
    test_followup_depth_cap()
    test_concurrent_duplicate_answer_race()

    try:
        test_live_integration()
    except Exception as exc:  # noqa: BLE001
        print(f"[live] SKIPPED/FAILED (requires NVIDIA_API_KEY + built index): {exc}")

    print("\nAll Phase 2 tests passed.")


if __name__ == "__main__":
    main()
