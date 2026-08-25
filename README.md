# RAG Question Generator

Given a startup's self-reported information, this app retrieves relevant
due-diligence context from a knowledge base built from
[Startup Science's due diligence checklist](https://www.startupscience.io/articles/startup-due-diligence-checklist),
generates the **top 10** ranked, probing due-diligence questions, and then
runs an **interactive questioning session**: the user answers one question
at a time, and the system decides — grounded in the RAG context and the
conversation so far — whether a follow-up question is needed before moving
on. Once the interview is complete, Phase 3 analyzes the full transcript and
produces a structured, evidence-based **investor due-diligence report**.

```
Phase 1: Top 10 questions
    -> Phase 2: interactive interview (follow-ups, contradiction checks)
        -> Phase 3: due-diligence report (scores, risks, gaps, recommendations)
```

## Phase 1 — Top 10 question generation

A local, single-pass RAG pipeline: no API server, database, or conversation
history.

```
Startup info
    -> Fetch Startup Science webpage      (app/rag/ingest.py)
    -> Extract article content            (app/rag/ingest.py)
    -> Clean + chunk text                 (app/rag/chunking.py)
    -> Embed (local HuggingFace model)    (app/rag/vectorstore.py)
    -> Store in ChromaDB                  (app/rag/vectorstore.py)
    -> Retrieve relevant context          (app/rag/retriever.py)
    -> LLM (NVIDIA NIM)                   (app/question_engine/generator.py)
    -> TOP 10 ranked due-diligence questions
```

Each question includes: `question`, `category`, `priority` (High/Medium/Low),
`reason`, and `source_context` (the due-diligence section that grounded it).
Questions are deduplicated and ranked highest-priority first.

## Phase 2 — Interactive questioning + intelligent follow-ups

Turns the Top 10 questions into a live, one-at-a-time questioning session,
backed by a FastAPI server and a local SQLite database.

```
Top 10 Questions -> Question 1 -> User Answer
    -> Startup info + RAG context + current Q&A + session history
    -> LLM decides: is a follow-up needed?
        YES -> generate ONE grounded follow-up question, ask it next
        NO  -> move to the next unanswered Top 10 question
```

Follow-ups are linked to the Top-10 question they came from, are always
grounded in the founder's actual answer (never generic), and the system
guards against asking the same question twice.

New modules:

- `app/models.py` — `AssessmentSession`, `Question` (top-10 + follow-ups,
  via `parent_question_id`), `Answer` (SQLAlchemy ORM)
- `app/db.py` — SQLite engine/session setup (`./data/phase2.db`)
- `app/question_engine/context_builder.py` — assembles exactly the context
  (startup info, stage, RAG context, current/previous Q&A, existing
  follow-ups) sent to the LLM for a follow-up decision
- `app/question_engine/followup.py` — asks the LLM whether a follow-up is
  needed, using the same raw-JSON prompting approach as Phase 1's generator
- `app/session_manager.py` — orchestrates session creation, answer
  submission, and the follow-up / next-question flow
- `app/api.py` — the FastAPI endpoints below

Phase 1's ingestion, chunking, embeddings, ChromaDB, retriever, LLM
integration, and top-10 generator are reused unchanged.

### API endpoints

```
POST   /sessions                         create a session, seed its top-10 questions
GET    /sessions/{session_id}            session status
GET    /sessions/{session_id}/questions  all questions so far + current_question
POST   /questions/{question_id}/answer   submit an answer -> follow-up or next question
POST   /questions/{question_id}/follow-up  re-run follow-up analysis for an answered question
```

### Current-question sequencing

`current_question` (from `GET /sessions/{id}/questions`) always walks the
Top-10 questions in rank order and returns the oldest pending question in
the first not-yet-resolved thread (a Top-10 question plus its follow-ups).
A follow-up must always be answered before the next Top-10 question is
offered — this is **not** simply "the oldest pending question by
`created_at`": all 10 Top-10 questions are inserted in a single batch at
session creation, so their `created_at` values are all earlier than any
follow-up's. Picking by raw timestamp would incorrectly offer the next
Top-10 question ahead of a pending follow-up on an earlier one. See
`get_current_question` in `app/session_manager.py`.

### Running the Phase 2 API server

```bash
source .venv/bin/activate
uvicorn app.main:app --reload
```

Then, e.g.:

```bash
curl -X POST http://127.0.0.1:8000/sessions \
  -H "Content-Type: application/json" \
  -d '{"company_id":"acme-co","startup_info":"We are a B2B SaaS startup, 15 employees, ₹2 crore annual revenue.","startup_stage":"Seed"}'

curl http://127.0.0.1:8000/sessions/<session_id>/questions

curl -X POST http://127.0.0.1:8000/questions/<question_id>/answer \
  -H "Content-Type: application/json" \
  -d '{"answer":"70% of our revenue comes from our top five customers."}'
```

Interactive API docs: `http://127.0.0.1:8000/docs`

## Phase 3 — AI due-diligence analysis & investor report

Once a Phase 2 interview is complete (every Top-10 question answered, no
pending follow-ups), Phase 3 analyzes the full transcript and produces a
structured, evidence-grounded investor due-diligence report — answering:
what are the strengths, the major risks, the missing information, the
contradictions, what evidence backs each finding, how strong is each
business category, what needs more diligence, and what's the overall
assessment and risk level.

```
Completed Assessment Session
    -> Collect all Questions + Answers + Follow-ups   (app/report_engine/evidence.py)
    -> Retrieve relevant RAG evidence                 (app/report_engine/evidence.py)
    -> Analyze the full interview (LLM, structured)   (app/report_engine/analyzer.py)
    -> Apply deterministic risk/contradiction backstops (app/report_engine/risk_analyzer.py)
    -> Compute category scores + overall score + risk level (app/report_engine/scorer.py)
    -> Assemble + validate the final report           (app/report_engine/report_generator.py)
    -> Store report in SQLite                         (AssessmentReport)
    -> Expose report through FastAPI
```

New modules (`app/report_engine/`):

- `schemas.py` — Pydantic models for the report: `Strength`, `Risk`,
  `InformationGap`, `Contradiction`, `Recommendation`, `CategoryAssessment`,
  and the final `AssessmentReportSchema`. `InterviewAnalysis` is what the LLM
  must produce (everything except the numeric scores).
- `evidence.py` — collects the full interview transcript and retrieves
  relevant due-diligence reference material via the existing RAG retriever
  (`app/rag/retriever.py`, unchanged).
- `analyzer.py` — the single LLM call that analyzes the whole interview,
  using the same raw-JSON-prompt + Pydantic-validate approach as Phase 1/2
  (`app/question_engine/generator.py`, `followup.py`).
- `risk_analyzer.py` — deterministic backstops over the LLM's findings:
  drops non-genuine "contradictions" (identical claims), and escalates any
  HIGH/CRITICAL contradiction into an explicit risk.
- `scorer.py` — all numeric scoring. The LLM never assigns a 0-10 score
  directly (see below); this module converts its qualitative judgment into
  scores, deterministically.
- `report_generator.py` — orchestrates the above into the final,
  validated `AssessmentReportSchema`.

### Evidence discipline

The analysis prompt instructs the model to:

- use only information present in the interview transcript or retrieved
  reference material — never invent facts or benchmarks
- distinguish founder **claims** from **verified evidence** (a claim without
  supporting detail is not proof)
- treat missing evidence as a gap, not a positive signal
- label every piece of evidence with its source: `FOUNDER_ANSWER`,
  `KNOWLEDGE_BASE`, `MODEL_INFERENCE`, or `MISSING_EVIDENCE`
- only report a contradiction when two answers genuinely conflict
- only report an information gap when the interview indicates it's actually
  missing, not as a blanket checklist

Example: if a founder says "we have excellent retention" without a metric,
that becomes a **Moderate** traction assessment with an evidence gap and
**Low** confidence — not a strength.

### Category scoring

Ten categories, each scored 0-10 (9-10 Excellent, 7-8 Strong, 5-6 Moderate,
3-4 Weak, 0-2 Critical): Market, Product, Traction, Business Model,
Financials, Team, Competition, Technology, Go-To-Market, Risk.

The LLM assigns each category a qualitative `assessment`
(Excellent/Strong/Moderate/Weak/Critical) and an `evidence_strength`
(Low/Medium/High) — never a number. `scorer.py` maps that to a 0-10 score
deterministically: a base score per assessment tier, adjusted up for High
evidence strength and down for Low (lack of evidence lowers the score, not
just the stated confidence).

### Overall score

A weighted average of the ten category scores, computed in Python
(`CATEGORY_WEIGHTS` in `app/report_engine/scorer.py`):

| Category       | Weight |
|-----------------|-------|
| Market          | 10%   |
| Product         | 10%   |
| Traction        | 15%   |
| Business Model  | 10%   |
| Financials      | 15%   |
| Team            | 10%   |
| Competition     | 5%    |
| Technology      | 5%    |
| Go-To-Market    | 10%   |
| Risk            | 10%   |

Traction and Financials carry the most concrete, verifiable signal for an
investment decision, so they're weighted highest; Competition and
Technology are typically judged more qualitatively pre-Series-A, so they're
weighted lowest. Rounded to one decimal place.

### Risk level

Deterministic thresholds against the overall score:

| Score      | Risk level |
|-----------|-----------|
| 8.0 - 10.0 | LOW       |
| 6.0 - 7.9  | MEDIUM    |
| 4.0 - 5.9  | HIGH      |
| 0.0 - 3.9  | CRITICAL  |

**Override:** any single `CRITICAL`-severity risk in the report forces the
overall risk level to `CRITICAL`, regardless of the numeric score — one
severe red flag (e.g. unresolved litigation, no IP ownership) shouldn't be
masked by otherwise-strong category scores.

### Report structure

```json
{
  "overall_score": 7.2,
  "risk_level": "MEDIUM",
  "executive_summary": "...",
  "strengths": [{"title": "...", "description": "...", "category": "traction",
    "evidence": [{"source": "FOUNDER_ANSWER", "detail": "..."}], "confidence": "Medium"}],
  "risks": [{"title": "...", "description": "...", "category": "risk", "severity": "HIGH",
    "impact": "...", "evidence": [...], "confidence": "High"}],
  "information_gaps": [{"topic": "...", "why_it_matters": "...", "priority": "HIGH"}],
  "contradictions": [{"topic": "...", "earlier_claim": "...", "later_claim": "...",
    "explanation": "...", "severity": "HIGH"}],
  "category_scores": {
    "market": 8.0, "product": 7.0, "traction": 9.0, "business_model": 7.0,
    "financials": 5.0, "team": 8.0, "competition": 7.0, "technology": 8.0,
    "go_to_market": 6.0, "risk": 5.0
  },
  "recommendations": [{"action": "...", "reason": "...", "priority": "HIGH"}]
}
```

### API endpoints

```
POST   /sessions/{session_id}/complete   generate (or return existing) the report
GET    /sessions/{session_id}/report     fetch the stored report
```

`POST /sessions/{session_id}/complete`:

1. Verifies the session exists (404 if not)
2. Verifies the interview is actually complete — every Top-10 question
   answered, no pending follow-up (409 `IncompleteSessionError` if not)
3. If a report already exists for this session, returns it as-is —
   **idempotent**, calling this twice never creates a second report
4. Otherwise: collects the full Q&A, retrieves RAG evidence, runs the
   analysis, computes scores, stores the report, and returns it
5. Returns 502 if RAG retrieval, the LLM call, or report validation fails
   (a malformed LLM response is never stored)

`GET /sessions/{session_id}/report` returns the stored report, or 404 if
none has been generated yet. It never triggers generation itself.

```bash
curl -X POST http://127.0.0.1:8000/sessions/<session_id>/complete
curl http://127.0.0.1:8000/sessions/<session_id>/report
```

## Tech

- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (local, free, no API key)
- **LLM**: NVIDIA NIM, `meta/llama-3.1-70b-instruct`
- **Vector store**: ChromaDB (persisted to `./data/chroma`)
- **Orchestration**: LangChain
- **API server**: FastAPI + Uvicorn (Phase 2)
- **Session/answer/report storage**: SQLite via SQLAlchemy (persisted to
  `./data/phase2.db`, Phase 2 + Phase 3)
- **Report validation**: Pydantic (Phase 3)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Add your NVIDIA API key to `.env` (never commit this file — it's gitignored):

```
NVIDIA_API_KEY=your-key-here
```

## Usage

**1. Build the knowledge base** (fetches the webpage, chunks it, embeds it, stores it):

```bash
python scripts/build_index.py
```

Re-run this any time the source article changes.

**2. Generate the top 10 questions** for a piece of startup information:

```bash
python scripts/generate_questions.py "We have 10,000 customers and ₹2 crore annual revenue."
```

If no argument is given, a default example is used.

## Testing

```bash
python tests/test_pipeline.py
```

Runs a smoke test over ingestion, chunking, retrieval, and top-10 question
generation end-to-end against the live vector store (run
`scripts/build_index.py` first). Verifies exactly 10 questions are returned,
each fully populated, with no duplicates.

```bash
python tests/test_phase2.py
```

Runs the Phase 2 test suite: answer storage, follow-up generation, the
no-follow-up-needed path, duplicate prevention, a multi-turn conversation,
grounding, contradiction detection, and current-question sequencing (a
pending follow-up must be answered before the next Top-10 question is
offered) — all against an in-memory SQLite DB with a stubbed LLM for
deterministic, offline results, plus one live end-to-end test against the
real NVIDIA LLM and vector store (requires `NVIDIA_API_KEY` and a built
index).

```bash
python tests/test_phase3.py
```

Runs the Phase 3 test suite: report generation, incomplete-session
rejection, strength/risk/information-gap/contradiction detection,
category-score and overall-score bounds, risk-level thresholds (including
the critical-risk override), idempotency of `POST /sessions/{id}/complete`,
and the API endpoints themselves (via FastAPI's `TestClient`) — all against
an in-memory SQLite DB with a stubbed LLM for deterministic, offline
results.

## Notes

- If the source webpage cannot be fetched or its structure can't be parsed,
  the pipeline stops with an explicit error rather than silently falling
  back to another source.
- API keys are read from environment variables only — never hard-coded.
- `ChatNVIDIA` defaults `max_tokens` to 1024, which is enough for Phase 1's
  ten short questions or Phase 2's single follow-up decision, but was
  observed to truncate Phase 3's larger structured report mid-JSON. Phase 3
  sets `REPORT_LLM_MAX_TOKENS` (in `app/config.py`, currently 4096)
  explicitly on its `ChatNVIDIA` instance for this reason.
