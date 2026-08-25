# RAG Question Generator

Given a startup's self-reported information, this app retrieves relevant
due-diligence context from a knowledge base built from
[Startup Science's due diligence checklist](https://www.startupscience.io/articles/startup-due-diligence-checklist),
generates the **top 10** ranked, probing due-diligence questions, and then
runs an **interactive questioning session**: the user answers one question
at a time, and the system decides — grounded in the RAG context and the
conversation so far — whether a follow-up question is needed before moving
on.

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

## Tech

- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (local, free, no API key)
- **LLM**: NVIDIA NIM, `meta/llama-3.1-70b-instruct`
- **Vector store**: ChromaDB (persisted to `./data/chroma`)
- **Orchestration**: LangChain
- **API server**: FastAPI + Uvicorn (Phase 2)
- **Session/answer storage**: SQLite via SQLAlchemy (persisted to `./data/phase2.db`, Phase 2)

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
grounding, and contradiction detection — all against an in-memory SQLite DB
with a stubbed LLM for deterministic, offline results, plus one live
end-to-end test against the real NVIDIA LLM and vector store (requires
`NVIDIA_API_KEY` and a built index).

## Notes

- If the source webpage cannot be fetched or its structure can't be parsed,
  the pipeline stops with an explicit error rather than silently falling
  back to another source.
- API keys are read from environment variables only — never hard-coded.
