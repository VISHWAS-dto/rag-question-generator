# RAG Question Generator

A helper for people who invest money in startups.

Before investing, an investor has to ask a young company a lot of careful
questions to check it is a good, honest, safe bet. That homework is called
**due diligence**. This tool does that homework for you.

## What it does

1. **Makes the questions.** You type a few lines about a company. The tool
   writes the **10 most important questions** to ask it, in order of
   importance.
2. **Runs the interview.** It asks those questions one at a time. After each
   answer it decides: is this clear enough, or should I ask a smart
   follow-up? It also notices when two answers don't match and asks about it.
3. **Writes the report.** When the interview is done, it produces a report
   with an overall score (0–10), a risk level (LOW / MEDIUM / HIGH /
   CRITICAL), the strengths, the risks, the missing information, the
   contradictions, and recommended next steps.

The AI only gives opinions in words. The final scores are worked out by
fixed rules, so they are consistent and repeatable.

## How it's built

The project has three phases, each building on the last:

| Phase | Name | What it adds |
|-------|------|--------------|
| 1 | The 10 Questions | Company info in → 10 ranked questions out. Nothing saved. |
| 2 | The Interview | Turns the questions into a live Q&A with follow-ups. Saves the conversation. |
| 3 | The Report | Reads the finished interview and produces the scored report. |

## Tech in one line each

- **RAG** – looks up a real published due-diligence checklist first, then
  gives it to the AI, so questions are grounded in a real guide, not guesswork.
- **ChromaDB** – the small searchable library on disk that stores the
  checklist in pieces.
- **NVIDIA LLM** – the AI that writes the questions, decides follow-ups, and
  analyzes the interview.
- **SQLite** – the local file that remembers each interview and its report.
- **LangGraph** – wraps every AI call in a self-repair loop (see below).

## LangGraph: self-repairing AI calls

The AI is asked to reply as strict JSON, but sometimes it returns broken
output — cut off, wrapped in extra text, or missing a field. Before, that
failed the whole request (HTTP 500): no questions, no report.

Each AI call now runs through a small LangGraph loop:

```
ask the model → try to parse it
                     │
        parsed OK ───┴─── broken → send it back with the error,
             │                     ask for corrected JSON (retry up to 2×)
           done                          │
                              still broken after retries → raise the error
```

So a request that used to fail on the first bad response now usually
succeeds, with no change to the questions, the report, or the API. This is
applied to all three AI steps: question generation, follow-up decisions, and
report analysis (`app/graph/repair_graph.py`; retry count is
`LLM_MAX_REPAIR_ATTEMPTS` in `app/config.py`).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Add your NVIDIA API key to a `.env` file (do not commit it):

```
NVIDIA_API_KEY=your-key-here
```

## Usage

**1. Build the knowledge base** (run once):

```bash
python scripts/build_index.py
```

**2. Get the top 10 questions for a company:**

```bash
python scripts/generate_questions.py "We are a B2B SaaS startup, 15 staff, ₹2 crore yearly revenue."
```

**3. Run the full interview + report (web + API):**

```bash
./scripts/run_servers.sh
```

- Backend (API):  http://127.0.0.1:8001
- Frontend (web): http://127.0.0.1:3000
- API docs:       http://127.0.0.1:8001/docs

### Main API endpoints

```
POST /sessions                   start a session, create its 10 questions
GET  /sessions/{id}/questions    all questions + the current one to answer
POST /questions/{id}/answer      submit an answer → follow-up or next question
POST /sessions/{id}/complete     generate the due-diligence report
GET  /sessions/{id}/report       fetch the saved report
```

## Testing

```bash
python tests/test_pipeline.py     # Phase 1
python tests/test_phase2.py       # Phase 2
python tests/test_phase3.py       # Phase 3
python tests/test_repair_graph.py # LangGraph self-repair loop
```

Phase 2 and 3 tests run offline. `test_pipeline.py` needs a built index and
`NVIDIA_API_KEY`.

## Note

This is the engine only — there is no login or hosted website yet. What
works today is the question, interview, and report logic behind the scenes.
