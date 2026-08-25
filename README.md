# RAG Question Generator — Phase 1

Given a startup's self-reported information, this app retrieves relevant
due-diligence context from a knowledge base built from
[Startup Science's due diligence checklist](https://www.startupscience.io/articles/startup-due-diligence-checklist)
and generates the **top 10** ranked, probing due-diligence questions — the
kind of questions a sharp investor would ask next, not questions the startup
already answered.

This is **Phase 1 only**: a local, single-pass RAG pipeline. No API server,
database, frontend, auth, conversation history, or multi-agent orchestration.

## Pipeline

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

## Tech

- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (local, free, no API key)
- **LLM**: NVIDIA NIM, `meta/llama-3.1-70b-instruct`
- **Vector store**: ChromaDB (persisted to `./data/chroma`)
- **Orchestration**: LangChain

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

## Notes

- If the source webpage cannot be fetched or its structure can't be parsed,
  the pipeline stops with an explicit error rather than silently falling
  back to another source.
- API keys are read from environment variables only — never hard-coded.
