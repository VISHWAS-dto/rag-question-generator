"""Central configuration for the RAG question-generation app (Phase 1)."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Source ---
SOURCE_URL = "https://www.startupscience.io/articles/startup-due-diligence-checklist"

# --- Storage ---
CHROMA_PERSIST_DIR = str(PROJECT_ROOT / "data" / "chroma")
CHROMA_COLLECTION_NAME = "due_diligence_knowledge"

# --- Chunking ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# --- Embeddings (local, no API key required) ---
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- LLM (NVIDIA NIM) ---
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")
LLM_MODEL_NAME = "meta/llama-3.1-70b-instruct"
LLM_TEMPERATURE = 0.4

# --- Phase 3: report analysis LLM ---
# ChatNVIDIA defaults max_tokens to 1024, which is enough for Phase 1's ten
# short questions or Phase 2's single follow-up decision, but not for a full
# InterviewAnalysis (mandatory assessments for all 10 categories, plus
# strengths/risks/gaps/contradictions/recommendations) — that response was
# observed to be truncated mid-JSON at the default limit. Set explicitly,
# well above what a full report response needs.
REPORT_LLM_MAX_TOKENS = 4096

# --- Retrieval ---
# Broader than a single-question pipeline needs, since we must ground
# 10 distinct, non-overlapping questions across multiple due-diligence
# categories (team, market, product, traction, financial, legal).
RETRIEVAL_TOP_K = 10

# --- Question generation ---
NUM_QUESTIONS = 10

# --- Phase 2: sessions / answers / follow-ups ---
SQLITE_DB_PATH = str(PROJECT_ROOT / "data" / "phase2.db")
SQLITE_URL = f"sqlite:///{SQLITE_DB_PATH}"

# How much RAG context to retrieve when grounding a follow-up decision.
# Narrower than the top-10 generation step, since we're now grounding a
# single question+answer pair rather than covering many categories at once.
FOLLOWUP_RETRIEVAL_TOP_K = 4

# Maximum follow-up questions allowed per Top-10 question before the system
# forces a move to the next Top-10 question, regardless of the LLM's
# decision. Without this cap, a chain of vague or evasive answers can keep
# opening new, individually-reasonable follow-up threads indefinitely
# (observed live: 11+ follow-ups deep on a single Top-10 question with no
# sign of converging), which would prevent the interview from ever reaching
# SessionStatus.COMPLETED.
MAX_FOLLOWUPS_PER_QUESTION = 3


def require_nvidia_api_key() -> str:
    if not NVIDIA_API_KEY:
        raise RuntimeError(
            "NVIDIA_API_KEY is not set. Add it to your .env file "
            "(never hard-code it in source)."
        )
    return NVIDIA_API_KEY
