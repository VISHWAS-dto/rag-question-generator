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


def require_nvidia_api_key() -> str:
    if not NVIDIA_API_KEY:
        raise RuntimeError(
            "NVIDIA_API_KEY is not set. Add it to your .env file "
            "(never hard-code it in source)."
        )
    return NVIDIA_API_KEY
