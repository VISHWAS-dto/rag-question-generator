#!/usr/bin/env bash
#
# Start the backend and the static frontend for local use.
#
#   ./scripts/run_servers.sh
#
# Backend  -> http://127.0.0.1:8001   (FastAPI / uvicorn)
# Frontend -> http://127.0.0.1:3000   (plain static file server)
#
# Why not `uvicorn app.main:app --reload --port 8001` directly:
# `--reload` watches the ENTIRE working directory, including .venv/ and
# data/chroma/. Files written *during* a request (ChromaDB's sqlite WAL,
# freshly compiled .pyc files) trip the reloader, which kills the worker
# mid-request. POST /sessions runs a ~20-40s LLM call, so that window is
# wide and the browser sees the dropped connection as "Failed to fetch".
#
# This script scopes reload to app/ only. Pass --no-reload to disable it
# entirely (recommended if you still see mid-request restarts).

set -euo pipefail
cd "$(dirname "$0")/.."

RELOAD_ARGS=(--reload --reload-dir app)
if [[ "${1:-}" == "--no-reload" ]]; then
    RELOAD_ARGS=()
fi

cleanup() { kill 0 2>/dev/null || true; }
trap cleanup EXIT INT TERM

echo "Backend  -> http://127.0.0.1:8001"
python -m uvicorn app.main:app --port 8001 "${RELOAD_ARGS[@]}" &

echo "Frontend -> http://127.0.0.1:3000"
python -m http.server 3000 --directory frontend --bind 127.0.0.1 &

wait
