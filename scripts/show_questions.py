"""Print all questions for a Phase 2 session as a clean numbered list.

Usage:
    python scripts/show_questions.py <session_id>
    python scripts/show_questions.py <session_id> --api-url http://127.0.0.1:8000
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests

DEFAULT_API_URL = "http://127.0.0.1:8000"


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print("Usage: python scripts/show_questions.py <session_id> [--api-url URL]")
        sys.exit(1)

    session_id = args[0]
    api_url = DEFAULT_API_URL
    if "--api-url" in args:
        api_url = args[args.index("--api-url") + 1]

    url = f"{api_url}/sessions/{session_id}/questions"

    try:
        response = requests.get(url, timeout=10)
    except requests.RequestException as exc:
        print(f"ERROR: could not reach {url}")
        print(f"  {exc}")
        print("\nIs the server running? Start it with: uvicorn app.main:app --reload")
        sys.exit(1)

    if response.status_code == 404:
        print(f"ERROR: session '{session_id}' not found.")
        print("Check the session_id, or create a new session with POST /sessions.")
        sys.exit(1)

    if response.status_code != 200:
        print(f"ERROR: unexpected response {response.status_code}")
        print(response.text)
        sys.exit(1)

    data = response.json()
    questions = data.get("questions", [])

    if not questions:
        print(f"Session '{session_id}' has no questions yet.")
        sys.exit(0)

    print(f"\n{'=' * 60}")
    print(f"QUESTIONS FOR SESSION {session_id}")
    print(f"{'=' * 60}\n")

    for i, q in enumerate(questions, start=1):
        tag = " (follow-up)" if q.get("is_followup") else ""
        status = q.get("status", "")
        print(f"{i}. [{status}]{tag} {q['question']}")

    current = data.get("current_question")
    print()
    if current:
        print(f"NEXT QUESTION TO ANSWER: {current['question']}")
    else:
        print("All questions in this session have been answered.")


if __name__ == "__main__":
    main()
