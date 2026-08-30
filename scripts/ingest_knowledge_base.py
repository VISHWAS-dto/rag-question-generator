"""Trigger knowledge-base ingestion on the RAG service.

Usage:
    # against a running rag service
    python scripts/ingest_knowledge_base.py --url http://localhost:8100

    # in-process (no running service; uses RAG_* env / .env)
    python scripts/ingest_knowledge_base.py --in-process
"""

from __future__ import annotations

import argparse
import sys

import httpx


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8100", help="RAG service base URL")
    parser.add_argument("--in-process", action="store_true", help="Run ingestion in this process")
    parser.add_argument("--no-recreate", action="store_true", help="Skip if already populated")
    parser.add_argument("--source-url", default=None, help="Override the source article URL")
    args = parser.parse_args()

    recreate = not args.no_recreate

    if args.in_process:
        from rag_service import service
        from shared.config import get_rag_settings
        from shared.logging import configure_logging

        configure_logging(json_output=False, service="ingest")
        settings = get_rag_settings()
        try:
            sections, chunks, url, collection = service.ingest(
                settings, source_url=args.source_url, recreate=recreate
            )
        except service.IngestionFailed as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        print(f"OK: {collection} <- {sections} sections, {chunks} chunks from {url}")
        return 0

    payload = {"recreate": recreate}
    if args.source_url:
        payload["source_url"] = args.source_url
    try:
        resp = httpx.post(f"{args.url.rstrip('/')}/ingest", json=payload, timeout=300)
    except httpx.HTTPError as exc:
        print(f"ERROR: could not reach RAG service at {args.url}: {exc}", file=sys.stderr)
        return 1
    if resp.status_code != 200:
        print(f"ERROR: ingest returned {resp.status_code}: {resp.text}", file=sys.stderr)
        return 1
    body = resp.json()
    print(
        f"OK: {body['collection']} <- {body['sections_extracted']} sections, "
        f"{body['chunks_indexed']} chunks from {body['source_url']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
