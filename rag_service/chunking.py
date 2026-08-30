"""Clean and chunk extracted article content into metadata-tagged pieces.

Ported from app/rag/chunking.py. Returns plain dicts ({text, metadata})
rather than LangChain Documents so the vector-store layer stays free to use
any backend.
"""

from __future__ import annotations

import re
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_service.ingest import ArticleContent


def clean_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return text.strip()


def chunk_article(
    article: ArticleContent, *, chunk_size: int, chunk_overlap: int
) -> list[dict[str, Any]]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[dict[str, Any]] = []
    for section in article.sections:
        section_text = clean_text(section.text)
        if not section_text:
            continue
        for idx, piece in enumerate(splitter.split_text(section_text)):
            chunks.append(
                {
                    "text": f"{section.heading}\n\n{piece}",
                    "metadata": {
                        "source_url": article.source_url,
                        "title": article.title,
                        "section": section.heading,
                        "section_level": section.level,
                        "chunk_index": idx,
                    },
                }
            )
    return chunks
