"""Clean and chunk extracted article content into LangChain Documents with metadata."""

import re

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.config import CHUNK_OVERLAP, CHUNK_SIZE
from app.rag.ingest import ArticleContent


def clean_text(text: str) -> str:
    """Normalize whitespace and strip stray artifacts from extracted text."""
    text = text.replace(" ", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return text.strip()


def chunk_article(article: ArticleContent) -> list[Document]:
    """Turn article sections into overlapping, metadata-tagged chunks.

    Each chunk carries: source_url, title, section (heading), section_level,
    and chunk_index (position within that section).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    documents: list[Document] = []

    for section in article.sections:
        section_text = clean_text(section.text)
        if not section_text:
            continue

        # Prefix each chunk with its section heading so retrieved chunks
        # remain self-describing even out of context.
        pieces = splitter.split_text(section_text)

        for idx, piece in enumerate(pieces):
            content = f"{section.heading}\n\n{piece}"
            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "source_url": article.source_url,
                        "title": article.title,
                        "section": section.heading,
                        "section_level": section.level,
                        "chunk_index": idx,
                    },
                )
            )

    return documents
