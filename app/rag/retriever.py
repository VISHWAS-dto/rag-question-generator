"""Retrieval of relevant due-diligence context chunks for a given startup info query."""

from langchain_core.documents import Document

from app.config import RETRIEVAL_TOP_K
from app.rag.vectorstore import load_vectorstore


def retrieve_context(startup_info: str, k: int = RETRIEVAL_TOP_K) -> list[Document]:
    """Return the top-k most relevant due-diligence chunks for the given startup info."""
    vectorstore = load_vectorstore()
    return vectorstore.similarity_search(startup_info, k=k)


def format_context(documents: list[Document]) -> str:
    """Render retrieved documents into a single text block for the LLM prompt."""
    blocks = []
    for doc in documents:
        section = doc.metadata.get("section", "Unknown section")
        blocks.append(f"[Section: {section}]\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks)
