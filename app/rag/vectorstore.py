"""Embedding model + ChromaDB persistent vector store setup."""

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import CHROMA_COLLECTION_NAME, CHROMA_PERSIST_DIR, EMBEDDING_MODEL_NAME

_embeddings = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """Lazily construct the (local, no-API-key) embedding model."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return _embeddings


def build_vectorstore(documents: list[Document]) -> Chroma:
    """Embed documents and persist them to a fresh ChromaDB collection.

    Recreates the collection from scratch so re-running the index build
    doesn't accumulate duplicate chunks from prior runs.
    """
    embeddings = get_embeddings()

    # Drop any existing collection with the same name so this is idempotent.
    Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    ).delete_collection()

    return Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
    )


def load_vectorstore() -> Chroma:
    """Load the existing persisted ChromaDB collection."""
    return Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_PERSIST_DIR,
    )
