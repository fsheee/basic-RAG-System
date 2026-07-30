from langchain_chroma import Chroma

from rag.config import CHROMA_DB_DIR
from rag.embeddings import get_embedding_model


def create_vector_store(chunks):
    """
    Create and save a ChromaDB vector store.
    """

    embeddings = get_embedding_model()

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR
    )

    return vector_store


def load_vector_store():
    """
    Load an existing ChromaDB vector store.
    """

    embeddings = get_embedding_model()

    vector_store = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings
    )

    return vector_store