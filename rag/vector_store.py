from langchain_chroma import Chroma

from rag.config import CHROMA_DB_DIR
from rag.embeddings import get_embedding_model


def ensure_vector_store():
    """
    Build the vector store from data/ PDFs if it is missing or empty.

    The chroma_db/ directory is gitignored and therefore not deployed, so on
    Streamlit Cloud the store starts out empty. Rebuilding it at startup keeps
    the app functional on a fresh deploy.
    """

    try:
        store = load_vector_store()
        if store._collection.count() > 0:
            return store
    except Exception:
        pass

    from rag.loader import load_documents
    from rag.splitter import split_documents

    documents = load_documents()
    chunks = split_documents(documents)
    return create_vector_store(chunks)


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