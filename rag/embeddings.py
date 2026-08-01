

from rag.config import EMBEDDING_MODEL  # noqa: E402  (config must be imported first: sets HF offline mode)

from langchain_huggingface import HuggingFaceEmbeddings


def get_embedding_model():
    print("Loading embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    print("Embedding model loaded.")

    return embeddings