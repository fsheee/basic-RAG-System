from langchain_huggingface import HuggingFaceEmbeddings

from rag.config import EMBEDDING_MODEL


def get_embedding_model():
    """
    Load and return the Hugging Face embedding model.
    """

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    return embeddings