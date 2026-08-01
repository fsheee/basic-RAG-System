

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from langchain_huggingface import HuggingFaceEmbeddings
from rag.config import EMBEDDING_MODEL

def get_embedding_model():
    print("Loading embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    print("Embedding model loaded.")

    return embeddings