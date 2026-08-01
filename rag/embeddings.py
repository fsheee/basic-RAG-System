

import os

from langchain_huggingface import HuggingFaceEmbeddings
from rag.config import EMBEDDING_MODEL

def get_embedding_model():
    print("Loading embedding model...")

    # Streamlit can close huggingface_hub's HTTP client between reruns, so use
    # offline mode when the model is already cached locally.
    from huggingface_hub import scan_cache_dir
    model_cached = False
    try:
        model_cached = any(
            model in repo.repo_id
            for repo in scan_cache_dir().repos
        )
    except Exception:
        model_cached = False

    if model_cached:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    print("Embedding model loaded.")

    return embeddings