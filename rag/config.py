from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")

CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "chroma_db")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def _model_is_cached(model_name):
    cache_root = os.environ.get(
        "HF_HUB_CACHE",
        Path.home() / ".cache" / "huggingface" / "hub",
    )
    repo_dir = Path(cache_root) / ("models--" + model_name.replace("/", "--"))
    return repo_dir.is_dir()


# huggingface_hub and transformers read HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE at
# import time, so force offline mode BEFORE anything imports them. Streamlit in
# particular imports huggingface_hub during its own import, so this must happen
# in config.py, which is imported first by every entry point.
if _model_is_cached(EMBEDDING_MODEL):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"