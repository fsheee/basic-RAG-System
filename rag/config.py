from dotenv import load_dotenv
import os

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")

CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")