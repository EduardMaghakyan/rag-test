import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

APP_ENV = os.environ.get("APP_ENV", "dev")

SOURCE_FILES_DIR = Path(os.environ.get("SOURCE_FILES_DIR", "research_papers"))

# Single URI — directory path for dev (Chroma), URL for prod (Qdrant)
VECTOR_STORE_URI = os.environ.get("VECTOR_STORE_URI", "chroma_db")
VECTOR_STORE_COLLECTION = os.environ.get("VECTOR_STORE_COLLECTION", "research_papers")

# Single model name — Ollama model for dev, OpenAI model for prod
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "nomic-embed-text")

LLM_MODEL = os.environ.get("LLM_MODEL", "llama3.1:8b")
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVAL_K = 4
MAX_HISTORY_MESSAGES = 10

logging.basicConfig(level=LOG_LEVEL, format="%(levelname)s: %(message)s")
