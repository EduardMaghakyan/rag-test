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
RETRIEVAL_FETCH_K = 20  # MMR candidate pool
RETRIEVAL_LAMBDA_MULT = 0.7  # MMR diversity (0=diverse, 1=relevant)
ENABLE_HYBRID_SEARCH = True  # BM25 + vector ensemble
BM25_WEIGHT = 0.3  # BM25 weight in ensemble
VECTOR_WEIGHT = 0.7  # Vector weight in ensemble
ENABLE_RERANKING = True  # FlashRank reranking
RERANK_FETCH_K = 20  # Over-fetch candidates for reranker
RERANK_MODEL = "ms-marco-MiniLM-L-12-v2"
MAX_HISTORY_MESSAGES = 10

logging.basicConfig(level=LOG_LEVEL, format="%(levelname)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
