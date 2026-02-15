from pathlib import Path

PAPERS_DIR = Path("research_papers")
CHROMA_DIR = Path("chroma_db")

LLM_MODEL = "llama3.1:8b"
EMBEDDING_MODEL = "nomic-embed-text"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVAL_K = 4
MAX_HISTORY_MESSAGES = 10
