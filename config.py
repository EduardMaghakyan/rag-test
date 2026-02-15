from pathlib import Path

# Paths
PAPERS_DIR = Path("research_papers")
CHROMA_DIR = Path("chroma_db")

# Ollama models
LLM_MODEL = "llama3.1:8b"
EMBEDDING_MODEL = "nomic-embed-text"

# Chunking
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval
RETRIEVAL_K = 4

# Conversation
MAX_HISTORY_MESSAGES = 10
