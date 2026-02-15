"""RAG chain: retrieval, context formatting, and answer generation with conversation history."""

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from config import LLM_MODEL, MAX_HISTORY_MESSAGES, RETRIEVAL_K
from ingest import get_vector_store

SYSTEM_PROMPT = """\
You are a helpful research assistant. \
Answer questions based ONLY on the provided context from research papers.

Rules:
- Use only the information in the context below to answer
- Cite sources by filename and page number (e.g., "According to 2510.06664v1.pdf, page 3...")
- If the context doesn't contain enough information, say so clearly
- Be concise and accurate

Context:
{context}"""


class RAGChain:
    """RAG chain with conversation history."""

    def __init__(self) -> None:
        self.vector_store = get_vector_store()
        self.llm = ChatOllama(model=LLM_MODEL)
        self.history: list[HumanMessage | AIMessage] = []

    def retrieve(self, query: str) -> list[Document]:
        """Similarity search for relevant chunks."""
        return self.vector_store.similarity_search(query, k=RETRIEVAL_K)

    def format_context(self, docs: list[Document]) -> str:
        """Format retrieved documents as numbered context with source refs."""
        parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "?")
            parts.append(f"[{i}] (Source: {source}, Page {page})\n{doc.page_content}")
        return "\n\n".join(parts)

    def format_sources(self, docs: list[Document]) -> str:
        """Deduplicated source list."""
        seen: set[str] = set()
        sources: list[str] = []
        for doc in docs:
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "?")
            key = f"{source}, Page {page}"
            if key not in seen:
                seen.add(key)
                sources.append(f"  - {key}")
        return "\n".join(sources)

    def ask(self, question: str) -> dict[str, str]:
        """Retrieve context, generate answer, return answer + sources."""
        docs = self.retrieve(question)
        context = self.format_context(docs)
        sources = self.format_sources(docs)

        system_msg = SystemMessage(content=SYSTEM_PROMPT.format(context=context))

        # Trim history to last N messages
        trimmed_history = self.history[-MAX_HISTORY_MESSAGES:]

        messages = [system_msg, *trimmed_history, HumanMessage(content=question)]
        response = self.llm.invoke(messages)

        # Update history
        self.history.append(HumanMessage(content=question))
        self.history.append(AIMessage(content=response.content))

        return {
            "answer": response.content,
            "sources": sources,
        }

    def reset(self) -> None:
        """Clear conversation history."""
        self.history.clear()
