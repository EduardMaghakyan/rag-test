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
    def __init__(self) -> None:
        self.vector_store = get_vector_store()
        self.llm = ChatOllama(model=LLM_MODEL)
        self.history: list[HumanMessage | AIMessage] = []

    @staticmethod
    def format_context(docs: list[Document]) -> str:
        return "\n\n".join(
            f"[{i}] (Source: {d.metadata.get('source', 'unknown')}, "
            f"Page {d.metadata.get('page', '?')})\n{d.page_content}"
            for i, d in enumerate(docs, 1)
        )

    @staticmethod
    def format_sources(docs: list[Document]) -> str:
        keys = dict.fromkeys(
            f"{d.metadata.get('source', 'unknown')}, Page {d.metadata.get('page', '?')}"
            for d in docs
        )
        return "\n".join(f"  - {k}" for k in keys)

    def ask(self, question: str) -> dict[str, str]:
        docs = self.vector_store.similarity_search(question, k=RETRIEVAL_K)
        context = self.format_context(docs)
        sources = self.format_sources(docs)

        messages = [
            SystemMessage(content=SYSTEM_PROMPT.format(context=context)),
            *self.history[-MAX_HISTORY_MESSAGES:],
            HumanMessage(content=question),
        ]
        answer = str(self.llm.invoke(messages).content)

        self.history.append(HumanMessage(content=question))
        self.history.append(AIMessage(content=answer))

        return {"answer": answer, "sources": sources}

    def reset(self) -> None:
        self.history.clear()
