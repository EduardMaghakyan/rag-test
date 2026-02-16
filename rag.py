import logging

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore

from config import (
    BM25_WEIGHT,
    ENABLE_HYBRID_SEARCH,
    ENABLE_RERANKING,
    MAX_HISTORY_MESSAGES,
    RERANK_FETCH_K,
    RERANK_MODEL,
    RETRIEVAL_FETCH_K,
    RETRIEVAL_K,
    RETRIEVAL_LAMBDA_MULT,
    VECTOR_WEIGHT,
)
from ingest import get_all_documents, get_llm, get_vector_store

logger = logging.getLogger(__name__)

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
    def __init__(
        self,
        *,
        vector_store: VectorStore | None = None,
        llm: BaseChatModel | None = None,
    ) -> None:
        self.vector_store = vector_store or get_vector_store()
        self.llm = llm or get_llm()
        self.history: list[HumanMessage | AIMessage] = []
        self.retriever = self._build_retriever()

    def _build_retriever(self) -> BaseRetriever:
        candidate_k = RERANK_FETCH_K if ENABLE_RERANKING else RETRIEVAL_K

        # Base: MMR retriever for diversity
        base_retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": candidate_k,
                "fetch_k": RETRIEVAL_FETCH_K,
                "lambda_mult": RETRIEVAL_LAMBDA_MULT,
            },
        )
        retriever = base_retriever

        # + Hybrid: BM25 + vector ensemble
        if ENABLE_HYBRID_SEARCH:
            try:
                from langchain_classic.retrievers import EnsembleRetriever
                from langchain_community.retrievers import BM25Retriever

                all_docs = get_all_documents(self.vector_store)
                if all_docs:
                    bm25_retriever = BM25Retriever.from_documents(all_docs, k=candidate_k)
                    retriever = EnsembleRetriever(
                        retrievers=[bm25_retriever, base_retriever],
                        weights=[BM25_WEIGHT, VECTOR_WEIGHT],
                    )
                    logger.info("Hybrid search enabled (BM25 + vector)")
                else:
                    logger.warning("No documents found for BM25; using vector-only retrieval")
            except Exception:
                logger.exception("Failed to enable hybrid search; falling back to vector-only")

        # + Rerank: FlashRank reranking
        if ENABLE_RERANKING:
            try:
                from flashrank import Ranker
                from langchain_classic.retrievers import ContextualCompressionRetriever
                from langchain_community.document_compressors import FlashrankRerank

                compressor = FlashrankRerank(
                    client=Ranker(model_name=RERANK_MODEL),
                    top_n=RETRIEVAL_K,
                )
                retriever = ContextualCompressionRetriever(
                    base_compressor=compressor,
                    base_retriever=retriever,
                )
                logger.info("FlashRank reranking enabled (model=%s)", RERANK_MODEL)
            except Exception:
                logger.exception("Failed to enable reranking; using retriever without reranking")

        return retriever

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
        docs = self.retriever.invoke(question)
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
