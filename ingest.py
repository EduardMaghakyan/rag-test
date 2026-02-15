import logging
import shutil
import sys
from pathlib import Path

import pymupdf
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    APP_ENV,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    LLM_MODEL,
    SOURCE_FILES_DIR,
    VECTOR_STORE_COLLECTION,
    VECTOR_STORE_URI,
)

logger = logging.getLogger(__name__)


def load_pdfs(directory: str | Path | None = None) -> list[Document]:
    papers_dir = Path(directory) if directory else SOURCE_FILES_DIR
    pdf_files = sorted(papers_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDF files found in %s", papers_dir)
        return []

    documents: list[Document] = []

    for pdf_path in pdf_files:
        logger.debug("Loading %s...", pdf_path.name)
        doc = pymupdf.open(pdf_path)
        for page_num, page in enumerate(doc):  # type: ignore[arg-type]
            text = page.get_text()
            if text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": pdf_path.name,
                            "page": page_num + 1,
                        },
                    )
                )
        doc.close()

    logger.info("Loaded %d pages from %d PDFs", len(documents), len(pdf_files))
    return documents


def chunk_documents(documents: list[Document]) -> list[Document]:
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    ).split_documents(documents)
    logger.debug("Split into %d chunks", len(chunks))
    return chunks


def get_llm() -> BaseChatModel:
    if APP_ENV == "prod":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=LLM_MODEL)

    from langchain_ollama import ChatOllama

    return ChatOllama(model=LLM_MODEL)


def _get_embeddings() -> Embeddings:
    if APP_ENV == "prod":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=EMBEDDING_MODEL)

    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(model=EMBEDDING_MODEL)


def get_vector_store() -> VectorStore:
    embeddings = _get_embeddings()
    if APP_ENV == "prod":
        from langchain_qdrant import QdrantVectorStore

        return QdrantVectorStore.from_existing_collection(
            url=VECTOR_STORE_URI,
            collection_name=VECTOR_STORE_COLLECTION,
            embedding=embeddings,
        )

    from langchain_chroma import Chroma

    return Chroma(
        persist_directory=VECTOR_STORE_URI,
        embedding_function=embeddings,
    )


def ingest() -> VectorStore:
    chunks = chunk_documents(load_pdfs())
    embeddings = _get_embeddings()
    if APP_ENV == "prod":
        from langchain_qdrant import QdrantVectorStore

        vector_store = QdrantVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            url=VECTOR_STORE_URI,
            collection_name=VECTOR_STORE_COLLECTION,
        )
        logger.info("Stored %d chunks in Qdrant at %s", len(chunks), VECTOR_STORE_URI)
    else:
        from langchain_chroma import Chroma

        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=VECTOR_STORE_URI,
        )
        logger.info("Stored %d chunks in ChromaDB at %s", len(chunks), VECTOR_STORE_URI)
    return vector_store


if __name__ == "__main__":
    store_path = Path(VECTOR_STORE_URI)
    if APP_ENV != "prod" and store_path.exists():
        print(f"Vector store already exists at {store_path}. Delete it to re-ingest.")
        response = input("Delete and re-ingest? [y/N] ").strip().lower()
        if response != "y":
            sys.exit(0)
        shutil.rmtree(store_path)

    ingest()
