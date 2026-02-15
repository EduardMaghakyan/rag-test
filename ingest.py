import logging
import os
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
    MAX_PDF_FILES,
    SOURCE_FILES_DIR,
    VECTOR_STORE_COLLECTION,
    VECTOR_STORE_URI,
)

logger = logging.getLogger(__name__)


def load_pdfs(directory: str | Path | None = None) -> list[Document]:
    papers_dir = Path(directory) if directory else SOURCE_FILES_DIR

    if not papers_dir.exists():
        logger.error("Directory does not exist: %s", papers_dir)
        return []
    if not papers_dir.is_dir():
        logger.error("Path is not a directory: %s", papers_dir)
        return []
    if not os.access(papers_dir, os.R_OK):
        logger.error("Directory is not readable: %s", papers_dir)
        return []

    pdf_files = sorted(papers_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDF files found in %s", papers_dir)
        return []

    if len(pdf_files) > MAX_PDF_FILES:
        logger.warning(
            "Found %d PDFs, exceeding limit of %d — truncating",
            len(pdf_files),
            MAX_PDF_FILES,
        )
        pdf_files = pdf_files[:MAX_PDF_FILES]

    documents: list[Document] = []
    failed_files: list[str] = []

    for pdf_path in pdf_files:
        logger.debug("Loading %s...", pdf_path.name)
        try:
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
        except Exception:
            logger.exception("Failed to load %s, skipping", pdf_path.name)
            failed_files.append(pdf_path.name)

    if failed_files:
        logger.warning("Failed to load %d file(s): %s", len(failed_files), failed_files)

    logger.info("Loaded %d pages from %d PDFs", len(documents), len(pdf_files) - len(failed_files))
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
