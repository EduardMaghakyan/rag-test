import logging
import os
import shutil
import sys
from collections.abc import Iterator
from itertools import groupby
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


def load_pdfs(directory: str | Path | None = None) -> Iterator[Document]:
    papers_dir = Path(directory) if directory else SOURCE_FILES_DIR

    if not papers_dir.exists():
        logger.error("Directory does not exist: %s", papers_dir)
        return
    if not papers_dir.is_dir():
        logger.error("Path is not a directory: %s", papers_dir)
        return
    if not os.access(papers_dir, os.R_OK):
        logger.error("Directory is not readable: %s", papers_dir)
        return

    pdf_files = sorted(papers_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDF files found in %s", papers_dir)
        return

    failed_files: list[str] = []
    page_count = 0

    for pdf_path in pdf_files:
        logger.debug("Loading %s...", pdf_path.name)
        try:
            doc = pymupdf.open(pdf_path)
            for page_num, page in enumerate(doc):  # type: ignore[arg-type]
                text = page.get_text()
                if text.strip():
                    page_count += 1
                    yield Document(
                        page_content=text,
                        metadata={
                            "source": pdf_path.name,
                            "page": page_num + 1,
                        },
                    )
            doc.close()
        except Exception:
            logger.exception("Failed to load %s, skipping", pdf_path.name)
            failed_files.append(pdf_path.name)

    if failed_files:
        logger.warning("Failed to load %d file(s): %s", len(failed_files), failed_files)

    logger.info("Loaded %d pages from %d PDFs", page_count, len(pdf_files) - len(failed_files))


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


def _create_empty_vector_store() -> VectorStore:
    embeddings = _get_embeddings()
    if APP_ENV == "prod":
        from langchain_qdrant import QdrantVectorStore

        return QdrantVectorStore(
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
    vector_store = _create_empty_vector_store()
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    for source, pages in groupby(load_pdfs(), key=lambda d: d.metadata["source"]):
        chunks = splitter.split_documents(list(pages))
        vector_store.add_documents(chunks)
        logger.debug("Indexed %d chunks from %s", len(chunks), source)

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
