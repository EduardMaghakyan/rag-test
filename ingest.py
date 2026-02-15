import logging
import shutil
import sys
from pathlib import Path

import pymupdf
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    PAPERS_DIR,
)

logger = logging.getLogger(__name__)


def load_pdfs(directory: str | Path | None = None) -> list[Document]:
    papers_dir = Path(directory) if directory else PAPERS_DIR
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


def get_vector_store() -> Chroma:
    return Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=OllamaEmbeddings(model=EMBEDDING_MODEL),
    )


def ingest() -> Chroma:
    chunks = chunk_documents(load_pdfs())
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=OllamaEmbeddings(model=EMBEDDING_MODEL),
        persist_directory=str(CHROMA_DIR),
    )
    logger.info("Stored %d chunks in ChromaDB at %s", len(chunks), CHROMA_DIR)
    return vector_store


if __name__ == "__main__":
    if CHROMA_DIR.exists():
        print(f"Vector store already exists at {CHROMA_DIR}. Delete it to re-ingest.")
        response = input("Delete and re-ingest? [y/N] ").strip().lower()
        if response != "y":
            sys.exit(0)
        shutil.rmtree(CHROMA_DIR)

    ingest()
