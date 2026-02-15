"""PDF ingestion pipeline: load PDFs, chunk text, store in ChromaDB."""

import sys

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


def load_pdfs(directory: str | None = None) -> list[Document]:
    """Extract text from all PDFs in directory using pymupdf."""
    papers_dir = PAPERS_DIR if directory is None else __import__("pathlib").Path(directory)
    documents: list[Document] = []

    pdf_files = sorted(papers_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {papers_dir}")
        return documents

    for pdf_path in pdf_files:
        print(f"Loading {pdf_path.name}...")
        doc = pymupdf.open(pdf_path)
        for page_num, page in enumerate(doc):
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

    print(f"Loaded {len(documents)} pages from {len(pdf_files)} PDFs")
    return documents


def chunk_documents(documents: list[Document]) -> list[Document]:
    """Split documents into chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    return chunks


def create_vector_store(chunks: list[Document]) -> Chroma:
    """Embed chunks and store in ChromaDB."""
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    print(f"Stored {len(chunks)} chunks in ChromaDB at {CHROMA_DIR}")
    return vector_store


def get_vector_store() -> Chroma:
    """Load existing ChromaDB vector store."""
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )


def ingest() -> Chroma:
    """Run full ingestion pipeline."""
    documents = load_pdfs()
    chunks = chunk_documents(documents)
    return create_vector_store(chunks)


if __name__ == "__main__":
    if CHROMA_DIR.exists():
        print(f"Vector store already exists at {CHROMA_DIR}. Delete it to re-ingest.")
        response = input("Delete and re-ingest? [y/N] ").strip().lower()
        if response != "y":
            sys.exit(0)
        import shutil
        shutil.rmtree(CHROMA_DIR)

    ingest()
