"""Tests for PDF loading, chunking, and RAG formatting."""

from pathlib import Path

import pymupdf
from langchain_core.documents import Document

from ingest import chunk_documents, load_pdfs
from rag import RAGChain

# --- Helpers ---


def create_test_pdf(path: Path, pages: list[str]) -> None:
    """Create a simple PDF with given page texts."""
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    doc.save(str(path))
    doc.close()


# --- Test PDF Loading ---


class TestLoadPdfs:
    def test_loads_pdfs_with_metadata(self, tmp_path: Path) -> None:
        create_test_pdf(tmp_path / "paper1.pdf", ["Page one content", "Page two content"])
        create_test_pdf(tmp_path / "paper2.pdf", ["Another paper"])

        docs = load_pdfs(str(tmp_path))

        assert len(docs) == 3
        assert docs[0].metadata["source"] == "paper1.pdf"
        assert docs[0].metadata["page"] == 1
        assert "Page one content" in docs[0].page_content
        assert docs[2].metadata["source"] == "paper2.pdf"

    def test_empty_directory(self, tmp_path: Path) -> None:
        docs = load_pdfs(str(tmp_path))
        assert docs == []

    def test_skips_blank_pages(self, tmp_path: Path) -> None:
        doc = pymupdf.open()
        doc.new_page()  # blank
        page = doc.new_page()
        page.insert_text((72, 72), "Has content")
        doc.save(str(tmp_path / "mixed.pdf"))
        doc.close()

        docs = load_pdfs(str(tmp_path))
        assert len(docs) == 1
        assert docs[0].metadata["page"] == 2


# --- Test Chunking ---


class TestChunkDocuments:
    def test_splits_long_document(self) -> None:
        long_text = "word " * 500  # ~2500 chars
        docs = [Document(page_content=long_text, metadata={"source": "test.pdf", "page": 1})]

        chunks = chunk_documents(docs)

        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.metadata["source"] == "test.pdf"
            assert chunk.metadata["page"] == 1

    def test_short_document_not_split(self) -> None:
        docs = [Document(page_content="Short text.", metadata={"source": "test.pdf", "page": 1})]

        chunks = chunk_documents(docs)

        assert len(chunks) == 1
        assert chunks[0].page_content == "Short text."


# --- Test RAG Formatting ---


class TestRAGChainFormatting:
    def setup_method(self) -> None:
        self.docs = [
            Document(page_content="Content A", metadata={"source": "paper1.pdf", "page": 1}),
            Document(page_content="Content B", metadata={"source": "paper1.pdf", "page": 1}),
            Document(page_content="Content C", metadata={"source": "paper2.pdf", "page": 3}),
        ]

    def test_format_context_numbering(self) -> None:
        context = RAGChain.format_context(None, self.docs)

        assert "[1]" in context
        assert "[2]" in context
        assert "[3]" in context
        assert "paper1.pdf" in context
        assert "paper2.pdf" in context

    def test_format_sources_deduplication(self) -> None:
        sources = RAGChain.format_sources(None, self.docs)

        lines = sources.strip().split("\n")
        # paper1.pdf page 1 appears twice in docs but should be deduped
        assert len(lines) == 2
        assert "paper1.pdf" in lines[0]
        assert "paper2.pdf" in lines[1]
