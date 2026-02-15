from pathlib import Path
from unittest.mock import Mock

import pymupdf
from langchain_core.documents import Document

from ingest import chunk_documents, load_pdfs
from rag import RAGChain


def create_test_pdf(path: Path, pages: list[str]) -> None:
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    doc.save(str(path))
    doc.close()


def test_load_pdfs_with_metadata(tmp_path: Path) -> None:
    create_test_pdf(tmp_path / "paper1.pdf", ["Page one content", "Page two content"])
    create_test_pdf(tmp_path / "paper2.pdf", ["Another paper"])

    docs = load_pdfs(str(tmp_path))

    assert len(docs) == 3
    assert docs[0].metadata["source"] == "paper1.pdf"
    assert docs[0].metadata["page"] == 1
    assert "Page one content" in docs[0].page_content
    assert docs[2].metadata["source"] == "paper2.pdf"


def test_load_pdfs_empty_directory(tmp_path: Path) -> None:
    assert load_pdfs(str(tmp_path)) == []


def test_load_pdfs_skips_blank_pages(tmp_path: Path) -> None:
    create_test_pdf(tmp_path / "mixed.pdf", ["", "Has content"])

    docs = load_pdfs(str(tmp_path))
    assert len(docs) == 1
    assert docs[0].metadata["page"] == 2


def test_chunk_splits_long_document() -> None:
    long_text = "word " * 500
    docs = [Document(page_content=long_text, metadata={"source": "test.pdf", "page": 1})]

    chunks = chunk_documents(docs)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.metadata["source"] == "test.pdf"
        assert chunk.metadata["page"] == 1


def test_chunk_short_document_not_split() -> None:
    docs = [Document(page_content="Short text.", metadata={"source": "test.pdf", "page": 1})]

    chunks = chunk_documents(docs)

    assert len(chunks) == 1
    assert chunks[0].page_content == "Short text."


def test_format_context_numbering() -> None:
    docs = [
        Document(page_content="Content A", metadata={"source": "paper1.pdf", "page": 1}),
        Document(page_content="Content B", metadata={"source": "paper2.pdf", "page": 3}),
    ]
    context = RAGChain.format_context(docs)

    assert "[1]" in context
    assert "[2]" in context
    assert "paper1.pdf" in context
    assert "paper2.pdf" in context


def test_format_sources_deduplication() -> None:
    docs = [
        Document(page_content="Content A", metadata={"source": "paper1.pdf", "page": 1}),
        Document(page_content="Content B", metadata={"source": "paper1.pdf", "page": 1}),
        Document(page_content="Content C", metadata={"source": "paper2.pdf", "page": 3}),
    ]
    sources = RAGChain.format_sources(docs)

    lines = sources.strip().split("\n")
    assert len(lines) == 2
    assert "paper1.pdf" in lines[0]
    assert "paper2.pdf" in lines[1]


def test_ask_returns_answer_and_sources() -> None:
    mock_store = Mock()
    mock_store.similarity_search.return_value = [
        Document(page_content="text", metadata={"source": "a.pdf", "page": 1})
    ]
    mock_llm = Mock()
    mock_llm.invoke.return_value = Mock(content="answer")

    chain = RAGChain(vector_store=mock_store, llm=mock_llm)
    result = chain.ask("question")

    assert result["answer"] == "answer"
    assert "a.pdf" in result["sources"]
    mock_store.similarity_search.assert_called_once()
    mock_llm.invoke.assert_called_once()
