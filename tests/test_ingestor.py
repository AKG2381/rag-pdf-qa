"""
tests/test_ingestor.py
Tests for the PDF ingestion pipeline.
"""

import io
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

from app.ingestor import chunk_documents, _pypdf_fallback


# ---------------------------------------------------------------------------
# Unit tests — chunking
# ---------------------------------------------------------------------------

def make_doc(content: str, page: int = 1) -> Document:
    return Document(
        page_content=content,
        metadata={"source": "test.pdf", "page": page, "total_pages": 3},
    )


def test_chunk_documents_splits_long_text():
    """A long document should be split into multiple chunks."""
    long_text = "This is a sentence. " * 200   # ~3600 chars
    docs = [make_doc(long_text)]

    chunks = chunk_documents(docs)

    assert len(chunks) > 1, "Long text should produce multiple chunks"
    assert all(len(c.page_content) <= 1000 for c in chunks), "No chunk should far exceed chunk_size"


def test_chunk_documents_preserves_metadata():
    """Chunk metadata should inherit page and source from parent doc."""
    docs = [make_doc("Hello world. " * 10, page=2)]
    chunks = chunk_documents(docs)

    for chunk in chunks:
        assert chunk.metadata["source"] == "test.pdf"
        assert chunk.metadata["page"] == 2


def test_chunk_documents_short_text_stays_single():
    """Short text that fits in one chunk should not be split."""
    docs = [make_doc("Short document.")]
    chunks = chunk_documents(docs)
    assert len(chunks) == 1


def test_chunk_index_assigned():
    """Every chunk should have a chunk_index in its metadata."""
    docs = [make_doc("Word " * 300)]
    chunks = chunk_documents(docs)
    for i, chunk in enumerate(chunks):
        assert "chunk_index" in chunk.metadata


# ---------------------------------------------------------------------------
# Integration-style tests — ingest_pdf (mocked)
# ---------------------------------------------------------------------------

@patch("app.ingestor.get_or_create_vectorstore")
@patch("app.ingestor.extract_text_from_pdf")
def test_ingest_pdf_happy_path(mock_extract, mock_vs):
    """ingest_pdf should extract, chunk, and add documents to the vector store."""
    from app.ingestor import ingest_pdf

    mock_extract.return_value = [make_doc("Some text about AI. " * 20)]
    mock_store = MagicMock()
    mock_store.get.return_value = {"ids": []}   # not yet ingested
    mock_vs.return_value = mock_store

    fake_pdf = io.BytesIO(b"fake pdf bytes")
    result = ingest_pdf(fake_pdf, "test.pdf")

    assert result["skipped"] is False
    assert result["pages"] == 1
    assert result["chunks"] >= 1
    mock_store.add_documents.assert_called_once()


@patch("app.ingestor.get_or_create_vectorstore")
def test_ingest_pdf_deduplication(mock_vs):
    """Re-ingesting the same file should be skipped."""
    from app.ingestor import ingest_pdf

    mock_store = MagicMock()
    mock_store.get.return_value = {"ids": ["existing-id"]}   # already ingested
    mock_vs.return_value = mock_store

    fake_pdf = io.BytesIO(b"some pdf content")
    result = ingest_pdf(fake_pdf, "already_ingested.pdf")

    assert result["skipped"] is True
    mock_store.add_documents.assert_not_called()
