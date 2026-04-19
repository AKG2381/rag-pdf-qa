from unittest.mock import MagicMock, patch
from langchain.schema import Document
import io

from app.ingestor import _chunk
from app.chain import _fmt_context, Response


def doc(text, page=1):
    return Document(page_content=text, metadata={"source": "test.pdf", "page": page, "total_pages": 5})


# --- chunking ---

def test_chunk_splits_long():
    chunks = _chunk([doc("word " * 300)])
    assert len(chunks) > 1

def test_chunk_keeps_metadata():
    chunks = _chunk([doc("hello " * 20, page=3)])
    assert all(c.metadata["page"] == 3 for c in chunks)

def test_chunk_single_short():
    assert len(_chunk([doc("short")])) == 1

def test_chunk_index_set():
    chunks = _chunk([doc("word " * 300)])
    assert all("chunk_index" in c.metadata for c in chunks)


# --- context formatting ---

def test_fmt_context_numbered():
    ctx = _fmt_context([doc("first"), doc("second")])
    assert "[1]" in ctx and "[2]" in ctx

def test_fmt_context_has_source():
    ctx = _fmt_context([doc("content", page=4)])
    assert "test.pdf" in ctx and "p.4" in ctx

def test_fmt_context_empty():
    assert _fmt_context([]) == ""


# --- Response ---

def test_response_from_docs():
    docs = [doc("neural nets", page=2), doc("transformers", page=7)]
    r = Response.from_docs("answer [1]", docs, "question")
    assert len(r.sources) == 2
    assert r.sources[0]["index"] == 1
    assert r.sources[1]["page"] == 7

def test_response_excerpt_capped():
    r = Response.from_docs("a", [doc("x" * 500)], "q")
    assert len(r.sources[0]["excerpt"]) <= 310


# --- ingest dedup ---

@patch("app.ingestor.get_embeddings")
@patch("app.ingestor.Chroma")
def test_ingest_dedup(MockChroma, _):
    from app.ingestor import ingest
    mock_store = MagicMock()
    mock_store.get.return_value = {"ids": ["abc"]}
    MockChroma.return_value = mock_store

    res = ingest(io.BytesIO(b"pdf"), "file.pdf")
    assert res["skipped"] is True
    mock_store.add_documents.assert_not_called()
