"""
tests/test_chain.py
Tests for the RAG chain — retrieval + generation.
"""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

from app.chain import RAGResponse, _format_context


# ---------------------------------------------------------------------------
# Unit tests — context formatting
# ---------------------------------------------------------------------------

def make_doc(content: str, source: str = "doc.pdf", page: int = 1) -> Document:
    return Document(
        page_content=content,
        metadata={"source": source, "page": page, "total_pages": 5},
    )


def test_format_context_numbering():
    """Passages should be numbered [1], [2], etc."""
    docs = [make_doc("First passage."), make_doc("Second passage.")]
    context = _format_context(docs)
    assert "[1]" in context
    assert "[2]" in context


def test_format_context_includes_metadata():
    """Context should include source filename and page number."""
    docs = [make_doc("Content here.", source="report.pdf", page=3)]
    context = _format_context(docs)
    assert "report.pdf" in context
    assert "page 3" in context


def test_format_context_empty():
    """Empty doc list should return empty string."""
    assert _format_context([]) == ""


# ---------------------------------------------------------------------------
# RAGResponse dataclass
# ---------------------------------------------------------------------------

def test_rag_response_from_docs():
    """RAGResponse.from_docs should correctly map docs to source dicts."""
    docs = [
        make_doc("Chunk about neural networks.", source="ai_book.pdf", page=12),
        make_doc("Chunk about transformers.", source="ai_book.pdf", page=15),
    ]
    response = RAGResponse.from_docs(
        answer="Neural networks and transformers are discussed [1][2].",
        docs=docs,
        query="What is AI?",
    )

    assert response.answer.startswith("Neural")
    assert len(response.sources) == 2
    assert response.sources[0]["index"] == 1
    assert response.sources[0]["source"] == "ai_book.pdf"
    assert response.sources[1]["page"] == 15


def test_rag_response_excerpt_truncation():
    """Long chunk content should be truncated in the excerpt."""
    long_content = "x" * 500
    docs = [make_doc(long_content)]
    response = RAGResponse.from_docs("answer", docs, "query")
    assert len(response.sources[0]["excerpt"]) <= 310   # 300 chars + "..."


# ---------------------------------------------------------------------------
# Integration — ask() (fully mocked)
# ---------------------------------------------------------------------------

@patch("app.chain.get_llm")
@patch("app.chain.get_retriever")
def test_ask_returns_rag_response(mock_retriever, mock_llm):
    """ask() should return a RAGResponse with answer and sources."""
    mock_ret = MagicMock()
    mock_ret.invoke.return_value = [make_doc("The capital of France is Paris.", source="geo.pdf", page=1)]
    mock_retriever.return_value = mock_ret

    mock_chain_output = "The capital of France is Paris [1]."
    mock_lm = MagicMock()
    mock_lm.__or__ = MagicMock(return_value=MagicMock(
        __or__=MagicMock(return_value=MagicMock(invoke=MagicMock(return_value=mock_chain_output)))
    ))
    mock_llm.return_value = mock_lm

    from app.chain import ask
    # Patch the chain directly since LangChain's | operator is complex to mock fully
    with patch("app.chain.ChatPromptTemplate") as mock_prompt:
        mock_prompt.from_messages.return_value.__or__ = MagicMock(
            return_value=MagicMock(__or__=MagicMock(
                return_value=MagicMock(invoke=MagicMock(return_value=mock_chain_output))
            ))
        )
        response = ask("What is the capital of France?")

    assert isinstance(response, RAGResponse)


@patch("app.chain.get_retriever")
def test_ask_no_docs_returns_not_found(mock_retriever):
    """ask() with no retrieved docs should return the 'not found' message."""
    mock_ret = MagicMock()
    mock_ret.invoke.return_value = []
    mock_retriever.return_value = mock_ret

    from app.chain import ask
    response = ask("Something completely unrelated")

    assert "couldn't find" in response.answer.lower()
    assert response.sources == []
