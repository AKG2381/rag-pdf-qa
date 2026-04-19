import hashlib
import logging
from pathlib import Path
from typing import IO

import pdfplumber
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from pypdf import PdfReader

from app.config import settings
from app.retriever import get_embeddings

log = logging.getLogger(__name__)


def _extract(file: IO[bytes], filename: str) -> list[Document]:
    docs = []
    try:
        with pdfplumber.open(file) as pdf:
            total = len(pdf.pages)
            for i, page in enumerate(pdf.pages, 1):
                text = (page.extract_text() or "").strip()
                if text:
                    docs.append(Document(
                        page_content=text,
                        metadata={"source": filename, "page": i, "total_pages": total},
                    ))
    except Exception:
        file.seek(0)
        reader = PdfReader(file)
        for i, page in enumerate(reader.pages, 1):
            text = (page.extract_text() or "").strip()
            if text:
                docs.append(Document(
                    page_content=text,
                    metadata={"source": filename, "page": i, "total_pages": len(reader.pages)},
                ))
    return docs


def _chunk(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    for i, c in enumerate(chunks):
        c.metadata["chunk_index"] = i
    return chunks


def _store(collection: str) -> Chroma:
    return Chroma(
        collection_name=collection,
        embedding_function=get_embeddings(),
        persist_directory=settings.vectorstore_path,
    )


def ingest(file: IO[bytes], filename: str, collection: str = "default") -> dict:
    file.seek(0)
    file_hash = hashlib.md5(file.read()).hexdigest()
    file.seek(0)

    store = _store(collection)
    if store.get(where={"file_hash": file_hash})["ids"]:
        return {"filename": filename, "pages": 0, "chunks": 0, "skipped": True}

    pages = _extract(file, filename)
    if not pages:
        return {"filename": filename, "pages": 0, "chunks": 0, "skipped": True}

    chunks = _chunk(pages)
    for c in chunks:
        c.metadata["file_hash"] = file_hash

    store.add_documents(chunks)
    return {"filename": filename, "pages": len(pages), "chunks": len(chunks), "skipped": False}


def clear_collection(collection: str) -> None:
    import shutil
    # ChromaDB stores each collection inside the persist directory
    # Simplest approach: delete via client
    store = _store(collection)
    store.delete_collection()
