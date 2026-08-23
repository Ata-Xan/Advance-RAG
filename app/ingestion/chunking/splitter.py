from __future__ import annotations
from typing import List
import logfire
from collections.abc import Sequence


from langchain_text_splitters import RecursiveCharacterTextSplitter


DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_SEPARATORS = ("\n\n", "\n", ". ", " ", "")


def build_text_splitter(
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    separators: Sequence[str] | None = None,
) -> RecursiveCharacterTextSplitter:
    """
    Create the recursive splitter used during ingestion.

    Overlap helps preserve context between adjacent chunks, which usually improves
    retrieval for ideas that span chunk boundaries.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=list(separators or DEFAULT_SEPARATORS),
    )


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    separators: Sequence[str] | None = None,
) -> list[str]:
    """
    Split raw text into overlapping chunks for downstream embedding/retrieval.
    """
    with logfire.span("✂️ Text overlapping Chunking", text_length=len(text)):
        if not text or not text.strip():
            return []

        splitter = build_text_splitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
        )
        chunks = splitter.split_text(text.strip())
        logfire.info(f"✅ Generated {len(chunks)} chunks")
    return splitter.split_text(text.strip())