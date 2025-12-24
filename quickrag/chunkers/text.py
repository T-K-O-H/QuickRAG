"""Text chunking implementations for QuickRAG."""

import re
from typing import Any

from quickrag.chunkers.base import BaseChunker, Chunk
from quickrag.config import settings


class TextChunker(BaseChunker):
    """Simple fixed-size text chunker with overlap.

    Fast and predictable. Good for most use cases.
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ):
        """Initialize chunker.

        Args:
            chunk_size: Maximum characters per chunk. Defaults to CHUNK_SIZE.
            chunk_overlap: Characters to overlap between chunks. Defaults to CHUNK_OVERLAP.
        """
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        """Split text into fixed-size chunks with overlap.

        Args:
            text: Text to chunk.
            metadata: Optional metadata to attach to each chunk.

        Returns:
            List of chunks.
        """
        if not text.strip():
            return []

        metadata = metadata or {}
        chunks = []
        start = 0
        index = 0

        while start < len(text):
            # Get chunk
            end = start + self.chunk_size

            # Try to break at word boundary
            if end < len(text):
                # Look for space near the end
                space_pos = text.rfind(" ", start + self.chunk_size // 2, end)
                if space_pos > start:
                    end = space_pos

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunk_metadata = {
                    **metadata,
                    "chunk_index": index,
                    "start_char": start,
                    "end_char": end,
                }
                chunks.append(
                    Chunk(
                        content=chunk_text,
                        metadata=chunk_metadata,
                        index=index,
                    )
                )
                index += 1

            # Move start with overlap
            start = end - self.chunk_overlap
            if start >= len(text) - self.chunk_overlap:
                break

        return chunks


class RecursiveChunker(BaseChunker):
    """Recursive character text splitter.

    Tries to split on natural boundaries (paragraphs, sentences, words)
    before falling back to character splits.
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        separators: list[str] | None = None,
    ):
        """Initialize chunker.

        Args:
            chunk_size: Maximum characters per chunk.
            chunk_overlap: Characters to overlap between chunks.
            separators: List of separators to try, in order of preference.
        """
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.separators = separators or [
            "\n\n",  # Paragraphs
            "\n",  # Lines
            ". ",  # Sentences
            ", ",  # Clauses
            " ",  # Words
            "",  # Characters
        ]

    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        """Split text recursively on natural boundaries.

        Args:
            text: Text to chunk.
            metadata: Optional metadata to attach to each chunk.

        Returns:
            List of chunks.
        """
        if not text.strip():
            return []

        metadata = metadata or {}
        final_chunks = self._split_text(text, self.separators)

        # Convert to Chunk objects
        chunks = []
        for i, chunk_text in enumerate(final_chunks):
            if chunk_text.strip():
                chunk_metadata = {
                    **metadata,
                    "chunk_index": i,
                }
                chunks.append(
                    Chunk(
                        content=chunk_text.strip(),
                        metadata=chunk_metadata,
                        index=i,
                    )
                )

        return chunks

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text."""
        if not separators:
            return [text]

        separator = separators[0]
        remaining_separators = separators[1:]

        # Split on current separator
        if separator:
            splits = text.split(separator)
        else:
            # Character-level split
            splits = list(text)

        # Merge splits that are small enough
        chunks = []
        current_chunk = ""

        for split in splits:
            piece = split + separator if separator and split != splits[-1] else split

            if len(current_chunk) + len(piece) <= self.chunk_size:
                current_chunk += piece
            else:
                if current_chunk:
                    # If current chunk is too big, split further
                    if len(current_chunk) > self.chunk_size and remaining_separators:
                        chunks.extend(self._split_text(current_chunk, remaining_separators))
                    else:
                        chunks.append(current_chunk)

                # Start new chunk, possibly with overlap
                if self.chunk_overlap and chunks:
                    overlap_text = chunks[-1][-self.chunk_overlap :]
                    current_chunk = overlap_text + piece
                else:
                    current_chunk = piece

        # Don't forget the last chunk
        if current_chunk:
            if len(current_chunk) > self.chunk_size and remaining_separators:
                chunks.extend(self._split_text(current_chunk, remaining_separators))
            else:
                chunks.append(current_chunk)

        return chunks

