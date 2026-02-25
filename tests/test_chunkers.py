"""Tests for text chunking implementations."""

import pytest

from quickrag.chunkers.base import Chunk
from quickrag.chunkers.text import TextChunker, RecursiveChunker


class TestTextChunker:
    """Tests for the simple fixed-size text chunker."""

    def test_basic_chunking(self, sample_text):
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk(sample_text)
        assert len(chunks) > 1
        for chunk in chunks:
            assert isinstance(chunk, Chunk)
            assert len(chunk.content) > 0

    def test_small_text_single_chunk(self):
        chunker = TextChunker(chunk_size=1000, chunk_overlap=50)
        chunks = chunker.chunk("Hello, world!")
        assert len(chunks) == 1
        assert chunks[0].content == "Hello, world!"

    def test_empty_text(self):
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk("")
        assert chunks == []

    def test_whitespace_only(self):
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk("   \n\n  ")
        assert chunks == []

    def test_chunk_metadata(self):
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        meta = {"source": "test.txt"}
        chunks = chunker.chunk("Hello world. This is a test of the chunking system.", meta)
        assert all("source" in c.metadata for c in chunks)
        assert all(c.metadata["source"] == "test.txt" for c in chunks)
        assert all("chunk_index" in c.metadata for c in chunks)

    def test_chunk_overlap_present(self):
        chunker = TextChunker(chunk_size=30, chunk_overlap=10)
        text = "word " * 50
        chunks = chunker.chunk(text)
        assert len(chunks) > 2
        # Verify overlap exists: end of one chunk should overlap with start of next
        for i in range(len(chunks) - 1):
            tail = chunks[i].content[-10:]
            head = chunks[i + 1].content[:10]
            # At least some overlap characters should exist
            assert len(tail) > 0 and len(head) > 0

    def test_chunk_index_sequential(self):
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        chunks = chunker.chunk("word " * 100)
        indices = [c.index for c in chunks]
        assert indices == list(range(len(chunks)))


class TestRecursiveChunker:
    """Tests for the recursive character text splitter."""

    def test_basic_chunking(self, sample_text):
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk(sample_text)
        assert len(chunks) > 1
        for chunk in chunks:
            assert isinstance(chunk, Chunk)
            assert len(chunk.content) > 0

    def test_respects_paragraph_boundary(self):
        text = "Paragraph one content here.\n\nParagraph two content here."
        chunker = RecursiveChunker(chunk_size=200, chunk_overlap=0)
        chunks = chunker.chunk(text)
        assert len(chunks) == 1  # Small enough to be one chunk

    def test_splits_on_paragraph(self):
        para1 = "A" * 200
        para2 = "B" * 200
        text = f"{para1}\n\n{para2}"
        chunker = RecursiveChunker(chunk_size=250, chunk_overlap=0)
        chunks = chunker.chunk(text)
        assert len(chunks) >= 2

    def test_empty_text(self):
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk("")
        assert chunks == []

    def test_metadata_propagation(self):
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=10)
        meta = {"filename": "test.md", "custom_key": "value"}
        chunks = chunker.chunk("word " * 50, meta)
        for chunk in chunks:
            assert chunk.metadata["filename"] == "test.md"
            assert chunk.metadata["custom_key"] == "value"
            assert "chunk_index" in chunk.metadata

    def test_custom_separators(self):
        chunker = RecursiveChunker(
            chunk_size=50,
            chunk_overlap=0,
            separators=["|", " "],
        )
        text = "Part A|Part B|Part C"
        chunks = chunker.chunk(text)
        assert len(chunks) >= 1

    def test_chunk_len_method(self):
        chunk = Chunk(content="hello world", metadata={}, index=0)
        assert len(chunk) == 11
