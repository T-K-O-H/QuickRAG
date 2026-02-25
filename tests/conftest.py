"""Shared fixtures for QuickRAG tests."""

import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def tmp_dir():
    """Provide a temporary directory that is cleaned up after the test."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def sample_text():
    """Provide sample text for chunking and loader tests."""
    return (
        "QuickRAG is a fast, plug-and-play Retrieval-Augmented Generation framework. "
        "It supports multiple execution modes including local, cloud, and hybrid. "
        "The framework is built on top of LangGraph for stateful workflows.\n\n"
        "Key features include hybrid search combining semantic and BM25 retrieval, "
        "decorator-based customization, and a beautiful Next.js web interface.\n\n"
        "QuickRAG can ingest documents from various sources such as PDF files, "
        "text files, CSV files, JSON files, DOCX documents, and web pages."
    )


@pytest.fixture
def sample_csv_content():
    """Provide sample CSV content."""
    return "name,description,category\nWidget A,A small widget,tools\nWidget B,A large widget,tools\n"


@pytest.fixture
def sample_json_content():
    """Provide sample JSON content."""
    return '[{"title": "Doc 1", "text": "Hello world"}, {"title": "Doc 2", "text": "Goodbye world"}]'


@pytest.fixture
def sample_jsonl_content():
    """Provide sample JSONL content."""
    return '{"title": "Line 1", "text": "First line"}\n{"title": "Line 2", "text": "Second line"}\n'
