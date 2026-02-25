"""Tests for document loaders."""

import json
from pathlib import Path

import pytest

from quickrag.loaders.base import LoadedDocument
from quickrag.loaders.text import TextLoader
from quickrag.loaders.csv_loader import CSVLoader
from quickrag.loaders.json_loader import JSONLoader
from quickrag.loaders.docx_loader import DocxLoader
from quickrag.loaders.auto import AutoLoader, load


class TestTextLoader:
    """Tests for the text file loader."""

    def test_supports_txt(self):
        loader = TextLoader()
        assert loader.supports("file.txt")
        assert loader.supports("file.md")
        assert loader.supports("file.markdown")
        assert loader.supports("file.rst")

    def test_does_not_support_other(self):
        loader = TextLoader()
        assert not loader.supports("file.pdf")
        assert not loader.supports("file.csv")
        assert not loader.supports("https://example.com")

    def test_load_txt_file(self, tmp_dir):
        filepath = tmp_dir / "test.txt"
        filepath.write_text("Hello, world!", encoding="utf-8")

        loader = TextLoader()
        docs = loader.load(filepath)

        assert len(docs) == 1
        assert docs[0].content == "Hello, world!"
        assert docs[0].metadata["filename"] == "test.txt"
        assert docs[0].metadata["extension"] == ".txt"

    def test_load_md_file(self, tmp_dir):
        filepath = tmp_dir / "README.md"
        filepath.write_text("# Title\n\nSome content", encoding="utf-8")

        loader = TextLoader()
        docs = loader.load(filepath)

        assert len(docs) == 1
        assert "# Title" in docs[0].content

    def test_file_not_found(self, tmp_dir):
        loader = TextLoader()
        with pytest.raises(FileNotFoundError):
            loader.load(tmp_dir / "nonexistent.txt")


class TestCSVLoader:
    """Tests for the CSV file loader."""

    def test_supports_csv(self):
        loader = CSVLoader()
        assert loader.supports("data.csv")
        assert loader.supports("data.tsv")
        assert not loader.supports("data.txt")

    def test_load_csv_combined(self, tmp_dir, sample_csv_content):
        filepath = tmp_dir / "test.csv"
        filepath.write_text(sample_csv_content, encoding="utf-8")

        loader = CSVLoader(combine_rows=True)
        docs = loader.load(filepath)

        assert len(docs) == 1
        assert "Widget A" in docs[0].content
        assert "Widget B" in docs[0].content

    def test_load_csv_per_row(self, tmp_dir, sample_csv_content):
        filepath = tmp_dir / "test.csv"
        filepath.write_text(sample_csv_content, encoding="utf-8")

        loader = CSVLoader(combine_rows=False)
        docs = loader.load(filepath)

        assert len(docs) == 2
        assert "Widget A" in docs[0].content
        assert "Widget B" in docs[1].content

    def test_content_columns(self, tmp_dir, sample_csv_content):
        filepath = tmp_dir / "test.csv"
        filepath.write_text(sample_csv_content, encoding="utf-8")

        loader = CSVLoader(content_columns=["name"], combine_rows=False)
        docs = loader.load(filepath)

        assert len(docs) == 2
        assert "description" not in docs[0].content.lower().split(":")[0]

    def test_metadata_columns(self, tmp_dir, sample_csv_content):
        filepath = tmp_dir / "test.csv"
        filepath.write_text(sample_csv_content, encoding="utf-8")

        loader = CSVLoader(metadata_columns=["category"], combine_rows=False)
        docs = loader.load(filepath)

        assert docs[0].metadata.get("category") == "tools"

    def test_empty_csv(self, tmp_dir):
        filepath = tmp_dir / "empty.csv"
        filepath.write_text("col1,col2\n", encoding="utf-8")

        loader = CSVLoader()
        docs = loader.load(filepath)
        assert docs == []

    def test_tsv_file(self, tmp_dir):
        filepath = tmp_dir / "test.tsv"
        filepath.write_text("name\tdesc\nAlpha\tFirst\n", encoding="utf-8")

        loader = CSVLoader(combine_rows=True)
        docs = loader.load(filepath)
        assert len(docs) == 1
        assert "Alpha" in docs[0].content


class TestJSONLoader:
    """Tests for the JSON file loader."""

    def test_supports_json(self):
        loader = JSONLoader()
        assert loader.supports("data.json")
        assert loader.supports("data.jsonl")
        assert not loader.supports("data.csv")

    def test_load_json_array(self, tmp_dir, sample_json_content):
        filepath = tmp_dir / "test.json"
        filepath.write_text(sample_json_content, encoding="utf-8")

        loader = JSONLoader()
        docs = loader.load(filepath)

        assert len(docs) == 2
        assert "Hello world" in docs[0].content

    def test_load_json_with_content_key(self, tmp_dir, sample_json_content):
        filepath = tmp_dir / "test.json"
        filepath.write_text(sample_json_content, encoding="utf-8")

        loader = JSONLoader(content_key="text")
        docs = loader.load(filepath)

        assert len(docs) == 2
        assert docs[0].content == "Hello world"
        assert docs[1].content == "Goodbye world"

    def test_load_json_with_metadata_keys(self, tmp_dir, sample_json_content):
        filepath = tmp_dir / "test.json"
        filepath.write_text(sample_json_content, encoding="utf-8")

        loader = JSONLoader(content_key="text", metadata_keys=["title"])
        docs = loader.load(filepath)

        assert docs[0].metadata["title"] == "Doc 1"

    def test_load_single_object(self, tmp_dir):
        filepath = tmp_dir / "single.json"
        filepath.write_text('{"key": "value", "data": "content"}', encoding="utf-8")

        loader = JSONLoader()
        docs = loader.load(filepath)
        assert len(docs) == 1

    def test_load_jsonl(self, tmp_dir, sample_jsonl_content):
        filepath = tmp_dir / "test.jsonl"
        filepath.write_text(sample_jsonl_content, encoding="utf-8")

        loader = JSONLoader(content_key="text")
        docs = loader.load(filepath)

        assert len(docs) == 2
        assert docs[0].content == "First line"
        assert docs[1].content == "Second line"

    def test_file_not_found(self, tmp_dir):
        loader = JSONLoader()
        with pytest.raises(FileNotFoundError):
            loader.load(tmp_dir / "nonexistent.json")


class TestDocxLoader:
    """Tests for the DOCX file loader."""

    def test_supports_docx(self):
        loader = DocxLoader()
        assert loader.supports("doc.docx")
        assert not loader.supports("doc.doc")
        assert not loader.supports("doc.pdf")

    def test_file_not_found(self, tmp_dir):
        loader = DocxLoader()
        with pytest.raises(FileNotFoundError):
            loader.load(tmp_dir / "nonexistent.docx")


class TestAutoLoader:
    """Tests for the auto-detecting document loader."""

    def test_supports_text(self):
        loader = AutoLoader()
        assert loader.supports("file.txt")
        assert loader.supports("file.md")
        assert loader.supports("file.csv")
        assert loader.supports("file.json")
        assert loader.supports("file.jsonl")
        assert loader.supports("file.docx")
        assert loader.supports("file.pdf")

    def test_supports_url(self):
        loader = AutoLoader()
        assert loader.supports("https://example.com")
        assert loader.supports("http://example.com")

    def test_load_directory(self, tmp_dir):
        (tmp_dir / "a.txt").write_text("File A", encoding="utf-8")
        (tmp_dir / "b.md").write_text("File B", encoding="utf-8")

        loader = AutoLoader()
        docs = loader.load(tmp_dir)
        assert len(docs) == 2

    def test_load_function_single(self, tmp_dir):
        filepath = tmp_dir / "test.txt"
        filepath.write_text("Test content", encoding="utf-8")

        docs = load(str(filepath))
        assert len(docs) == 1
        assert docs[0].content == "Test content"

    def test_load_function_list(self, tmp_dir):
        f1 = tmp_dir / "a.txt"
        f2 = tmp_dir / "b.txt"
        f1.write_text("A", encoding="utf-8")
        f2.write_text("B", encoding="utf-8")

        docs = load([str(f1), str(f2)])
        assert len(docs) == 2

    def test_unsupported_file(self, tmp_dir):
        filepath = tmp_dir / "file.xyz"
        filepath.write_text("data", encoding="utf-8")

        loader = AutoLoader()
        with pytest.raises(ValueError, match="No loader found"):
            loader.load(filepath)
