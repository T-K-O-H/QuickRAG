"""CSV file loader for QuickRAG."""

import csv
import io
from pathlib import Path
from typing import Any

from quickrag.loaders.base import BaseLoader, LoadedDocument


class CSVLoader(BaseLoader):
    """Loader for CSV files.

    Each row is converted to a text block. You can specify which columns
    to include as content and which to include as metadata.
    """

    EXTENSIONS = {".csv", ".tsv"}

    def __init__(
        self,
        content_columns: list[str] | None = None,
        metadata_columns: list[str] | None = None,
        combine_rows: bool = True,
    ):
        """Initialize CSV loader.

        Args:
            content_columns: Columns to use as content. If None, uses all.
            metadata_columns: Columns to include as metadata.
            combine_rows: If True, combine all rows into a single document.
                          If False, each row becomes a separate document.
        """
        self.content_columns = content_columns
        self.metadata_columns = metadata_columns
        self.combine_rows = combine_rows

    def supports(self, source: str | Path) -> bool:
        """Check if source is a CSV file."""
        path = Path(source)
        return path.suffix.lower() in self.EXTENSIONS

    def _row_to_text(self, row: dict[str, str], columns: list[str] | None) -> str:
        """Convert a CSV row to readable text."""
        cols = columns or list(row.keys())
        parts = []
        for col in cols:
            if col in row and row[col].strip():
                parts.append(f"{col}: {row[col].strip()}")
        return "\n".join(parts)

    def load(self, source: str | Path) -> list[LoadedDocument]:
        """Load a CSV file.

        Args:
            source: Path to the CSV file.

        Returns:
            List of LoadedDocuments.
        """
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
        text = path.read_text(encoding="utf-8")
        reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
        rows = list(reader)

        if not rows:
            return []

        base_metadata: dict[str, Any] = {
            "filename": path.name,
            "extension": path.suffix,
            "size_bytes": path.stat().st_size,
            "row_count": len(rows),
        }

        if self.combine_rows:
            row_texts = [
                self._row_to_text(row, self.content_columns)
                for row in rows
                if self._row_to_text(row, self.content_columns).strip()
            ]
            combined = "\n\n".join(row_texts)
            return [
                LoadedDocument(
                    content=combined,
                    source=str(path.absolute()),
                    metadata=base_metadata,
                )
            ]

        documents = []
        for i, row in enumerate(rows):
            content = self._row_to_text(row, self.content_columns)
            if not content.strip():
                continue

            row_meta = {**base_metadata, "row_index": i}
            if self.metadata_columns:
                for col in self.metadata_columns:
                    if col in row:
                        row_meta[col] = row[col]

            documents.append(
                LoadedDocument(
                    content=content,
                    source=str(path.absolute()),
                    metadata=row_meta,
                )
            )

        return documents
