"""JSON file loader for QuickRAG."""

import json
from pathlib import Path
from typing import Any

from quickrag.loaders.base import BaseLoader, LoadedDocument


class JSONLoader(BaseLoader):
    """Loader for JSON and JSONL files.

    Supports:
    - Single JSON objects
    - JSON arrays of objects
    - JSONL (one JSON object per line)

    You can specify a jq-style path to extract content from nested structures.
    """

    EXTENSIONS = {".json", ".jsonl"}

    def __init__(
        self,
        content_key: str | None = None,
        metadata_keys: list[str] | None = None,
        text_key: str | None = None,
    ):
        """Initialize JSON loader.

        Args:
            content_key: Key path to extract content (e.g. "text", "body", "content").
                         If None, serializes the entire object.
            metadata_keys: Keys to extract as metadata.
            text_key: Alias for content_key (for compatibility).
        """
        self.content_key = content_key or text_key
        self.metadata_keys = metadata_keys

    def supports(self, source: str | Path) -> bool:
        """Check if source is a JSON file."""
        path = Path(source)
        return path.suffix.lower() in self.EXTENSIONS

    def _extract_content(self, obj: Any) -> str:
        """Extract text content from a JSON object."""
        if self.content_key and isinstance(obj, dict):
            value = obj.get(self.content_key, "")
            if isinstance(value, str):
                return value
            return json.dumps(value, indent=2, ensure_ascii=False)

        if isinstance(obj, str):
            return obj

        if isinstance(obj, dict):
            parts = []
            for key, value in obj.items():
                if isinstance(value, str):
                    parts.append(f"{key}: {value}")
                else:
                    parts.append(f"{key}: {json.dumps(value, ensure_ascii=False)}")
            return "\n".join(parts)

        return json.dumps(obj, indent=2, ensure_ascii=False)

    def _extract_metadata(self, obj: Any) -> dict[str, Any]:
        """Extract metadata from a JSON object."""
        if not self.metadata_keys or not isinstance(obj, dict):
            return {}
        return {k: obj[k] for k in self.metadata_keys if k in obj}

    def load(self, source: str | Path) -> list[LoadedDocument]:
        """Load a JSON or JSONL file.

        Args:
            source: Path to the JSON file.

        Returns:
            List of LoadedDocuments.
        """
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        text = path.read_text(encoding="utf-8")
        base_metadata: dict[str, Any] = {
            "filename": path.name,
            "extension": path.suffix,
            "size_bytes": path.stat().st_size,
        }

        # Handle JSONL
        if path.suffix.lower() == ".jsonl":
            return self._load_jsonl(text, str(path.absolute()), base_metadata)

        data = json.loads(text)

        # Handle JSON array
        if isinstance(data, list):
            documents = []
            for i, item in enumerate(data):
                content = self._extract_content(item)
                if content.strip():
                    meta = {**base_metadata, "array_index": i, **self._extract_metadata(item)}
                    documents.append(
                        LoadedDocument(
                            content=content,
                            source=str(path.absolute()),
                            metadata=meta,
                        )
                    )
            return documents

        # Handle single object
        content = self._extract_content(data)
        meta = {**base_metadata, **self._extract_metadata(data)}
        return [
            LoadedDocument(
                content=content,
                source=str(path.absolute()),
                metadata=meta,
            )
        ]

    def _load_jsonl(
        self, text: str, source: str, base_metadata: dict[str, Any]
    ) -> list[LoadedDocument]:
        """Load JSONL format (one JSON object per line)."""
        documents = []
        for i, line in enumerate(text.strip().splitlines()):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            content = self._extract_content(obj)
            if content.strip():
                meta = {**base_metadata, "line_index": i, **self._extract_metadata(obj)}
                documents.append(
                    LoadedDocument(
                        content=content,
                        source=source,
                        metadata=meta,
                    )
                )
        return documents
