"""Web page loader for QuickRAG."""

from pathlib import Path
import re

from quickrag.loaders.base import BaseLoader, LoadedDocument


class WebLoader(BaseLoader):
    """Loader for web pages using requests + BeautifulSoup."""

    def supports(self, source: str | Path) -> bool:
        """Check if source is a URL."""
        source_str = str(source)
        return source_str.startswith(("http://", "https://"))

    def load(self, source: str | Path) -> list[LoadedDocument]:
        """Load a web page.

        Args:
            source: URL to load.

        Returns:
            List with single LoadedDocument.
        """
        import httpx
        from bs4 import BeautifulSoup
        from markdownify import markdownify

        url = str(source)

        # Fetch page
        response = httpx.get(url, follow_redirects=True, timeout=30)
        response.raise_for_status()

        # Parse HTML
        soup = BeautifulSoup(response.text, "html.parser")

        # Remove script and style elements
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()

        # Get title
        title = soup.title.string if soup.title else url

        # Get main content (try common content containers)
        main_content = soup.find("main") or soup.find("article") or soup.find("body")

        if main_content:
            # Convert to markdown for cleaner text
            content = markdownify(str(main_content), heading_style="ATX")
        else:
            content = soup.get_text(separator="\n")

        # Clean up whitespace
        content = re.sub(r"\n{3,}", "\n\n", content)
        content = content.strip()

        return [
            LoadedDocument(
                content=content,
                source=url,
                metadata={
                    "title": title,
                    "url": url,
                    "content_type": response.headers.get("content-type", ""),
                },
            )
        ]

