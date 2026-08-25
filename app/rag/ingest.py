"""Fetch and extract meaningful article content from the Startup Science webpage.

This module is intentionally strict: if the page cannot be fetched or the
expected article structure cannot be found, it raises instead of silently
falling back to something else.
"""

from dataclasses import dataclass, field

import requests
from bs4 import BeautifulSoup

from app.config import SOURCE_URL

REQUEST_TIMEOUT_SECONDS = 15
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

# Tags that carry the actual prose/list content we want to keep.
CONTENT_TAGS = {"p", "li", "h2", "h3", "h4"}


@dataclass
class Section:
    """One heading-delimited section of the article."""

    heading: str
    level: str  # "h2" or "h3"
    paragraphs: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(self.paragraphs)


@dataclass
class ArticleContent:
    source_url: str
    title: str
    sections: list[Section]


class FetchError(RuntimeError):
    """Raised when the webpage cannot be fetched."""


class ExtractionError(RuntimeError):
    """Raised when the expected article structure cannot be found on the page."""


def fetch_html(url: str = SOURCE_URL) -> str:
    """Fetch raw HTML for the given URL. Raises FetchError on any failure."""
    try:
        response = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        raise FetchError(f"Failed to reach {url}: {exc}") from exc

    if response.status_code != 200:
        raise FetchError(
            f"Unexpected HTTP status {response.status_code} when fetching {url}"
        )

    if not response.text or len(response.text) < 500:
        raise FetchError(
            f"Response from {url} was empty or suspiciously short "
            f"({len(response.text)} chars)."
        )

    return response.text


def extract_article(html: str, source_url: str = SOURCE_URL) -> ArticleContent:
    """Extract the title and heading-delimited body sections from the article HTML.

    Raises ExtractionError if the known article container / title element is
    missing, rather than guessing at a fallback container.
    """
    soup = BeautifulSoup(html, "html.parser")

    title_el = soup.select_one("h1.title-article-2026") or soup.select_one("h1")
    if title_el is None or not title_el.get_text(strip=True):
        raise ExtractionError(
            "Could not find the article title (expected <h1> on the page). "
            "The page structure may have changed."
        )
    title = title_el.get_text(strip=True)

    content_el = soup.select_one("div.article-content")
    if content_el is None:
        raise ExtractionError(
            "Could not find the main article content container "
            "(expected <div class=\"article-content\">). "
            "The page structure may have changed."
        )

    sections: list[Section] = []
    current = Section(heading=title, level="h1")

    for el in content_el.find_all(list(CONTENT_TAGS)):
        text = el.get_text(" ", strip=True)
        if not text:
            continue

        if el.name in ("h2", "h3"):
            if current.paragraphs:
                sections.append(current)
            current = Section(heading=text, level=el.name)
        else:
            current.paragraphs.append(text)

    if current.paragraphs:
        sections.append(current)

    if not sections:
        raise ExtractionError(
            "Article content container was found but contained no "
            "extractable paragraph/list text."
        )

    return ArticleContent(source_url=source_url, title=title, sections=sections)


def load_article(url: str = SOURCE_URL) -> ArticleContent:
    """Convenience wrapper: fetch + extract in one call."""
    html = fetch_html(url)
    return extract_article(html, source_url=url)
