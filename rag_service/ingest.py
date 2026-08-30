"""Fetch and extract article content from the configured source webpage.

Ported from the original app/rag/ingest.py, unchanged in behaviour: strict by
design - if the page cannot be fetched or the expected structure is missing,
it raises instead of silently falling back.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import requests
from bs4 import BeautifulSoup

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

CONTENT_TAGS = {"p", "li", "h2", "h3", "h4"}


@dataclass
class Section:
    heading: str
    level: str
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
    """The webpage could not be fetched."""


class ExtractionError(RuntimeError):
    """The expected article structure could not be found on the page."""


def fetch_html(url: str, *, timeout_seconds: int = 15) -> str:
    try:
        response = requests.get(
            url, headers={"User-Agent": USER_AGENT}, timeout=timeout_seconds
        )
    except requests.RequestException as exc:
        raise FetchError(f"Failed to reach {url}: {exc}") from exc

    if response.status_code != 200:
        raise FetchError(f"Unexpected HTTP status {response.status_code} when fetching {url}")

    if not response.text or len(response.text) < 500:
        raise FetchError(
            f"Response from {url} was empty or suspiciously short ({len(response.text)} chars)."
        )

    return response.text


def extract_article(html: str, source_url: str) -> ArticleContent:
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
            'Could not find the main article content container '
            '(expected <div class="article-content">). The page structure may have changed.'
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
            "Article content container was found but contained no extractable text."
        )

    return ArticleContent(source_url=source_url, title=title, sections=sections)


def load_article(url: str, *, timeout_seconds: int = 15) -> ArticleContent:
    return extract_article(fetch_html(url, timeout_seconds=timeout_seconds), source_url=url)
