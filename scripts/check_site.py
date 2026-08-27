"""Validate the dependency-free portfolio landing page before deployment."""

from __future__ import annotations

import re
from html import unescape
from html.parser import HTMLParser
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SITE = _ROOT / "site"
_REQUIRED_IDS = {
    "main-content",
    "top",
    "personas",
    "how-it-works",
    "architecture",
    "engineering",
    "stack",
}
_REQUIRED_TEXT = (
    "Real-Time AI Voice Agent",
    "Call Live Demo",
    "(267) 573-8471",
    "Care Coordinator",
    "Financial Services Assistant",
    "Travel Concierge",
    "History Guide",
    "portfolio demonstration only",
    "Do not provide real medical, financial, authentication, account",
)
_SECRET_PATTERNS = (
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"ASIA[0-9A-Z]{16}"),
    re.compile(r"AC[0-9a-fA-F]{32}"),
    re.compile(r"(?i)(?:auth|secret|access)[_-]?token\s*[=:]\s*['\"][^'\"]+"),
)


class _PageAudit(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.ids: set[str] = set()
        self.duplicate_ids: set[str] = set()
        self.links: list[dict[str, str]] = []
        self.heading_ones = 0
        self.scripts = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = {name: value or "" for name, value in attrs}
        element_id = attributes.get("id")
        if element_id:
            if element_id in self.ids:
                self.duplicate_ids.add(element_id)
            self.ids.add(element_id)
        if tag == "a":
            self.links.append(attributes)
        elif tag == "h1":
            self.heading_ones += 1
        elif tag == "script":
            self.scripts += 1


def _fail(message: str) -> None:
    raise SystemExit(f"site check failed: {message}")


def main() -> None:
    index_path = _SITE / "index.html"
    css_path = _SITE / "styles.css"
    required_files = (
        index_path,
        css_path,
        _SITE / "favicon.svg",
    )
    for required_file in required_files:
        if not required_file.is_file():
            _fail(f"missing {required_file.relative_to(_ROOT)}")

    html = index_path.read_text(encoding="utf-8")
    css = css_path.read_text(encoding="utf-8")
    parser = _PageAudit()
    parser.feed(html)
    parser.close()

    if parser.duplicate_ids:
        _fail(f"duplicate element IDs: {sorted(parser.duplicate_ids)}")
    missing_ids = _REQUIRED_IDS - parser.ids
    if missing_ids:
        _fail(f"missing sections: {sorted(missing_ids)}")
    if parser.heading_ones != 1:
        _fail(f"expected one h1, found {parser.heading_ones}")
    if parser.scripts:
        _fail("JavaScript is not required for this static page")

    internal_targets = {
        link["href"].removeprefix("#")
        for link in parser.links
        if link.get("href", "").startswith("#")
    }
    missing_targets = internal_targets - parser.ids
    if missing_targets:
        _fail(f"links target missing IDs: {sorted(missing_targets)}")

    telephone_links = [link for link in parser.links if link.get("href") == "tel:+12675738471"]
    if len(telephone_links) < 3:
        _fail("the public demo telephone link is not prominent enough")

    for link in parser.links:
        if link.get("target") == "_blank" and "noreferrer" not in link.get("rel", "").split():
            _fail(f"external link lacks noreferrer: {link.get('href', '')}")

    normalized_text = " ".join(unescape(html).replace("\u2011", "-").split())
    for required_text in _REQUIRED_TEXT:
        if required_text.casefold() not in normalized_text.casefold():
            _fail(f"missing required copy: {required_text}")

    public_files = "\n".join(path.read_text(encoding="utf-8") for path in required_files)
    for pattern in _SECRET_PATTERNS:
        if pattern.search(public_files):
            _fail(f"possible credential matched {pattern.pattern}")

    if css.count("{") != css.count("}"):
        _fail("unbalanced CSS blocks")
    if "@import" in css or re.search(r"url\(\s*['\"]?https?://", css):
        _fail("landing page must not depend on remote CSS assets")
    if re.search(r"\banimation\s*:", css) and (
        "prefers-reduced-motion: reduce" not in css or "animation: none" not in css
    ):
        _fail("CSS motion must provide a reduced-motion override")
    if len(html.encode()) > 40_000 or len(css.encode()) > 35_000:
        _fail("static page exceeds the size budget")

    print(
        "site check passed: "
        f"{len(parser.ids)} IDs, {len(parser.links)} links, "
        f"{len(html.encode()) + len(css.encode())} HTML/CSS bytes"
    )


if __name__ == "__main__":
    main()
