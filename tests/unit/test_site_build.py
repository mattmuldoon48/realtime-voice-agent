"""Deployment assets must not reuse cache keys after their contents change."""

import runpy
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit

build_site = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts" / "build_site.py"))[
    "build_site"
]


class _AssetLinks(HTMLParser):
    def __init__(self, html: str) -> None:
        super().__init__()
        self.links: dict[str, str] = {}
        self.feed(html)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "link":
            attributes = dict(attrs)
            self.links[str(attributes["rel"])] = str(attributes["href"])


def test_changed_stylesheet_cannot_reuse_previous_cache_entry(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "public"
    source.mkdir()
    (source / "index.html").write_text(
        '<link rel="stylesheet" href="./styles.css"><link rel="icon" href="./favicon.svg">',
        encoding="utf-8",
    )
    (source / "styles.css").write_text("body { background: black; }", encoding="utf-8")
    (source / "favicon.svg").write_text("<svg></svg>", encoding="utf-8")
    build_site(source, output)
    first = _AssetLinks((output / "index.html").read_text(encoding="utf-8")).links
    old_cache = {first["stylesheet"]: (output / "styles.css").read_text(encoding="utf-8")}

    (source / "styles.css").write_text("body { background: ivory; }", encoding="utf-8")
    build_site(source, output)
    second = _AssetLinks((output / "index.html").read_text(encoding="utf-8")).links

    assert second["stylesheet"] not in old_cache
    assert second["icon"] == first["icon"]
    asset_path = output / urlsplit(second["stylesheet"]).path
    assert asset_path.read_text(encoding="utf-8") == "body { background: ivory; }"

    build_site(source, output)
    repeated = _AssetLinks((output / "index.html").read_text(encoding="utf-8")).links
    assert repeated == second
